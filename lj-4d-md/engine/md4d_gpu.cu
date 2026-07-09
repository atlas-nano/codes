/* md4d_gpu.cu  —  Single-GPU CUDA port of the 4D Lennard-Jones MD engine (md4d.c).
 *
 * Same CLI and physics as md4d.c: force-shifted LJ at rc, NVT Nose-Hoover chain
 * (length 3, MTK), PBC minimum image, optional Berendsen barostat / Frenkel-Ladd
 * springs / g(r).  FP64 throughout to match the CPU engine (validate via E_cons).
 * O(N^2) force loop = 1 thread per atom (full force, no Newton-3rd; U,virial halved).
 *
 * Build:  nvcc -O3 -arch=sm_89 -o md4d_gpu md4d_gpu.cu
 * Usage:  ./md4d_gpu n=6 rho=1.0 T=4.0 ...   (identical keys to md4d)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#define DIM 4
#define TPB 128
#define CK(call) do{ cudaError_t e=(call); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

__constant__ int    d_N;
__constant__ double d_L, d_Linv, d_rc, d_rc2, d_fshift, d_ushift, d_fllam;

/* per-atom force, potential (full pair sum), virial (full), + FL spring */
__global__ void k_forces(const double*x,const double*xref,double*f,double*u,double*w,double*flW){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=d_N) return;
  double xi[DIM]; for(int k=0;k<DIM;k++) xi[k]=x[i*DIM+k];
  double fi[DIM]={0,0,0,0}, ui=0.0, wi=0.0;
  for(int j=0;j<d_N;j++){ if(j==i) continue;
    double d2=0.0, dr[DIM];
    for(int k=0;k<DIM;k++){ double s=xi[k]-x[j*DIM+k]; s-=d_L*rint(s*d_Linv); dr[k]=s; d2+=s*s; }
    if(d2>=d_rc2) continue;
    double r2i=1.0/d2, r6i=r2i*r2i*r2i, r12i=r6i*r6i, r=sqrt(d2);
    double fpair=(24.0*(2.0*r12i-r6i))*r2i - d_fshift/r;       // (F/r), force-shifted
    ui += 4.0*(r12i-r6i) - d_ushift + (r-d_rc)*d_fshift;        // force-shifted potential
    for(int k=0;k<DIM;k++) fi[k]+=fpair*dr[k];
    wi += fpair*d2;
  }
  double flw=0.0;
  if(d_fllam>0.0)
    for(int k=0;k<DIM;k++){ double dr=xi[k]-xref[i*DIM+k]; dr-=d_L*rint(dr*d_Linv);
      fi[k]+=-2.0*d_fllam*dr; flw+=dr*dr; }
  for(int k=0;k<DIM;k++) f[i*DIM+k]=fi[k];
  u[i]=ui; w[i]=wi; flW[i]=flw;
}
__global__ void k_kick (double*v,const double*f,double a,int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) v[i]+=a*f[i];}
__global__ void k_drift(double*x,const double*v,double dt,int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]+=dt*v[i];}
__global__ void k_wrap (double*x,double L,double Li,int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]-=L*floor(x[i]*Li);}
__global__ void k_scale(double*v,double s,int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) v[i]*=s;}
__global__ void k_sq   (const double*v,double*o,int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=v[i]*v[i];}
__global__ void k_smul (double*x,double mu,int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]*=mu;}
__global__ void k_gr(const double*x,unsigned long long*h,double L,double Li,double rm2,double idr,int nb,int N){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
  for(int j=i+1;j<N;j++){ double d2=0.0;
    for(int k=0;k<DIM;k++){ double s=x[i*DIM+k]-x[j*DIM+k]; s-=L*rint(s*Li); d2+=s*s; }
    if(d2<rm2){ int b=(int)(sqrt(d2)*idr); if(b<nb) atomicAdd(&h[b],1ULL); } }
}

/* globals (host) */
static int N, Ndof, nblk;
static double L, Linv, rho, Tset, dt, rc, rc2, ushift, fshift;
static double *dx,*dv,*df,*dxref,*du,*dw,*dflW,*dtmp;
#define MNHC 3
static double xi[MNHC], vxi[MNHC], Qm[MNHC];

static double dsum(double*dptr){ thrust::device_ptr<double> p(dptr); return thrust::reduce(p,p+N*DIM,0.0);} // for v^2 over N*DIM
static double dsumN(double*dptr){ thrust::device_ptr<double> p(dptr); return thrust::reduce(p,p+N,0.0);}      // per-atom arrays

static double kinetic(){ k_sq<<<(N*DIM+TPB-1)/TPB,TPB>>>(dv,dtmp,N*DIM); return 0.5*dsum(dtmp);}
static void forces(){ k_forces<<<nblk,TPB>>>(dx,dxref,df,du,dw,dflW); }
static double Upot(){ return 0.5*dsumN(du);}        // full pair sum halved
static double virial(){ return 0.5*dsumN(dw);}

static void nhc_half(){
  double dt2=0.5*dt,dt4=0.25*dt,dt8=0.125*dt;
  double KE2=2.0*kinetic(), G[MNHC];
  G[0]=(KE2-Ndof*Tset)/Qm[0];
  for(int m=1;m<MNHC;m++) G[m]=(Qm[m-1]*vxi[m-1]*vxi[m-1]-Tset)/Qm[m];
  vxi[MNHC-1]+=G[MNHC-1]*dt4;
  for(int m=MNHC-2;m>=0;m--){ double ef=exp(-dt8*vxi[m+1]); vxi[m]=vxi[m]*ef*ef+G[m]*dt4*ef; }
  double scale=exp(-dt2*vxi[0]);
  k_scale<<<(N*DIM+TPB-1)/TPB,TPB>>>(dv,scale,N*DIM);
  for(int m=0;m<MNHC;m++) xi[m]+=dt2*vxi[m];
  KE2*=scale*scale; G[0]=(KE2-Ndof*Tset)/Qm[0];
  for(int m=0;m<MNHC-1;m++){ double ef=exp(-dt8*vxi[m+1]); vxi[m]=vxi[m]*ef*ef+G[m]*dt4*ef;
    G[m+1]=(Qm[m]*vxi[m]*vxi[m]-Tset)/Qm[m+1]; }
  vxi[MNHC-1]+=G[MNHC-1]*dt4;
}
static double nhc_energy(){ double e=Ndof*Tset*xi[0]; for(int m=1;m<MNHC;m++) e+=Tset*xi[m];
  for(int m=0;m<MNHC;m++) e+=0.5*Qm[m]*vxi[m]*vxi[m]; return e; }

/* deterministic RNG + lattices (host) */
static double uniform(){ static unsigned long long s=88172645463325252ULL; s^=s<<13;s^=s>>7;s^=s<<17; return (s>>11)*(1.0/9007199254740992.0);}
static double gauss(){ double u1=uniform(),u2=uniform(); if(u1<1e-300)u1=1e-300; return sqrt(-2.0*log(u1))*cos(2.0*M_PI*u2);}
static double getf(int c,char**v,const char*k,double def){size_t kl=strlen(k);
  for(int i=1;i<c;i++) if(!strncmp(v[i],k,kl)&&v[i][kl]=='=') return atof(v[i]+kl+1); return def;}
static void gets_(int c,char**v,const char*k,char*o,const char*def){size_t kl=strlen(k);
  for(int i=1;i<c;i++) if(!strncmp(v[i],k,kl)&&v[i][kl]=='='){strcpy(o,v[i]+kl+1);return;} strcpy(o,def);}

int main(int argc,char**argv){
  int nside=(int)getf(argc,argv,"n",6);
  rho=getf(argc,argv,"rho",1.0); Tset=getf(argc,argv,"T",4.0);
  dt=getf(argc,argv,"dt",0.00371); rc=getf(argc,argv,"rc",2.5);
  long nequil=(long)getf(argc,argv,"nequil",8000), nprod=(long)getf(argc,argv,"nprod",16000);
  int thermo_pr=(int)getf(argc,argv,"thermoevery",500), velstride=(int)getf(argc,argv,"velstride",1);
  int thermostat=(int)getf(argc,argv,"thermostat",1), prodtherm=(int)getf(argc,argv,"prodthermostat",1);
  double tdamp=getf(argc,argv,"tdamp",0.10);
  int baro_on=(int)getf(argc,argv,"barostat",0); double Pset=getf(argc,argv,"Pset",0.0);
  double pdamp=getf(argc,argv,"pdamp",2.0), pkappa=getf(argc,argv,"pkappa",0.02);
  int gr_every=(int)getf(argc,argv,"grevery",0), gr_nbins=(int)getf(argc,argv,"grbins",400);
  double gr_rmax=getf(argc,argv,"grrmax",0.0); double fl_lambda=getf(argc,argv,"fllambda",0.0);
  char prefix[512]; gets_(argc,argv,"prefix",prefix,"run");
  char lattice[32]; gets_(argc,argv,"lattice",lattice,"hc"); int is_d4=(strcmp(lattice,"d4")==0);

  N=1; for(int k=0;k<DIM;k++) N*=nside; if(is_d4) N/=2;
  L=pow((double)N/rho,1.0/DIM); Linv=1.0/L; rc2=rc*rc; Ndof=DIM*N-DIM; nblk=(N+TPB-1)/TPB;
  if(L<=2.0*rc){ fprintf(stderr,"ERROR L=%.3f<=2rc=%.3f\n",L,2*rc); return 1; }
  { double r2i=1.0/rc2,r6i=r2i*r2i*r2i,r12i=r6i*r6i; ushift=4.0*(r12i-r6i); fshift=(24.0*(2.0*r12i-r6i))/rc; }

  double*hx=(double*)malloc(sizeof(double)*N*DIM),*hv=(double*)malloc(sizeof(double)*N*DIM);
  // lattice
  { double a=L/nside; int idx=0;
    for(int i0=0;i0<nside;i0++)for(int i1=0;i1<nside;i1++)for(int i2=0;i2<nside;i2++)for(int i3=0;i3<nside;i3++){
      if(is_d4){ if((i0+i1+i2+i3)&1) continue; hx[idx*DIM+0]=i0*a;hx[idx*DIM+1]=i1*a;hx[idx*DIM+2]=i2*a;hx[idx*DIM+3]=i3*a; }
      else { hx[idx*DIM+0]=(i0+0.5)*a;hx[idx*DIM+1]=(i1+0.5)*a;hx[idx*DIM+2]=(i2+0.5)*a;hx[idx*DIM+3]=(i3+0.5)*a; }
      idx++; } }
  // MB velocities, remove COM, rescale to Tset
  for(int i=0;i<N*DIM;i++) hv[i]=gauss();
  { double p[DIM]={0}; for(int i=0;i<N;i++)for(int k=0;k<DIM;k++)p[k]+=hv[i*DIM+k];
    for(int k=0;k<DIM;k++)p[k]/=N; for(int i=0;i<N;i++)for(int k=0;k<DIM;k++)hv[i*DIM+k]-=p[k];
    double ke=0; for(int i=0;i<N*DIM;i++)ke+=hv[i]*hv[i]; double s=sqrt(Tset/(ke/Ndof)); for(int i=0;i<N*DIM;i++)hv[i]*=s; }

  CK(cudaMalloc(&dx,sizeof(double)*N*DIM)); CK(cudaMalloc(&dv,sizeof(double)*N*DIM));
  CK(cudaMalloc(&df,sizeof(double)*N*DIM)); CK(cudaMalloc(&dxref,sizeof(double)*N*DIM));
  CK(cudaMalloc(&du,sizeof(double)*N)); CK(cudaMalloc(&dw,sizeof(double)*N));
  CK(cudaMalloc(&dflW,sizeof(double)*N)); CK(cudaMalloc(&dtmp,sizeof(double)*N*DIM));
  CK(cudaMemcpy(dx,hx,sizeof(double)*N*DIM,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dxref,hx,sizeof(double)*N*DIM,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dv,hv,sizeof(double)*N*DIM,cudaMemcpyHostToDevice));
  CK(cudaMemcpyToSymbol(d_N,&N,sizeof(int)));
  CK(cudaMemcpyToSymbol(d_rc,&rc,sizeof(double))); CK(cudaMemcpyToSymbol(d_rc2,&rc2,sizeof(double)));
  CK(cudaMemcpyToSymbol(d_ushift,&ushift,sizeof(double))); CK(cudaMemcpyToSymbol(d_fshift,&fshift,sizeof(double)));
  CK(cudaMemcpyToSymbol(d_fllam,&fl_lambda,sizeof(double)));
  auto setLcuda=[&](){ CK(cudaMemcpyToSymbol(d_L,&L,sizeof(double))); CK(cudaMemcpyToSymbol(d_Linv,&Linv,sizeof(double))); };
  setLcuda();

  Qm[0]=Ndof*Tset*tdamp*tdamp; for(int m=1;m<MNHC;m++)Qm[m]=Tset*tdamp*tdamp;
  for(int m=0;m<MNHC;m++){xi[m]=0;vxi[m]=0;}
  forces(); CK(cudaDeviceSynchronize());

  char fn[600]; snprintf(fn,sizeof fn,"%s.thermo",prefix); FILE*fth=fopen(fn,"w");
  fprintf(fth,"# 4D LJ MD (GPU) N=%d rho=%.5f L=%.5f T=%.4f dt=%.6f rc=%.3f thermostat=%d Ndof=%d\n",N,rho,L,Tset,dt,rc,thermostat,Ndof);
  fprintf(fth,"# phase step time T P Upot Econs\n");
  snprintf(fn,sizeof fn,"%s.vel",prefix); FILE*fv=NULL;
  unsigned long long*dhist=NULL; double gr_dr=0; long gr_samp=0; double fl_Wacc=0; long fl_Wcnt=0;
  float*hbuf=(float*)malloc(sizeof(float)*N*DIM);

  long step=0;
  for(int phase=0;phase<2;phase++){
    long nsteps=(phase==0)?nequil:nprod;
    int use_thermo=thermostat && ((phase==0)?1:prodtherm);
    int baro_phase=baro_on && (phase==0);
    double Lsum=0; long Lcnt=0;
    if(phase==1&&velstride>0) fv=fopen(fn,"wb");
    if(phase==1&&gr_every>0){ if(gr_rmax<=0||gr_rmax>0.5*L)gr_rmax=0.5*L; gr_dr=gr_rmax/gr_nbins;
      CK(cudaMalloc(&dhist,sizeof(unsigned long long)*gr_nbins)); CK(cudaMemset(dhist,0,sizeof(unsigned long long)*gr_nbins)); }
    for(long s=0;s<nsteps;s++,step++){
      if(use_thermo) nhc_half();
      k_kick<<<(N*DIM+TPB-1)/TPB,TPB>>>(dv,df,0.5*dt,N*DIM);
      k_drift<<<(N*DIM+TPB-1)/TPB,TPB>>>(dx,dv,dt,N*DIM);
      k_wrap<<<(N*DIM+TPB-1)/TPB,TPB>>>(dx,L,Linv,N*DIM);
      forces();
      k_kick<<<(N*DIM+TPB-1)/TPB,TPB>>>(dv,df,0.5*dt,N*DIM);
      if(use_thermo) nhc_half();
      if(baro_phase){
        double T=2.0*kinetic()/Ndof, W=virial(), V=pow(L,DIM), Pinst=(N*T+W/DIM)/V;
        double mu=pow(1.0-pkappa*(dt/pdamp)*(Pset-Pinst),1.0/DIM); if(mu<0.98)mu=0.98; if(mu>1.02)mu=1.02;
        L*=mu; Linv=1.0/L; setLcuda(); k_smul<<<(N*DIM+TPB-1)/TPB,TPB>>>(dx,mu,N*DIM); forces();
        if(s>=nsteps/2){ Lsum+=L; Lcnt++; }
      }
      if(phase==1&&fv&&(s%velstride)==0){
        CK(cudaMemcpy(hv,dv,sizeof(double)*N*DIM,cudaMemcpyDeviceToHost));
        for(int i=0;i<N*DIM;i++) hbuf[i]=(float)hv[i]; fwrite(hbuf,sizeof(float),N*DIM,fv);
      }
      if(phase==1&&dhist&&(s%gr_every)==0){ k_gr<<<nblk,TPB>>>(dx,dhist,L,Linv,gr_rmax*gr_rmax,1.0/gr_dr,gr_nbins,N); gr_samp++; }
      if(phase==1&&fl_lambda>0.0){ fl_Wacc+=dsumN(dflW); fl_Wcnt++; }
      if((step%thermo_pr)==0){
        double ke=kinetic(), T=2.0*ke/Ndof, W=virial(), U=Upot(), V=pow(L,DIM), P=(N*T+W/DIM)/V;
        double Ec=ke+U+(use_thermo?nhc_energy():0.0);
        fprintf(fth,"%d %ld %.6f %.6f %.6f %.8f %.8f\n",phase,step,step*dt,T,P,U,Ec); fflush(fth);
      }
    }
    if(phase==0&&baro_on&&Lcnt>0){ double Lm=Lsum/Lcnt,mu=Lm/L; L=Lm;Linv=1.0/L; setLcuda();
      k_smul<<<(N*DIM+TPB-1)/TPB,TPB>>>(dx,mu,N*DIM); rho=(double)N/pow(L,DIM); forces();
      fprintf(stderr,"[barostat] L=%.5f rho*=%.5f\n",L,rho); }
    if(phase==0) fprintf(stderr,"[equil done] step=%ld\n",step);
  }
  if(fv)fclose(fv);
  if(dhist){ unsigned long long*hh=(unsigned long long*)malloc(sizeof(unsigned long long)*gr_nbins);
    CK(cudaMemcpy(hh,dhist,sizeof(unsigned long long)*gr_nbins,cudaMemcpyDeviceToHost));
    snprintf(fn,sizeof fn,"%s.gr",prefix); FILE*fg=fopen(fn,"w");
    fprintf(fg,"# r g(r) (4D rho=%.6f samples=%ld rmax=%.4f)\n",rho,gr_samp,gr_rmax); double hp=0.5*M_PI*M_PI;
    for(int b=0;b<gr_nbins;b++){ double r0=b*gr_dr,r1=(b+1)*gr_dr,rc_=(b+0.5)*gr_dr;
      double shell=hp*(r1*r1*r1*r1-r0*r0*r0*r0), ideal=(double)N*0.5*rho*shell;
      fprintf(fg,"%.5f %.6f\n",rc_,(double)hh[b]/((double)gr_samp*ideal)); } fclose(fg); free(hh); }
  fclose(fth);
  snprintf(fn,sizeof fn,"%s.meta",prefix); FILE*fm=fopen(fn,"w");
  fprintf(fm,"dim=%d\nN=%d\nnside=%d\nrho_reduced=%.8f\nL_reduced=%.8f\nTset=%.8f\ndt_reduced=%.8f\nrc=%.4f\n"
             "thermostat=%d\nprodthermostat=%d\ntdamp=%.5f\nnequil=%ld\nnprod=%ld\nvelstride=%d\nNdof=%d\n"
             "barostat=%d\nPset=%.5f\nfl_lambda=%.8g\nfl_W_mean=%.10g\nfl_W_samples=%ld\n"
             "sigma_A=3.405\nepsilon_K=119.78\nmass_amu=39.948\n",
             DIM,N,nside,rho,L,Tset,dt,rc,thermostat,prodtherm,tdamp,nequil,nprod,velstride,Ndof,
             baro_on,Pset,fl_lambda,(fl_Wcnt>0?fl_Wacc/fl_Wcnt:0.0),fl_Wcnt);
  fclose(fm);
  fprintf(stderr,"[done GPU] N=%d rho=%.4f T=%.4f (%ld prod frames stride %d)\n",N,rho,Tset,(velstride>0?nprod/velstride:0),velstride);
  return 0;
}
