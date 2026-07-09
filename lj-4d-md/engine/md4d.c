/* md4d.c  —  Standalone 4-dimensional Lennard-Jones (12-6) molecular dynamics.
 *
 * Purpose: generate equilibrium 4D-LJ liquid trajectories and per-step velocities
 * for the 3PT cage-entropy p = 1/d cross-dimensional (d = 4) test.
 *
 * Physics / conventions
 *   - DIM = 4 fixed at compile time (arrays are [N][4]); reduced LJ units throughout
 *     (sigma = epsilon = mass = 1).  Reduced time unit tau = sigma*sqrt(m/eps).
 *     Map to argon in post-processing (sigma=3.405 A, eps/kB=119.78 K, m=39.948).
 *   - LJ 12-6 with a *force-shifted* cutoff at rc:  both U and F are continuous and
 *     F(rc)=0, so the inverted memory kernel is free of a cutoff force discontinuity.
 *       u_sf(r) = 4(r^-12 - r^-6) - u(rc) - (r-rc)*u'(rc)
 *       f_sf(r) = f(r) - f(rc)            (f = -du/dr)
 *   - Ensemble: NVT via a Nose-Hoover chain (length M=3, Martyna-Tuckerman-Klein
 *     reversible propagator) OR NVE (thermostat off) for production if requested.
 *     A *deterministic* thermostat is used on purpose: a stochastic (Langevin)
 *     thermostat would corrupt the velocity memory kernel that 3PT analyses.
 *   - PBC, minimum image, brute-force O(N^2) pair loop (robust for N up to a few
 *     thousand; OpenMP-parallel).  Requires box L > 2 rc (checked at startup).
 *
 * Output
 *   - <prefix>.thermo : step time T P U_pot E_cons  (whitespace columns, header '#')
 *   - <prefix>.vel    : binary float32 velocities, production phase only.
 *                       Layout per frame: N*DIM float32 (atom-major: a0x a0y a0z a0w a1x ...).
 *                       Frames written every `velstride` production steps.
 *   - <prefix>.meta   : key=value run metadata for the analysis pipeline.
 *
 * Build:  gcc -O3 -march=native -fopenmp -o md4d md4d.c -lm
 * Usage:  ./md4d key=val ...   (see defaults in main(); e.g. n=6 rho=0.60 T=1.6)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define DIM 4

/* ----------------------------- globals -------------------------------------- */
static int     N;             /* number of atoms                                */
static double  L, Linv;       /* box length (cube) and 1/L                       */
static double  rho;           /* reduced number density N/L^DIM                  */
static double  Tset;          /* target reduced temperature                      */
static double  dt;            /* reduced timestep                                */
static double  rc, rc2;       /* cutoff and cutoff^2                             */
static double  ushift, fshift;/* force-shift constants u(rc), -du/dr|rc          */
static int     thermostat;    /* 1 = NHC on, 0 = NVE                             */
static double  tdamp;         /* thermostat time constant (reduced)              */
static int     Ndof;          /* degrees of freedom = DIM*N - DIM (COM removed)  */

static double *x, *v, *f;     /* [N*DIM] positions, velocities, forces           */
static double *x_ref;         /* [N*DIM] Frenkel-Ladd reference lattice sites     */

/* Nose-Hoover chain (length 3) */
#define MNHC 3
static double xi[MNHC], vxi[MNHC], Qm[MNHC];

/* ----------------------------- utilities ------------------------------------ */
static double rng_state = 0.0;
static double uniform(void){           /* simple deterministic LCG in [0,1)      */
    /* xorshift64-ish on a 64-bit seed held in a double's bits is fragile; use
       a plain 64-bit LCG via unsigned long long. */
    static unsigned long long s = 88172645463325252ULL;
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return (s >> 11) * (1.0/9007199254740992.0);
}
static double gauss(void){             /* Box-Muller standard normal             */
    double u1 = uniform(), u2 = uniform();
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0*log(u1)) * cos(2.0*M_PI*u2);
}

static double getf(int argc, char**argv, const char*key, double def){
    size_t kl = strlen(key);
    for (int i=1;i<argc;i++)
        if (!strncmp(argv[i],key,kl) && argv[i][kl]=='=') return atof(argv[i]+kl+1);
    (void)rng_state;
    return def;
}
static void gets_(int argc, char**argv, const char*key, char*out, const char*def){
    size_t kl = strlen(key);
    for (int i=1;i<argc;i++)
        if (!strncmp(argv[i],key,kl) && argv[i][kl]=='='){ strcpy(out,argv[i]+kl+1); return; }
    strcpy(out,def);
}

/* ----------------------------- forces --------------------------------------- */
/* Compute forces and potential energy; returns U_pot (force-shifted). */
static double forces(void){
    for (int i=0;i<N*DIM;i++) f[i]=0.0;
    double U = 0.0;
    #pragma omp parallel
    {
        double *floc = calloc(N*DIM, sizeof(double));
        double Uloc = 0.0;
        #pragma omp for schedule(dynamic,16)
        for (int i=0;i<N;i++){
            const double *xi_ = x + i*DIM;
            for (int j=i+1;j<N;j++){
                const double *xj = x + j*DIM;
                double d2 = 0.0, dr[DIM];
                for (int k=0;k<DIM;k++){
                    double s = xi_[k]-xj[k];
                    s -= L*rint(s*Linv);        /* minimum image */
                    dr[k]=s; d2 += s*s;
                }
                if (d2 >= rc2) continue;
                double r2i = 1.0/d2;
                double r6i = r2i*r2i*r2i;
                double r12i= r6i*r6i;
                /* u(r) = 4(r^-12 - r^-6); f_mag = -du/dr = 24(2 r^-12 - r^-6)/r */
                double r = sqrt(d2);
                double fmag = (24.0*(2.0*r12i - r6i))*r2i;   /* (-du/dr)/r        */
                /* force-shift: subtract f(rc): fmag_eff = fmag - fshift/r        */
                double fpair = fmag - fshift/r;              /* this is (F/r)-ish */
                /* potential (force-shifted) */
                Uloc += 4.0*(r12i - r6i) - ushift - (r-rc)*(-fshift);
                for (int k=0;k<DIM;k++){
                    double comp = fpair*dr[k];
                    floc[i*DIM+k] += comp;
                    floc[j*DIM+k] -= comp;
                }
            }
        }
        #pragma omp critical
        {
            for (int i=0;i<N*DIM;i++) f[i]+=floc[i];
            U += Uloc;
        }
        free(floc);
    }
    return U;
}

/* virial (sum r·F over pairs) for pressure; recompute cheaply alongside if needed */
static double virial(void){
    double W = 0.0;
    #pragma omp parallel for reduction(+:W) schedule(dynamic,16)
    for (int i=0;i<N;i++){
        const double *xi_ = x + i*DIM;
        for (int j=i+1;j<N;j++){
            const double *xj = x + j*DIM;
            double d2=0.0, dr[DIM];
            for (int k=0;k<DIM;k++){ double s=xi_[k]-xj[k]; s-=L*rint(s*Linv); dr[k]=s; d2+=s*s; }
            if (d2>=rc2) continue;
            double r2i=1.0/d2, r6i=r2i*r2i*r2i, r12i=r6i*r6i;
            double r=sqrt(d2);
            double fmag=(24.0*(2.0*r12i - r6i))*r2i;
            double fpair=fmag - fshift/r;       /* (F/r) */
            W += fpair*d2;                      /* r·F = (F/r)*r^2 */
        }
    }
    return W;
}

/* ----------------------------- kinetic / COM -------------------------------- */
static double kinetic(void){
    double ke=0.0;
    for (int i=0;i<N*DIM;i++) ke += v[i]*v[i];
    return 0.5*ke;          /* unit mass */
}
static void remove_com_velocity(void){
    double p[DIM]={0};
    for (int i=0;i<N;i++) for(int k=0;k<DIM;k++) p[k]+=v[i*DIM+k];
    for (int k=0;k<DIM;k++) p[k]/=N;
    for (int i=0;i<N;i++) for(int k=0;k<DIM;k++) v[i*DIM+k]-=p[k];
}

/* ----------------------------- NHC thermostat ------------------------------- */
/* Single Nose-Hoover chain of length MNHC, MTK reversible update over dt/2.
 * Scales velocities; updates chain variables.  No Suzuki-Yoshida/multistep
 * (nc=1,nys=1): adequate for gentle thermostatting; extended energy tracked. */
static void nhc_half(void){
    double dt2=0.5*dt, dt4=0.25*dt, dt8=0.125*dt;
    double KE2 = 2.0*kinetic();                 /* sum m v^2                      */
    double G[MNHC];
    G[0] = (KE2 - Ndof*Tset)/Qm[0];
    for (int m=1;m<MNHC;m++) G[m] = (Qm[m-1]*vxi[m-1]*vxi[m-1] - Tset)/Qm[m];
    /* update chain velocities from the tail inward */
    vxi[MNHC-1] += G[MNHC-1]*dt4;
    for (int m=MNHC-2;m>=0;m--){
        double ef = exp(-dt8*vxi[m+1]);
        vxi[m] = vxi[m]*ef*ef + G[m]*dt4*ef;
    }
    /* scale particle velocities */
    double scale = exp(-dt2*vxi[0]);
    for (int i=0;i<N*DIM;i++) v[i]*=scale;
    /* advance chain positions */
    for (int m=0;m<MNHC;m++) xi[m]+=dt2*vxi[m];
    /* recompute G[0] with scaled KE, update chain velocities outward */
    KE2 *= scale*scale;
    G[0] = (KE2 - Ndof*Tset)/Qm[0];
    for (int m=0;m<MNHC-1;m++){
        double ef = exp(-dt8*vxi[m+1]);
        vxi[m] = vxi[m]*ef*ef + G[m]*dt4*ef;
        G[m+1] = (Qm[m]*vxi[m]*vxi[m] - Tset)/Qm[m+1];
    }
    vxi[MNHC-1] += G[MNHC-1]*dt4;
}
/* conserved-quantity contribution of the chain (for E_cons diagnostic) */
static double nhc_energy(void){
    double e = Ndof*Tset*xi[0];
    for (int m=1;m<MNHC;m++) e += Tset*xi[m];
    for (int m=0;m<MNHC;m++) e += 0.5*Qm[m]*vxi[m]*vxi[m];
    return e;
}

/* ----------------------------- init ----------------------------------------- */
static void init_lattice(int nside){
    /* simple hypercubic lattice of nside^4 sites, jittered slightly */
    double a = L/nside;
    int idx=0;
    for (int i0=0;i0<nside;i0++)
    for (int i1=0;i1<nside;i1++)
    for (int i2=0;i2<nside;i2++)
    for (int i3=0;i3<nside;i3++){
        x[idx*DIM+0]=(i0+0.5)*a; x[idx*DIM+1]=(i1+0.5)*a;
        x[idx*DIM+2]=(i2+0.5)*a; x[idx*DIM+3]=(i3+0.5)*a;
        idx++;
    }
}
/* D4 checkerboard lattice (4D close-packed; kissing number 24): hypercubic grid
   sites with even coordinate-index sum.  N = nside^4 / 2.  Stable 4D LJ crystal. */
static void init_lattice_d4(int nside){
    double a = L/nside;
    int idx=0;
    for (int i0=0;i0<nside;i0++)
    for (int i1=0;i1<nside;i1++)
    for (int i2=0;i2<nside;i2++)
    for (int i3=0;i3<nside;i3++){
        if (((i0+i1+i2+i3) & 1) != 0) continue;   /* keep even-sum sites */
        x[idx*DIM+0]=i0*a; x[idx*DIM+1]=i1*a;
        x[idx*DIM+2]=i2*a; x[idx*DIM+3]=i3*a;
        idx++;
    }
}
static void init_velocities(void){
    for (int i=0;i<N*DIM;i++) v[i]=gauss();
    remove_com_velocity();
    /* rescale to exactly Tset */
    double T = 2.0*kinetic()/Ndof;
    double s = sqrt(Tset/T);
    for (int i=0;i<N*DIM;i++) v[i]*=s;
}

/* ----------------------------- main ----------------------------------------- */
int main(int argc, char**argv){
    int    nside   = (int)getf(argc,argv,"n",6);        /* N = nside^4            */
    rho            = getf(argc,argv,"rho",0.60);
    Tset           = getf(argc,argv,"T",1.6);
    dt             = getf(argc,argv,"dt",0.00371);      /* ~8 fs in argon units   */
    rc             = getf(argc,argv,"rc",3.0);
    long nequil    = (long)getf(argc,argv,"nequil",50000);
    long nprod     = (long)getf(argc,argv,"nprod",31250);
    int  thermo_pr = (int)getf(argc,argv,"thermoevery",500);
    int  velstride = (int)getf(argc,argv,"velstride",1);
    thermostat     = (int)getf(argc,argv,"thermostat",1);   /* 1 NVT, 0 NVE       */
    int  prodtherm = (int)getf(argc,argv,"prodthermostat",1);/* thermostat in prod*/
    tdamp          = getf(argc,argv,"tdamp",0.10);      /* reduced; ~ paper-ish   */
    int  baro_on   = (int)getf(argc,argv,"barostat",0); /* 1: Berendsen P-couple in equil */
    double Pset    = getf(argc,argv,"Pset",0.0);        /* target reduced pressure */
    double pdamp   = getf(argc,argv,"pdamp",2.0);       /* barostat relax time (reduced) */
    double pkappa  = getf(argc,argv,"pkappa",0.02);     /* compressibility coupling const */
    int  gr_every  = (int)getf(argc,argv,"grevery",0);  /* >0: accumulate g(r) every N prod steps */
    int  gr_nbins  = (int)getf(argc,argv,"grbins",400);
    double gr_rmax = getf(argc,argv,"grrmax",0.0);       /* 0 -> L/2 at runtime */
    double fl_lambda = getf(argc,argv,"fllambda",0.0);   /* Frenkel-Ladd spring strength (>0 on) */
    char prefix[512]; gets_(argc,argv,"prefix",prefix,"run");
    char lattice[32]; gets_(argc,argv,"lattice",lattice,"hc");   /* "hc" or "d4" */
    int is_d4 = (strcmp(lattice,"d4")==0);

    N = 1; for(int k=0;k<DIM;k++) N*=nside;
    if (is_d4) N/=2;                          /* D4 = even-sum sites of the hc grid */
    L = pow((double)N/rho, 1.0/DIM); Linv=1.0/L;
    rc2 = rc*rc;
    Ndof = DIM*N - DIM;

    if (L <= 2.0*rc){
        fprintf(stderr,"ERROR: box L=%.3f <= 2*rc=%.3f (minimum image violated). "
                       "Increase n or rc.\n", L, 2.0*rc); return 1;
    }
    /* force-shift constants at rc */
    {
        double r2i=1.0/rc2, r6i=r2i*r2i*r2i, r12i=r6i*r6i;
        ushift = 4.0*(r12i - r6i);
        fshift = (24.0*(2.0*r12i - r6i))/rc;   /* = -du/dr at rc = F(rc) magnitude */
    }
    x=malloc(sizeof(double)*N*DIM); v=malloc(sizeof(double)*N*DIM); f=malloc(sizeof(double)*N*DIM);

    /* thermostat masses Q_m = (Ndof or 1)*T*tdamp^2 */
    Qm[0]=Ndof*Tset*tdamp*tdamp;
    for (int m=1;m<MNHC;m++) Qm[m]=Tset*tdamp*tdamp;
    for (int m=0;m<MNHC;m++){ xi[m]=0.0; vxi[m]=0.0; }

    if (is_d4) init_lattice_d4(nside); else init_lattice(nside);
    x_ref=malloc(sizeof(double)*N*DIM);            /* Frenkel-Ladd reference sites */
    for (int i=0;i<N*DIM;i++) x_ref[i]=x[i];
    init_velocities();
    double U = forces();

    FILE *flog=fopen("/dev/stdout","w");
    char fn[600];
    snprintf(fn,sizeof fn,"%s.thermo",prefix); FILE *fth=fopen(fn,"w");
    fprintf(fth,"# 4D LJ MD  N=%d nside=%d rho=%.5f L=%.5f Tset=%.4f dt=%.6f rc=%.3f "
                "thermostat=%d tdamp=%.4f Ndof=%d\n",N,nside,rho,L,Tset,dt,rc,thermostat,tdamp,Ndof);
    fprintf(fth,"# phase step time T P Upot Econs\n");

    /* .meta is written AFTER the run (below) so rho/L reflect the barostat-
       equilibrated production density. */
    snprintf(fn,sizeof fn,"%s.vel",prefix); FILE *fv=NULL;

    /* g(r) accumulator (production phase) */
    long *grhist=NULL; double gr_dr=0.0; long gr_samples=0;
    /* gr_rmax / gr_dr set after the (possible) barostat equilibration, since L may change */

    double Pid_kin; /* pressure ideal part uses instantaneous T */
    long step=0;
    int use_thermo;

    /* ---- run a phase: equilibration then production ---- */
    double Lsum=0.0; long Lcnt=0;          /* mean box length over 2nd half of equil */
    double fl_Wacc=0.0; long fl_Wcnt=0;    /* Frenkel-Ladd <sum |dr|^2> accumulator   */
    for (int phase=0; phase<2; phase++){
        long nsteps = (phase==0)? nequil : nprod;
        use_thermo = thermostat && ( (phase==0) ? 1 : prodtherm );
        int baro_phase = baro_on && (phase==0);   /* couple pressure only in equil */
        if (phase==1 && velstride>0){ fv=fopen(fn,"wb"); }
        if (phase==1 && gr_every>0){               /* set up g(r) after equilibration */
            if (gr_rmax<=0.0) gr_rmax=0.5*L;
            if (gr_rmax>0.5*L) gr_rmax=0.5*L;
            gr_dr=gr_rmax/gr_nbins;
            grhist=calloc(gr_nbins,sizeof(long));
        }
        for (long s=0;s<nsteps;s++,step++){
            if (use_thermo) nhc_half();
            /* velocity Verlet: v += f*dt/2; x += v*dt; recompute f; v += f*dt/2 */
            for (int i=0;i<N*DIM;i++) v[i]+=0.5*dt*f[i];
            for (int i=0;i<N*DIM;i++){ x[i]+=dt*v[i]; }
            /* wrap into box [0,L) */
            for (int i=0;i<N*DIM;i++){ x[i]-=L*floor(x[i]*Linv); }
            U=forces();
            /* Frenkel-Ladd harmonic tether: U += lambda*sum|dr|^2, F += -2 lambda dr */
            if (fl_lambda>0.0){
                double W=0.0;
                for (int i=0;i<N;i++){
                    for (int k=0;k<DIM;k++){
                        double dr=x[i*DIM+k]-x_ref[i*DIM+k];
                        dr-=L*rint(dr*Linv);
                        f[i*DIM+k] += -2.0*fl_lambda*dr;
                        W += dr*dr;
                    }
                }
                U += fl_lambda*W;
                if (phase==1){ fl_Wacc+=W; fl_Wcnt++; }   /* <W> over production */
            }
            for (int i=0;i<N*DIM;i++) v[i]+=0.5*dt*f[i];
            if (use_thermo) nhc_half();

            /* Berendsen barostat (equilibration only): scale box+coords toward Pset */
            if (baro_phase){
                double ke=kinetic(); double Tinst=2.0*ke/Ndof; double W=virial();
                double V=pow(L,DIM); double Pinst=(N*Tinst + W/DIM)/V;
                double mu=pow(1.0 - pkappa*(dt/pdamp)*(Pset - Pinst), 1.0/DIM);
                if (mu<0.98) mu=0.98; if (mu>1.02) mu=1.02;   /* clamp per-step scaling */
                L*=mu; Linv=1.0/L;
                for (int i=0;i<N*DIM;i++) x[i]*=mu;
                U=forces();
                if (s >= nsteps/2){ Lsum+=L; Lcnt++; }        /* average late equil */
            }

            /* dump production velocities */
            if (phase==1 && fv && velstride>0 && (s % velstride)==0){
                float *buf=malloc(sizeof(float)*N*DIM);
                for (int i=0;i<N*DIM;i++) buf[i]=(float)v[i];
                fwrite(buf,sizeof(float),N*DIM,fv);
                free(buf);
            }
            /* accumulate g(r) (production only, every gr_every steps) */
            if (phase==1 && grhist && (s % gr_every)==0){
                double rmax2=gr_rmax*gr_rmax;
                #pragma omp parallel
                {
                    long *hloc=calloc(gr_nbins,sizeof(long));
                    #pragma omp for schedule(dynamic,16)
                    for (int i=0;i<N;i++){
                        const double *xi_=x+i*DIM;
                        for (int j=i+1;j<N;j++){
                            const double *xj=x+j*DIM; double d2=0.0;
                            for (int k=0;k<DIM;k++){ double s2=xi_[k]-xj[k]; s2-=L*rint(s2*Linv); d2+=s2*s2; }
                            if (d2<rmax2){ int b=(int)(sqrt(d2)/gr_dr); if(b<gr_nbins) hloc[b]++; }
                        }
                    }
                    #pragma omp critical
                    { for(int b=0;b<gr_nbins;b++) grhist[b]+=hloc[b]; }
                    free(hloc);
                }
                gr_samples++;
            }
            if ((step % thermo_pr)==0){
                double ke=kinetic();
                double T=2.0*ke/Ndof;
                double W=virial();
                /* P = rho*T + virial/(DIM*V) ; V=L^DIM ; rho=N/V */
                double V=pow(L,DIM);
                double P = (N*T + W/DIM)/V;
                double Econs = ke+U + (use_thermo? nhc_energy():0.0);
                fprintf(fth,"%d %ld %.6f %.6f %.6f %.8f %.8f\n",
                        phase, step, step*dt, T, P, U, Econs);
                fflush(fth);
                (void)Pid_kin;
            }
        }
        if (phase==0){
            if (baro_on && Lcnt>0){
                double Lmean=Lsum/Lcnt; double mu=Lmean/L;
                L=Lmean; Linv=1.0/L;
                for (int i=0;i<N*DIM;i++) x[i]*=mu;
                rho=(double)N/pow(L,DIM);
                U=forces();
                fprintf(stderr,"[barostat] equilibrated to L=%.5f rho*=%.5f (Pset=%.4f)\n",
                        L,rho,Pset);
                if (L<=2.0*rc) fprintf(stderr,"WARNING: L=%.3f <= 2*rc after barostat\n",L);
            }
            fprintf(stderr,"[equil done] step=%ld\n",step);
        }
    }
    if (fv) fclose(fv);
    /* write g(r): g(r)=<hist>/[(N/2) rho shell_vol], 4D shell_vol=(pi^2/2)((r+dr)^4-r^4) */
    if (grhist && gr_samples>0){
        snprintf(fn,sizeof fn,"%s.gr",prefix); FILE *fg=fopen(fn,"w");
        fprintf(fg,"# r  g(r)   (4D, rho=%.6f, samples=%ld, rmax=%.4f)\n",rho,gr_samples,gr_rmax);
        double half_pi2=0.5*M_PI*M_PI;
        for (int b=0;b<gr_nbins;b++){
            double r0=b*gr_dr, r1=(b+1)*gr_dr, rc_=(b+0.5)*gr_dr;
            double shell=half_pi2*(r1*r1*r1*r1 - r0*r0*r0*r0);
            double ideal=(double)N*0.5*rho*shell;
            double g=(double)grhist[b]/((double)gr_samples*ideal);
            fprintf(fg,"%.5f %.6f\n",rc_,g);
        }
        fclose(fg); free(grhist);
    }
    fclose(fth);
    /* write .meta now: rho/L reflect the (possibly barostat-equilibrated) production state */
    snprintf(fn,sizeof fn,"%s.meta",prefix); FILE *fm=fopen(fn,"w");
    fprintf(fm,"dim=%d\nN=%d\nnside=%d\nrho_reduced=%.8f\nL_reduced=%.8f\nTset=%.8f\n"
               "dt_reduced=%.8f\nrc=%.4f\nthermostat=%d\nprodthermostat=%d\ntdamp=%.5f\n"
               "nequil=%ld\nnprod=%ld\nvelstride=%d\nNdof=%d\nbarostat=%d\nPset=%.5f\n"
               "fl_lambda=%.8g\nfl_W_mean=%.10g\nfl_W_samples=%ld\n"
               "# argon mapping for unit conversion in analysis:\n"
               "sigma_A=3.405\nepsilon_K=119.78\nmass_amu=39.948\n",
               DIM,N,nside,rho,L,Tset,dt,rc,thermostat,prodtherm,tdamp,nequil,nprod,velstride,Ndof,
               baro_on,Pset, fl_lambda, (fl_Wcnt>0? fl_Wacc/fl_Wcnt:0.0), fl_Wcnt);
    fclose(fm);
    if (flog && flog!=stdout) fclose(flog);
    fprintf(stderr,"[done] N=%d rho=%.4f T=%.4f wrote %s.vel (%ld prod frames, stride %d)\n",
            N,rho,Tset, prefix, (velstride>0? nprod/velstride:0), velstride);
    free(x);free(v);free(f);free(x_ref);
    return 0;
}
