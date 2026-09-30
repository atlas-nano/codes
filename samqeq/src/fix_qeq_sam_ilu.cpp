// clang-format off
/* ----------------------------------------------------------------------
   samQEq (fix qeq/sam): BLOCK-2x2 ILUT preconditioner for the ACKS2 saddle.
   Methods are members of class FixQEqSam.

   The metal limit makes the saddle ill-conditioned -> the diagonal (Jacobi)
   preconditioner needs thousands of BiCGStab iters and is proc-sensitive on large metals
   (e.g. benzene/Au). An ILU of the saddle converges in a few iters, scale-independent. SCALAR ILU(0) is unstable on the large indefinite saddle; this
   uses BLOCK-2x2 ILUT (per-atom [s_i,u_i]) whose diagonal blocks [[eta+ridge,1],[1,X_diag]]
   are nonsingular (det = (eta+ridge)*X_diag - 1 != 0) -> stable, no scalar pivoting.

   PARALLEL = CENTRALIZED / REPLICATED. Per-proc block-Jacobi (additive Schwarz)
   DROPS the cross-proc Au-Au couplings that make the metal near-singular -> BiCGStab
   breaks down at nprocs>1; a piecewise-constant per-proc coarse correction (two-level
   Schwarz) is too weak (it still breaks down). Instead every rank assembles the
   FULL global saddle (gather each proc's H/X couplings by global atom tag), factors the
   EXACT serial block-ILUT, and applies it to the GLOBALLY-gathered residual. The solve is
   replicated and identical on every rank -> the preconditioner is decomposition-independent
   by construction (proc-stable to the np=1 answer) and converges like serial. The fine
   matvec/forces/kspace stay distributed; only the (cheap, ~2N) preconditioner is replicated
   -> ideal for the interfacial metal systems here (thousands of atoms, <=tens of ranks),
   not for >>1e4-atom metals. Enable with fix_modify <id> precond ilu [drop_tol].

   LAYOUT: BiCGStab vectors use NN=nlocal+nghost: s in [0,NN), u in [NN,2NN), constraints
   at [2NN],[2NN+1]. The ILU factors the GLOBAL block system (N=natoms atoms x 2 dof),
   indexed by global tag g=tag-1. Block vector index: 2*g = s_g, 2*g+1 = u_g. A bordered
   2x2 Schur handles the two global constraints (Sum s=0, Sum u=0).
-------------------------------------------------------------------------*/

#include "fix_qeq_sam.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "memory.h"
#include "update.h"

#include <cmath>
#include <vector>
#include <map>
#include <array>

using namespace LAMMPS_NS;

// 2x2 helpers (row-major [a00,a01,a10,a11])
static inline void mm22(const double *A, const double *B, double *C) {
  C[0]=A[0]*B[0]+A[1]*B[2]; C[1]=A[0]*B[1]+A[1]*B[3];
  C[2]=A[2]*B[0]+A[3]*B[2]; C[3]=A[2]*B[1]+A[3]*B[3];
}
static inline void inv22(const double *A, double *Ai) {
  double det = A[0]*A[3]-A[1]*A[2];
  if (std::fabs(det) < 1e-300) det = (det>=0?1e-300:-1e-300);
  double id = 1.0/det;
  Ai[0]= A[3]*id; Ai[1]=-A[1]*id; Ai[2]=-A[2]*id; Ai[3]= A[0]*id;
}

/* ----------------------------------------------------------------------*/

void FixQEqSam::ilu_free()
{
  memory->destroy(ilu_rp);   ilu_rp = nullptr;
  memory->destroy(ilu_ci);   ilu_ci = nullptr;
  memory->destroy(ilu_dptr); ilu_dptr = nullptr;
  memory->destroy(ilu_lu);   ilu_lu = nullptr;
  memory->destroy(ilu_Dinv); ilu_Dinv = nullptr;
  memory->destroy(ilu_Y0);   ilu_Y0 = nullptr;
  memory->destroy(ilu_Y1);   ilu_Y1 = nullptr;
  memory->destroy(ilu_scr);  ilu_scr = nullptr;
  memory->destroy(ilu_w);    ilu_w = nullptr;
  ilu_n = ilu_nnz = 0;
  ilu_valid = 0;             // freed factor ⇒ force a rebuild on the next solve
}

/* ----------------------------------------------------------------------
   Assemble the GLOBAL block saddle (replicated on every rank), factor block-ILUT,
   precompute pivots + bordered constraint Schur. Each rank emits its local atoms'
   diagonal + H/X half-list couplings keyed by global tag; an Allgatherv unions them
   (the half-list stores each physical pair on exactly ONE rank -> no double count),
   then every rank factors the identical global system.
-------------------------------------------------------------------------*/
void FixQEqSam::ilu_build()
{
  // CENTRALIZED preconditioner needs globally-unique, contiguous tags (g = tag-1 in [0,N)).
  if (!atom->tag_enable || !atom->tag_consecutive())
    error->all(FLERR, "fix qeq/sam: precond ilu requires consecutive atom IDs (atom_modify id yes)");
  if (atom->natoms > MAXSMALLINT)
    error->all(FLERR, "fix qeq/sam: precond ilu (centralized) too large: {} atoms", atom->natoms);

  const int n = (int) atom->natoms;     // GLOBAL block count (replicated factor)
  const int nl = atom->nlocal;
  int *type = atom->type;
  int *mask = atom->mask;               // needed to skip non-group rows (see the emit loop)
  tagint *tag = atom->tag;

  // --- emit this rank's contributions: (gi, gj, comp, value), comp 0 = s-s block, 3 = u-u block.
  //     diagonals are owner-unique; off-diagonals come from the H/X half-list (each pair once
  //     globally) and are emitted SYMMETRICALLY (gi,gj)+(gj,gi). The s-u identity (blocks [1],[2])
  //     is added locally during accumulation, not communicated. ---
  std::vector<double> sb;
  sb.reserve((size_t)nl * 32);          // ~diagonal + a handful of couplings per atom (grows as needed)
  auto emit = [&](int gi, int gj, int comp, double v) {
    sb.push_back((double)gi); sb.push_back((double)gj); sb.push_back((double)comp); sb.push_back(v);
  };
  for (int i = 0; i < nl; i++) {
    // H/X rows EXIST ONLY FOR GROUP ATOMS. compute_H (fix_qeq_sam.cpp) and compute_X
    // write firstnbr/numnbrs under `mask[i] & groupbit`; allocate_matrix uses memory->create (malloc,
    // not calloc), so for a non-group atom those entries are uninitialized and the inner loops below
    // would walk a wild jlist range. The guard matters whenever the group is a subset of all atoms.
    // precond_apply guards the same way (below).
    if (!(mask[i] & groupbit)) continue;
    int gi = (int)tag[i] - 1;
    emit(gi, gi, 0, eta[type[i]] + lr_ridge + ilu_shift);   // s-s diagonal (+ optional Manteuffel shift)
    emit(gi, gi, 3, X_diag[i] - ilu_shift);                 // u-u diagonal
    for (int p = H.firstnbr[i]; p < H.firstnbr[i]+H.numnbrs[i]; p++) {
      int gj = (int)tag[H.jlist[p]] - 1; double v = H.val[p];
      if (gj == gi) emit(gi, gi, 0, v);                     // self-periodic-image -> s-s diagonal
      else { emit(gi, gj, 0, v); emit(gj, gi, 0, v); }
    }
    for (int p = X.firstnbr[i]; p < X.firstnbr[i]+X.numnbrs[i]; p++) {
      int gj = (int)tag[X.jlist[p]] - 1; double v = X.val[p];
      if (gj == gi) emit(gi, gi, 3, v);
      else { emit(gi, gj, 3, v); emit(gj, gi, 3, v); }
    }
  }

  // --- Allgatherv the contributions to every rank ---
  const int nprocs = comm->nprocs;
  int myn = (int) sb.size();
  std::vector<int> counts(nprocs), displs(nprocs);
  MPI_Allgather(&myn, 1, MPI_INT, counts.data(), 1, MPI_INT, world);
  long tot = 0; for (int p = 0; p < nprocs; p++) { displs[p] = (int)tot; tot += counts[p]; }
  std::vector<double> ab((size_t)tot);
  MPI_Allgatherv(sb.data(), myn, MPI_DOUBLE, ab.data(), counts.data(), displs.data(), MPI_DOUBLE, world);

  // --- accumulate into the global per-atom block adjacency ---
  std::vector<std::map<int, std::array<double,4>>> blk(n);
  for (int i = 0; i < n; i++) { auto &dii = blk[i][i]; dii[1] = 1.0; dii[2] = 1.0; }  // s-u identity
  // Mark which global rows are group rows. Each rank knows only its own atoms' masks, but with
  // the guard above every group atom (and only a group atom) emits its own diagonals, and every
  // emitted column is group-guarded too -- so the gathered stream identifies the group set exactly.
  std::vector<char> ingrp(n, 0);
  for (long e = 0; e+3 < tot; e += 4) {
    int gi = (int)ab[e], gj = (int)ab[e+1], comp = (int)ab[e+2];
    blk[gi][gj][comp] += ab[e+3];
    ingrp[gi] = 1;
  }

  // --- block-ILUT(drop_tol): per-row L (col<i) + U (col>=i); fill kept above a relative
  //     drop tolerance. Pivots from U_ii^{-1} stored in ilu_Dinv. (Serial kernel, applied
  //     over the N global block rows.) ---
  ilu_free();
  ilu_n = n;
  memory->create(ilu_Dinv, 4*n, "samqeq:ilu_Dinv");
  std::vector<std::vector<std::pair<int,std::array<double,4>>>> Lrows(n), Urows(n);
  auto frob = [](const double *b){ return std::sqrt(b[0]*b[0]+b[1]*b[1]+b[2]*b[2]+b[3]*b[3]); };
  for (int i = 0; i < n; i++) {
    std::map<int, std::array<double,4>> w = blk[i];         // working row = A row i
    double rn = 0.0; for (auto &kv : w) { const double *b=kv.second.data(); rn += b[0]*b[0]+b[1]*b[1]+b[2]*b[2]+b[3]*b[3]; }
    double tau = ilu_droptol * std::sqrt(rn);
    auto it = w.begin();
    while (it != w.end() && it->first < i) {                // eliminate cols k<i (increasing)
      int k = it->first;
      double Mik[4]; mm22(it->second.data(), &ilu_Dinv[4*k], Mik);   // M_ik = A_ik U_kk^{-1}
      if (frob(Mik) <= tau) { it = w.erase(it); continue; }          // drop tiny L
      for (auto &uj : Urows[k]) {                                    // subtract M_ik * U[k][j>k]
        int j = uj.first; if (j == k) continue;
        double t[4]; mm22(Mik, uj.second.data(), t);
        auto &wj = w[j];                                             // fill (value-init zeros)
        for (int q=0;q<4;q++) wj[q] -= t[q];
      }
      Lrows[i].push_back({k, {Mik[0],Mik[1],Mik[2],Mik[3]}});
      it = w.erase(it);
    }
    for (auto jt = w.begin(); jt != w.end(); ) {            // drop tiny U off-diagonals (col>i)
      if (jt->first > i && frob(jt->second.data()) <= tau) jt = w.erase(jt);
      else ++jt;
    }
    for (auto &kv : w) Urows[i].push_back({kv.first, kv.second});   // U row (cols>=i, sorted)
    inv22(w[i].data(), &ilu_Dinv[4*i]);                            // U_ii^{-1}
  }
  // --- flatten L+U to combined block-CSR (L cols<i, diagonal at dptr, U cols>i) ---
  long nb = 0; for (int i = 0; i < n; i++) nb += (long)Lrows[i].size() + (long)Urows[i].size();
  ilu_nnz = (int)nb;
  memory->create(ilu_rp,  n+1,    "samqeq:ilu_rp");
  memory->create(ilu_ci,  ilu_nnz,"samqeq:ilu_ci");
  memory->create(ilu_lu,  4*ilu_nnz, "samqeq:ilu_lu");
  memory->create(ilu_dptr,n,      "samqeq:ilu_dptr");
  int m = 0;
  for (int i = 0; i < n; i++) {
    ilu_rp[i] = m;
    for (auto &e : Lrows[i]) { ilu_ci[m]=e.first; for(int t=0;t<4;t++) ilu_lu[4*m+t]=e.second[t]; m++; }
    for (auto &e : Urows[i]) { if (e.first==i) ilu_dptr[i]=m; ilu_ci[m]=e.first; for(int t=0;t<4;t++) ilu_lu[4*m+t]=e.second[t]; m++; }
  }
  ilu_rp[n] = m;

  // bordered constraints (global block vector, 2*n): C col0 (Sum u) = ones on u-components (odd);
  // C col1 (Sum s) = ones on s-components (even). Y = M2^{-1} C, Schur S = C^T Y (2x2).
  memory->create(ilu_Y0, 2*n, "samqeq:ilu_Y0");
  memory->create(ilu_Y1, 2*n, "samqeq:ilu_Y1");
  memory->create(ilu_scr, 2*n, "samqeq:ilu_scr");
  memory->create(ilu_w,  2*n, "samqeq:ilu_w");
  // The constraints are Sum_{i in GROUP} s_i = 0 and Sum_{i in GROUP} u_i = 0, so the C columns
  // carry ones on GROUP rows only. A one on a non-group row would feed a phantom +1 per excluded atom
  // into the Schur sums below -- still a legal nonsingular
  // preconditioner, but a distorted projection that costs iterations. With C zero there, Y0/Y1 are
  // exactly zero on those rows (decoupled [[0,1],[1,0]] block, zero rhs), so the sums need no masking.
  for (int i = 0; i < n; i++) { ilu_scr[2*i] = 0.0; ilu_scr[2*i+1] = ingrp[i] ? 1.0 : 0.0; }  // C col0 (u)
  ilu_solveM2(ilu_scr, ilu_Y0);
  for (int i = 0; i < n; i++) { ilu_scr[2*i] = ingrp[i] ? 1.0 : 0.0; ilu_scr[2*i+1] = 0.0; }  // C col1 (s)
  ilu_solveM2(ilu_scr, ilu_Y1);
  double s00=0,s01=0,s10=0,s11=0;                 // replicated -> global Schur, no MPI reduction
  for (int i = 0; i < n; i++) {
    s00 += ilu_Y0[2*i+1]; s01 += ilu_Y1[2*i+1];   // C col0 = u-components
    s10 += ilu_Y0[2*i];   s11 += ilu_Y1[2*i];     // C col1 = s-components
  }
  double det = s00*s11 - s01*s10;
  if (std::fabs(det) < 1e-300) det = (det >= 0 ? 1e-300 : -1e-300);
  ilu_Sinv[0] =  s11/det; ilu_Sinv[1] = -s01/det;
  ilu_Sinv[2] = -s10/det; ilu_Sinv[3] =  s00/det;
}

/* block-col binary search within block-row i; -1 if absent*/
int FixQEqSam::ilu_find(int i, int col)
{
  int lo = ilu_rp[i], hi = ilu_rp[i+1]-1;
  while (lo <= hi) {
    int mid = (lo+hi)/2, c = ilu_ci[mid];
    if (c == col) return mid;
    if (c < col) lo = mid+1; else hi = mid-1;
  }
  return -1;
}

/* solve the block system M2 sol = rhs (block vector, 2*ilu_n) via block-ILUT*/
void FixQEqSam::ilu_solveM2(const double *rhs, double *sol)
{
  const int n = ilu_n;
  for (int i = 0; i < n; i++) {                  // forward: unit block-L
    double y0 = rhs[2*i], y1 = rhs[2*i+1];
    for (int p = ilu_rp[i]; p < ilu_dptr[i]; p++) {
      int k = ilu_ci[p]; const double *M = &ilu_lu[4*p];
      y0 -= M[0]*sol[2*k] + M[1]*sol[2*k+1];
      y1 -= M[2]*sol[2*k] + M[3]*sol[2*k+1];
    }
    sol[2*i] = y0; sol[2*i+1] = y1;
  }
  for (int i = n-1; i >= 0; i--) {               // back: U sol = y, with D_i^{-1}
    double y0 = sol[2*i], y1 = sol[2*i+1];
    for (int p = ilu_dptr[i]+1; p < ilu_rp[i+1]; p++) {
      int j = ilu_ci[p]; const double *U = &ilu_lu[4*p];
      y0 -= U[0]*sol[2*j] + U[1]*sol[2*j+1];
      y1 -= U[2]*sol[2*j] + U[3]*sol[2*j+1];
    }
    const double *Di = &ilu_Dinv[4*i];
    sol[2*i]   = Di[0]*y0 + Di[1]*y1;
    sol[2*i+1] = Di[2]*y0 + Di[3]*y1;
  }
}

/* ----------------------------------------------------------------------
   preconditioner apply (override). precond_mode 0 -> base diagonal; 1 -> CENTRALIZED
   block-ILUT saddle: gather the local residual into the global block rhs (Allreduce by
   tag; each atom owned once), run the replicated bordered ILUT solve (identical on every
   rank), scatter each rank's owned atoms back. Decomposition-independent by construction.
-------------------------------------------------------------------------*/
void FixQEqSam::precond_apply(double *in, double *out)
{
  if (precond_mode == 0) { FixACKS2Sam::precond_apply(in, out); return; }

  const int n = ilu_n;          // global block count
  const int nl = atom->nlocal;
  tagint *tag = atom->tag;
  int *mask = atom->mask;

  // gather local residual -> global block rhs (zero-fill owned entries, Allreduce-sum)
  for (int g = 0; g < 2*n; g++) ilu_scr[g] = 0.0;
  for (int i = 0; i < nl; i++)
    if (mask[i] & groupbit) { int g = (int)tag[i]-1; ilu_scr[2*g] = in[i]; ilu_scr[2*g+1] = in[NN+i]; }
  MPI_Allreduce(MPI_IN_PLACE, ilu_scr, 2*n, MPI_DOUBLE, MPI_SUM, world);   // global rhs on every rank

  ilu_solveM2(ilu_scr, ilu_w);                  // y0 = M2^{-1} f (replicated)

  // bordered constraint correction: mu = S^{-1}(C^T y0 - g). C^T y0 summed over the GLOBAL
  // system (replicated -> no MPI). g=in[2NN..] lives on last_rows_rank only -> Bcast it.
  double cty0_0 = 0.0, cty0_1 = 0.0;
  for (int g = 0; g < n; g++) { cty0_0 += ilu_w[2*g+1]; cty0_1 += ilu_w[2*g]; }   // Sum u, Sum s
  double gg[2] = {0.0, 0.0};
  if (last_rows_flag) { gg[0] = in[2*NN]; gg[1] = in[2*NN+1]; }
  MPI_Bcast(gg, 2, MPI_DOUBLE, last_rows_rank, world);
  double r0 = cty0_0 - gg[0], r1 = cty0_1 - gg[1];
  double mu0 = ilu_Sinv[0]*r0 + ilu_Sinv[1]*r1;
  double mu1 = ilu_Sinv[2]*r0 + ilu_Sinv[3]*r1;

  // scatter w = y0 - Y mu to this rank's owned atoms
  for (int i = 0; i < nl; i++)
    if (mask[i] & groupbit) {
      int g = (int)tag[i]-1;
      out[i]    = ilu_w[2*g]   - mu0*ilu_Y0[2*g]   - mu1*ilu_Y1[2*g];
      out[NN+i] = ilu_w[2*g+1] - mu0*ilu_Y0[2*g+1] - mu1*ilu_Y1[2*g+1];
    }
  if (last_rows_flag) { out[2*NN] = mu0; out[2*NN+1] = mu1; }
}

/* ----------------------------------------------------------------------
   One retry of a GROSSLY unconverged uncapped saddle solve under the block-ILUT preconditioner.
   Called from FixACKS2Sam::pre_force (rel-residual > 10x tolerance after BiCGStab exhausted imax).
   The diagonal Jacobi preconditioner scales the u-block by 1/X_diag; a monatomic ion
   has X_diag ~ -1e-3 (no intra edges), so its u-row is scaled 1e3x
   against the charge rows and BiCGStab can stall where ILU on the same operator converges in
   tens of matvecs. MINRES(diag) also converges such systems, but far more slowly. The switch is kept for the rest of the run
   (a preconditioner only sets the iteration count) and announced once. Restart from s = 0 (q = q0): the
   exhausted iterate is not a usable initial guess. Returns matvecs used; -1 when no retry applies.
-------------------------------------------------------------------------*/
int FixQEqSam::saddle_fallback(double *b, double *x)
{
  if (precond_mode == 1 || acks2_use_minres) return -1;   // ILU already in use, or MINRES (unpreconditioned): nothing to escalate to
  if (comm->me == 0)
    error->warning(FLERR, "samqeq: ACKS2 saddle BiCGStab exhausted its iterations at step {} (rel-residual"
                          "{:.2e} under the diagonal preconditioner) -- retrying ONCE with `precond ilu`"
                          "(block-ILUT, centralized) and keeping it for the rest of this run. A preconditioner"
                          "changes the iteration count, never the answer. If this repeats, put `fix_modify {}"
                          "precond ilu` in the deck; if the retry fails too, the run stops (see the next error)",
                   update->ntimestep, acks2_relresid, id);
  precond_mode = 1; ilu_valid = 0;
  ilu_build(); ilu_valid = 1;
  const int sz = 2*NN + 2;
  for (int k = 0; k < sz; k++) x[k] = 0.0;
  return BiCGStab(b, x);
}
