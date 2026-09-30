/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation. Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software. This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
-------------------------------------------------------------------------*/

/* ----------------------------------------------------------------------
   pppm/samqeq : per-atom reciprocal-potential PPPM for fix qeq/sam, with no
   ELECTRODE-package dependency. The compute_vector / start_compute /
   make_rho_in_brick / project_psi routines are the per-atom-potential subset
   of PPPMElectrode (Ahrens-Iwers, Tee, Meissner, J. Chem. Phys. 155, 104104
   (2021)), reimplemented as a minimal subclass of the current PPPM so it
   inherits stock grid setup (new Grid3d API) and only adds the source-group
   density bricks + potential projection.
-------------------------------------------------------------------------*/

#include "pppm_samqeq.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "fft3d_wrap.h"
#include "grid3d.h"
#include "math_const.h"
#include "memory.h"
#include "update.h"

#include <cstring>

using namespace LAMMPS_NS;

static constexpr FFT_SCALAR ZEROF = 0.0;

// these enum tags are passed to the inherited PPPM::pack/unpack_forward_grid and
// pack/unpack_reverse_grid callbacks, so their VALUES must match PPPM's own
// (see pppm.cpp): REVERSE_RHO=0; FORWARD_IK=0, FORWARD_AD=1, ...
enum { REVERSE_RHO };
enum { FORWARD_IK, FORWARD_AD, FORWARD_IK_PERATOM, FORWARD_AD_PERATOM };

/* ----------------------------------------------------------------------*/

PPPMSamqeq::PPPMSamqeq(LAMMPS *lmp) :
    PPPM(lmp), electrolyte_density_brick(nullptr), electrolyte_density_fft(nullptr), compute_step(-1)
{
}

/* ----------------------------------------------------------------------
   free all memory
-------------------------------------------------------------------------*/

PPPMSamqeq::~PPPMSamqeq()
{
  if (copymode) return;

  PPPMSamqeq::deallocate();
  if (peratom_allocate_flag) PPPMSamqeq::deallocate_peratom();
}

/* ----------------------------------------------------------------------
   PPPM::init() first (sets up triclinic/slabflag from the current box/kspace_modify
   state, per the base class, incl. the slab boundary checks: x/y periodic + z
   boundary f f), then reject the geometries this class's own compute_vector/
   make_rho_in_brick/project_psi (the fix qeq/sam charge-solve RHS) does not
   implement: triclinic boxes and slab nozforce (slabflag==2). The EW3DC slab
   correction (kspace_modify slab <volfactor>, slabflag==1) IS supported: the
   inherited PPPM machinery already extends the grid/volume by slab_volfactor and
   PPPM::compute() (chained to by our compute()) calls PPPM::slabcorr() for the
   force/energy dipole correction; compute_vector below adds the SAME dipole term
   analytically to the per-atom potential so the charge solve and the forces
   derive from one energy functional (else: energy nonconservation, the
   recip_self-mismatch class of bug). Placed in init() (not compute_vector)
   because init() is the standard once-per-setup contract-validation hook (also
   re-run on triclinic/box-style changes via PPPM::init()'s own
   triclinic_check()), so this fires before any solve rather than on the first
   matvec, and does not need a one-time-flag dance in a hot per-step function.
-------------------------------------------------------------------------*/

void PPPMSamqeq::init()
{
  PPPM::init();

  // ★ RUN-BOUNDARY FIX (, triage of the P0 reset_timestep crash): PPPM::init() -- called at EVERY
  // `run` command -- deallocates + reallocates ALL grid arrays (our allocate() memsets them to 0), and
  // greensfn/fk*/vg are only recomputed by PPPM::setup(), which Verlet::setup runs AFTER the fix's
  // setup_pre_force. start_compute()'s self-heal sentinel (`compute_step == -1`) only detected the very
  // FIRST setup ever (constructor init), so on a 2nd+ run whose boundary solve is allowed to proceed --
  // exactly the `reset_timestep 0` production idiom, where ntimestep==0 masquerades as a fresh first
  // setup and defeats the #9 boundary skip AND leaves compute_step (= last step of the previous run)
  // > ntimestep -- compute_vector ran the ENTIRE boundary charge solve on an all-zero greensfn: zero
  // reciprocal operator while recip_self is still subtracted from the diagonal (~4.9 eV/e softening)
  // -> indefinite operator, CG stalls at cap, garbage/over-polarized charges -> "Out of range atoms -
  // cannot compute PPPM" crash lottery (gdb-verified : compute_step=100, ntimestep=0, greensfn==0
  // through make_rho_in_brick). init() is precisely the point where the grid is invalidated, so re-arm
  // the sentinel here: the next compute_vector self-heals via setup(). Byte-identical for single-run
  // decks (constructor already set -1); multi-run decks get one redundant-but-identical setup() per run.
  compute_step = -1;

  if (slabflag == 2)
    error->all(FLERR, "pppm/samqeq does not support kspace_modify slab nozforce");
  if (domain->triclinic)
    error->all(FLERR, "pppm/samqeq does not support triclinic boxes");
}

/* ----------------------------------------------------------------------
   allocate = PPPM::allocate + source-group density bricks; also take
   ownership of u_brick (compute_vector needs it every step, whereas stock
   PPPM only allocates it in allocate_peratom for the ik scheme).
-------------------------------------------------------------------------*/

void PPPMSamqeq::allocate()
{
  PPPM::allocate();

  memory->create3d_offset(electrolyte_density_brick, nzlo_out, nzhi_out, nylo_out, nyhi_out,
                          nxlo_out, nxhi_out, "pppm/samqeq:electrolyte_density_brick");
  memory->create(electrolyte_density_fft, nfft_both, "pppm/samqeq:electrolyte_density_fft");

  // u_brick: for the ad scheme PPPM::allocate already made it; for the ik
  // scheme (default) it does not, so own it here (and skip it in
  // allocate_peratom / deallocate_peratom).
  if (differentiation_flag != 1)
    memory->create3d_offset(u_brick, nzlo_out, nzhi_out, nylo_out, nyhi_out, nxlo_out, nxhi_out,
                            "pppm:u_brick");

  // ★ UNINITIALISED-MEMORY FIX (valgrind, session 16): the FIRST compute_vector() runs in the qeq fix's
  // setup_pre_force (recip_self calibration + the first solve) — BEFORE any PPPM::compute(), which is what
  // normally fills/zeros these grid + FFT-domain buffers each step. So on the first call (and after every
  // grid re-setup/resize, which reallocs fresh uninitialised heap) the path reads uninitialised cells; the
  // garbage flows through the reciprocal potential -> the QEq matvec/CG -> spurious forces -> atoms drift off
  // the PPPM grid -> "Out of range atoms - cannot compute PPPM" on re-setup/large grids. Zero everything the
  // compute_vector path touches so the very first call is well-defined (one-time per allocate; negligible cost).
  memset(&(electrolyte_density_brick[nzlo_out][nylo_out][nxlo_out]), 0, ngrid * sizeof(FFT_SCALAR));
  memset(electrolyte_density_fft, 0, nfft_both * sizeof(FFT_SCALAR));
  if (differentiation_flag != 1)
    memset(&(u_brick[nzlo_out][nylo_out][nxlo_out]), 0, ngrid * sizeof(FFT_SCALAR));
  memset(density_fft, 0, nfft_both * sizeof(FFT_SCALAR));
  memset(work1, 0, 2 * nfft_both * sizeof(FFT_SCALAR));
  memset(work2, 0, 2 * nfft_both * sizeof(FFT_SCALAR));
  // greensfn is (re)computed in PPPM::setup() but compute_gf_* can leave some k-entries unset (e.g. the k=0 /
  // boundary modes); the compute_vector path multiplies work1*greensfn, so any unset entry leaks uninitialised
  // garbage into the reciprocal potential -> the recip_self calibration + the solve. Pre-zero so unset stays 0.
  memset(greensfn, 0, nfft_both * sizeof(FFT_SCALAR));
}

/* ----------------------------------------------------------------------*/

void PPPMSamqeq::deallocate()
{
  memory->destroy3d_offset(electrolyte_density_brick, nzlo_out, nylo_out, nxlo_out);
  electrolyte_density_brick = nullptr;
  memory->destroy(electrolyte_density_fft);
  electrolyte_density_fft = nullptr;

  // u_brick owned here for the ik scheme; PPPM::deallocate only frees it for ad.
  if (differentiation_flag != 1) {
    memory->destroy3d_offset(u_brick, nzlo_out, nylo_out, nxlo_out);
    u_brick = nullptr;
  }

  PPPM::deallocate();
  // #31 exit-segfault fix: PPPM::deallocate() does raw `delete gc/fft1/fft2/remap` WITHOUT nulling, so it
  // is NOT idempotent. We call it here (needed for the mid-run grid-resize realloc), but the base ~PPPM()
  // ALSO calls PPPM::deallocate() at destruction -> a SECOND raw delete of those now-dangling pointers ->
  // double-free / "Address not mapped" segfault at program exit. Null them so the base's second call is a
  // no-op (delete nullptr / memory->destroy(nullptr) are both safe). (PPPMElectrode avoids this by
  // re-implementing the full deallocate with nulling instead of chaining to PPPM::deallocate.)
  gc = nullptr;
  fft1 = nullptr;
  fft2 = nullptr;
  remap = nullptr;
}

/* ----------------------------------------------------------------------
   per-atom (virial) bricks, as in PPPM::allocate_peratom but WITHOUT u_brick,
   which pppm/samqeq owns in allocate(). Only used if a per-atom kspace
   compute (e.g. compute pe/atom) is requested.
-------------------------------------------------------------------------*/

void PPPMSamqeq::allocate_peratom()
{
  peratom_allocate_flag = 1;

  memory->create3d_offset(v0_brick, nzlo_out, nzhi_out, nylo_out, nyhi_out, nxlo_out, nxhi_out,
                          "pppm:v0_brick");
  memory->create3d_offset(v1_brick, nzlo_out, nzhi_out, nylo_out, nyhi_out, nxlo_out, nxhi_out,
                          "pppm:v1_brick");
  memory->create3d_offset(v2_brick, nzlo_out, nzhi_out, nylo_out, nyhi_out, nxlo_out, nxhi_out,
                          "pppm:v2_brick");
  memory->create3d_offset(v3_brick, nzlo_out, nzhi_out, nylo_out, nyhi_out, nxlo_out, nxhi_out,
                          "pppm:v3_brick");
  memory->create3d_offset(v4_brick, nzlo_out, nzhi_out, nylo_out, nyhi_out, nxlo_out, nxhi_out,
                          "pppm:v4_brick");
  memory->create3d_offset(v5_brick, nzlo_out, nzhi_out, nylo_out, nyhi_out, nxlo_out, nxhi_out,
                          "pppm:v5_brick");

  memset(&v0_brick[nzlo_out][nylo_out][nxlo_out], 0, ngrid * sizeof(FFT_SCALAR));
  memset(&v1_brick[nzlo_out][nylo_out][nxlo_out], 0, ngrid * sizeof(FFT_SCALAR));
  memset(&v2_brick[nzlo_out][nylo_out][nxlo_out], 0, ngrid * sizeof(FFT_SCALAR));
  memset(&v3_brick[nzlo_out][nylo_out][nxlo_out], 0, ngrid * sizeof(FFT_SCALAR));
  memset(&v4_brick[nzlo_out][nylo_out][nxlo_out], 0, ngrid * sizeof(FFT_SCALAR));
  memset(&v5_brick[nzlo_out][nylo_out][nxlo_out], 0, ngrid * sizeof(FFT_SCALAR));

  // use the same GC ghost grid object for per-atom communication, but the
  // buffers must be large enough for the per-atom comm count.
  if (differentiation_flag) npergrid = 6;
  else npergrid = 7;

  memory->destroy(gc_buf1);
  memory->destroy(gc_buf2);
  memory->create(gc_buf1, npergrid * ngc_buf1, "pppm:gc_buf1");
  memory->create(gc_buf2, npergrid * ngc_buf2, "pppm:gc_buf2");
}

/* ----------------------------------------------------------------------*/

void PPPMSamqeq::deallocate_peratom()
{
  peratom_allocate_flag = 0;

  memory->destroy3d_offset(v0_brick, nzlo_out, nylo_out, nxlo_out);
  memory->destroy3d_offset(v1_brick, nzlo_out, nylo_out, nxlo_out);
  memory->destroy3d_offset(v2_brick, nzlo_out, nylo_out, nxlo_out);
  memory->destroy3d_offset(v3_brick, nzlo_out, nylo_out, nxlo_out);
  memory->destroy3d_offset(v4_brick, nzlo_out, nylo_out, nxlo_out);
  memory->destroy3d_offset(v5_brick, nzlo_out, nylo_out, nxlo_out);

  // u_brick is owned by allocate()/deallocate(); do NOT free it here.
}

/* ----------------------------------------------------------------------
   refresh the particle->grid map once per timestep (lazy; cheap on repeats).
   On the very first call, make sure the grid is set up.
-------------------------------------------------------------------------*/

void PPPMSamqeq::start_compute()
{
  if (compute_step < update->ntimestep) {
    if (compute_step == -1) setup();
    boxlo = domain->boxlo;
    // extend size of per-atom arrays if necessary
    if (atom->nmax > nmax) {
      memory->destroy(part2grid);
      nmax = atom->nmax;
      memory->create(part2grid, nmax, 3, "pppm/samqeq:part2grid");
    }
    particle_map();
    compute_step = update->ntimestep;
  }
}

/* ----------------------------------------------------------------------
   reciprocal energy/forces with qsqsum refreshed from the LIVE charges.
   Stock PPPM::compute() reuses qsqsum/qsum measured at setup() and only
   re-measures them when the atom COUNT changes (pppm.cpp) -- valid for fixed
   point charges, but fix qeq/sam's charges fluctuate every step. The cached
   value enters the reciprocal self-energy term (-g_ewald*qsqsum/sqrt(pi)) and
   the net-charge term, so a stale qsqsum makes the reported E_long wrong by
   qscale*g_ewald*(qsqsum_setup - qsqsum_now)/sqrt(pi) (large when the setup
   charges are far from equilibrated, e.g. straight off `set`). The term is
   position-independent => it contributes NO force, so dynamics are unaffected;
   this only fixes the reported energy. qsum_qsq() is a single scalar Allreduce
   (negligible vs PPPM's per-step FFTs) and does not touch the grid/forces.
-------------------------------------------------------------------------*/

void PPPMSamqeq::compute(int eflag, int vflag)
{
  qsum_qsq(0);                 // re-measure qsum/qsqsum from the current (solved) charges (0 = no per-step neutrality warning; setup already warns once)
  PPPM::compute(eflag, vflag); // stock reciprocal solve; now uses the fresh qsqsum in its self-energy.
                               // Under slabflag==1 this also runs the inherited PPPM::slabcorr()
                               // (force/energy dipole correction), whose non-neutral qsum terms now
                               // see the fresh qsum too — matching compute_vector's slab term exactly.
}

/* ----------------------------------------------------------------------
   per-atom reciprocal potential of the source group's charges sampled at the
   sensor group's atoms. ACCUMULATES into vec (no qqrd2e prefactor).
-------------------------------------------------------------------------*/

void PPPMSamqeq::compute_vector(double *vec, int sensor_grpbit, int source_grpbit,
                                bool invert_source)
{
  start_compute();

  // borrow brick2fft() for the electrolyte (source-group) density by
  // temporarily swapping in the electrolyte density pointers.
  FFT_SCALAR ***density_brick_real = density_brick;
  FFT_SCALAR *density_fft_real = density_fft;
  // E1 (scaling audit): under DYNAMICS, positions are fixed within a timestep, so start_compute()'s lazy
  // once-per-step particle_map() above is already current for every matvec of the step's solve(s) —
  // unconditionally recomputing the identical map here (~30x/step at the observed CG budget) was pure
  // waste. MINIMIZATION is the one caller where positions DO change within a single ntimestep (line-search
  // energy evaluations share the iteration's step number), so keep the per-call refresh whenever this is
  // not a dynamics run (update->whichflag: 1 = dynamics, 2 = minimize, 0 = between runs — refresh kept for
  // both non-dynamics cases out of caution). Bit-preserving: when skipped, the map is identical anyway.
  if (update->whichflag != 1) particle_map();
  make_rho_in_brick(source_grpbit, electrolyte_density_brick, invert_source);
  density_brick = electrolyte_density_brick;
  density_fft = electrolyte_density_fft;
  gc->reverse_comm(Grid3d::KSPACE, this, REVERSE_RHO, 1, sizeof(FFT_SCALAR), gc_buf1, gc_buf2,
                   MPI_FFT_SCALAR);
  brick2fft();
  density_brick = density_brick_real;
  density_fft = density_fft_real;

  // transform electrolyte charge density (r -> k)
  for (int i = 0, n = 0; i < nfft; i++) {
    work1[n++] = electrolyte_density_fft[i];
    work1[n++] = ZEROF;
  }
  fft1->compute(work1, work1, -1);

  // multiply by Green's function, transform back (k -> r) into u_brick
  for (int i = 0, n = 0; i < nfft; i++) {
    work2[n] = work1[n] * greensfn[i];
    n++;
    work2[n] = work1[n] * greensfn[i];
    n++;
  }
  fft2->compute(work2, work2, 1);
  for (int k = nzlo_in, n = 0; k <= nzhi_in; k++)
    for (int j = nylo_in; j <= nyhi_in; j++)
      for (int i = nxlo_in; i <= nxhi_in; i++) {
        u_brick[k][j][i] = work2[n];
        n += 2;
      }
  gc->forward_comm(Grid3d::KSPACE, this, FORWARD_AD, 1, sizeof(FFT_SCALAR), gc_buf1, gc_buf2,
                   MPI_FFT_SCALAR);
  project_psi(vec, sensor_grpbit);

  // ★ EW3DC SLAB CORRECTION to the per-atom potential (Yeh-Berkowitz, J. Chem. Phys. 111, 3155;
  // non-neutral extension J. Chem. Phys. 131, 094107). The grid solve above ran in the vacuum-extended
  // box (V_slab = xprd*yprd*zprd*slab_volfactor, set up by PPPM::setup()); the inter-slab dipole
  // interaction it still contains is cancelled on the FORCE/ENERGY side by the inherited
  // PPPM::slabcorr() (called from PPPM::compute() when slabflag==1), whose energy is
  //   E_slab = (2pi/V) * (M_z^2 - qsum*S2 - qsum^2*L^2/12), [RAW: slabcorr applies qqrd2e*scale]
  // with M_z = sum_j q_j z_j, S2 = sum_j q_j z_j^2, L = zprd_slab. The charge solve must see the SAME
  // functional, so add its exact charge-gradient to the potential:
  //   phi_slab(i) = dE_slab/dq_i = (2pi/V) * (2 z_i M_z - S2 - qsum z_i^2 - qsum L^2/6).
  // Consistency checks: (a) 1/2 sum_i q_i phi_slab(i) == E_slab exactly (quadratic form, symmetric
  // kernel K_ij = (2pi/V)(2 z_i z_j - z_i^2 - z_j^2 - L^2/6), so the CG operator stays symmetric);
  // (b) -dE_slab/dz_i reproduces slabcorr()'s force f_z(i) = -(4pi/V) q_i (M_z - qsum z_i).
  // Group semantics: the sums run over the SOURCE group (same membership predicate as
  // make_rho_in_brick, incl. invert_source) and the potential is added at SENSOR atoms only —
  // identical to the grid path's source/sensor contract. Coordinates: raw x[i][2] exactly as
  // slabcorr() reads them (z is boundary f under slab PPPM, so no wrap/image issue). RAW units
  // (no qqrd2e), matching this method's contract; fix qeq/sam applies qqrd2e downstream.
  // NB fix qeq/sam side: calibrate_recip_self()'s neutral +1/-1 probe pair picks up
  // (2pi/V)(z_A-z_B)^2 from this term; the fix subtracts it analytically (fix_qeq_sam_lr.cpp) so
  // recip_self stays the pure grid self-coefficient.
  if (slabflag == 1) {
    double *q = atom->q;
    double **x = atom->x;
    int *mask = atom->mask;
    const int nlocal = atom->nlocal;
    const double zprd_slab = domain->zprd * slab_volfactor;

    double sums[3] = {0.0, 0.0, 0.0};    // {qsum, M_z, S2} over the source group (local part)
    for (int i = 0; i < nlocal; i++) {
      bool const i_in_source = !!(mask[i] & source_grpbit) != invert_source;
      if (!i_in_source) continue;
      sums[0] += q[i];
      sums[1] += q[i] * x[i][2];
      sums[2] += q[i] * x[i][2] * x[i][2];
    }
    double sums_all[3];
    MPI_Allreduce(sums, sums_all, 3, MPI_DOUBLE, MPI_SUM, world);
    const double qsum_src = sums_all[0];
    const double dipole_src = sums_all[1];
    const double dipole_r2_src = sums_all[2];

    // volume was set to xprd*yprd*zprd_slab by PPPM::setup(); use it exactly as slabcorr() does
    const double pref = MathConst::MY_2PI / volume;
    for (int i = 0; i < nlocal; i++) {
      if (!(mask[i] & sensor_grpbit)) continue;
      vec[i] += pref *
          (2.0 * x[i][2] * dipole_src - dipole_r2_src - qsum_src * x[i][2] * x[i][2] -
           qsum_src * zprd_slab * zprd_slab / 6.0);
    }
  }
}

/* ----------------------------------------------------------------------
   interpolate u_brick back to the sensor atoms with the order-n stencil.
-------------------------------------------------------------------------*/

void PPPMSamqeq::project_psi(double *vec, int sensor_grpbit)
{
  double **x = atom->x;
  int *mask = atom->mask;
  const bigint ngridtotal = (bigint) nx_pppm * ny_pppm * nz_pppm;
  const double scaleinv = 1.0 / ngridtotal;

  for (int i = 0; i < atom->nlocal; i++) {
    if (!(mask[i] & sensor_grpbit)) continue;
    double v = 0.0;
    int nix = part2grid[i][0];
    int niy = part2grid[i][1];
    int niz = part2grid[i][2];
    FFT_SCALAR dix = nix + shiftone - (x[i][0] - boxlo[0]) * delxinv;
    FFT_SCALAR diy = niy + shiftone - (x[i][1] - boxlo[1]) * delyinv;
    FFT_SCALAR diz = niz + shiftone - (x[i][2] - boxlo[2]) * delzinv;
    compute_rho1d(dix, diy, diz);
    for (int ni = nlower; ni <= nupper; ni++) {
      double iz0 = rho1d[2][ni];
      int miz = ni + niz;
      for (int mi = nlower; mi <= nupper; mi++) {
        double iy0 = iz0 * rho1d[1][mi];
        int miy = mi + niy;
        for (int li = nlower; li <= nupper; li++) {
          int mix = li + nix;
          double ix0 = iy0 * rho1d[0][li];
          v += ix0 * u_brick[miz][miy][mix];
        }
      }
    }
    vec[i] += v * scaleinv;
  }
}

/* ----------------------------------------------------------------------
   map the source group's charges onto a scratch density brick (= PPPM::make_rho
   masked by source_grpbit; invert_source flips membership).
-------------------------------------------------------------------------*/

void PPPMSamqeq::make_rho_in_brick(int source_grpbit, FFT_SCALAR ***scratch_brick,
                                   bool invert_source)
{
  int l, m, n, nx, ny, nz, mx, my, mz;
  FFT_SCALAR dx, dy, dz, x0, y0, z0;

  // clear 3d density array
  memset(&(scratch_brick[nzlo_out][nylo_out][nxlo_out]), 0, ngrid * sizeof(FFT_SCALAR));

  double *q = atom->q;
  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    bool const i_in_source = !!(mask[i] & source_grpbit) != invert_source;
    if (!i_in_source) continue;
    nx = part2grid[i][0];
    ny = part2grid[i][1];
    nz = part2grid[i][2];
    dx = nx + shiftone - (x[i][0] - boxlo[0]) * delxinv;
    dy = ny + shiftone - (x[i][1] - boxlo[1]) * delyinv;
    dz = nz + shiftone - (x[i][2] - boxlo[2]) * delzinv;

    compute_rho1d(dx, dy, dz);

    z0 = delvolinv * q[i];
    for (n = nlower; n <= nupper; n++) {
      mz = n + nz;
      y0 = z0 * rho1d[2][n];
      for (m = nlower; m <= nupper; m++) {
        my = m + ny;
        x0 = y0 * rho1d[1][m];
        for (l = nlower; l <= nupper; l++) {
          mx = l + nx;
          scratch_brick[mz][my][mx] += x0 * rho1d[0][l];
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   R4: EXACT per-atom grid self-coefficient (REQUEST_recip_self_wander_guard R4; the TODO in the
   calibrate_recip_self banner). compute_vector() forms
       prec_i = (1/N_grid) sum_g W_i(g) u(g), u = B[ G. F[rho] ], rho(g') = delvolinv sum_j q_j W_j(g')
   with F/B the unnormalized forward/backward FFTs, so atom i's OWN contribution to prec_i is exactly
       c_i = (delvolinv/N_grid) sum_{g,g'} W_i(g) W_i(g') Kr(g-g'), Kr(D) = B[G](D) = sum_k G(k) e^{+ik.D}.
   Kr is needed only for |D_x|,|D_y|,|D_z| <= P-1 (the stencil support): one backward FFT of greensfn, the
   (2P-1)^3 entries gathered by Allreduce (replicated, decomposition-independent). Per atom the double sum
   factorizes through the 1-d stencil autocorrelations C_x(D) = sum_a W_x(a) W_x(a+D):
       c_i = (delvolinv/N_grid) sum_D C_x(D_x) C_y(D_y) C_z(D_z) Kr(D) ((2P-1)^3 = 729 terms at P=5).
   ★ WHAT THE OPERATOR NEEDS IS NOT K_ii BUT K_ii − ξ (measured 2026-09-02, the first build of this route).
   K_ii is the grid's k≠0 self-potential of a LONE charge: it contains the charge's interaction with its own
   periodic images (the k=0 mode is dropped ⇒ neutralizing background): ξ = ψ_{k≠0}(0) − 2α/√π (+ Σ_{n≠0}
   erfc(αn)/n), the Wigner-type constant → −2.837/L for a cube. The Ewald energy the forces integrate keeps
   ½ξΣq² (a real term of the periodic system), so the solve's diagonal must be η + ξ, i.e. the quantity to
   subtract from the grid diagonal is 2α/√π − smearing = K_ii − ξ. A neutral PROBE PAIR cancels ξ and so
   measures exactly that (water box: K_ii 0.16864, ξ −0.21453, K_ii − ξ 0.38317 vs probe 0.38207; 24.8 Å box:
   0.27054 + 0.114 vs 0.38303). The remaining 3e-3 is the probe's own short-range grid pair error (SELFDIAG
   pairs at 1–2 spacings scatter 0.33–0.375), which this route does not have. ξ is evaluated analytically on
   the reciprocal lattice of the (slab-extended) box with the same g_ewald; the influence function equals the
   exact kernel at low k to O((kh)^{2P}), so ξ_grid = ξ to well below 1e-4. The per-atom sub-grid variation of
   K_ii is ≤1e-5 at order 5 (B-spline autocorrelations are nearly shift-invariant), so the practical content of
   this route is: calibration-free, drift-free, box-size-exact. EW3DC: the Yeh-Berkowitz term is added AFTER
   project_psi in compute_vector, so it is not part of c_i -- consistent with the scalar route. Validate with
   SAMQEQ_FIELDDIAG (SELFDIAG prints the probe S_A, K_A, ξ and c_A = K_A − ξ side by side).
-------------------------------------------------------------------------*/
void PPPMSamqeq::build_self_kernel()
{
  const int P = order, W = 2*order - 1;
  if (nx_pppm < W || ny_pppm < W || nz_pppm < W)
    error->all(FLERR, "pppm/samqeq: per-atom self-term needs >= {} grid points per direction (order {}); grid is {}x{}x{}",
               W, P, nx_pppm, ny_pppm, nz_pppm);
  for (int i = 0, n = 0; i < nfft; i++) { work1[n++] = greensfn[i]; work1[n++] = ZEROF; }
  fft2->compute(work1, work1, 1);                        // backward, unnormalized: same convention as compute_vector
  self_kern.assign((size_t)W*W*W, 0.0);
  for (int k = nzlo_in, n = 0; k <= nzhi_in; k++)
    for (int j = nylo_in; j <= nyhi_in; j++)
      for (int i = nxlo_in; i <= nxhi_in; i++, n += 2) {
        int dx, dy, dz;                                 // grid index -> signed offset, or skip
        if (i <= P-1) dx = i; else if (i >= nx_pppm-(P-1)) dx = i - nx_pppm; else continue;
        if (j <= P-1) dy = j; else if (j >= ny_pppm-(P-1)) dy = j - ny_pppm; else continue;
        if (k <= P-1) dz = k; else if (k >= nz_pppm-(P-1)) dz = k - nz_pppm; else continue;
        self_kern[((size_t)(dz+P-1)*W + (dy+P-1))*W + (dx+P-1)] = work1[n];
      }
  MPI_Allreduce(MPI_IN_PLACE, self_kern.data(), W*W*W, MPI_DOUBLE, MPI_SUM, world);
  // analytic image self-potential of the grid-periodic box (slab: the vacuum-extended z period, as the grid)
  {
    const double a = g_ewald;
    const double Lx = domain->xprd, Ly = domain->yprd;
    const double Lz = domain->zprd * ((slabflag == 1) ? slab_volfactor : 1.0);
    const double V = Lx*Ly*Lz, kx = MathConst::MY_2PI/Lx, ky = MathConst::MY_2PI/Ly, kz = MathConst::MY_2PI/Lz;
    const double kmax = sqrt(4.0*a*a*36.0);           // e^{-k^2/4a^2} < 2e-16 beyond this
    const int mx = (int)(kmax/kx)+1, my = (int)(kmax/ky)+1, mz = (int)(kmax/kz)+1;
    double sk = 0.0;
    for (int i = -mx; i <= mx; i++)
      for (int j = -my; j <= my; j++)
        for (int k = -mz; k <= mz; k++) {
          if (i == 0 && j == 0 && k == 0) continue;
          const double k2 = (i*kx)*(i*kx) + (j*ky)*(j*ky) + (k*kz)*(k*kz);
          if (k2 > kmax*kmax) continue;
          sk += 4.0*MathConst::MY_PI/(V*k2) * exp(-k2/(4.0*a*a));
        }
    double sr = 0.0;                                  // real-space images: erfc(a n)/n, a few shells suffice
    const double rmax = 6.0/a;
    const int nx = (int)(rmax/Lx)+1, ny = (int)(rmax/Ly)+1, nz = (int)(rmax/Lz)+1;
    for (int i = -nx; i <= nx; i++)
      for (int j = -ny; j <= ny; j++)
        for (int k = -nz; k <= nz; k++) {
          if (i == 0 && j == 0 && k == 0) continue;
          const double n = sqrt((i*Lx)*(i*Lx) + (j*Ly)*(j*Ly) + (k*Lz)*(k*Lz));
          sr += erfc(a*n)/n;
        }
    self_xi = sk - 2.0*a/MathConst::MY_PIS + sr;
  }
  self_kern_nx = nx_pppm; self_kern_ny = ny_pppm; self_kern_nz = nz_pppm;
  self_kern_gewald = g_ewald;
  self_kern_prd[0] = domain->xprd; self_kern_prd[1] = domain->yprd; self_kern_prd[2] = domain->zprd;
}

bool PPPMSamqeq::compute_self_peratom(double *out, int grpbit)
{
  start_compute();                                       // particle_map current (same gate as compute_vector)
  if (self_kern.empty() || self_kern_nx != nx_pppm || self_kern_ny != ny_pppm || self_kern_nz != nz_pppm ||
      self_kern_gewald != g_ewald || self_kern_prd[0] != domain->xprd || self_kern_prd[1] != domain->yprd ||
      self_kern_prd[2] != domain->zprd)
    build_self_kernel();
  const int P = order, W = 2*order - 1;
  const bigint ngridtotal = (bigint) nx_pppm * ny_pppm * nz_pppm;
  const double pref = delvolinv / (double) ngridtotal;
  double **x = atom->x; int *mask = atom->mask;
  std::vector<double> cx(W), cy(W), cz(W);
  for (int i = 0; i < atom->nlocal; i++) {
    if (!(mask[i] & grpbit)) continue;
    const int nix = part2grid[i][0], niy = part2grid[i][1], niz = part2grid[i][2];
    const FFT_SCALAR dix = nix + shiftone - (x[i][0] - boxlo[0]) * delxinv;
    const FFT_SCALAR diy = niy + shiftone - (x[i][1] - boxlo[1]) * delyinv;
    const FFT_SCALAR diz = niz + shiftone - (x[i][2] - boxlo[2]) * delzinv;
    compute_rho1d(dix, diy, diz);
    for (int d = -(P-1); d <= P-1; d++) {                // 1-d stencil autocorrelations
      double sx = 0.0, sy = 0.0, sz = 0.0;
      for (int a = nlower; a <= nupper; a++) {
        const int b = a + d;
        if (b < nlower || b > nupper) continue;
        sx += rho1d[0][a]*rho1d[0][b]; sy += rho1d[1][a]*rho1d[1][b]; sz += rho1d[2][a]*rho1d[2][b];
      }
      cx[d+P-1] = sx; cy[d+P-1] = sy; cz[d+P-1] = sz;
    }
    double s = 0.0;
    for (int kz = 0; kz < W; kz++) {
      const double wz = cz[kz]; if (wz == 0.0) continue;
      for (int ky = 0; ky < W; ky++) {
        const double wyz = wz*cy[ky]; if (wyz == 0.0) continue;
        const double *kr = &self_kern[((size_t)kz*W + ky)*W];
        for (int kx = 0; kx < W; kx++) s += wyz*cx[kx]*kr[kx];
      }
    }
    out[i] = pref * s - self_xi;                        // K_ii − ξ: what the neutral probe pair measures, exactly
  }
  return true;
}
