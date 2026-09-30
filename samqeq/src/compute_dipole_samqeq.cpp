/* ----------------------------------------------------------------------
   compute dipole/samqeq — total + per-molecule dipole for samQEq/samQEq.

   Usage: compute ID group-ID dipole/samqeq [debye]
     scalar = mean per-molecule |dipole|, in DEBYE (the FQ headline number; molecule-ID 0 EXCLUDED, see below)
     vector[0-2]= total dipole (Σ q_i r_i, unwrapped), e·Å (or Debye with the "debye" keyword)
     vector[3] = |total dipole|, e·Å (or Debye)
     vector[4] = mean per-molecule |dipole|, DEBYE (== scalar; always Debye)

   Charges are the live atom->q solved by fix qeq/sam. Per-molecule dipoles
   use unwrapped coords and are origin-independent for neutral molecules
   (samQEq enforces per-molecule neutrality).
   Molecule-ID 0 (the pooled non-molecular fragment -- bare ions, electrode atoms in atom_style
   full/molecular with no real bond topology) is EXCLUDED from the per-molecule mean |dipole| (scalar,
   vector[4]) -- lumping unrelated atoms into one fake "molecule" gives an arbitrary-origin, physically
   meaningless number. It still contributes to the system TOTAL dipole (vector[0-3]). Any molecule
   (including the mol-0 pool) that ends up with a nonzero net charge has an origin/unwrap-convention-dependent
   dipole; this is flagged with a one-time warning.
-------------------------------------------------------------------------*/

#include "compute_dipole_samqeq.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "update.h"
#include "comm.h"

#include <cmath>
#include <cstring>

// Molecule-ID 0 pools every non-molecular atom (bare ions, electrode atoms in atom_style full/molecular
// with no real bond topology) into one fake "molecule". Its "dipole" is an artifact of that pooling -- an
// arbitrary-origin sum over physically unrelated atoms -- so it must not pollute the headline mean-|dipole|
// scalar. Also: a molecule (any molecule id, incl. the mol-0 pool) that carries a nonzero net charge has an
// unwrap-image-convention-dependent (not physically intrinsic) dipole; warn once, not silently.
// `warned_nonneutral_mol` is a translation-unit-local flag, not a class member -- it is process-lifetime, so it fires once per LAMMPS run even across multiple
// `compute .../dipole` instances (a reasonable trade-off for a diagnostic notice, not a physics-affecting
// quantity).
namespace {
bool warned_nonneutral_mol = false;
}

using namespace LAMMPS_NS;

/* ----------------------------------------------------------------------*/

ComputeDipoleSamQEq::ComputeDipoleSamQEq(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg), moldip(nullptr), moldip_all(nullptr),
    molcnt(nullptr), molcnt_all(nullptr)
{
  if (narg < 3 || narg > 4) error->all(FLERR, "Illegal compute dipole/samqeq command");

  unit_debye = 0;
  if (narg == 4) {
    if (strcmp(arg[3], "debye") == 0) unit_debye = 1;
    else error->all(FLERR, "Illegal compute dipole/samqeq keyword: {}", arg[3]);
  }

  scalar_flag = 1;
  vector_flag = 1;
  size_vector = 5;
  extscalar = 0;
  extvector = 0;

  conv = 4.8032047;          // 1 e·Å = 4.8032047 Debye
  nmol_alloc = 0;
  vector = new double[size_vector];
  for (int i = 0; i < size_vector; i++) vector[i] = 0.0;

  if (!atom->molecule_flag)
    error->all(FLERR, "compute dipole/samqeq requires a molecular atom_style (per-molecule dipoles)");
}

/* ----------------------------------------------------------------------*/

ComputeDipoleSamQEq::~ComputeDipoleSamQEq()
{
  delete[] vector;
  delete[] moldip; delete[] moldip_all;
  delete[] molcnt; delete[] molcnt_all;
}

/* ----------------------------------------------------------------------*/

void ComputeDipoleSamQEq::grow_mol(bigint nmol)
{
  delete[] moldip; delete[] moldip_all;
  delete[] molcnt; delete[] molcnt_all;
  moldip     = new double[3 * nmol];
  moldip_all = new double[3 * nmol];
  molcnt     = new double[nmol];
  molcnt_all = new double[nmol];
  nmol_alloc = nmol;
}

/* ----------------------------------------------------------------------*/

void ComputeDipoleSamQEq::compute_vector()
{
  invoked_vector = update->ntimestep;

  double **x = atom->x;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
  imageint *image = atom->image;
  double *q = atom->q;
  int nlocal = atom->nlocal;

  // largest molecule-id present in the group, across all procs
  tagint maxmol_local = 0;
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) maxmol_local = MAX(maxmol_local, molecule[i]);
  tagint maxmol = 0;
  MPI_Allreduce(&maxmol_local, &maxmol, 1, MPI_LMP_TAGINT, MPI_MAX, world);

  bigint nmol = (bigint) maxmol + 1;
  if (nmol > nmol_alloc) grow_mol(nmol);

  for (bigint m = 0; m < 3 * nmol; m++) moldip[m] = 0.0;
  for (bigint m = 0; m < nmol; m++) molcnt[m] = 0.0;

  // Per-molecule NET CHARGE (local scratch, not a persistent member -- see the file-scope comment above;
  // compute_vector runs at most once/timestep, so a fresh nmol-sized buffer here is cheap next to the
  // existing moldip/molcnt Allreduce of the same size).
  double *molq = new double[nmol]();
  double *molq_all = new double[nmol];

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      double u[3];
      domain->unmap(x[i], image[i], u);
      tagint m = molecule[i];
      moldip[3 * m + 0] += q[i] * u[0];
      moldip[3 * m + 1] += q[i] * u[1];
      moldip[3 * m + 2] += q[i] * u[2];
      molcnt[m] += 1.0;
      molq[m] += q[i];
    }

  MPI_Allreduce(moldip, moldip_all, (int)(3 * nmol), MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(molcnt, molcnt_all, (int) nmol, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(molq, molq_all, (int) nmol, MPI_DOUBLE, MPI_SUM, world);

  double total[3] = {0.0, 0.0, 0.0};
  double permol_sum = 0.0;
  bigint npop = 0;
  for (bigint m = 0; m < nmol; m++) {
    if (molcnt_all[m] > 0.0) {
      double dx = moldip_all[3 * m + 0];
      double dy = moldip_all[3 * m + 1];
      double dz = moldip_all[3 * m + 2];
      total[0] += dx; total[1] += dy; total[2] += dz;
      // (a) mol==0 is the pooled non-molecular fragment (see the file-scope comment) -- excluded from the
      // per-molecule mean |dipole| headline stat (the FQ-water number this compute exists for). It still
      // contributes to the system TOTAL dipole above (a legitimate, if origin-dependent-for-a-charged-
      // system, Sigma q_i r_i quantity).
      if (m != 0) { permol_sum += sqrt(dx * dx + dy * dy + dz * dz); npop++; }
      // (b) any molecule (incl. the mol-0 pool) with a nonzero net charge has an origin/unwrap-convention-
      // dependent dipole -- flag it once instead of silently reporting a convention-dependent number.
      if (!warned_nonneutral_mol && fabs(molq_all[m]) > 1.0e-6) {
        warned_nonneutral_mol = true;
        if (comm->me == 0)
          error->warning(FLERR, "compute dipole/samqeq: molecule {} has net charge {:.6f}; its dipole is"
                         "origin-dependent", (long) m, molq_all[m]);
      }
    }
  }

  delete[] molq; delete[] molq_all;

  const double sc = unit_debye ? conv : 1.0;
  vector[0] = total[0] * sc;
  vector[1] = total[1] * sc;
  vector[2] = total[2] * sc;
  vector[3] = sqrt(total[0] * total[0] + total[1] * total[1] + total[2] * total[2]) * sc;
  vector[4] = (npop > 0) ? conv * permol_sum / (double) npop : 0.0;   // always Debye
}

/* ----------------------------------------------------------------------*/

double ComputeDipoleSamQEq::compute_scalar()
{
  if (invoked_vector != update->ntimestep) compute_vector();
  invoked_scalar = update->ntimestep;
  scalar = vector[4];          // mean per-molecule |dipole| in Debye
  return scalar;
}
