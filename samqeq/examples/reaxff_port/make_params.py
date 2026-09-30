#!/usr/bin/env python3
"""Write the samQEq parameter files of this example from the ReaxFF force fields.

ReaxFF doubles its tabulated hardness (eta = 2 * gamma_EEM) and fix qeq/reaxff works in eV with a Coulomb
constant of 14.4 eV A. Under units real samQEq reads chi and eta in kcal/mol and uses the exact Coulomb
constant 332.06371 kcal A/mol, so the ReaxFF operator is reproduced exactly by scaling chi and eta by
332.06371/14.4. The fourth column is ReaxFF's shielding parameter gamma, the cube-root kernel's parameter.
"""
S = 332.06371 / 14.4


def eem(ffield, elements):
    lines = open(ffield).read().splitlines()
    i = next(k for k, l in enumerate(lines) if "Nr of atoms" in l)
    nat = int(lines[i].split()[0]); i += 4
    out = {}
    for _ in range(nat):
        l1, l2 = lines[i].split(), lines[i + 1].split()
        out[l1[0]] = (float(l2[5]), 2.0 * float(l2[6]), float(l1[6]))   # chi, eta, gamma
        i += 4
    return [out[e] for e in elements]


for name, ff, els in (("rdx", "ffield.reax", "CHON"), ("water", "qeq_ff.water", "OH")):
    with open(f"{name}.param", "w") as f:
        f.write("# gamma_align kappa_bond r_ov r_loc lr_alpha lr_ewald  (units real: energies in kcal/mol)\n"
                "0.30  0.0  2.0  2.5  0.0  2\n")
        for t, (el, (chi, eta, g)) in enumerate(zip(els, eem(ff, els)), 1):
            f.write(f"{t}  {chi*S:.12f}  {eta*S:.12f}  {g:.4f}  0.0  {-7*S:.6f}  {3*S:.6f}   # {el}\n")
