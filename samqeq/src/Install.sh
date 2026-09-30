# Install/unInstall package files in LAMMPS
# mode = 0/1/2 for uninstall/install/update
# (mirrors src/QEQ/Install.sh convention)

mode=$1

# enforce using portable C locale
LC_ALL=C
export LC_ALL

# arg1 = file, arg2 = file it depends on
action () {
  if (test $mode = 0) then
    rm -f ../$1
  elif (! cmp -s $1 ../$1) then
    if (test -z "$2" || test -e ../$2) then
      cp $1 ..
      if (test $mode = 2) then
        echo "  updating src/$1"
      fi
    fi
  elif (test -n "$2") then
    if (test ! -e ../$2) then
      rm -f ../$1
    fi
  fi
}

# all package files
# pppm_samqeq depends on the KSPACE package (PPPM base class); only install it
# when KSPACE is present (arg2 of action gates on ../pppm.h existing).
# fix_qeq_base_sam + fix_acks2_sam are the self-contained ACKS2/QEq base, so
# SAMQEQ does not require the REAXFF package; core-only deps.
action fix_qeq_base_sam.cpp
action fix_qeq_base_sam.h
action fix_acks2_sam.cpp
action fix_acks2_sam.h
action fix_qeq_sam.cpp
action fix_qeq_sam.h
action fix_qeq_sam_lr.cpp
action fix_qeq_sam_xl.cpp
action fix_qeq_sam_quartic.cpp
action fix_qeq_sam_modify.cpp
action fix_qeq_sam_ilu.cpp
action pair_coul_shield_intra.cpp
action pair_coul_shield_intra.h
action compute_dipole_samqeq.cpp
action compute_dipole_samqeq.h
action pppm_samqeq.cpp pppm.h
action pppm_samqeq.h pppm.h
