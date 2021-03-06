/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(sticky,FixSticky)

#else

#ifndef LMP_FIX_STICKY_H
#define LMP_FIX_STICKY_H

#include "fix.h"

namespace LAMMPS_NS {

class FixSticky : public Fix {
 public:
  FixSticky(class LAMMPS *, int, char **);
  ~FixSticky();
  int setmask();
  double compute_scalar();
//  void init();
  void post_integrate();
/*
  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *);
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);
  void grow_arrays(int);
  void copy_arrays(int, int, int);
  int pack_exchange(int, double *);
  int unpack_exchange(int, double *);
  double compute_vector(int);
  double memory_usage();
*/
 private:
  int me;
  int iatomtype,jatomtype;
  double cutoff;

  int localSwap, globalSwap;
  int totalSwap;

//  tagint **created;

//  tagint *copy;

//  class RanMars *random;
//  class NeighList *list;

//  int countflag,commflag;

//  int dedup(int, int, tagint *);

  // DEBUG

  void print_bb();
//  void print_copy(const char *, tagint, int, int, int, int *);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Invalid atom type in fix infect command

Self-explanatory.

E: Invalid bond type in fix infect command

Self-explanatory.

E: Cannot use fix infect with non-molecular systems

Only systems with bonds that can be changed can be used.  Atom_style
template does not qualify.

E: Inconsistent iparam/jparam values in fix infect command

If itype and jtype are the same, then their maxbond and newtype
settings must also be the same.

E: Fix infect cutoff is longer than pairwise cutoff

This is not allowed because bond creation is done using the
pairwise neighbor list.

E: Fix infect angle type is invalid

Self-explanatory.

E: Fix infect dihedral type is invalid

Self-explanatory.

E: Fix infect improper type is invalid

Self-explanatory.

E: Cannot yet use fix infect with this improper style

This is a current restriction in LAMMPS.

E: Fix infect needs ghost atoms from further away

This is because the fix needs to walk bonds to a certain distance to
acquire needed info, The comm_modify cutoff command can be used to
extend the communication range.

E: New bond exceeded bonds per atom in fix infect

See the read_data command for info on setting the "extra bond per
atom" header value to allow for additional bonds to be formed.

E: New bond exceeded special list size in fix infect

See the special_bonds extra command for info on how to leave space in
the special bonds list to allow for additional bonds to be formed.

E: Fix infect induced too many angles/dihedrals/impropers per atom

See the read_data command for info on setting the "extra angle per
atom", etc header values to allow for additional angles, etc to be
formed.

E: Special list size exceeded in fix infect

See the read_data command for info on setting the "extra special per
atom" header value to allow for additional special values to be
stored.

W: Fix infect is used multiple times or with fix bond/break - may not work as expected

When using fix infect multiple times or in combination with
fix bond/break, the individual fix instances do not share information
about changes they made at the same time step and thus it may result
in unexpected behavior.

*/
