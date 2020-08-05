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

E: Illegal fix sticky command: incorrect number of arguments.

Self-explanatory. Fix sticky requires 7 arguments.

E: Illegal fix sticky command: Nevery <= 0.

Self-explanatory.

E: Illegal fix sticky command: Cutoff <= 0.

Self-explanatory.

E: Invalid atom type(s) in fix sticky command.

Self-explanatory.Fix sticky only works for defined atom types.

E: Fix sticky requires atom style sphere.

Self-explanatory.

*/
