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

FixStyle(infect,FixInfect)

#else

#ifndef LMP_FIX_INFECT_H
#define LMP_FIX_INFECT_H

#include "fix.h"

namespace LAMMPS_NS {

class FixInfect : public Fix {
 public:
  FixInfect(class LAMMPS *, int, char **);
  ~FixInfect();
  int setmask();
  void init();
  void init_list(int, class NeighList *);
  void setup(int);
  void post_integrate();
  double compute_scalar();
  double memory_usage();

 private:
  int me;
  int iatomtype,jatomtype;
  double cutoff;
  double cutsq;

  int localInfect,globalInfect;
  int totalInfect;

  tagint *copy;

  class RanMars *random;
  class NeighList *list;

  // DEBUG

  void print_bb();
//  void print_copy(const char *, tagint, int, int, int, int *);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal fix infect command:incorrect number of arguments.

Self-explanatory. Fix infect requires 7 arguments.

E: Illegal fix infect command: Nevery <= 0.

Self-explanatory.

E: Invalid atom type(s) in fix infect command.

Self-explanatory.Fix infect only works for defined atom types.

E: Illegal fix infect command: Negative cutoff distance.

Self-explanatory.

E: Fix infect requires atom style sphere.

Self-explanatory.

E: Fix infect cutoff is longer than pairwise cutoff.

Self-explanatory.

*/
