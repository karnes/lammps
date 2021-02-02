/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
Contributing Author: John Karnes (karnes1@llnl.gov)
------------------------------------------------------------------------- */

#include "fix_sticky.h"
#include "citeme.h"
#include <mpi.h>
#include <cstring>
#include "update.h"
#include "atom.h"
#include "domain.h"
#include "force.h"
#include "modify.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;

static const char cite_fix_sticky[] =
  "fix sticky:\n\n"
  "@Article{Gissinger17,\n"
  " author = {J. R. Gissinger, B. D. Jensen, K. E. Wise},\n"
  " title = {Modeling chemical reactions in classical molecular dynamics simulations},\n"
  " journal = {Polymer},\n"
  " year =    2017,\n"
  " volume =  128,\n"
  " pages =   {211--217}\n"
  "}\n\n";

#define BIG 1.0e20
#define DELTA 16

/* ---------------------------------------------------------------------- */
/* What this fix should do:
   1) check atom type sphere
   2) have particle types i and j
   3) particle type 'i' will mutate into type 'j' if within distance r_cut
   4) loop over particles type 'i' 
           loop over particle type 'j'
		        if r_ij < r_cut
				   mutate i type into j type
   5) count number of i that mutate into j
   5) end?
   6) additional: if z position if any particle i is less than z_cut
                     mutate i into j
*/
FixSticky::FixSticky(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (lmp->citeme) lmp->citeme->add(cite_fix_sticky);

  if (narg != 7) error->all(FLERR,"Illegal fix sticky command: "
                           "incorrect number of arguments");

//  MPI_Comm_rank(world,&me); //jjk not sure if needed

//  nevery = force->inumeric(FLERR,arg[3]);
  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  if (nevery <= 0) error->all(FLERR,"Illegal fix sticky command");

//  iatomtype = force->inumeric(FLERR,arg[4]);
  iatomtype = utils::inumeric(FLERR,arg[4],false,lmp);
//  jatomtype = force->inumeric(FLERR,arg[5]);
  jatomtype = utils::inumeric(FLERR,arg[5],false,lmp);
//  double cutoff = force->numeric(FLERR,arg[6]);
  double cutoff = utils::numeric(FLERR,arg[6],false,lmp);
  cutoff = atof(arg[6]);
  if (cutoff < 0.0) error->all(FLERR,"Illegal fix sticky command");

  if (iatomtype < 1 || iatomtype > atom->ntypes ||
      jatomtype < 1 || jatomtype > atom->ntypes ||
	  iatomtype == jatomtype)
    error->all(FLERR,"Invalid atom type(s) in fix sticky command");

  if (!atom->sphere_flag)
    error->all(FLERR,"Fix sticky requires atom style sphere");
  scalar_flag = 1;
  totalSwap = 0;
}

/* ---------------------------------------------------------------------- */

FixSticky::~FixSticky() // jjk address this much later
{
  // unregister callbacks to this fix from Atom class

//  atom->delete_callback(id,0);

  // delete locally stored arrays

//  memory->destroy(type);
}

/* ---------------------------------------------------------------------- */

double FixSticky::compute_scalar()
{
  return (double) totalSwap;
}
/* ---------------------------------------------------------------------- */

int FixSticky::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= POST_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixSticky::post_integrate()
{
  int i;
  double **x = atom->x;

  if (update->ntimestep % nevery) return;

  int nLocal = atom->nlocal;
  int *type = atom->type;
  localSwap = 0;
  for(i=0; i<nLocal;i++){
    if(atom->type[i]==iatomtype){
	  if(atom->mask[i] & groupbit){
        if(x[i][2] < domain->boxlo[2] + cutoff){
		  atom->type[i] = jatomtype;
		  localSwap++;
		}
	  }
	}
  }
  MPI_Allreduce(&localSwap,&globalSwap,1,MPI_INT,MPI_SUM,world);
  if(comm->me==0){
	totalSwap+=globalSwap;
  }
  // DEBUG
//  print_bb();
}
/*
double FixSticky::memory_usage()
{
  int nmax = 0;
  return 0;
}
*/
/* ---------------------------------------------------------------------- */

void FixSticky::print_bb()
{
    printf("\n");
    printf("domain->boxlo = %f, LJdiameter2 = %f/n",domain->boxlo[2],cutoff);
}

