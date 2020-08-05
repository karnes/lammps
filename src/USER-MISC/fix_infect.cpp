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

#include "fix_infect.h"
#include "citeme.h"
#include <mpi.h>
#include <cstring>
#include "update.h"
#include "atom.h"
#include "force.h"
#include "modify.h"
#include "pair.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "random_mars.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;

static const char cite_fix_infect[] =
  "fix infect:\n\n"
  "@Article{Gissinger17,\n"
  " author = {J. R. Gissinger, B. D. Jensen, K. E. Wise},\n"
  " title = {Modeling chemical reactions in classical molecular dynamics simulations},\n"
  " journal = {Polymer},\n"
  " year =    2017,\n"
  " volume =  128,\n"
  " pages =   {211--217}\n"
  "}\n\n";


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
FixInfect::FixInfect(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
   list(NULL)  //jjk not sure if i need these
{
  if (lmp->citeme) lmp->citeme->add(cite_fix_infect);

  if (narg != 7) error->all(FLERR,"Illegal fix infect command: "
                           "incorrect number of arguments.");

//  MPI_Comm_rank(world,&me); //jjk not sure if needed

  nevery = force->inumeric(FLERR,arg[3]);
  if (nevery <= 0) error->all(FLERR,"Illegal fix infect command: Nevery <= 0.");

  iatomtype = force->inumeric(FLERR,arg[4]);
  jatomtype = force->inumeric(FLERR,arg[5]);
  double cutoff = force->numeric(FLERR,arg[6]);
  cutsq = cutoff*cutoff;

  if (iatomtype < 1 || iatomtype > atom->ntypes ||
      jatomtype < 1 || jatomtype > atom->ntypes ||
	  iatomtype == jatomtype)
    error->all(FLERR,"Invalid atom type(s) in fix infect command.");
  if (cutoff < 0.0) error->all(FLERR,"Illegal fix infect command: Negative cutoff distance.");

  if (!atom->sphere_flag)
    error->all(FLERR,"Fix infect requires atom style sphere.");
  
//  atom->add_callback(0); //jjk don't know what this is
//  countflag = 0; // jjk don't know what this is

  // set comm sizes needed by this fix
  // forward is big due to comm of broken bonds and 1-2 neighbors

//  comm_forward = MAX(2,2+atom->maxspecial); // jjk don't know what this is
//  comm_reverse = 2;  // jjk don't know what this is

  // allocate arrays local to this fix

//  nmax = 0; // jjk not sure if needed
//  distsq = NULL; // jjk not sure if needed

  scalar_flag = 1;
  totalInfect = globalInfect = localInfect = 0;
// Debug printout
  // printf("Nevery = %d i_type = %d j_type = %d cutoff = %e\n",nevery,iatomtype,jatomtype,cutoff);
}

/* ---------------------------------------------------------------------- */

FixInfect::~FixInfect() // jjk address this much later
{
  // unregister callbacks to this fix from Atom class

//  atom->delete_callback(id,0);


  // delete locally stored arrays

//  memory->destroy(bondcount);
//  memory->destroy(partner);
//  memory->destroy(finalpartner);
//  memory->destroy(distsq);
//  memory->destroy(created);
//  delete [] copy;
}

/* ---------------------------------------------------------------------- */

int FixInfect::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
//  mask |= POST_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixInfect::init()
{

  // check cutoff for iatomtype,jatomtype
  // jjk -- should check versus max distance in neighborlist. ghost cutoff?

  if (force->pair == NULL || cutsq > force->pair->cutsq[iatomtype][jatomtype] ||
    cutsq > force->pair->cutsq[iatomtype][iatomtype])
    error->all(FLERR,"Fix infect cutoff is longer than pairwise cutoff.");

  // need a half neighbor list, built every Nevery steps

  // jjk look at other files, see if this is best choice

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->fix = 1;
  neighbor->requests[irequest]->occasional = 1;

//  lastcheck = -1; // jjk don't know what this is
}

/* ---------------------------------------------------------------------- */

void FixInfect::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixInfect::setup(int /*vflag*/)
{
//  int i,j,m;

  // compute initial bondcount if this is first run
  // can't do this earlier, in constructor or init, b/c need ghost info

//  if (countflag) return;
//  countflag = 1;

  // count bonds stored with each bond I own
  // if newton bond is not set, just increment count on atom I
  // if newton bond is set, also increment count on atom J even if ghost
  // bondcount is long enough to tally ghost atom counts

//  int *num_bond = atom->num_bond;
//  int **bond_type = atom->bond_type;
//  tagint **bond_atom = atom->bond_atom;
//  int nlocal = atom->nlocal;
//  int nghost = atom->nghost;
//  int nall = nlocal + nghost;
//  int newton_bond = force->newton_bond;

//  for (i = 0; i < nall; i++) bondcount[i] = 0;
/*
  for (i = 0; i < nlocal; i++)
    for (j = 0; j < num_bond[i]; j++) {
      if (bond_type[i][j] == btype) {
        bondcount[i]++;
        if (newton_bond) {
          m = atom->map(bond_atom[i][j]);
          if (m < 0)
            error->one(FLERR,"Fix infect needs ghost atoms "
                       "from further away");
          bondcount[m]++;
        }
      }
    }
*/
  // if newton_bond is set, need to sum bondcount

//  commflag = 1;
//  if (newton_bond) comm->reverse_comm_fix(this,1);
}

/* ---------------------------------------------------------------------- */

void FixInfect::post_integrate()
{
  int i,j,k,m,n,ii,jj,inum,jnum,itype,jtype,n1,n2,n3,possible;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  int *ilist,*jlist,*numneigh,**firstneigh;
  tagint *slist;

  if (update->ntimestep % nevery) return;

  // check that all procs have needed ghost atoms within ghost cutoff
  // only if neighbor list has changed since last check
  // needs to be <= test b/c neighbor list could have been re-built in
  //   same timestep as last post_integrate() call, but afterwards
  // NOTE: no longer think is needed, due to error tests on atom->map()
  // NOTE: if delete, can also delete lastcheck and check_ghosts()

  //if (lastcheck <= neighbor->lastcall) check_ghosts();

  // acquire updated ghost atom positions
  // necessary b/c are calling this after integrate, but before Verlet comm
/*
  comm->forward_comm();

  // forward comm of bondcount, so ghosts have it
// jjk don't think i need this
  commflag = 1;
  comm->forward_comm_fix(this,1);

  int nlocal = atom->nlocal;
  int nall = atom->nlocal + atom->nghost;

  memory->create(distsq,nmax,"infect:distsq");
  }

  int nlocal = atom->nlocal;
  int nall = atom->nlocal + atom->nghost;
*/
  // loop over neighbors of my atoms
  // each atom sets one closest eligible partner atom ID
  localInfect = 0;

  neighbor->build_one(list);

  double **x = atom->x;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  neighbor->build_one(list,1); //jjk not sure about this one
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    if (!(mask[i] & groupbit)) continue;
    itype = type[i];
	if (itype!=iatomtype && itype!=jatomtype) continue;
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      if (!(mask[j] & groupbit)) continue;
      jtype = type[j];
	  if (jtype==itype) continue;
	  if (jtype!=iatomtype && jtype!=jatomtype) continue;
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      if (rsq >= cutsq) continue;

      if (rsq < cutsq) {
	    atom->type[i]=atom->type[j]=jatomtype;	 
	    localInfect++;	
	  }		
    }
  }

  MPI_Allreduce(&localInfect,&globalInfect,1,MPI_INT,MPI_SUM,world);
  if(comm->me==0){
    totalInfect+=globalInfect;
  }
}

/* ---------------------------------------------------------------------- */

double FixInfect::compute_scalar()
{
//  if (n == 0) return (double) createcount;
  return (double) totalInfect;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixInfect::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = nmax * sizeof(int);
  bytes = 2*nmax * sizeof(tagint);
  bytes += nmax * sizeof(double);
  return bytes;
}

/* ---------------------------------------------------------------------- */

void FixInfect::print_bb()
{
    printf("\n");
//    printf("TAG " TAGINT_FORMAT ": %d %d %d nspecial: ",atom->tag[i],
//           atom->nspecial[i][0],atom->nspecial[i][1],atom->nspecial[i][2]);
//    for (int j = 0; j < atom->nspecial[i][2]; j++) {
//      printf(" " TAGINT_FORMAT,atom->special[i][j]);
//    }
//    printf("\n");
//  }
}

/* ---------------------------------------------------------------------- */
/*
void FixInfect::print_copy(const char *str, tagint m,
                              int n1, int n2, int n3, int *v)
{
  printf("%s " TAGINT_FORMAT ": %d %d %d nspecial: ",str,m,n1,n2,n3);
  for (int j = 0; j < n3; j++) printf(" %d",v[j]);
  printf("\n");
}
*/
