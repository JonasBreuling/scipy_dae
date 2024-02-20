/*
 * $Log:        demo_recall.c,v $
 * Revision 3.3  2008/05/09  19:29  arge
 * DASSLC version
 *
 */

#include "../src/dasslc.h"

DASSLC_RES residuals;
DASSLC_JAC jacobian;
DASSLC_PSOL psolver;

#define MANY_SETUPS

//#define NO_INPUTFILE

void
main (int argc, char **argv)
{
 PTR_ROOT root;
 BOOL error;
 char *inputfile;
 REAL t, tout;
 int calls = 0;
#ifdef NO_INPUTFILE
 int rank = 2;
 double y[2], yp[2];
#else
 double *y=NULL, *yp=NULL;
 int rank = 0;
#endif

 if (argc > 1) inputfile = argv[1];
#ifdef NO_INPUTFILE
 else inputfile = "?";
#else
 else inputfile = "../examples/demo.dat";
#endif

do
{
 t = 0.; tout = 0.1; calls++;

#ifdef MANY_SETUPS
 if (error = daSetup (inputfile, &root, residuals, rank, t, y, yp, NULL, NULL, NULL), error)
#else
 if (calls==1 && (error = daSetup (inputfile, &root, residuals, rank, t, y, yp, NULL, NULL, NULL), error))
#endif
   {
    printf ("Setup error = %d\n", error);
    exit(1);
   }

 if (calls==1 && *inputfile == '?') // redirect the statistics output from screen to a file
 {
  root.filename = "demo.save";
 
  if (!(root.savefile = fopen (root.filename, "w+")))
  {
   printf ("Cannot open savefile \"%s\"\n", root.filename);
   exit(1);
  }

  root.y[0]=root.y[1]=1;
  root.yp[0]=root.yp[1]=0;

  daPrint_header (root.savefile, inputfile); // header
  daPrint_run (root.savefile, &root); // show the solver parameters
 }

#ifndef MANY_SETUPS
 root.y[0]=root.y[1]=1;
 root.yp[0]=root.yp[1]=0;
 root.bdf.h = 0;
#endif

 if (error = dasslc (INITIAL_COND, &root, residuals, &t, tout, jacobian, psolver), error < 0)
   printf ("INITIAL error = %d\n", error);
 else
   for (; tout <= 1.; tout += .1)
      if (error = dasslc (TRANSIENT, &root, residuals, &t, tout, jacobian, psolver), error < 0)
	{
	 printf ("TRANSIENT error = %d\n", error);
	 break;
	}

 daStat (root.savefile, &root);

 root.iter.atol[0] = 1e-20;
 if (!root.iter.stol) root.iter.atol[1] = 1e-20;

 if (error >= 0)
   if (error = dasslc (STEADY_STATE, &root, residuals, &t, tout, jacobian, psolver), error < 0)
     printf ("STEADY error = %d\n", error);

 daStat (root.savefile, &root);

#ifdef MANY_SETUPS
 daFree (&root);
#endif

 printf ("call = %d\n", calls);

}while(error >= 0 && calls < 1000);

}					/* main */

BOOL
residuals (PTR_ROOT *root, REAL t, REAL *y, REAL *yp, REAL *res, BOOL *jac)
{
 BOOL error = FALSE;
 FAST int i, k;
 int rank, *index;
 PRIV REAL d = -.01, e = .01;

 if (*jac)
   {
    rank = root -> jac.rank;
    index = root -> jac.index;
   }
 else rank = root -> rank;

 for (k = 0; k < rank; k++)
    {
     i = (*jac ? index[k] : k);
     switch (i)
	   {
	    case 0: res[i] = yp[0] - d * y[0] - y[1] / e; break;
	    case 1: res[i] = yp[1] + y[1] / e; break;
	    default: error = -1;
	   }
    }

 return (error);
}					/* residuals */

#define PD(i,j) (*(pd + rank * (i) + j))

BOOL
jacobian (PTR_ROOT *root, REAL t, REAL *y, REAL *yp, REAL cj, void *ja, DASSLC_RES *residuals)
{
 int rank = root -> rank;
 REAL *pd = (REAL *)ja;
 PRIV REAL d = -.01, e = .01;

 PD(0,0) = cj - d;
 PD(0,1) = -1.0 / e;
 PD(1,1) = cj + 1.0 / e;

 return FALSE;
}					/* jacobian */


/* preconditioner: P x = b, where b also returns x */
BOOL
psolver (PTR_ROOT *root, REAL *b, DASSLC_RES *residuals)
{
 PRIV REAL d = -.01, e = .01;
 REAL cj = root -> iter.cj, c = 1. / (1. + e * cj);

 b[0] = (b[0] + b[1] * c) / (cj - d);
 b[1] *= e * c;

 return FALSE;
}					/* psolver */
