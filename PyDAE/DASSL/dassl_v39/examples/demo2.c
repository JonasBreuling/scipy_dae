/*
 * $Log:        demo2.c,v $
 * Revision 2.5  2005/07/09  10:20  arge
 * DASSLC version
 *
 */

#include "../src/dasslc.h"

DASSLC_RES residuals;
DASSLC_JAC jacobian;
DASSLC_PSOL psolver;

void
main (int argc, char **argv)
{
 PTR_ROOT root;
 BOOL error;
 char *inputfile;
 REAL t = 0., tout = 0.1, *y = NULL;
 int rank = 0;

 if (argc > 1) inputfile = argv[1];
 else
	{
	 inputfile = "?";
	 rank = 2;
	 y = NEW(REAL, rank, "Y");
	 y[0] = y[1] = 1.0;
	}

 if (error = daSetup (inputfile, &root, residuals, rank, t, y, NULL, NULL, NULL, NULL), error)
 {
  printf ("Setup error = %d\n", error);
  exit(1);
 }

 if (*inputfile == '?') // redirect the statistics output from screen to a file
 {
  root.filename = "Demo2.save";
 
  if (!(root.savefile = fopen (root.filename, "w+")))
  {
   printf ("Cannot open savefile \"%s\"\n", root.filename);
   exit(1);
  }

  daPrint_header (root.savefile, inputfile); // header
  daPrint_run (root.savefile, &root); // show the solver parameters
 }

 if (error = dasslc (INITIAL_COND, &root, residuals, &t, tout, jacobian, psolver), error < 0)
   printf ("error = %d\n", error);
 else
   for (; tout <= 1.; tout += .1)
      if (error = dasslc (TRANSIENT, &root, residuals, &t, tout, jacobian, psolver), error < 0)
	{
	 printf ("error = %d\n", error);
	 break;
	}

 daStat (root.savefile, &root);

 root.iter.atol[0] = 1e-20;
 if (!root.iter.stol) root.iter.atol[1] = 1e-20;

 if (error >= 0)
   if (error = dasslc (STEADY_STATE, &root, residuals, &t, tout, jacobian, psolver), error < 0)
     printf ("error = %d\n", error);

 daStat (root.savefile, &root);
 daFree (&root);
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
