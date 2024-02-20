/*
 * $Log:        pend.c,v $
 * Revision 3.0  2007/07/12  10:25  arge
 * DASSLC version
 *
 */

#include "../src/dasslc.h"

BOOL residuals (PTR_ROOT *, REAL, REAL *, REAL *, REAL *, BOOL *);
BOOL jacobian (PTR_ROOT *, REAL, REAL *, REAL *, REAL, void *, BOOL (*)(PTR_ROOT *, REAL, REAL *, REAL *, REAL *, BOOL *));

int dae_index=3;
double g=9.8, L=1;

void
main (int argc, char **argv)
{
 PTR_ROOT root;
 BOOL error;
 char *inputfile;
 REAL t = 0., tout = 0.1, *y;
 FILE *fp=fopen("../examples/pend.mat","wt");  // MAT file to open with "load -ascii" in MATLAB
 int index[4][5] = {{0, 0, 0, 0, 0},	// index 0 formulation (with drift-off effect)
					{1, 1, 1, 1, 1},	// index 1 formulation
					{1, 1, 2, 2, 2},	// index 2 formulation
					{1, 1, 2, 2, 3}};	// index 3 formulation

 if (argc > 1) inputfile = argv[1];
 else inputfile = "../examples/pend.dat";

 printf("Index formulation [0, 1, 2, or 3] = "); scanf("%d", &dae_index);

 if (dae_index < 0 || dae_index > 3)
	{
	 printf("******* error: Invalid differential index of the DAE system\n");
	 exit(1);
	}

 daSetup (inputfile, &root, residuals, 0, t, NULL, NULL, index[dae_index], NULL, NULL);
 y=root.y;

 if (error = dasslc (INITIAL_COND, &root, residuals, &t, tout, jacobian, NULL), error < 0)
   printf ("error = %d\n", error);
 else
  {
   fprintf(fp,"%g %g %g %g %g %g\n",t,y[0],y[1],y[2],y[3],y[4]);
   for (; tout <= 100.; tout += .1)
      if (error = dasslc (TRANSIENT, &root, residuals, &t, tout, jacobian, NULL), error < 0)
	{
	 if (root.savefile) fprintf (root.savefile,"******* error = %d\n", error);
	 break;
	}
   else fprintf(fp,"%g %g %g %g %g %g\n",t,y[0],y[1],y[2],y[3],y[4]);
  }

 fclose(fp);
 daStat (root.savefile, &root);
 daFree (&root);
}					/* main */

BOOL
residuals (PTR_ROOT *root, REAL t, REAL *y, REAL *yp, REAL *res, BOOL *jac)
{
 BOOL error = FALSE;
 FAST int i, k;
 int rank, *index;

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
	    case 0: res[0] = yp[0] - y[2] ; break;
	    case 1: res[1] = yp[1] - y[3] ; break;
	    case 2: res[2] = yp[2] + y[4]*y[0] ; break;
	    case 3: res[3] = yp[3] + y[4]*y[1] + g ; break;
 	    case 4: switch (dae_index)
				{
				 case 0: res[4] = yp[4] + 3*y[3]*g/(L*L); break;			  // index 0 formulation
				 case 1: res[4] = y[2]*y[2]+y[3]*y[3]-g*y[1]-L*L*y[4]; break; // index 1 formulation
				 case 2: res[4] = y[0]*y[2]+y[1]*y[3]; break;				  // index 2 formulation
				 case 3: res[4] = y[0]*y[0]+y[1]*y[1]-L*L; break;			  // index 3 formulation
				}
				break;
	    default: error = -1;
	   }
    }

 return (error);
}					/* residuals */


#define PD(i,j) (*(pd + rank * (i) + j))

BOOL
jacobian (PTR_ROOT *root, REAL t, REAL *y, REAL *yp, REAL cj, void *ja,
	  BOOL (*residuals)(PTR_ROOT *, REAL, REAL *, REAL *, REAL *, BOOL *))
{
 int rank = root -> rank;
 REAL *pd = (REAL *)ja;

 PD(0,0) = cj;
 PD(0,2) = -1;
 PD(1,1) = cj;
 PD(1,3) = -1;
 PD(2,0) = y[4];
 PD(2,2) = cj;
 PD(2,4) = y[0];
 PD(3,1) = y[4];
 PD(3,3) = cj;
 PD(3,4) = y[1];

 switch (dae_index)
	{
	 case 0:
 	// index 0 formulation
	 PD(4,3) = 3*g/(L*L);
	 PD(4,4) = cj;
	 break;

	// index 1 formulation
	 case 1:
	 PD(4,1) = -g;
	 PD(4,2) = 2*y[2];
	 PD(4,3) = 2*y[3];
	 PD(4,4) = -L*L;
	 break;

 	// index 2 formulation
	 case 2:
	 PD(4,0) = y[2];
	 PD(4,1) = y[3];
	 PD(4,2) = y[0];
	 PD(4,3) = y[1];
	 break;
 	// index 3 formulation
	 case 3:
	 PD(4,0) = 2*y[0];
	 PD(4,1) = 2*y[1];
	}

 return FALSE;
}					/* jacobian */

