/*
 * $Log:        dasslc.c,v $
 * Revision 3.8  12/06/04  01:30 {arge}
 * DASSLC version
 *
 * Revision 1.0  92/09/29  11:10 {arge}
 * - extract code from DAWRS package
 * Revision 1.1  96/06/21  01:15 {arge}
 * - create new sparse structure list
 * - set arSet_option as external function
 * Revision 1.2  96/07/24  09:19 {arge}
 * - minor changes in condition to step at tstop
 *   to match revisions up to 91/06/24 of dassl.f
 * - change type BOOL and SET to signed char
 *   because this is not default for CRAY compilers
 * Revision 1.3  97/01/29  21:25 {arge, fabiola} (based on DASPK)
 * - use norm vector with reciprocal weights
 * - correct N-R iterations counter
 * Revision 2.0  97/05/22  12:25 {arge, fabiola} (based on DASPK)
 * - default values for dampi and damps set to 1.0
 * - root -> t allways contains the current time
 * - parameter convtol added to iter structure
 * - created the iteration matrix types NONE, BAND_MODE, and USER_BAND
 *   and changed the order of the types
 * - current residual vector added in 'root' structure to use in 'jacobian'
 *   and/or 'psolver' routines
 * - 'residuals' routine added to 'jacobian' routine as argument
 * - preconditioner added to dasslc's arguments as an user's routine
 * - created iterative method to solve linear system (using Krylov method)
 * Revision 2.1  97/07/23  16:40 {arge, fabiola}
 * - minimize the excessive printing in debug matrix for sparse structure
 * Revision 2.2  97/11/12  10:45 {arge}
 * - correct bug in root.alloc for multiple calls of daSetup()
 * Revision 2.3  97/12/16  21:30 {arge}
 * - free iteration matrix in daFree() correctly, according to 'mtype'
 * Revision 2.4  99/12/04  22:10 {arge}
 * - move sparse modes to dasslc.h
 * - make sparse structure building more reliable
 * Revision 2.4.1 00/10/19  22:00 {arge}
 * - removed bug in arIs_integer() for Linux
 * Revision 2.4.2 02/10/24  15:30 {arge}
 * - correct use of daFree() when savefile is not given
 * Revision 2.4.3 03/03/29  23:40 {arge}
 * - initialize bdf structure before calling daInitial() during initial call (multiple restarts)
 * Revision 2.5 05/07/07  22:10 {arge}
 * - included user supplied linear algebra for iteration matrix (USER_ALGEBRA),
 *   the user must provide pointer to the functions: ujacFactor(), ujacSolve(), ujacPrint(), ujacFree()
 *   besides the <jacobian>() function
 * - daSetup() now may not receive any inputfile by setting inputfile="?" and the user
 *   may initialize the parameters in ROOT structure by hand through a pointer to the user_init(root)
 *   function provided by the user (if not used then it must point to a NULL)
 * - replaced exits in daSetup() by returning ERROR
 * Revision 3.0  07/07/12  08:30 {arge}
 * - included the differential index vector of the variables in daSetup() arguments to
 *   treat high-index problems, index=NULL can be used for index-0 and index-1 DAE systems
 * - created daNorm2() to deal with high-index DAEs
 * - increased the scratch area in PTR_ROOT to 3 * rank
 * Revision 3.1  07/10/08  11:25 {arge}
 * - make iters and krys pointers to their structures inside root.
 * - created arSetOption() function to set options externally (after calling daSetup()).
 * Revision 3.2  07/10/19  08:15 {arge}
 * - created set_rank() function to remove rank information when not necessary.
 * - changed atol default value to 1e-10.
 * Revision 3.3  08/03/07  00:45 {arge}
 * - remove memory leakage in sparse structure.
 * Revision 3.4  08/05/09  23:05 {arge}
 * - remove memory leakage in dafree().
 * Revision 3.5  09/01/26  16:25 {arge}
 * - fixed bug in initial stepsize and tolerances in function arSetOption().
 * Revision 3.6  09/05/19  10:20 {arge}
 * - fixed bug in daPrint_matrix() when debugging band matrix.
 * Revision 3.7  10/10/22  12:30 {arge}
 * - changed dependency matrix allocation form in Make_graph to solve larger problems.
 * - improve the sparse structure evaluation to avoid missing nonzeros
 * Revision 3.8  12/05/22  18:30 {arge, paulo}
 * - reduced the lower bound for numerical perturbation in Make_graph() to sround^1.5
 */

/*
 * C version of the well-known DASSL code of the Linda R. Petzold (LLNL),
 * 28/12/89 with a lot of new features (Revisions 1.0, 1.1 and 1.2), added
 * some features from DASPK of Linda R. Petzold, Peter N. Brown, Alan C.
 * Hindmarsh, and Clement W. Ulrich, 01/03/96 (Revisions 1.3 and 2.0).
 *
 *                          by
 *
 *                  Argimiro R. Secchi {Revision 1.0 - 3.2}
 *                          and
 *                  Fabiola A. Pereira {Revision 1.3 - 2.0}
 *
 *
 *            more details in DASSLC user manual.
 */

/*
  ***************************************************************************
  *             DIFFERENTIAL-ALGEBRAIC SYSTEM SOLVER (DASSLC)               *
  ***************************************************************************

			  (c) Copyright 1992-2007

			  Argimiro R. Secchi, UFRGS

			Simulation Laboratory - LASIM
            GIMSCOP (Group of Integration, Modeling, Simulation,
			         Control, and Optimization of Processes)
		      Department of Chemical Engineering
		    Federal University of Rio Grande do Sul
		      Porto Alegre, RS - Brazil
			 e-mail: arge@enq.ufrgs.br
*/

#ifndef lint
static char copyright[] =
    "DASSLC: Copyright (c) 1992,2007 by Argimiro R. Secchi - UFRGS";
static char RCSid[] =
    "@(#)$Header: dasslc.c,v 3.0 07/07/12 08:30:29 arge Exp $";
#endif /* lint */

#include "dasslc.h"

#include <sys/types.h>
#include <ctype.h>
#include <math.h>
#ifdef OLDUNIX
#  include <strings.h>
#  include <sys/time.h>
#  include <sys/resource.h>
#else
#  include <string.h>
#  include <time.h>
#endif /* UNIX */
#ifdef CRAY
#  include <memory.h>
#endif /* CRAY */

#ifdef HUGE
#  undef HUGE
#endif /* HUGE */
#define HUGE		1e200		/* redefined for setup use only */

#ifdef tolower
#  undef tolower
#endif /* tolower */
#define tolower(c)	(isupper (c) ? ((c) - 'A' + 'a') : (c))

#ifdef toupper
#  undef toupper
#endif /* toupper */
#define toupper(c)	(islower (c) ? ((c) + 'A' - 'a') : (c))

#define pfree(p)	if (p) free (p)
#define EXIT	{fprintf (stderr, \
		 "***ERROR Make_graph: read/write error in pertfile\n"); \
		 exit (1);}

typedef struct var_data		/* individual restrictions */
{
 char set;			/* 8-bits set control - print: sbit 1 */
 BOOL nonneg;			/* set to nonnegative solution - sbit 2 */
 int nonblanks;			/* number of nonblanks - sbit 3 */
 int *index;			/* index of nonblanks - sbit 3 */
 REAL rtol;			/* relative error tolerance - sbit 4 */
 REAL atol;			/* absolute error tolerance - sbit 4 */
} VAR_DATA;                     /* var_data */

typedef struct graph		/* graph structure used in sparse structure */
{
 char *dep;			/* the diff.-algebraic dependency matrix */
 int size;			/* row size */
 int *index;			/* column index vector */
} GRAPH;			/* graph */

typedef struct link		/* List of array information */
{
 int index;
 struct link *next;
} LINK;				/* link */

PRIV BOOL daPC (PTR_ROOT *, DASSLC_RES *, REAL *, REAL *, REAL *, DASSLC_JAC *, DASSLC_PSOL *);
PRIV BOOL daStatus (PTR_ROOT *, SET, BOOL, REAL, REAL);
PRIV BOOL daInitial (PTR_ROOT *, DASSLC_RES *, REAL, REAL *, REAL *, DASSLC_JAC *, DASSLC_PSOL *);
PRIV BOOL daSteady (PTR_ROOT *, DASSLC_RES *, REAL, REAL *, DASSLC_JAC *, DASSLC_PSOL *);
PRIV BOOL get_information (FILE *, char *);
PRIV BOOL set_inputfile (char *, char *, int);
PRIV BOOL set_database (FILE *, char *, int);
PRIV BOOL set_print (char *, int);
PRIV BOOL set_debug (char *, int);
PRIV BOOL set_rank (char *, int);
PRIV BOOL get_initial (char *, int *, int *);
PRIV BOOL get_value (char *, int *, int *, int);
PRIV BOOL get_residual (char *, int *, int *, int);
PRIV BOOL get_sparse (char *, int *, int *);
PRIV BOOL no_end_of_data (FILE *, char *, int *);
PRIV BOOL res_dep (GRAPH *, DASSLC_RES *, REAL *, char *, int, REAL, REAL, REAL);
PRIV BOOL set_band (FAST SPARSE_LIST *, BOOL, int, int, int *);
PRIV BOOL daPsolver (PTR_ROOT *, REAL *, DASSLC_RES *);
PRIV BOOL daAxV (PTR_ROOT *, int, REAL *, REAL *, REAL *, REAL *, REAL *,
		 long *, BOOL *, DASSLC_RES *, DASSLC_PSOL *);
PRIV BOOL daSpigm (PTR_ROOT *, KRYLOV *, int, REAL *, int *, int, BOOL *,
		   DASSLC_RES *, DASSLC_PSOL *);
PRIV SET set_mtype (char *, int *);
PRIV SET set_linearmode (char *, int *);
PRIV SET set_sparsemode (char *, int *);
PRIV SET set_savefile (char *, int *);
PRIV SET set_pertfile (char *, int *);
PRIV SET set_savepert (char *, int *);
PRIV void set_priority (void);
PRIV void init_globals (PTR_ROOT *);
PRIV void update_root (void);
PRIV void sparse_structure (DASSLC_RES *);
PRIV void copy_index (int, int, int *);
PRIV void free_prob (DATABASE *);
PRIV void Print_no_convergence (FILE *, PTR_ROOT *, REAL);
PRIV void heLS (int, int, REAL *, REAL *, REAL *);
PRIV struct table *is_command (char *);
PRIV GRAPH *Make_graph (DASSLC_RES *);
PRIV int get_text (FILE *, char *, int *);
PRIV int heQR (int, int, REAL *, REAL *, int);
PRIV char *is_option (char *, REALvar *, SHORTvar *, BOOLvar *, FUNCvar *);
PRIV REAL orth (int, int, int, int, REAL *, REAL *, REAL *);

#ifdef SPARSE
PRIV char *daSparse_matrix (int);
#endif /* SPARSE */


/* dasslc; solve a system of differential-algebraic
 * equations of the form F(t,y,yp) = 0 from time t
 * to tout.   The structure root is defined in the
 * header file DASSL.H.
 */
BOOL
dasslc (SET mode, PTR_ROOT *root, DASSLC_RES *residuals, REAL *t, REAL tout, DASSLC_JAC *jacobian,
	    DASSLC_PSOL *psolver)
{
 ITER_SET *iter = &root -> iter;
 BDF_DATA *bdf = &root -> bdf;
 BOOL done;
 FAST int i;
 int rank = root -> rank, n, sgh, perf;
 long *timepnt = &iter -> timepnt, nli, ncfl, newton, convfail;
 FAST REAL *rtol = iter -> rtol, *atol = iter -> atol;
 REAL tn, tstop, *wt = iter -> wt, r, fouround = 4. * iter -> roundoff,
      *hmin = &iter -> hmin, *h = &bdf -> h, upround = 25. * fouround,
      *phi = bdf -> phi, *y = root -> y, *yp = root -> yp, t0 = -tused ();

 root -> t = *t;

 if (root -> mode == ERROR) return daStatus (root, mode, INV_RUN, tout, t0);

 if ((root -> jac.mtype == NONE && iter -> linearmode == DIR_MODE) ||
     (root -> jac.mtype >= USER_DENSE && !jacobian))
   return daStatus (root, mode, INV_MTYPE, tout, t0);

 if (*t == tout) return daStatus (root, mode, INV_TOUT, tout, t0);

 for (done = TRUE, n = (iter -> stol ? 1 : rank), i = 0; i < n; i++)
    {					/* check tolerances */
     if (*rtol > 0. || *atol > 0.) done = FALSE;
     if (*rtol++ < 0.) return daStatus (root, mode, INV_RTOL, tout, t0);
     if (*atol++ < 0.) return daStatus (root, mode, INV_ATOL, tout, t0);
    }
 if (done) return daStatus (root, mode, ZERO_TOL, tout, t0);
 rtol = iter -> rtol; atol = iter -> atol;

 if (mode != root -> mode)		/* initial call */
   {
    *timepnt = 0;
					/* error weight vector */
    if (daWeights (rank, iter -> stol, rtol, atol, y, wt))
      return daStatus (root, mode, INV_WT, tout, t0);

    if (mode != STEADY_STATE)
      {
       FAST REAL *phi1;
       REAL hset = *h;
					/* minimum stepsize */
       *hmin = fouround * MAX(ABS(*t), ABS(tout));

       tn = tout - *t;			/* check initial interval */
       if (ABS(tn) < *hmin) return daStatus (root, mode, TOO_CLOSE, tout, t0);

       if (!*h) 			/* no initial stepsize */
	 {				/* this is very conservative */
	  *h = 1.e-3 * tn;
	  r = daNorm (rank, yp, wt);	/* derivative vector norm */
	  if (r > .5 / ABS(*h)) *h = .5 * SGNP(tn) / r;
	 }
       else				/* initial stepsize was input */
	 if (tn * *h < 0.) return daStatus (root, mode, TOUT_BEH_T, tout, t0);

       sgh = SGNP(*h);			/* upper-lower bounds to stepsize */
       if (iter -> hmax && iter -> hmax < ABS(*h)) *h = iter -> hmax * sgh;
       if (ABS(*h) < *hmin) *h = *hmin * sgh;

       if (iter -> iststop)		/* tstop was imposed */
	 {
	  tstop = iter -> tstop;
	  if ((tstop - *t) * sgh <= 0.)
	    return daStatus (root, mode, TSTOP_BEH_T, tout, t0);
	  if ((*t + *h - tstop) * sgh > 0.) *h = tstop - *t;
	  if ((tstop - tout) * sgh < 0.)
	    return daStatus (root, mode, TSTOP_BEH_TOUT, tout, t0);
	 }

       root -> mode = mode;

       if (mode == TRANSIENT) r = *h;
       else r = 1.0;

       phi1 = phi + rank;		/* bdf initialization */
       for (i = 0; i < rank; i++)
	  {
	   *phi++ = y[i]; *phi1++ = r * yp[i];
	  }

       bdf -> phase = FALSE;
       bdf -> tfar = tn = *t;
       bdf -> order = 1;
       bdf -> orderold = 0;
       bdf -> updateorder = 0;
       bdf -> ratefactor = 100.;
       *bdf -> psi = *h;

       iter -> jac = JAC;
       iter -> cj = iter -> cjold = 1. / *h;

       if (mode == INITIAL_COND)	/* compute initial derivative */
	 {
	  bdf -> hold = *h;
	  *h = hset;			/* keep user defined stepsize */

	  done = daInitial (root, residuals, *t, y, yp, jacobian, psolver);
	  return daStatus (root, mode, done, tout, t0);
	 }

       phi = bdf -> phi;
       bdf -> hold = 0.;

       if (root -> debug.bdf) daPrint_bdf (stdout, root);
      }
    else				/* compute steady-state */
      {
       root -> mode = mode;

       done = daSteady (root, residuals, *t, y, jacobian, psolver);
       return daStatus (root, mode, done, tout, t0);
      }
   }
 else					/* non-initial call */
   {
    if (mode == STEADY_STATE)		/* iterative mode for steady-state */
      {
       done = daSteady (root, residuals, *t, y, jacobian, psolver);
       return daStatus (root, mode, done, tout, t0);
      }
    else
      if (mode == INITIAL_COND)		/* iterative mode for initial deriv. */
	{
	 done = daInitial (root, residuals, *t, y, yp, jacobian, psolver);
	 return daStatus (root, mode, done, tout, t0);
	}

    *timepnt = 0;

    sgh = SGNP(*h);
    if (iter -> hmax && iter -> hmax < ABS(*h)) *h = iter -> hmax * sgh;

    if ((*t - tout) * sgh > 0.)
      return daStatus (root, mode, TOUT_BEH_T, tout, t0);

    tn = bdf -> tfar;			/* last timepoint */
    *hmin = fouround * MAX(ABS(tn), ABS(tout));

    if (iter -> iststop)		/* check for stopping point */
      {
       tstop = iter -> tstop;
       if ((tstop - tn) * sgh <= 0.)
	 return daStatus (root, mode, TSTOP_BEH_T, tout, t0);
       if ((tstop - tout) * sgh < 0.)
	 return daStatus (root, mode, TSTOP_BEH_TOUT, tout, t0);
      }

    if ((tn - tout) * sgh >= 0.)	/* check for continuation */
      {
       *t = tout; done = INTERPOL;
      }
    else if (iter -> istall && (tn - *t) * *h >= *hmin)
	   {
	    *t = tn; done = INTERMED;
	   }
	 else if (iter -> iststop)
		if (fabs ((double)(tn - tstop)) <= upround * (ABS(tn) + ABS(*h)))
		  {
		   *t = tstop; done = EXACT;
		  }
		else if ((tn + *h - tstop) * sgh > 0.) *h = tstop - tn;

    if (done)				/* reached any point */
      {
       (*timepnt)++; iter -> tottpnt++;
       root -> t = *t;
       daInterpol (rank, bdf, *t, y, yp);
      }
   }

 if (done)                              /* reached tout or tn */
   return daStatus (root, mode, done, tout, t0);

 if (iter -> linearmode == ITER_MODE)   /* save counters for analysis */
   {
    perf = 0;
    nli = root -> kry.nli;
    ncfl = root -> kry.ncfl;
    convfail = iter -> convfail;
    newton = iter -> newton;
   }

 do
   {
    if (++(*timepnt) > iter -> maxlen)	/* check for maximum wavelen */
      {
       (*timepnt)--;
       return daStatus (root, mode, MAX_TIMEPNT, tout, t0);
      }

    if (iter -> linearmode == ITER_MODE && *timepnt > 10 &&
	iter -> newton > newton)        /* performance analysis */
      {
       REAL dnw = iter -> newton - newton,
	    rli = (root -> kry.nli - nli) / dnw,
	    rncf = (iter -> convfail - convfail) / (REAL)*timepnt,
	    rlcf = (root -> kry.ncfl - ncfl) / dnw;

       if (rli > (REAL)root -> kry.maxl || rncf > 0.9 || rlcf > 0.9)
	 {
	  root -> kry.perf++;

	  if (root -> debug.nr)
	    if (perf++ < 10)
	      {
	       fprintf (stdout, "***WARNING dasslc: Poor iterative algorithm performance (time=%g)\n",
			*t);

	       if (rli > (REAL)root -> kry.maxl)
		 fprintf (stdout, "***DEBUG dasslc: Average no. of linear iterations = %g\n",
			  rli);

	       if (rncf > 0.9)
		 fprintf (stdout, "***DEBUG dasslc: Nonlinear conv. failure rate = %g\n",
			  rncf);

	       if (rlcf > 0.9)
		 fprintf (stdout, "***DEBUG dasslc: Linear conv. failure rate = %g\n",
			  rlcf);
	      }
	    else if (perf == 11)
		   fprintf (stdout, "***WARNING dasslc: Too many poor iterative performance...\n");
	 }
      }
					/* update wt */
    if (daWeights (rank, iter -> stol, rtol, atol, phi, wt))
      return daStatus (root, mode, INV_WT, tout, t0);

					/* check for too much accuracy */
    if ((r = daNorm (rank, phi, wt) * upround) > 1.)
      {
       for (n = (iter -> stol ? 1 : rank), i = 0; i < n; i++)
	  {
	   *rtol++ *= r; *atol++ *= r;
	  }
       rtol = iter -> rtol; atol = iter -> atol;
       fprintf (stderr, "***WARNING dassl: too much accuracy requested, error tolerances were multiplied by the factor = %g\n",
		(double)r);
      }

    *hmin = fouround * MAX(ABS(tn), ABS(tout));

					/* predictor-corrector */
    if ((done = daPC (root, residuals, t, y, yp, jacobian, psolver)) < 0)
      {
       (*timepnt)--; root -> t = *t;
       return daStatus (root, mode, done, tout, t0);
      }

    iter -> tottpnt++;
    tn = bdf -> tfar;

    if ((tn - tout) * sgh >= 0.)	/* check for continuation */
      {
       *t = tout; done = INTERPOL;
       daInterpol (rank, bdf, *t, y, yp);
      }
    else if (iter -> iststop)
	   if (fabs ((double)(tn - tstop)) <= upround * (ABS(tn) + ABS(*h)))
	     {
	      *t = tstop; done = EXACT;
	      daInterpol (rank, bdf, *t, y, yp);
	     }
	   else if ((tn + *h - tstop) * sgh > 0.) *h = tstop - tn;

    if (iter -> istall) done = INTERMED;

   } while (!done);

 root -> t = *t;

 return daStatus (root, mode, done, tout, t0);
}					/* dasslc */


/* daWeights; compute the inverse of the weights for norm vector
 * function. Returns TRUE if any weight is <= 0, or FALSE otherwise.
 */
BOOL
daWeights (int rank, BOOL stol, FAST REAL *rtol, FAST REAL *atol,
	   FAST REAL *y, FAST REAL *wt)
{
 FAST int i;

 for (i = 0; i < rank; i++, y++)
    {
     wt[i] = *rtol * ABS(*y) + *atol;

     if (!stol)
       {
	rtol++; atol++;
       }
    }

 for (i = 0; i < rank; i++, wt++)
    if (*wt <= 0.) return TRUE;
    else *wt = 1.0 / *wt;

 return FALSE;
}					/* daWeights */


/* daNorm; compute the weighted root-mean-square norm */
REAL
daNorm (int rank, FAST REAL *y, FAST REAL *wt)
{
 FASTER REAL ymax = fabs ((double)(*y * *wt)), sum;
 FAST int i;

 for (i=1; i < rank; i++)
    if (fabs ((double)(y[i] * wt[i])) > ymax) ymax = fabs ((double)(y[i] * wt[i]));

 if (!ymax) return 0.;

 for (sum = i = 0; i < rank; i++, y++, wt++)
    sum += (*y * *wt) / ymax * (*y * *wt) / ymax;

 return ymax * sqrt ((double)(sum / rank));
}					/* daNorm */


/* daNorm2; compute the weighted root-mean-square norm dealing with high-index DAEs */
REAL
daNorm2 (int rank, FAST REAL *y, FAST REAL *wt, REAL *hp)
{
 FASTER REAL ymax, sum;
 FAST int i;

 if (hp)
   {
	ymax = fabs ((double)(*y * *wt * *hp));
 
	for (i=1; i < rank; i++)
        if (fabs ((double)(y[i] * wt[i] * hp[i])) > ymax) ymax = fabs ((double)(y[i] * wt[i] * hp[i]));

    if (!ymax) return 0.;

    for (sum = i = 0; i < rank; i++, y++, wt++, hp++)
       sum += (*y * *wt * *hp) / ymax * (*y * *wt * *hp) / ymax;
   }
 else
   {
	ymax = fabs ((double)(*y * *wt));
    for (i=1; i < rank; i++)
    if (fabs ((double)(y[i] * wt[i])) > ymax) ymax = fabs ((double)(y[i] * wt[i]));

    if (!ymax) return 0.;

    for (sum = i = 0; i < rank; i++, y++, wt++)
       sum += (*y * *wt) / ymax * (*y * *wt) / ymax;
   }

 return ymax * sqrt ((double)(sum / rank));
}					/* daNorm2 */


/*
 * daInterpol; approximates the solution ans its derivatives at time t
 * by evaluating the polynomials used in the predictor-corrector algorithm,
 * and their derivatives, there.
 */
void
daInterpol (int rank, BDF_DATA *bdf, REAL t, REAL *y, REAL *yp)
{
 SHORT koldp1 = bdf -> orderold + 1;
 FAST int i, j;
 FAST REAL *phi = bdf -> phi, *psi = bdf -> psi;
 FASTER REAL gamma, hy = 1., hyp = 0.;
 REAL delta = t - bdf -> tfar;

 for (i = 0; i < rank; i++)
    {
     y[i] = *phi++; yp[i] = 0.;
    }

 for (gamma = delta / *psi, j = 1; j < koldp1; j++, psi++)
    {
     hyp = hyp * gamma + hy / *psi;
     hy *= gamma;
     gamma = (delta + *psi) / *(psi + 1);
     for (i = 0; i < rank; i++)
	{
	 y[i] += hy * *phi;
	 yp[i] += hyp * *phi++;
	}
    }
}					/* daInterpol */


/*
 * daPC; solve a system of differential-algebraic equations of the form
 * F(t,y,yp) = 0, for one step (from tn to tn + h).
 * The method used are modified divided difference, fixed leading coefficient
 * forms of backward differentiation formulas.  The code adjusts the stepsize
 * and order to control the local error per step.
 */
PRIV BOOL
daPC (PTR_ROOT *root, DASSLC_RES *residuals, REAL *t, REAL *y, REAL *yp, DASSLC_JAC *jacobian,
      DASSLC_PSOL *psolver)
{
 ITER_SET *iter = &root -> iter;
 BDF_DATA *bdf = &root -> bdf;
 BOOL converged, err, lerr, *jac = &iter -> jac, dbnr = root -> debug.nr,
      *phase = &bdf -> phase, dummy = FALSE;
 SHORT *k = &bdf -> order, *kold = &bdf -> orderold, kp1, kp2, km1, knew, nsp1;
 FAST SHORT convfail = 0, singmatrix = 0, errorfail = 0, newton;
 int rank = root -> rank;
 FAST int i, j;
 REAL told, tn = bdf -> tfar, *h = &bdf -> h, *hold = &bdf -> hold, cjr, ck,
      *cjold = &iter -> cjold, *cj = &iter -> cj, alpha0, alphas,
      ynorm, oldnorm, newnorm, *ratefactor = &bdf -> ratefactor,
      upround = 1.e2 * iter -> roundoff, est, *wt = iter -> wt, r,
      *delta = root -> scratch, *accum = delta + rank, *hp = root -> index ? accum + rank : NULL;
 FAST REAL *phi;
 FASTER REAL beta, gamma;

 nsp1 = bdf -> updateorder + 1;

 do
   {                                    /* compute bdf coefficients */
    kp2 = (kp1 = *k + 1) + 1; km1 = *k - 1; told = tn;
    if (*h != *hold || *k != *kold) nsp1 = 1;
    nsp1 = (bdf -> updateorder = MIN(nsp1, *kold + 2)) + 1;
    if (kp2 >= nsp1)
      {
       *bdf -> alpha = *bdf -> beta = *bdf -> sigma = 1.;
       *bdf -> gamma = 0.; beta = *h;
       for (i = 1; i < kp1; i++)
		  {
		   j = i - 1;
		   gamma = bdf -> psi[j];
		   bdf -> psi[j] = beta;
		   bdf -> beta[i] = bdf -> beta[j] * beta / gamma;
		   beta = gamma + *h;
		   bdf -> alpha[i] = *h / beta;
		   bdf -> sigma[i] = i * bdf -> sigma[j] * bdf -> alpha[i];
		   bdf -> gamma[i] = bdf -> gamma[j] + bdf -> alpha[j] / *h;
		  }
       bdf -> psi[*k] = beta;

       if (root -> debug.bdf) daPrint_bdf (stdout, root);
      }

    for (alpha0 = alphas = 0., i = 1; i < kp1; i++)
       {
		alpha0 -= bdf -> alpha[i-1];
		alphas -= 1. / (REAL)i;
       }

    r = *cj;
    *cj = - alphas / *h;		/* compute leading coefficients */
    cjr = *cj / *cjold;

    if (*cj != r) *ratefactor = 100.;

    ck = fabs ((double)(bdf -> alpha[*k] + alphas - alpha0));
    ck = MAX(ck, bdf -> alpha[*k]);	/* variable stepsize error coeff. */

					/* decide whether new jac. is needed */
    if (cjr <= iter -> jacrate || cjr * iter -> jacrate >= 1.) *jac = JAC;

    if (kp1 >= nsp1)			/* change phi to phi star */
      for (j = bdf -> updateorder, phi = bdf -> phi + j * rank; j < kp1; j++)
		 for (beta = bdf -> beta[j], i = 0; i < rank; i++) *phi++ *= beta;

    root -> t = *t = (tn += *h);	/* update time */

    do
      {
					/* predictor */
       for (phi = bdf -> phi, i = 0; i < rank; i++)
	      {
	       y[i] = *phi++; yp[i] = 0.;
	      }

       for (j = 1; j < kp1; j++)
	      for (gamma = bdf -> gamma[j], i = 0; i < rank; i++)
	         {
	          y[i] += *phi; yp[i] += gamma * *phi++;
	         }

	   if (hp) for (r = ABS(*h), i = 0; i < rank; i++)
				  if (r < 1) hp[i] = pow(r,MAX(0,root->index[i]-1));
				  else hp[i] = 1;

       ynorm = upround * daNorm2 (rank, y, wt, hp);

       converged = TRUE;		/* corrector */
       newton = 0;
       iter -> reseval++;
       if (err = (*residuals) (root, tn, y, yp, delta, &dummy), err)
		 {
		  if (dbnr) fprintf (stdout, "***DEBUG daPC: error %d in residuals (time=%g)\n",
							 err, (double)tn);
		  converged = FALSE; break;
		 }
					/* reevaluate the iteration matrix,
					 * pd = dF/dy + cj * dF/dyp,
					 * where  F(t,y,yp) = 0.
					 */
       if (*jac == JAC)
		 {
		  iter -> jaceval++;
		  *cjold = *cj; cjr = 1.;
		  *ratefactor = 100.;
		  if (err = daIteration_matrix (root, residuals, tn, y, yp, delta, accum, jacobian), err)
		    {
			 if (dbnr) fprintf (stdout, "***DEBUG daPC: error %d in iteration matrix (time=%g)\n",
								err, (double)tn);
			 converged = FALSE; break;
		    }
		  *jac = JAC_DONE;
		  singmatrix = 0;
		 }
					/* initialize err. accumulate vector */
       memset (accum, 0, rank * sizeof (REAL));

       do				/* corrector loop */
		 {
		  iter -> newton++;
					/* convergence accelerate factor */
		  r = (iter -> factor ? 2. / (1. + cjr) : 1.);
		  for (i = 0; i < rank; i++) delta[i] *= r;

		  if (iter -> linearmode == DIR_MODE)
	        daLUsolve (root, delta);	/* forward-backward substitution */
		  else
			{
			 lerr = daSolve_krylov (root, delta, &err, residuals, psolver);

			 if (err || lerr)
			   {
			    converged = FALSE;

			    if (!err)
				  {
				   if (lerr == ERROR) err = ERROR;
				   else if (!psolver && lerr == 3) err = SINGULAR;
			      }

				break;
			   }
	        }

		  for (phi = delta, i = 0; i < rank; i++)
			 {				/* update variables */
			  y[i] -= *phi; accum[i] -= *phi;
			  yp[i] -= *cj * *phi++;
			 }
					/* test for iteration convergence */
		  newnorm = daNorm2 (rank, delta, wt, hp);
       /* if (newnorm <= ynorm) break; */ /* comm: too early to exit loop */

	      if (newton > 0)		/* compute convergence rate */
			if ((r = pow ((double)(newnorm / oldnorm), 1. / (double)newton)) <= .9)
			  *ratefactor = r / (1. - r);

			else			/* the corrector has not yet conv. */
			 {
			  if (dbnr) fprintf (stdout, "***DEBUG daPC: High convergence rate %g in N-R (time=%g)\n",
							    (double)r, (double)tn);
			  converged = FALSE; break;
			 }
		  else
			{
			 if (newnorm <= ynorm) break;

			 oldnorm = newnorm;
			}

		  if (*ratefactor * newnorm <= iter -> convtol) break;

					/* many iterations without conv. */
		  if (++newton >= iter -> maxnewton)
			{
			 if (dbnr) fprintf (stdout, "***DEBUG daPC: Maximum newton iterations %d reached (time=%g)\n",
						 	    newton, (double)tn);
			 converged = FALSE; break;
			}

		  iter -> reseval++;		/* evaluate the residuals */
		  if (err = (*residuals) (root, tn, y, yp, delta, &dummy), err)
			{
			 if (dbnr) fprintf (stdout, "***DEBUG daPC: error %d in residuals (time=%g)\n",
								err, (double)tn);
			 converged = FALSE;
			}

	     } while (!err);

       if (err) break;

       if (converged)			/* the iteration has converged */
		 {
		  SHORT kdiff;
		  REAL erkm1, erkp1, terk, terkm1, terkm2, terkp1;

		  if (iter -> nonneg)		/* set the solution nonnegative */
			{
			 for (i = 0; i < rank; i++) delta[i] = MIN(y[i], 0.);
			 newnorm = daNorm2 (rank, delta, wt, hp);
			 if (newnorm > iter -> convtol)	/* the change is too large */
			   {
				if (dbnr) fprintf (stdout, "***DEBUG daPC: Norm (%g) too large to get nonnegative solution (time=%g)\n",
								  (double)newnorm, (double)tn);
				converged = FALSE; break;
			   }
			 for (i = 0; i < rank; i++) accum[i] -= delta[i];
			}

		  *jac = NO_JAC;

					/* estimate errors orders at k, k-1, k-2
					 * as if constant stepsize was used.
					 */
		  r = daNorm2 (rank, accum, wt, hp);
	      terk = kp1 * (est = bdf -> sigma[*k] * r); knew = *k;

	      if (*k > 1)
			{
			 for (phi = bdf -> phi + *k * rank, i = 0; i < rank; i++)
				delta[i] = accum[i] + *phi++;
			 terkm1 = *k * (erkm1 = bdf -> sigma[km1] * daNorm2 (rank, delta, wt, hp));

	         if (*k > 2)
			   {
				for (phi = bdf -> phi + km1 * rank, i = 0; i < rank; i++)
				   delta[i] += *phi++;
				terkm2 = km1 * bdf -> sigma[km1-1] * daNorm2 (rank, delta, wt, hp);

					/* lower order */
				if (MAX(terkm1, terkm2) <= terk)
				  {
				   knew = km1; est = erkm1;
				  }
			   }
			 else if (terkm1 <= .5 * terk)
					{
					 knew = km1; est = erkm1;
					}
			}

					/* calculate the local error, and
					 * test whether the current step
					 * is successful.
					 */
		  if (ck * r > 1.)
			{
			 if (dbnr) fprintf (stdout, "***DEBUG daPC: Local error %g > 1.0 too large (time=%g)\n",
						 	   (double)ck * r, (double)tn);
			 break;
			}

					/* determine best order and stepsize */
		  kdiff = *k - *kold; *kold = *k;
		  *hold = *h;

		  if (knew == km1 || *k == iter -> maxorder) *phase = TRUE;

		  if (*phase)
			{
					/* check whether decided to lower order,
					 * or already using maximum order,
					 * or stepsize not constant, or
					 * order raised in previous step.
					 */
			if (knew != km1 && *k < iter -> maxorder && kp2 < nsp1
				&& kdiff != 1)
			  {
			   for (phi = bdf -> phi + kp1 * rank, i = 0; i < rank; i++)
				  delta[i] = accum[i] - *phi++;
			   erkp1 = (terkp1 = daNorm2 (rank, delta, wt, hp)) / kp2;

			   if (*k > 1)
				 {			/* lower order */
				  if (terkm1 <= MIN(terk, terkp1))
					{
					 knew = km1; est = erkm1;
					}
					/* raise order */
				  else if (terkp1 < terk && *k < iter -> maxorder)
						 {
						  knew = kp1; est = erkp1;
						 }
				 }
			   else if (terkp1 < .5 * terk)
					  {
					   knew = kp1; est = erkp1;
					  }
			  }
			else if (knew == km1)      /* lower order */
				   {
					knew = km1; est = erkm1;
				   }
					/* determine the appropriate stepsize
					 * for the next step.
					 */
	        r = pow ((double)(2. * est + 1.e-4), -1. / (double)(knew + 1));
	        if (r >= 2.) *h *= 2.;
	        else if (r <= 1.) *h *= MAX(.5, MIN(.9, r));
	       }
					/* if phase = 0, increase order by one
					 * and double the stepsize.
					 */
	      else
	       {
	        knew = kp1; *h *= 2.;
	       }
					/* upper bound to stepsize */
	      if (iter -> hmax && iter -> hmax < ABS(*h))
            *h = iter -> hmax * SGNP(*h);

		  *k = knew;
					/* update differences for next step */
		  if (*kold < iter -> maxorder)
			for (phi = bdf -> phi + kp1 * rank, i = 0; i < rank; i++)
			   *phi++ = accum[i];
		  for (phi = bdf -> phi + *kold * rank, i = 0; i < rank; i++)
			 *phi++ += accum[i];
		  for (phi--, j = km1; j >= 0; j--)
			 for (i = 0; i < rank; i++, phi--) *(phi-rank) += *phi;

		  bdf -> tfar = tn;
		  return FALSE;
	     }
					/* if the iteration matrix is not
					 * current, re-do the step with a
					 * new iteration matrix.
					 */
       if (*jac == NO_JAC) *jac = JAC;
       if (*jac == JAC) iter -> rejnewton += newton;
      } while (*jac == JAC);

					/* no covergence with current iteration
					 * matrix, or singular iteration matrix,
					 * or unsuccessful step.
					 */
    if (!converged && !err) *jac = NO_JAC;
    iter -> rejnewton += newton;
    iter -> timepnt++;
    iter -> rejtpnt++;
    *phase = TRUE;
    tn = told;				/* restore time */

    if (kp1 >= nsp1)			/* restore the differences */
      for (j = bdf -> updateorder, phi = bdf -> phi + j * rank; j < kp1; j++)
	for (beta = 1 / bdf -> beta[j], i = 0; i < rank; i++) *phi++ *= beta;

					/* restore psi */
    for (i = 1; i < kp1; i++) bdf -> psi[i-1] = bdf -> psi[i] - *h;

    if (converged)			/* failure due to error test */
      {
       iter -> errorfail++;
       if (dbnr) fprintf (stdout, "***DEBUG daPC: %d test error failures.\n",
			  errorfail + 1);
       if (++errorfail == 1)
		 {
		  *k = knew;			/* keep order const. or lower by one */
		  r = .9 * pow ((double)(2. * est + 1.e-4), -1. / (double)(knew+1));
		  *h *= MAX(.25, MIN(.9, r));	/* compute new stepsize based on
					 * differences of the solutions.
					 */
		 }
       else if (errorfail > 2)		/* reduce stepsize by a factor of .25 */
			  {
			   *k = 1; *h *= .25;	/* set order to one */
			  }
			else
			  {
			   *k = knew; *h *= .25;	/* keep order const. or lower by one */
			  }
					/* go back and try this step again */
	   if (ABS(*h) >= iter -> hmin) continue;
       if (dbnr) fprintf (stdout, "***DEBUG daPC: Minimum stepsize %g reached\n",
			 (double)iter -> hmin);
       err = ERROR_FAIL;
      }

    else
      {					/* the newton iteration failed to
					 * convergence with a current
					 * iteration matrix.
					 */
       iter -> convfail++;
       if (err == SINGULAR)		/* the iteration matrix is singular */
		 {
		  if (dbnr) fprintf (stdout, "***DEBUG daPC: %d singular iteration matrix.\n",
						    singmatrix + 1);
		  *h *= .25;
					/* check to many failures of
					 * singular iteration matrix.
					 */
		  if (++singmatrix < iter -> maxsingular && ABS(*h) >= iter -> hmin)
	      continue;
	     }
					/* failure on residual or psolver functions */
       else if (err == ERROR) err = CALL_ERROR;
			else
			  {				/* check to many conv. failures */
			   if (dbnr) fprintf (stdout, "***DEBUG daPC: %d test convergence failures.\n",
								  convfail + 1);
			   *h *= .25;
	           if (++convfail < iter -> maxconvfail &&
		       ABS(*h) >= iter -> hmin) continue;
	           if (errorfail >= iter -> maxerrorfail) err = MAX_ERR_FAIL;
	           else if (err) err = FAIL;
					else err = CONV_FAIL;
			  }
      }
					/* restore y and yp to their last
					 * values at tn.
					 */
    *t = tn;
    daInterpol (rank, bdf, *t, y, yp);

    return err;
   } while (TRUE);
}					/* daPC */


/*
 * daLUsolve; manages the solution of linear system arising in the newton
 * iteration.  The adequate routine is called according to information
 * in root -> jac.mtype.
 */
void
daLUsolve (PTR_ROOT *root, REAL *delta)
{
 switch (root -> jac.mtype)
       {
	case USER_DENSE:
	case DENSE_MODE:
	     daSolve (root -> rank, 1, root -> jac.pivot,
		      (REAL *)root -> jac.matrix, delta);
	     break;

#ifdef SPARSE
	case USER_SPARSE:
	case SPARSE_MODE:
	     daSparse_solve (root -> jac.matrix, delta);
	     break;
#endif /* SPARSE */

	case USER_BAND:
	case BAND_MODE:
	     daSolveb (root -> rank, root -> jac.lband, root -> jac.uband,
		       1, root -> jac.pivot, (REAL *)root -> jac.matrix, delta);
	     break;

	case USER_ALGEBRA:
		 (*ujacSolve) (root, delta, delta);
		 break;

	case NONE:;
       }
}					/* daLUsolve */


/*
 * daIteration_matrix; computes the iteration matrix (dF/dy + cj * dFdyp,
 * where F(t,y,yp) = 0) either by numerical finite differencing or by
 * the user-supplied routine.
 */
BOOL
daIteration_matrix (PTR_ROOT *root, DASSLC_RES *residuals, REAL t, REAL *y,
		    REAL *yp, REAL *delta1, REAL *delta2, DASSLC_JAC *jacobian)
{
 JACOBIAN *jac = &root -> jac;
 BOOL err;
 int rank = root -> rank;
 REAL cj = root -> iter.cj;

 switch (jac -> mtype)
       {
	case USER_DENSE:		/* user-defined dense matrix */
	    {
	     REAL *matrix = (REAL *)jac -> matrix;

	     root -> res = delta1;
	     memset (matrix, 0, rank * rank * sizeof (REAL));
	     err = (*jacobian) (root, t, y, yp, cj, matrix, residuals);
	     if (root -> debug.matrix) daPrint_matrix (stdout, root, t);
	     if (!err)
	       if (root -> iter.linearmode == ITER_MODE)
	         root -> kry.decomp = TRUE;
	       else
	         if (daLU (rank, jac -> pivot, matrix)) err = SINGULAR;
	     break;
	    }

	case DENSE_MODE:		/* dense finite-difference matrix */
	    {
	     FAST SPARSE_LIST *spl = jac -> spl;
	     BOOL flag = TRUE;
	     FAST int i, j, *index, size;
	     FAST REAL *matrix = (REAL *)jac -> matrix;
	     FASTER REAL ysave, ypsave, del;
	     REAL sround = sqrt ((double)root -> iter.roundoff),
		  bound = pow ((double)sround, (double)1.5), h = root -> bdf.h,
		  *wt = root -> iter.wt, iwt;

	     memset (matrix, 0, rank * rank * sizeof (REAL));
	     for (j = 0; j < rank; j++, wt++, spl++)
		{
		 size = jac -> rank = spl -> size;
		 index = jac -> index = spl -> index;
		 ysave = y[j]; ypsave = yp[j];
		 iwt = (*wt ? 1. / ABS(*wt) : 0.);
		 del = h * ypsave;
		 del = (ysave + SGNP(del) * sround *
			MAX(ABS(ysave), MAX(iwt, ABS(del)))) - ysave;
		 if (ABS(del) < bound) del = SGNP(del) * bound;
		 y[j] += del; yp[j] += cj * del;
		 if (err = (*residuals) (root, t, y, yp, delta2, &flag), err)
		   return err;

		 del = 1. / del;
		 matrix = (REAL *)(jac -> matrix) + j;
					/* sparse evaluation */
		 if (flag && size < rank)
		   for (; size > 0; size--)
		      {
		       i = *index++;
		       *(matrix + i * rank) = (delta2[i] - delta1[i]) * del;
		      }
		 else			/* full evaluation */
		    for (flag = TRUE, i = 0; i < rank; i++, matrix += rank)
		       *matrix = (delta2[i] - delta1[i]) * del;

		 y[j] = ysave; yp[j] = ypsave;
		}
	     jac -> rank = 0;		/* clean jacobian structure */

	     if (root -> debug.matrix) daPrint_matrix (stdout, root, t);
	     if (root -> iter.linearmode == ITER_MODE)
	       root -> kry.decomp = TRUE;
	     else
	       if (daLU (rank, jac -> pivot, (REAL *)jac -> matrix))
		 err = SINGULAR;

	     break;
	    }

#ifdef SPARSE
	case USER_SPARSE:
	    {
	     char *matrix = jac -> matrix;

	     root -> res = delta1;
	     daSparse_clear (matrix);
	     err = (*jacobian) (root, t, y, yp, cj, matrix, residuals);
	     if (root -> debug.matrix) daPrint_matrix (stdout, root, t);
	     if (!err)
	       if (root -> iter.linearmode == ITER_MODE)
	         root -> kry.decomp = TRUE;
	       else
	         if (daSparse_LU (matrix)) err = SINGULAR;

	     break;
	    }

	case SPARSE_MODE:
	    {
	     FAST SPARSE_LIST *spl = jac -> spl;
	     BOOL flag = TRUE;
	     char *matrix = jac -> matrix;
	     FAST int i, j, *index, size;
	     FASTER REAL ysave, ypsave, del;
	     REAL sround = sqrt ((double)root -> iter.roundoff), pd, bound,
		  h = root -> bdf.h, *wt = root -> iter.wt, iwt;

	     bound = pow ((double)sround, (double)1.5);
	     daSparse_clear (matrix);
	     for (j = 0; j < rank; j++, wt++, spl++)
		{
		 size = jac -> rank = spl -> size;
		 index = jac -> index = spl -> index;
		 ysave = y[j]; ypsave = yp[j];
		 iwt = (*wt ? 1. / ABS(*wt) : 0.);
		 del = h * ypsave;
		 del = (ysave + SGNP(del) * sround *
			MAX(ABS(ysave), MAX(iwt, ABS(del)))) - ysave;
		 if (ABS(del) < bound) del = SGNP(del) * bound;
		 y[j] += del; yp[j] += cj * del;
		 if (err = (*residuals) (root, t, y, yp, delta2, &flag), err)
		   return err;

		 del = 1. / del;
					/* sparse evaluation */
		 if (flag && size < rank)
		   for (; size > 0; size--)
		      {
		       i = *index++;
		       daSparse_set (matrix, i, j, (delta2[i] - delta1[i]) * del);
		      }
		 else			/* full evaluation */
		    for (flag = TRUE, i = 0; i < rank; i++)
		       if ((pd = (delta2[i] - delta1[i]) * del) != 0.)
			 daSparse_set (matrix, i, j, pd);

		 y[j] = ysave; yp[j] = ypsave;
		}
	     jac -> rank = 0;		/* clean jacobian structure */

	     if (root -> debug.matrix) daPrint_matrix (stdout, root, t);
	     if (root -> iter.linearmode == ITER_MODE)
	       root -> kry.decomp = TRUE;
             else
	       if (daSparse_LU (matrix)) err = SINGULAR;

	     break;
	    }
#endif /* SPARSE */

	case USER_BAND:			/* user-defined band matrix */
	    {
	     REAL *matrix = (REAL *)jac -> matrix;

	     root -> res = delta1;
	     memset (matrix + jac -> lband * rank, 0, (jac -> lband +
		     jac -> uband + 1) * rank * sizeof (REAL));
	     err = (*jacobian) (root, t, y, yp, cj, matrix, residuals);
	     if (root -> debug.matrix) daPrint_matrix (stdout, root, t);
	     if (!err)
	       if (root -> iter.linearmode == ITER_MODE)
	         root -> kry.decomp = TRUE;
	       else
	         if (daLUb (rank, jac -> lband, jac -> uband, jac -> pivot,
		            matrix)) err = SINGULAR;
	     break;
	    }

	case BAND_MODE:			/* band finite-difference matrix */
	    {
	     FAST SPARSE_LIST *spl = jac -> spl;
	     BOOL flag = TRUE;
	     FAST int i, j, *index, size;
             int m = jac -> lband + jac -> uband;
	     FAST REAL *matrix = (REAL *)jac -> matrix;
	     FASTER REAL ysave, ypsave, del;
	     REAL sround = sqrt ((double)root -> iter.roundoff),
		  bound = pow ((double)sround, (double)1.5), h = root -> bdf.h,
		  *wt = root -> iter.wt, iwt;

             memset (matrix + jac -> lband * rank, 0, (m + 1) * rank * sizeof (REAL));
	     for (j = 0; j < rank; j++, wt++, spl++)
		{
		 size = jac -> rank = spl -> size;
		 index = jac -> index = spl -> index;
		 ysave = y[j]; ypsave = yp[j];
		 iwt = (*wt ? 1. / ABS(*wt) : 0.);
		 del = h * ypsave;
		 del = (ysave + SGNP(del) * sround *
			MAX(ABS(ysave), MAX(iwt, ABS(del)))) - ysave;
		 if (ABS(del) < bound) del = SGNP(del) * bound;
		 y[j] += del; yp[j] += cj * del;
		 if (err = (*residuals) (root, t, y, yp, delta2, &flag), err)
		   return err;

		 del = 1. / del;
		 matrix = (REAL *)(jac -> matrix) + j + (m - j) * rank;
					/* sparse evaluation */
		 if (flag && size < rank)
		   for (; size > 0; size--)
		      {
		       i = *index++;
		       *(matrix + i * rank) = (delta2[i] - delta1[i]) * del;
		      }
		 else			/* full evaluation */
		    for (flag = TRUE, i = 0; i < rank; i++, matrix += rank)
		       *matrix = (delta2[i] - delta1[i]) * del;

		 y[j] = ysave; yp[j] = ypsave;
		}
	     jac -> rank = 0;		/* clean jacobian structure */

	     if (root -> debug.matrix) daPrint_matrix (stdout, root, t);
	     if (root -> iter.linearmode == ITER_MODE)
	       root -> kry.decomp = TRUE;
	     else
	       if (daLUb (rank, jac -> lband, jac -> uband, jac -> pivot,
			  (REAL *)jac -> matrix))

	     break;
	    }

	case USER_ALGEBRA:
	    {
	     root -> res = delta1;
	     err = (*jacobian) (root, t, y, yp, cj, (void *)jac -> matrix, residuals);
	     if (root -> debug.matrix) daPrint_matrix (stdout, root, t);

	     break;
	    }

	case NONE: err = FALSE; root -> iter.jaceval--; break;

	default: fprintf (stderr, "***ERROR daIteration_matrix: type of iteration matrix evaluation = %d is unknown.\n",
			  jac -> mtype);
		 exit (1);
       }

 return err;
}					/* daIteration_matrix */


/*
 * daStatus; checks DASSL's return code and print an error message
 * when necessary.
*/
PRIV BOOL
daStatus (PTR_ROOT *root, SET mode, BOOL status, REAL tout, REAL ct)
{
 if (status < 0)
   {
    ITER_SET *iter = &root -> iter;
    char msg[80];
    REAL h = root -> bdf.h, t = root -> t;

    switch (status)
	  {
	   case MAX_TIMEPNT: break;
	   case INV_WT:
		sprintf (msg, "element of wt is or become < or = zero");
		break;
	   case ERROR_FAIL:
		sprintf (msg, "error test failed repeatedly");
		break;
	   case CONV_FAIL:
		sprintf (msg, "corrector could not converge after %d conv. fails",
			 (mode == STEADY_STATE ? STEADY_FACTOR :
			 (mode == INITIAL_COND ? INIT_FACTOR : 1))
			 * iter -> maxconvfail);
		break;
	   case SINGULAR:
		sprintf (msg, "iteration matrix was singular %d times sequentially",
			 iter -> maxsingular);
		break;
	   case MAX_ERR_FAIL:
		sprintf (msg, "non-conv. due %d repeated error test failures",
			 (mode == STEADY_STATE ? STEADY_FACTOR :
			 (mode == INITIAL_COND ? INIT_FACTOR : 1))
			 * iter -> maxerrorfail);
		break;
	   case FAIL:
		sprintf (msg, "non-conv. due %d repeated error in residuals",
			 (mode == STEADY_STATE ? STEADY_FACTOR :
			 (mode == INITIAL_COND ? INIT_FACTOR : 1))
			 * iter -> maxconvfail);
		break;
	   case CALL_ERROR:
		sprintf (msg, "incorrigible error in residuals");
		break;
	   case INV_MTYPE:
		sprintf (msg, "invalid matrix type to linear solver");
		break;
	   case INV_TOUT:
		sprintf (msg, "tin = tout = %g", (double)tout);
		break;
	   case INV_RUN:
		sprintf (msg, "last step was interrupted by an error");
		break;
	   case INV_RTOL:
		sprintf (msg, "some element of rtol is < or = zero");
		break;
	   case INV_ATOL:
		sprintf (msg, "some element of atol is < or = zero");
		break;
	   case ZERO_TOL:
		sprintf (msg, "all elements of rtol and atol are zero");
		break;
	   case TOO_CLOSE:
		sprintf (msg, "tout = %g is too close to tin = %g to start",
			 (double)tout, (double)t);
		break;
	   case TOUT_BEH_T:
		sprintf (msg, "tout = %g behind tin = %g (h = %g)",
			 (double)tout, (double)t, (double)h);
		break;
	   case TSTOP_BEH_T:
		sprintf (msg, "tstop = %g behind tin = %g (h = %g)",
			 (double)iter -> tstop, (double)t, (double)h);
		break;
	   case TSTOP_BEH_TOUT:
		sprintf (msg, "tstop = %g behind tout = %g (h = %g)",
			 (double)iter -> tstop, (double)tout, (double)h);
		break;
	   default: fprintf (stderr, "***ERROR daStatus: Unknown error status %d\n",
			     status);
		    exit (1);
	  }

    if (status != MAX_TIMEPNT)
      {
       if (root -> debug.conv) Print_no_convergence (stdout, root, t);

       fprintf (stderr, "***ERROR daStatus: Status = %d: %s\n", status, msg);

       if (root -> savefile != stderr)
	 fprintf (root -> savefile, "***ERROR daStatus: Status = %d: %s\n", status, msg);
      }
   }

 if (root -> print || root -> idxprint)
   daPrint_point (root -> savefile, root, root -> t);

 ct += tused ();
 switch (root -> mode)
       {
	case STEADY_STATE: root -> debug.t_info.t_steady += ct; break;
	case INITIAL_COND: root -> debug.t_info.t_initial += ct; break;
	case TRANSIENT: root -> debug.t_info.t_trans += ct; break;
	case ERROR: root -> debug.t_info.t_total += ct;
       }

 if (status < 0 && status != MAX_TIMEPNT) root -> mode = ERROR;

 return status;
}					/* daStatus */


/*
 * daInitial; take one step of size h or smaller with the backward Euler method
 * to find yp at the initial time t.  A modified damped Newton iteration is
 * used to solve the correction iteration.  The initial guess yp is used in
 * the prediction, and in forming the iteration matrix, but is not involved
 * in the error test.  This may have trouble converging it the initial guess
 * is not good, or if F(t,y,yp) depends nonlinearly on yp.
 */
PRIV BOOL
daInitial (PTR_ROOT *root, DASSLC_RES *residuals, REAL t, REAL *y, REAL *yp, DASSLC_JAC *jacobian,
		   DASSLC_PSOL *psolver )
{
 ITER_SET *iter = &root -> iter;
 BDF_DATA *bdf = &root -> bdf;
 BOOL converged, err, lerr, dbnr = root -> debug.nr, jac,
      dummy = FALSE;
 FAST SHORT convfail = 0, singmatrix = 0, errorfail = 0, newton;
 SHORT maxnewton = INIT_FACTOR * iter -> maxnewton;
 int rank = root -> rank;
 FAST int i;
 REAL tn, *h = &bdf -> h, *cj = &iter -> cj, ratefactor, est, *wt = iter -> wt,
      r, *delta = root -> scratch, *accum = delta + rank, *ys = bdf -> phi,
      *yps = bdf -> phi + rank, hs = *h, *hold = &bdf -> hold, oldnorm,
      ynorm, newnorm, *hp = root -> index ? accum + rank : NULL;
 FAST REAL *dy;

 *h = *hold;

 if (hp) for (r = ABS(*h), i = 0; i < rank; i++)
			if (r < 1) hp[i] = pow(r,MAX(0,root->index[i]-1));
			else hp[i] = 1;

 ynorm = 1.e2 * iter -> roundoff * daNorm2 (rank, ys, wt, hp);

 do
   {
    *cj = 1. / *h;			/* compute leading coefficient */

					/* update time */
    tn = t + *h;
					/* predictor */
    for (i = 0; i < rank; i++) y[i] += *h * yp[i];

    jac = JAC;
    converged = TRUE;
    newton = 0;

    do					/* corrector loop */
      {
       iter -> newton++;

       iter -> reseval++;
       if (err = (*residuals) (root, tn, y, yp, delta, &dummy), err)
		 {
		  if (dbnr) fprintf (stdout, "***DEBUG daInitial: error %d in residuals (time=%g)\n",
							err, (double)tn);
		  converged = FALSE; break;
		 }
					/* reevaluate the iteration matrix,
					 * pd = dF/dy + cj * dF/dyp,
					 * where  F(t,y,yp) = 0.
					 */
       if (jac == JAC)
		 {
		  iter -> jaceval++;
		  if (err = daIteration_matrix (root, residuals, tn, y, yp, delta, accum, jacobian), err)
			{
			 if (dbnr) fprintf (stdout, "***DEBUG daInitial: error %d in iteration matrix (time=%g)\n",
								err, (double)tn);
			 converged = FALSE; break;
			}
		  jac = JAC_DONE;
		  singmatrix = 0;
		  ratefactor = 1.e6;
		 }
					/* multiply res. by damping factor */
       for (i = 0; i < rank; i++) delta[i] *= iter -> dampi;

       if (iter -> linearmode == DIR_MODE)
		 daLUsolve (root, delta);	/* forward-backward substitution */
       else
		{
		 lerr = daSolve_krylov (root, delta, &err, residuals, psolver);

		 if (err || lerr)
		   {
			converged = FALSE;

			if (!err)
			  {
			   if (lerr == ERROR) err = ERROR;
			   else if (!psolver && lerr == 3) err = SINGULAR;
			  }

			break;
		   }
	    }

       for (dy = delta, i = 0; i < rank; i++)
		  {				/* update variables */
		   y[i] -= *dy;
		   yp[i] -= *cj * *dy++;
		  }
					/* test for iteration convergence */
       newnorm = daNorm2 (rank, delta, wt, hp);
       /* if (newnorm <= ynorm) break; */

       if (newton > 0)			/* compute convergence rate */
		 if (r = pow ((double)(newnorm / oldnorm), 1. / (double)newton), r <= .9)
		   ratefactor = r / (1. - r);
		 else				/* the corrector has not yet conv. */
		  {
		   if (dbnr) fprintf (stdout, "***DEBUG daInitial: High convergence rate %g in N-R (time=%g)\n",
							 (double)r, (double)tn);
		   converged = FALSE; break;
	      }
       else
		{
	     if (newnorm <= ynorm) break;

	     oldnorm = newnorm;
	    }

       if (ratefactor * newnorm <= iter -> convtol) break;

       if (++newton >= maxnewton)       /* many iterations without conv. */
		 {
		  if (dbnr) fprintf (stdout, "***DEBUG daInitial: Maximum newton iterations %d reached (time=%g)\n",
							 newton, (double)tn);
		  converged = FALSE; break;
		 }
					/* check for new iteration matrix */
       if (!(newton % iter -> maxjacfix)) jac = JAC;

      } while (TRUE);

    if (converged)			/* the iteration has converged */
      {
       if (iter -> nonneg)		/* set the solution nonnegative */
		 {
		  for (i = 0; i < rank; i++) delta[i] = MIN(y[i], 0.);
			 newnorm = daNorm2 (rank, delta, wt, hp);
		  if (newnorm > iter -> convtol)	/* the change is too large */
			{
			 if (dbnr) fprintf (stdout, "***DEBUG daInitial: Norm (%g) too large to get nonnegative solution (time=%g)\n",
								(double)newnorm, (double)tn);
			 converged = FALSE; goto fail;
			}

		  for (dy = delta, i = 0; i < rank; i++)
			 {
			  y[i] -= *dy;
			  yp[i] -= *cj * *dy++;
			 }
		 }

       for (i = 0; i < rank; i++)	/* error test */
		  accum[i] = y[i] - ys[i];

       if (est = daNorm2 (rank, accum, wt, hp), est <= 1.0)
		 {				/* restore initial values */
		  memcpy (y, ys, rank * sizeof (REAL));
		  *hold = *h;
		  *h = hs;

		  return converged;
		 }
       else
		if (dbnr) fprintf (stdout, "***DEBUG daInitial: Error %g (> 1.0) too large (time=%g)\n",
						  (double)est, (double)tn);
      }

					/* the backward Euler step failed.
					 * Restore y to the original
					 * values and relax yp by damping
					 * factor. Reduce stepsize an try
					 * again, if possible.
					 */
    fail:
    for (i = 0; i < rank; i++)
       yp[i] += iter -> dampi * (yps[i] - yp[i]);

    memcpy (y, ys, rank * sizeof (REAL));

    iter -> rejnewton += newton;
    iter -> timepnt++;
    iter -> rejtpnt++;

    if (converged)			/* failure due to error test */
      {
       iter -> errorfail++;
       if (dbnr) fprintf (stdout, "***DEBUG daInitial: %d test error failures.\n",
			  errorfail + 1);

       r = .9 / (2. * est + 1.e-4);
       *h *= MAX(.1, MIN(.5, r));	/* compute new stepsize based on
					 * differences of the solutions.
					 */
					/* go back and try this step again */
       if (ABS(*h) >= iter -> hmin && ++errorfail < INIT_FACTOR *
		   iter -> maxerrorfail)
		 err = 0;
       else
	     err = MAX_ERR_FAIL;
      }

    else
      {					/* the newton iteration failed to
					 * convergence with a current
					 * iteration matrix.
					 */
       iter -> convfail++;
       if (err == SINGULAR)		/* the iteration matrix is singular */
		 {
		  if (dbnr) fprintf (stdout, "***DEBUG daInitial: %d singular iteration matrix.\n",
							singmatrix + 1);
		  *h *= .25;
					/* check to many failures of
					 * singular iteration matrix.
					 */
		  if (++singmatrix < iter -> maxsingular && ABS(*h) >= iter -> hmin)
			err = 0;
		 }
					/* failure on residual or psolver functions */
       else if (err == ERROR) err = CALL_ERROR;
			else
			 {				/* check to many conv. failures */
			  if (dbnr) fprintf (stdout, "***DEBUG daInitial: %d test convergence failures.\n",
								 convfail + 1);
			  *h *= .25;
			  if (++convfail < iter -> maxconvfail * INIT_FACTOR &&
			      ABS(*h) >= iter -> hmin)
			    err = 0;
			  else
				if (err) err = FAIL;
				else err = CONV_FAIL;
			 }
      }

    if (hp) for (r = ABS(*h), i = 0; i < rank; i++)
			   if (r < 1) hp[i] = pow(r,MAX(0,root->index[i]-1));
			   else hp[i] = 1;

   } while (!err);

 if (ABS(*h) < iter -> hmin)
   {
    if (dbnr) fprintf (stdout, "***DEBUG daInitial: Minimum stepsize %g reached\n",
		       (double)iter -> hmin);
    err = ERROR_FAIL;
   }

 *hold = *h;				/* restore initial stepsize */
 *h = hs;

 return err;
}					/* daInitial */


/*
 * daSteady; solve a system of algebraic equations of the form F(to,y) = 0.
 */
PRIV BOOL
daSteady (PTR_ROOT *root, DASSLC_RES *residuals, REAL t, REAL *y, DASSLC_JAC *jacobian,
		  DASSLC_PSOL *psolver)
{
 ITER_SET *iter = &root -> iter;
 BOOL converged, err, lerr, jac, dbnr = root -> debug.nr, dummy = FALSE;
 FAST SHORT convfail = 0, singmatrix = 0, newton;
 SHORT maxnewton = STEADY_FACTOR * iter -> maxnewton;
 int rank = root -> rank;
 FAST int i;
 REAL oldnorm, newnorm, ratefactor, *wt = iter -> wt, r,
      *yp = root -> yp, *delta = root -> scratch, *res = delta + rank,
      upround = 1.e2 * iter -> roundoff, ynorm;
 FAST REAL *dy;

 memset (yp, 0, rank * sizeof (REAL));
 iter -> cj = 0;

 do
   {
    ynorm = upround * daNorm (rank, y, wt);
    jac = JAC;
    converged = FALSE;
    newton = 0;

    do
      {
       iter -> newton++;

       iter -> reseval++;
       if (err = (*residuals) (root, t, y, yp, delta, &dummy), err)
	 {
	  if (dbnr) fprintf (stdout, "***DEBUG daSteady: error %d in residuals\n",
			     err);
	  break;
	 }
					/* reevaluate the Jacobian,
					 * pd = dF/dy, where F(to,y) = 0.
					 */
       if (jac == JAC)
	 {
	  iter -> jaceval++;
	  if (err = daJacobian_matrix (root, residuals, t, y, yp, delta, res, jacobian), err)
	    {
	     if (dbnr) fprintf (stdout, "***DEBUG daSteady: error %d in Jacobian\n",
				err);
	     break;
	    }
	  jac = JAC_DONE;
	  singmatrix = 0;
	  ratefactor = 1.e6;
	 }

       if (iter -> linearmode == DIR_MODE)
	 daLUsolve (root, delta);	/* forward-backward substitution */
       else
	 {
	  lerr = daSolve_krylov (root, delta, &err, residuals, psolver);

	  if (err || lerr)
	    {
	     converged = FALSE;

	     if (!err)
	       {
		if (lerr == ERROR) err = ERROR;
		else if (!psolver && lerr == 3) err = SINGULAR;
	       }

	     break;
	    }
	 }

       for (dy = delta, i = 0; i < rank; i++) y[i] -= iter -> damps * *dy++;

					/* test for iteration convergence */
       newnorm = daNorm (rank, delta, wt);
       /* if (newnorm <= ynorm)
	 {
	  converged = TRUE; break;
	 } */

       if (newton > 0)			/* compute convergence rate */
	 if ((r = pow ((double)(newnorm / oldnorm), 1. / (double)newton)) <= .9)
	   ratefactor = r / (1. - r);
	 else				/* the corrector has not yet conv. */
	   {
	    if (dbnr) fprintf (stdout, "***DEBUG daSteady: High convergence rate %g in N-R\n",
			       (double)r);
	    break;
	   }
       else
	 {
	  if (newnorm <= ynorm)
	    {
	     converged = TRUE; break;
	    }

	  oldnorm = newnorm;
	 }

       if (ratefactor * newnorm <= iter -> convtol)
	 {
	  converged = TRUE; break;
	 }

       if (++newton >= maxnewton)	/* many iterations without conv. */
	 {
	  if (dbnr) fprintf (stdout, "***DEBUG daSteady: Maximum newton iterations %d reached\n",
			     newton);
	  break;
	 }

       if (!(newton % iter -> maxjacfix)) jac = JAC;

      } while (TRUE);

    if (converged)			/* the iteration has converged */
      {
					/* check convergence with new weights */
       if (daWeights (rank, iter -> stol, iter -> rtol, iter -> atol, y, wt))
	 return INV_WT;

       newnorm = daNorm (rank, delta, wt);
       if (!newton || ((r = pow ((double)(newnorm / oldnorm), 1. / (double)newton)) <= .9 &&
	   (r / (1. - r) * newnorm) <= iter -> convtol))
	 {
	  if (iter -> nonneg)		/* set the solution nonnegative */
	    {
	     for (i = 0; i < rank; i++) delta[i] = MIN(y[i], 0.);
	     newnorm = daNorm (rank, delta, wt);
	     if (newnorm > iter -> convtol)	/* the change is too large */
	       {
		if (dbnr) fprintf (stdout, "***DEBUG daSteady: Norm (%g) too large to get nonnegative solution \n",
				   (double)newnorm);
		goto fail;
	       }

	     for (dy = delta, i = 0; i < rank; i++) y[i] -= *dy++;
	    }

	  return converged;
	 }
      }
					/* no covergence with current Jacobian
					 * matrix, or singular Jacobian matrix.
					 */
    fail:
    iter -> rejnewton += newton;
    iter -> convfail++;

    if (err == SINGULAR)		/* the Jacobian matrix is singular */
      {
       if (dbnr) fprintf (stdout, "***DEBUG daSteady: %d singular Jacobian matrix.\n",
			  singmatrix + 1);
					/* check to many failures of
					 * singular Jacobian matrix.
					 */
       if (++singmatrix < iter -> maxsingular)
	 {
	  for (i = 0; i < rank; i++) y[i] = 0.75 * y[i] + upround;
	  err = 0;
	 }
      }
					/* failure on residual functions */
    else if (err == ERROR) err = CALL_ERROR;
	 else
	   {				/* check to many conv. failures */
	    if (dbnr) fprintf (stdout, "***DEBUG daSteady: %d test convergence failures.\n",
			       convfail + 1);
	    if (++convfail < iter -> maxconvfail * STEADY_FACTOR) err = 0;
	    else
	      if (err) err = FAIL;
	      else err = CONV_FAIL;
	   }
					/* update wt */
    if (daWeights (rank, iter -> stol, iter -> rtol, iter -> atol, y, wt))
      return INV_WT;

   } while (!err);

 return err;
}					/* daSteady */


/*
 * daJacobian_matrix; computes the Jacobian matrix (dF/dy, where F(to,y) = 0)
 * either by numerical finite differencing or by the user-supplied routine.
 */
BOOL
daJacobian_matrix (PTR_ROOT *root, DASSLC_RES *residuals, REAL t, REAL *y, REAL *yp,
		   REAL *delta1, REAL *delta2, DASSLC_JAC *jacobian)
{
 JACOBIAN *jac = &root -> jac;
 BOOL err;
 int rank = root -> rank;

 switch (jac -> mtype)
       {
	case USER_DENSE:		/* user-defined dense matrix */
	    {
	     REAL *matrix = (REAL *)jac -> matrix;

	     root -> res = delta1;
	     memset (matrix, 0, rank * rank * sizeof (REAL));
	     err = (*jacobian) (root, t, y, yp, 0, matrix, residuals);
	     if (root -> debug.matrix) daPrint_matrix (stdout, root, t);
             if (!err)
               if (root -> iter.linearmode == ITER_MODE)
                 root -> kry.decomp = TRUE;
               else
                 if (daLU (rank, jac -> pivot, matrix)) err = SINGULAR;

	     break;
	    }

	case DENSE_MODE:		/* dense finite-difference matrix */
	    {
	     FAST SPARSE_LIST *spl = jac -> spl;
	     BOOL flag = TRUE;
	     FAST int i, j, *index, size;
	     FAST REAL *matrix = (REAL *)jac -> matrix;
	     FASTER REAL ysave, del;
	     REAL sround = sqrt ((double)root -> iter.roundoff), iwt,
		  *wt = root -> iter.wt, bound = pow ((double)sround, (double)1.5);

	     if (SGNP(root -> bdf.hold) < 0) sround = -sround;
	     memset (matrix, 0, rank * rank * sizeof (REAL));
	     for (j = 0; j < rank; j++, wt++, spl++)
		{
		 size = jac -> rank = spl -> size;
		 index = jac -> index = spl -> index;
		 ysave = y[j];
		 iwt = (*wt ? 1. / ABS(*wt) : 0.);
		 del = (ysave + sround * MAX(ABS(ysave), iwt)) - ysave;
		 if (ABS(del) < bound) del = SGNP(del) * bound;
		 y[j] += del;
		 if (err = (*residuals) (root, t, y, yp, delta2, &flag), err)
		   return err;

		 del = 1. / del;
		 matrix = (REAL *)(jac -> matrix) + j;
					/* sparse evaluation */
		 if (flag && size < rank)
		   for (; size > 0; size--)
		      {
		       i = *index++;
		       *(matrix + i * rank) = (delta2[i] - delta1[i]) * del;
		      }
		 else			/* full evaluation */
		    for (flag = TRUE, i = 0; i < rank; i++, matrix += rank)
		       *matrix = (delta2[i] - delta1[i]) * del;

		 y[j] = ysave;
		}
	     jac -> rank = 0;		/* clean jacobian structure */

	     if (root -> debug.matrix) daPrint_matrix (stdout, root, t);
             if (root -> iter.linearmode == ITER_MODE)
               root -> kry.decomp = TRUE;
             else
               if (daLU (rank, jac -> pivot, (REAL *)jac -> matrix))
                 err = SINGULAR;

	     break;
	    }

#ifdef SPARSE
	case USER_SPARSE:
	    {
	     char *matrix = jac -> matrix;

	     root -> res = delta1;
	     daSparse_clear (matrix);
	     err = (*jacobian) (root, t, y, yp, 0, matrix, residuals);
	     if (root -> debug.matrix) daPrint_matrix (stdout, root, t);
             if (!err)
               if (root -> iter.linearmode == ITER_MODE)
                 root -> kry.decomp = TRUE;
               else
                 if (daSparse_LU (matrix)) err = SINGULAR;

	     break;
	    }

	case SPARSE_MODE:
	    {
	     FAST SPARSE_LIST *spl = jac -> spl;
	     BOOL flag = TRUE;
	     char *matrix = jac -> matrix;
	     FAST int i, j, *index, size;
	     FASTER REAL ysave, del;
	     REAL sround = sqrt ((double)root -> iter.roundoff), pd, iwt,
		  *wt = root -> iter.wt, bound = pow ((double)sround, (double)1.5);

	     if (SGNP(root -> bdf.hold) < 0) sround = -sround;
	     daSparse_clear (matrix);
	     for (j = 0; j < rank; j++, wt++, spl++)
		{
		 size = jac -> rank = spl -> size;
		 index = jac -> index = spl -> index;
		 ysave = y[j];
		 iwt = (*wt ? 1. / ABS(*wt) : 0.);
		 del = (ysave + sround * MAX(ABS(ysave), iwt)) - ysave;
		 if (ABS(del) < bound) del = SGNP(del) * bound;
		 y[j] += del;
		 if (err = (*residuals) (root, t, y, yp, delta2, &flag), err)
		   return err;

		 del = 1. / del;
					/* sparse evaluation */
		 if (flag && size < rank)
		   for (; size > 0; size--)
		      {
		       i = *index++;
		       daSparse_set (matrix, i, j, (delta2[i] - delta1[i]) * del);
		      }
		 else			/* full evaluation */
		    for (flag = TRUE, i = 0; i < rank; i++)
		       if ((pd = (delta2[i] - delta1[i]) * del) != 0.)
			 daSparse_set (matrix, i, j, pd);

		 y[j] = ysave;
		}
	     jac -> rank = 0;		/* clean jacobian structure */

	     if (root -> debug.matrix) daPrint_matrix (stdout, root, t);
             if (root -> iter.linearmode == ITER_MODE)
               root -> kry.decomp = TRUE;
             else
               if (daSparse_LU (matrix)) err = SINGULAR;

	     break;
	    }
#endif /* SPARSE */

	case USER_BAND:			/* user-defined band matrix */
	    {
	     REAL *matrix = (REAL *)jac -> matrix;

	     root -> res = delta1;
	     memset (matrix + jac -> lband * rank, 0, (jac -> lband +
		     jac -> uband + 1) * rank * sizeof (REAL));
	     err = (*jacobian) (root, t, y, yp, 0, matrix, residuals);
	     if (root -> debug.matrix) daPrint_matrix (stdout, root, t);
             if (!err)
               if (root -> iter.linearmode == ITER_MODE)
                 root -> kry.decomp = TRUE;
               else
                 if (daLUb (rank, jac -> lband, jac -> uband, jac -> pivot,
		            matrix)) err = SINGULAR;

	     break;
	    }

	case BAND_MODE:			/* band finite-difference matrix */
	    {
	     FAST SPARSE_LIST *spl = jac -> spl;
	     BOOL flag = TRUE;
	     FAST int i, j, *index, size;
	     int m = jac -> lband + jac -> uband;
	     FAST REAL *matrix = (REAL *)jac -> matrix;
	     FASTER REAL ysave, del;
	     REAL sround = sqrt ((double)root -> iter.roundoff), iwt,
		  *wt = root -> iter.wt, bound = pow ((double)sround, (double)1.5);

	     if (SGNP(root -> bdf.hold) < 0) sround = -sround;
             memset (matrix + jac -> lband * rank, 0, (m + 1) * rank * sizeof (REAL));
	     for (j = 0; j < rank; j++, wt++, spl++)
		{
		 size = jac -> rank = spl -> size;
		 index = jac -> index = spl -> index;
		 ysave = y[j];
		 iwt = (*wt ? 1. / ABS(*wt) : 0.);
		 del = (ysave + sround * MAX(ABS(ysave), iwt)) - ysave;
		 if (ABS(del) < bound) del = SGNP(del) * bound;
		 y[j] += del;
		 if (err = (*residuals) (root, t, y, yp, delta2, &flag), err)
		   return err;

		 del = 1. / del;
		 matrix = (REAL *)(jac -> matrix) + j + (m - j) * rank;
					/* sparse evaluation */
		 if (flag && size < rank)
		   for (; size > 0; size--)
		      {
		       i = *index++;
		       *(matrix + i * rank) = (delta2[i] - delta1[i]) * del;
		      }
		 else			/* full evaluation */
		    for (flag = TRUE, i = 0; i < rank; i++, matrix += rank)
		       *matrix = (delta2[i] - delta1[i]) * del;

		 y[j] = ysave;
		}
	     jac -> rank = 0;		/* clean jacobian structure */

	     if (root -> debug.matrix) daPrint_matrix (stdout, root, t);
             if (root -> iter.linearmode == ITER_MODE)
               root -> kry.decomp = TRUE;
             else
               if (daLUb (rank, jac -> lband, jac -> uband, jac -> pivot,
			  (REAL *)jac -> matrix))
                 err = SINGULAR;

	     break;
	    }

	case USER_ALGEBRA:
	    {
	     root -> res = delta1;
	     err = (*jacobian) (root, t, y, yp, 0, (void *)jac -> matrix, residuals);
	     if (root -> debug.matrix) daPrint_matrix (stdout, root, t);

	     break;
	    }

	case NONE: err = FALSE; root -> iter.jaceval--; break;

	default: fprintf (stderr, "***ERROR daJacobian_matrix: type of Jacobian matrix evaluation = %d is unknown.\n",
			  jac -> mtype);
		 exit (1);
       }

 return err;
}					/* daJacobian_matrix */


#define B(i,j)	(*(b + n * (j) + i))
/*
		 LINEAR EQUATIONS SOLVER - DENSE MATRIX

	  LU(n*n) * [X(ns*n)]transp. = [B(ns*n)]transp.

     where  LU is an upper triangular matrix ( U )
	    and a lower triangular matrix ( L ), i.e.,  A = L * U,
	    L is a unit lower triangular matrix with changed signals.

	    the solutions X return in the rows of B.

	    ip --> n-dimensional integer vector:
		   if ip[i] =! i then the ith row was changed
		   by the ip[i] row during A factorization.

	  note: if B is the identity matrix then X is the
		transpose inverse of LU.
*/
void
daSolve (int n, int ns, FAST int *ip, REAL *a, REAL *b)
{
 FAST int i, j, k, p;
 int n1 = n - 1;
 FAST REAL *lu, *rhs;
 FASTER REAL a1, a2;

 for (i = 0; i < n1; i++)
    {
     p = *ip++;			   	/* pivot element */

     for (k = 0; k < ns; k++)	     	/* Forward substituition */
		{
		 a1 = B(p,k);
		 if (p != i)			/* row interchange */
		 {
		  B(p,k) = B(i,k); B(i,k) = a1;
		 }

		 if (a1)
		 {
	      lu = a + n * (j = i + 1) + i;
		  for (rhs = b + n * k + j; j < n; j++, lu += n) *rhs++ += a1 * *lu;
		 }
		}
    }

 for (i = n1; i >= 0; i--)		/* Backward substituition */
    {
     for (a1 = 1 / *(a + n * i + i), k = 0; k < ns; k++)
		{
		 a2 = (B(i,k) *= a1);
		 if (i && a2)
		   for (rhs = b + n * k, lu = a + i, j = 0; j < i; j++, lu += n)
				*rhs++ -= a2 * *lu;
		}
    }
}					/* daSolve */


#define A(i,j)	(*(a + n *(i) + j))
/*
	      LU FACTORIZATION OF DENSE MATRIX A(n*n)


	    A return the upper triangular matrix ( U )
	    and lower triangular matrix ( L ), i.e., A = L * U,
	    L is an unit lower triangular matrix with the
	    changed signals of its elements.

	    ip --> n-dimensional integer vector with the
		   pivoting index:
		   if ip[i] =! i then the ith row was changed
		   by the ip[i] row during the A factorization.

            daLU returns k != 0 if U(k,k) = 0, and zero otherwise.
*/
int
daLU (int n, FAST int *ip, REAL *a)
{
 int n1 = n - 1, info = 0;
 FAST int i, j, k, p, ip1;
 FAST REAL *r;
 FASTER REAL a1;

 for (ip[n1] = n1, i = 0; i < n1; i++)
    {
					/* column pivoting */

     for (a1 = fabs ((double)A(i,i)), ip1 = j = (p = i) + 1; j < n; j++)
		if (fabs ((double)A(j,i)) > a1)
		{
		 a1 = fabs ((double)A(j,i)); p = j;
		}

     if (!A(p,i))
       {
		info = p + 1; continue;
       }

     if (p != i)			/* pivot interchange */
       {
		a1 = A(p,i); A(p,i) = A(i,i); A(i,i) = a1;
       }

					/* factorization */
     a1 = - 1 / A(i,i);
     for (j = ip1, r = a + n * j + i; j < n; j++, r += n) *r *= a1;

     for (*ip++ = p, j = ip1; j < n; j++)
		{
		 a1 = A(p,j);
		 if (p != i)			/* row interchange */
		 {
	      A(p,j) = A(i,j); A(i,j) = a1;
		 }

		 if (a1)
		   for (r = a + n * ip1, k = ip1; k < n; k++, r += n)
			  *(r + j) += a1 * *(r + i);
		}
    }

 if (!A(n1,n1)) return n;
 return info;
}					/* daLU */


/*
	     LINEAR EQUATIONS SOLVER - DENSE MATRIX

	  A(n*n) * [X(ns*n)]transp. = [B(ns*n)]transp.

	   the solutions X return in the rows of B.

	  note: if B is the identity matrix then X return the
		transpose inverse of A.
*/
int
daLUback (int n, int ns, REAL *a, REAL *b)
{
 int n1, *ip = (int *)must_malloc (n * sizeof (int), "LU");

 if (!(n1 = daLU (n, ip, a), n1)) daSolve (n, ns, ip, a, b);

 free (ip);

 return (n1);
}					/* daLUback */


/*
		 LINEAR EQUATIONS SOLVER - BAND MATRIX

	  LU((2*lb+ub+1)*n) * [X(ns*n)]transp. = [B(ns*n)]transp.

     where  LU is an upper triangular matrix ( U )
	    and a lower triangular matrix ( L ), i.e.,  A = L * U,
	    L is a unit lower triangular matrix with changed signals.

	    the solutions X return in the rows of B.

	    ip --> n-dimensional integer vector:
		   if ip[i] =! i then the ith row was changed
		   by the ip[i] row during A factorization.

	  note: if B is the identity matrix then X is the
		transpose inverse of LU.

 ref: LINPACK, 08/14/78.
*/
void
daSolveb (int n, int lb, int ub, int ns, FAST int *ip, REAL *a, REAL *b)
{
 FAST int i, j, k, p;
 int n1 = n - 1, m = lb + ub, lm;
 FAST REAL *lu, *rhs;
 FASTER REAL a1, a2;

 if (lb > 0)
   {
    for (i = 0; i < n1; i++)
       {
	lm = m + MIN(lb,n1-i);
	p = *ip++;		   	/* pivot element */

	for (k = 0; k < ns; k++)       	/* Forward substituition */
	   {
	    a1 = B(p,k);
	    if (p != i)			/* row interchange */
	      {
	       B(p,k) = B(i,k); B(i,k) = a1;
	      }

	    if (a1)
	      {
	       lu = a + n * (j = m + 1) + i;
	       for (rhs = b + n * k + i + 1; j <= lm; j++, lu += n)
		  *rhs++ += a1 * *lu;
	      }
	   }
       }
   }

 for (i = n1; i >= 0; i--)		/* Backward substituition */
    {
     for (a1 = 1 / *(a + n * m + i), k = 0; k < ns; k++)
		{
		 a2 = (B(i,k) *= a1);
		 if (i && a2)
		 {
	      lm = MIN(i,m);
	      lu = a + (m - lm) * n + i;
	      for (rhs = b + n * k + i - lm, j = 0; j < lm; j++, lu += n)
	         *rhs++ -= a2 * *lu;
		 }
		}
    }
}					/* daSolveb */


/*
	      LU FACTORIZATION OF BAND MATRIX A((2*lb+ub+1)*n)

	    A matrix with lb diagonals below the main diagonal,
	      and ub diagonals above the main diagonal, which
	      can be set up as follow:

	    m = lb + ub;
	    for (j = 0; j < n; j++)
	       {
		i1 = MAX(0,j-ub);
		i2 = MIN(n,j+lb+1);
		for (i = i1; i < i2; i++)
		   {
		    k = i - j + m;
		    A(k,j) = A_full(i,j);
		   }
	       }

	    that is, the columns of the full matrix are stored in
	    the columns of A and the diagonals of the full matrix
	    are stored in rows lb through 2*lb+ub of A.

	    A return the upper triangular matrix ( U )
	      and lower triangular matrix ( L ), i.e., A = L * U,
	      L is an unit lower triangular matrix with the
	      changed signals of its elements.

	    ip --> n-dimensional integer vector with the
		   pivoting index:
		   if ip[i] =! i then the ith row was changed
		   by the ip[i] row during the A factorization.

            daLUb returns k != 0 if U(k,k) = 0, and zero otherwise.

  ref.: LINPACK, 08/14/78.
*/
int
daLUb (int n, int lb, int ub, int *ip, REAL *a)
{
 int n1 = n - 1, m = lb + ub, j1 = MIN(n1,m), i1, ju = 0, lm, mm, info = 0;
 FAST int i, j, k, p;
 FAST REAL *r;
 FASTER REAL a1;

 for (i = m - j1 + 1; i < lb; i++)  /* zero initial fill-in columns */
    for (j = m - i; j < j1; j++) A(i,j) = 0;

				/* gaussian elimination with partial pivoting */
 for (ip[n1] = n1, i = 0; i < n1; i++ )
    {
     i1 = i + 1;

     if (j1 < n)
       {                        /* zero next fill-in column */
		if (lb > 0) for (k = 0; k < lb; k++) A(k,j1) = 0;
		j1++;
       }

     lm = m + MIN(lb,n1-i);	/* find p = pivot index */
     for (a1 = fabs (A(m,i)), j = (p = m) + 1; j <= lm; j++)
	 if (fabs (A(j,i)) > a1)
	  {
	   a1 = fabs (A(j,i)); p = j;
	  }

     if (!A(p,i))
       {
		info = i + 1; continue;
       }

     if (p != m)                /* interchange if necessary */
       {
		a1 = A(p,i); A(p,i) = A(m,i); A(m,i) = a1;
       }
				/* compute  multipliers */
     a1 = - 1 / A(m,i);
     for (j = m + 1, r = a + n * j + i; j <= lm; j++, r += n) *r *= a1;

     ju = MAX(ju,p+i-lb+1);
     ju = MIN(ju,n);
     mm = m;

     for (ip[i] = p + i - m, j = i1; j < ju; j++)
		{
		 p--; mm--;
		 a1 = A(p,j);
		 if (p != mm)		/* row interchange */
		 {
		  A(p,j) = A(mm,j); A(mm,j) = a1;
		 }

		 if (a1)                /* row elimination with column indexing */
		 {
	      i1 = (mm - m) * n;
	      for (k = m + 1, r = a + n * k; k <= lm; k++, r += n)
	         *(r + j + i1) += a1 * *(r + i);
		 }
		}
    }

 if (!A(m,n1)) info = n;
 return info;
}					/* daLUb */


/*
	     LINEAR EQUATIONS SOLVER -  BAND MATRIX

	  A((2*lb+ub+1)*n) * [X(ns*n)]transp. = [B(ns*n)]transp.

	   the solutions X return in the rows of B.

	  note: if B is the identity matrix then X return the
		transpose inverse of A.
*/
int
daLUbackb (int n, int lb, int ub, int ns, REAL *a, REAL *b)
{
 int n1, *ip = (int *)must_malloc (n * sizeof (int), "LU");

 if (!(n1 = daLUb (n, lb, ub, ip, a), n1))
   daSolveb (n, lb, ub, ns, ip, a, b);

 free (ip);

 return (n1);
}					/* daLUback */


#ifdef SPARSE
/* daSparse_matrix; built the sparse matrix structure */
PRIV char *
daSparse_matrix (int rank)
{
 int error = FALSE;
 char *matrix = spCreate (rank, FALSE, &error);

 if (matrix == NULL || error)
   {
    fprintf (stderr, "***ERROR daSparse_matrix: Not enough memory \n");
    exit (1);
   }

 return matrix;
}					/* daSparse_matrix */
#endif /* SPARSE */


/* daPsolver; default preconditionning routine using LU factorization
 *            with back- and forward substitution
 */
PRIV BOOL
daPsolver (PTR_ROOT *root, REAL *z, DASSLC_RES *residuals)
{
 BOOL err = FALSE;

 switch (root -> jac.mtype)
       {
	case USER_DENSE:
	case DENSE_MODE:
	     if (root -> kry.decomp && daLU (root -> rank,
		 root -> jac.pivot, (REAL *)root -> jac.matrix))
	       err = TRUE;
	     else
	       daSolve (root -> rank, 1, root -> jac.pivot,
			(REAL *)root -> jac.matrix, z);
	     break;

#ifdef SPARSE
	case USER_SPARSE:
	case SPARSE_MODE:
	     if (root -> kry.decomp &&
		 daSparse_LU (root -> jac.matrix)) err = TRUE;
	     else
	       daSparse_solve (root -> jac.matrix, z);

	     break;
#endif /* SPARSE */

	case USER_BAND:
	case BAND_MODE:
	     if (root -> kry.decomp && daLUb (root -> rank,
		 root -> jac.lband, root -> jac.uband, root -> jac.pivot,
		 (REAL *)root -> jac.matrix))
	       err = TRUE;
	     else
	       daSolveb (root -> rank, root -> jac.lband, root -> jac.uband,
		       1, root -> jac.pivot, (REAL *)root -> jac.matrix, z);
	     break;

	case USER_ALGEBRA:
	    {
	     if (root -> kry.decomp && (*ujacFactor) (root))
	       err = TRUE;
	     else
	       (*ujacSolve) (root, z, z);
	     break;
	    }

	case NONE: root -> kry.nps--;
       }

 return err;
}					/* daPsolver */


/* heLS; this is similar to the LINPACK routine DGESL except that
 * HES is an upper Hessenberg matrix, modified by Peter Brown, Lawrence
 * Livermore Natl. Lab. 'heLS' solves the least squares problem
 *                       MIN (b-HES*x,b-HES*x)
 * using the factors computed by 'heQR', where:
 *
 * HES = the upper triangular factor R in the QR decomposition of HES(n,n+1).
 *
 * q = the coefficients of the n givens rotations used in the QR
 *     factorization of HES.
 *
 * b = the right hand side vector. It returns the solution vector x.
 */
PRIV void
heLS (int maxlp1, int n, REAL *hes, REAL *q, REAL *b)
{
 FAST int j, k, km1, iq;
 FAST REAL *aux;
 REAL c, s, t1, t2;

 for (k = 1; k <= n; k++)
    {
     km1 = k - 1;
     iq = 2 * (k - 1);
     c = q[iq];
     s = q[iq+1];
     t1 = b[km1];
     t2 = b[k];
     b[km1] = c * t1 - s * t2;
     b[k] = s * t1 + c * t2;
    }

 k = n - 1;
 aux = hes + k * maxlp1;
 for (; k >= 0; k--, aux -= maxlp1)
    {
     c = (b[k] /= aux[k]);
     for (j = 0; j < k; j++) b[j] -= c * aux[j];
    }
}                            /* heLS */


/* heQR; this routine performs a QR decomposition of an upper Hessenberg
 * matrix HES(n,n+1). It is a modification of LINPACK by Peter Brown, Lawrence
 * Livermore Natl. Lab. There are two options available:
 *
 * ijob =  1: performing a fresh decomposition
 * ijob >= 2: updating the QR factors by adding a row and a column to
 *            the matrix HES.
 *
 * 'heQR' decomposes an upper Hessenberg matrix by using Givens rotations.
 * The factorization can be written q * HES = R, where q is a product of
 * Givens rotations (the factors c and s of each Givens rotation) and R is
 * upper triangular, returned on HES. If HES(k,k) = 0 is not an error
 * condition for this subroutine, but it does indicate that heLS will divide
 * by zero if if called, in this case it returns k, otherwise 0.
 */
PRIV int
heQR (int maxlp1, int n, REAL *hes, REAL *q, int ijob)
{
 FAST int i, j, k, km1, iq;
 int nm1, info = 0;
 FAST REAL *aux;
 REAL c, s, t, t1, t2;

 if (ijob <= 1)
   {
    for (aux = hes, k = 1; k <= n; k++, aux += maxlp1)
       {
	km1 = k - 1;

	if (km1 >= 1)
	  {
	   for (j = 0; j < km1; j++)
	      {
	       i = 2 * j;
	       t1 = aux[j];
	       t2 = aux[j+1];
	       c = q[i];
	       s = q[i+1];
	       aux[j] = c * t1 - s * t2;
	       aux[j+1] = s * t1 + c * t2;
	      }
	  }

	iq = 2 * km1;
	t1 = aux[km1];
	t2 = aux[k];

	if (!t2)
	  {
	   c = 1.0;
	   s = 0.0;
	  }
	else
	  {
	   if (ABS (t2) >= ABS (t1))
	     {
	      t = t1 / t2;
	      s = -1.0 / sqrt (1.0 + t * t);
	      c = -s * t;
	     }
	   else
	     {
	      t = t2 / t1;
	      c = 1.0 / sqrt (1.0 + t * t);
	      s = -c * t;
	     }
	  }

	q[iq] = c;
	q[iq+1] = s;
	aux[km1] = c * t1 - s * t2;

	if (!aux[km1]) info = k;
       }
   }
 else
  {
   nm1 = n - 1;
   aux = hes + maxlp1 * nm1;
   for (k = 1; k <= nm1; k++)
      {
       km1 = k - 1;
       i = 2 * km1;
       t1 = aux[km1];
       t2 = aux[k];
       c = q[i];
       s = q[i+1];
       aux[km1] = c * t1 - s * t2;
       aux[k] = s * t1 + c * t2;
      }

   t1 = aux[nm1];
   t2 = aux[n];

   if (!t2)
     {
      c = 1.0;
      s = 0.0;
     }
   else
     {
      if (ABS (t2) >= ABS(t1))
		{
		 t = t1 / t2;
		 s = -1.0 / sqrt (1.0 + t * t);
		 c = -s * t;
		}
      else
		{
		 t = t2 / t1;
		 c = 1.0 / sqrt (1.0 + t * t);
		 s = -c * t;
		}
     }

   iq = 2 * n - 1;
   q[iq-1] = c;
   q[iq] = s;
   aux[nm1] = c * t1 - s * t2;

   if (!aux[nm1]) info = n;
  }

 return info;
}                            /* heQR */


/* orth; this routine orthogonalizes the vector vnew against the previous
 * kmp vectors in the v array. It uses a modified Gram-Schmidt
 * orthogonalization procedure with conditional reorthogonalization.
 * It returns the L-2 norm of vnew.
 *
 * vnew = the vector of length n containing a scaled product of the Jacobian
 *        and the vector V(ll-1,*). It returns the new vector orthogonal to
 *        V(i0,*), where i0 = MAX(0, (ll - kmp)).
 *
 * v    = the n x ll array containing the previous ll orthogonal vectors
 *        V(0,*) to V(ll-1,*).
 *
 * hes  = An ll x ll upper Hessenberg matrix containing, in HES(k,i),
 *        k < ll, scaled inner products of A * V(k,*)' and V(i,*)'. It
 *        returns the upper Hessenberg matrix with line ll filled in with
 *        scaled inner products of A * V(ll-1,*)' and V(i,*)'.
 */
PRIV REAL
orth (int n, int maxlp1, int ll, int kmp, REAL *vnew, REAL *v, REAL *hes)
{
 FAST int i, j;
 int i0;
 FAST REAL *auxV, *auxH;
 REAL sum, sumdsq, vnrm, snormw;

 for (sum = j = 0; j < n; j++) sum += vnew[j] * vnew[j];

 vnrm = sqrt (sum);                    /* vnew norm */

/* Do Modified Gram-Schmidt on vnew = A * V(LL-1).
 * Scaled inner products give new column of HES.
 * Projections of earlier vectors are subtracted from vnew.
 */

 i = i0 = MAX(0, (ll - kmp));
 auxV = v + n * i;
 auxH = hes + maxlp1 * (ll - 1);

 for (; i < ll; i++, auxV += n)
    {				      /* inner product of V(i,*) and vnew */
     for (sum = j = 0; j < n; j++) sum += auxV[j] * vnew[j];
     for (j = 0; j < n; j++) vnew[j] -= sum * auxV[j];
     auxH[i] = sum;
    }

 for (sum = j = 0; j < n; j++) sum += vnew[j] * vnew[j];

 snormw = sqrt (sum);                 /* vneq norm */

/* If vnew is small compared to its input value (in norm), then
 * reorthogonalize vnew to V(0,*) through V(ll-1,*).
 * Correct if relative correction exceeds 1000*(unit roundoff).
 * Finally, correct snormw using the dot products involved.
 */

 if ((vnrm + 0.001 * snormw) == vnrm)
   {
    i = i0;
    auxV = v + n * i;
    for (sumdsq = 0.0; i < ll; i++, auxV += n)
       {                              /* inner product of V(i,*) and vnew */
		for (sum = j = 0; j < n; j++) sum -= auxV[j] * vnew[j];

		if ((auxH[i] + 0.001 * sum) == auxH[i]) continue;

		auxH[i] -= sum;
		for (j = 0; j < n; j++) vnew[j] += sum * auxV[j];

		sumdsq += sum * sum;
       }

    if (sumdsq)
      {
       sum = snormw * snormw - sumdsq;
       snormw = sqrt (MAX (0.0, sum));
      }
   }

 auxH[ll] = snormw;
 return snormw;
}                            /* orth */


/* daAxV; this routine computes the product
 *              z = (D-inverse)*(P-inverse)*(dF/dy)*(D*v),
 * where F(y) = G(t, y, cj*(y-A)), cj is a scalar proportional to 1/h,
 * and A involves the past history of y.  The quantity cj*(y-A) is
 * an approximation to the first derivative of y and is stored
 * in the array yp. Note that dF/dy = dG/dy + cj * dG/dyp (the iterative
 * matrix). D is a diagonal scaling matrix, and P is the left preconditioning
 * matrix. v is assumed to have L-2 norm equal to 1. The product is stored
 * in z and is computed by means of a difference quotient, a call to
 * 'residuals', and one call to 'psolver'.
 *
 * res = current residual vector of G(t,y,yp).
 *
 * v = real array of length n (can be the same array as z).
 *
 * wght = array of length n containing scale factors. 1/wght[i] are the
 *        diagonal elements of the matrix D.
 *
 * vtem = work array of length n used to store the unscaled version of v.
 *
 * cj = scalar proportional to current value of 1/(step size h).
 *
 * ires = error flag from 'residuals'.
 *
 * npsl = the number of calls to 'psolver'.
 *
 * It returns:
 *   TRUE   (1) if there was a recoverable error in 'psolver',
 *   ERROR (-1) if there was an unrecoverable error in 'psolver', and
 *   FALSE  (0) if there was no error.
 */
PRIV BOOL
daAxV (PTR_ROOT *root, int n, REAL *res, REAL *v, REAL *wght, REAL *z,
       REAL *vtem, long *npsl, BOOL *ires, DASSLC_RES *residuals, DASSLC_PSOL *psolver)
{
 FAST int i;
 BOOL jac = FALSE, ier = FALSE;
 REAL *yptem = root -> kry.yptem, cj = root -> iter.cj, tn = root -> t,
      *y = root -> y, *yp = root -> yp;

 for (i = 0; i < n; i++)
    {
     vtem[i] = v[i] / wght[i];
     yptem[i] = yp[i] + vtem[i] * cj;
     z[i] = y[i] + vtem[i];
    }

 *ires = (*residuals) (root, tn, z, yptem, vtem, &jac);

 root -> iter.reseval++;

 if (*ires) return ier;

 for (i = 0; i < n; i++) z[i] = vtem[i] - res[i];

 if (!(ier = (*psolver) (root, z, residuals), ier) && root -> kry.decomp)
   root -> kry.decomp = FALSE;
    
 (*npsl)++;

 if (!ier) for (i = 0; i < n; i++) z[i] *= wght[i];

 return ier;
}                            /* daAxV */


/* daSpigm; this routine solves the linear system A * z = r using a scaled
 * preconditioned version of the generalized minimum residual method.
 * An initial guess of z = 0 is assumed. The major variables are:
 *
 * kry -> maxl = the maximum allowable order of the matrix hes.
 *
 * kry -> kmp = the number of previous vectors that the new vector must
 *              be made orthogonal to.  (kmp <= maxl)
 *
 * kry -> lintol = tolerance on residuals r - A * z in weighted rms norm.
 *
 * kry -> r = the right hand side of the system A * z = r. It is also used as
 *            work space when computing the final approximation and will
 *            therefore be destroyed.
 *
 * kry -> z = the final computed approximation to the solution of the
 *            system A * z = r.
 *
 * wght = the n-vector containing the nonzeros elements of the diagonal
 *        scaling matrix.
 *
 * dl = real work array used for calculation of the residual norm rho when
 *      the method is incomplete (kmp < maxl) and/or when using restarting.
 *      It returns the scaled preconditioned residual,
 *                 (D-inverse)*(P-inverse)*(r-A*z).
 *      Only loaded when performing restarts of the Krylov iteration.
 *
 * nrsts = counter for the number of restarts on the current call to daSpigm.
 *         If nrsts > 0, then the residual (kry -> r) is already scaled, and
 *         so scaling of (kry -> r) is not necessary.
 *
 * lgmr = the number of iterations performed and the current order of the
 *        upper Hessenberg matrix hes.
 *
 * v = the (lgmr+1) by neq array containing the lgmr orthogonal vectors.
 *
 * hes = the upper triangular factor of the QR decomposition of the lgmr
 *       by (lgmr+1) upper Hessenberg matrix whose entries are the scaled
 *       inner-products of A * V(i,*)' and V(k,*)'.
 *
 * q = real array of length 2 * maxl containing the components of the givens
 *     rotations used in the QR decomposition of hes.  It is loaded in 'heQR'
 *     and used in 'heLS'.
 *
 * ires = error flag from 'residuals'.
 *
 * It returns the error flag:
 *	   0 means convergence in lgmr iterations, lgmr <= maxl,
 *           or error in 'residuals'.
 *         1 means the convergence test did not pass in maxl
 *           iterations, but the new residual norm (rho) is
 *           less than the old residual norm (rnrm), and so z is
 *           computed.
 *         2 means the convergence test did not pass in maxl
 *           iterations, new residual norm (rho) >= old residual
 *           norm (rnrm), and the initial guess, z = 0, is
 *           returned.
 *         3 means there was a recoverable error in 'psolver'
 *           caused by the preconditioner being out of date.
 *        -1 means there was an unrecoverable error in 'psolver'.
 */
PRIV BOOL
daSpigm (PTR_ROOT *root, KRYLOV *kry, int n, REAL *wght, int *lgmr,
	 int nrsts, BOOL *ires, DASSLC_RES *residuals, DASSLC_PSOL *psolver)
{
 FAST int i, j, ll, l, k;
 int info, i2, maxlp1 = kry -> maxl + 1, maxlm1 = kry -> maxl - 1, iflag = 0,
     ns = n * sizeof (REAL);
 BOOL ier = FALSE;
 FAST REAL *aux, *auxi;
 REAL sum, rnrm, c, dlnrm, prod, rho, s, snormw, *v = kry -> v, *q = kry -> q,
      *dl = kry -> dl;

 *lgmr = 0;

 memset (kry -> z, 0, ns);
 memset (kry -> hes, 0, kry -> maxl * maxlp1 * sizeof (REAL));

 if (nrsts == 0)
   {
    if (!(ier = (*psolver) (root, kry -> r, residuals), ier) && kry -> decomp)
      kry -> decomp = FALSE;

    (kry -> nps)++;

    if (ier) return (ier < 0 ? -1 : 3);

    for (j = 0; j < n; j++) v[j] = kry -> r[j] * wght[j];
   }
 else memcpy (v, kry -> r, ns);

 for (sum = j = 0; j < n; j++) sum += v[j] * v[j];

 rnrm = sqrt (sum);      /* weighted norm of final preconditioned residual */

 if (rnrm <= kry -> lintol) return iflag;

 for (sum = 1.0 / rnrm, j = 0; j < n; j++) v[j] *= sum;

 for (aux = v + n, ll = 1, prod = 1.0; ll <= kry -> maxl; ll++, aux += n)
    {
     l = ll - 1;
     *lgmr = ll;

     ier = daAxV (root, n, root -> res, v + l * n, wght, aux, dl, &kry -> nps,
		  ires, residuals, psolver);

     if (*ires) return iflag;

     if (ier) return (ier < 0 ? -1 : 3);

     snormw = orth (n, maxlp1, ll, kry -> kmp, aux, v, kry -> hes);
     info = heQR (maxlp1, ll, kry -> hes, q, ll);

     if (info == ll) return 2;

     prod *= q[2 * l + 1];
     rho = fabs (prod * rnrm);

     if ((ll > kry -> kmp) && (kry -> kmp < kry -> maxl))
       {
		if (ll == (kry -> kmp + 1))
		{
	     memcpy (dl, v, ns);

	     for (auxi = v + n, i = 0; i < kry -> kmp; i++, auxi += n)
	      {
	       i2 = i * 2;
	       s = q[i2 + 1];
	       c = q[i2];
	       for (k = 0; k < n; k++) dl[k] = s * dl[k] + c * auxi[k];
	      }
		}

	    s = q[2 * l + 1];
		c = q[2 * l] / snormw;

		for (k = 0; k < n; k++) dl[k] = s * dl[k] + c * aux[k];

		for (sum = j = 0; j < n; j++) sum += dl[j] * dl[j];

		dlnrm = sqrt (sum);      /* dl norm */
		rho *= dlnrm;
       }
		      /* weighted norm of final preconditioned residual */
     if (rho > kry -> lintol)
       {
		if (ll != kry -> maxl)
		{
	     for (sum = 1.0 / snormw, j = 0; j < n; j++) aux[j] *= sum;
	     continue;             /* next ll loop */
		}

	 if (rho >= rnrm) return 2; /* exit ll loop */

	 iflag = 1;

	 if (kry -> restart)       /* active restarting */
	  {
	   if (kry -> kmp == kry -> maxl)
	     {
	      memcpy (dl, v, ns);

	      for (auxi = v + n, i = 0; i < maxlm1; i++, auxi += n)
			{
			 i2 = i * 2;
			 s = q[i2 + 1];
			 c = q[i2];

			 for (k = 0; k < n; k++) dl[k] = s * dl[k] + c * auxi[k];
			}

	      s = q[2 * maxlm1 + 1];
	      c = q[2 * maxlm1] / snormw;

	      for (k = 0; k < n; k++) dl[k] = s * dl[k] + c * auxi[k];
	     }

	   for (sum = rnrm * prod, j = 0; j < n; j++) dl[j] *= sum;
	  }
     }
     break;
    }                            /* end of ll loop */

 ll = *lgmr;

 memset (kry -> r + 1, 0, ll * sizeof (REAL));

 *kry -> r = rnrm;

 heLS (maxlp1, ll, kry -> hes, q, kry -> r);

 for (auxi = v, i = 0; i < ll; i++, auxi += n)
    for (sum = kry -> r[i], j = 0; j < n; j++) kry -> z[j] += sum * auxi[j];

 for (i = 0; i < n; i++) kry -> z[i] /= wght[i];

 return iflag;
}                            /* daSpigm */


/* daSolve_krylov; uses a restart algorithm and interfaces to daSpigm for
 * the solution of the linear system arising from a Newton iteration,
 * where x is the right-hand side vector on input, and the solution vector
 * on output, of length neq. It returns the same 'daSpigm' error flag.
*/
BOOL
daSolve_krylov (PTR_ROOT *root, REAL *x, BOOL *ires, DASSLC_RES *residuals, DASSLC_PSOL *psolver)
{
 KRYLOV *kry = &root -> kry;
 BOOL flag;
 DASSLC_PSOL *psol = (psolver ? psolver : daPsolver);
 int nrsts, lgmr, n = root -> rank, ns = n * sizeof (REAL);
 FAST int i;
 REAL *ewt = root -> iter.wt;

 for (i = 0; i < n; i++) ewt[i] *= kry -> rsqrtn;

 root -> res = kry -> res;
 memcpy (root -> res, x, ns);
 memcpy (kry -> r, x, ns);
 memset (x, 0, ns);

 nrsts = -1;
 do
   {
    if (++nrsts > 0) memcpy (kry -> r, kry -> dl, ns);

    flag = daSpigm (root, kry, n, ewt, &lgmr, nrsts, ires, residuals, psol);

    kry -> nli += lgmr;

    for (i = 0; i < n; i++) x[i] += kry -> z[i];

   } while ((flag == 1) && (nrsts < kry -> maxrest) && (*ires == FALSE));

 if (*ires || flag) kry -> ncfl++;

 for (i = 0; i < n; i++) ewt[i] *= kry -> sqrtn;

 if (root -> debug.nr)
   if (*ires)
     fprintf (stdout, "***DEBUG daSolve_krylov: error %d in residuals (time=%g)\n",
	      *ires, (double)root -> t);
   else if (flag)
	  if (flag == ERROR || flag == 3)
	    fprintf (stdout, "***DEBUG daSolve_krylov: error %d in psolver (time=%g)\n",
		     flag, (double)root -> t);
	  else
	    {
	     fprintf (stdout, "***DEBUG daSolve_krylov: nonconvergence in %d linear iterations (time=%g)\n",
		      root -> kry.maxl, (double)root -> t);

	     if (flag == 1)
	       fprintf (stdout, "                         with norm reduction\n");
	     else
	       fprintf (stdout, "                         with no norm reduction\n");
	    }

 return flag;
}                            /* daSolve_krylov */


/*
 * daRoundoff; this routine computes the unit roundoff of the machine in double
 * precision.  this is defined as the smallest positive machine number
 * u such that  1.0e0 + u != 1.0e0.
 */
REAL
daRoundoff (void)
{
 REAL u = 1.0;

 while (((REAL)1.0 + (u *= (REAL)0.5)) != (REAL)1.0);

 return (u * (REAL)2.0);
}					/* daRoundoff */


/* daRandom; simple pseudo-random number generator between [0,1] */
REAL
daRandom (unsigned seed)
{
 PRIV long int a1, a2;
 PRIV char flag = 1;

 if (flag)
   {
    flag = 0; srand (seed);
    a1 = rand () / RAND_MAX * 137438L + 3;
    a2 = rand () / RAND_MAX * 7 + 11;
   }

 if (rand () / RAND_MAX > .5)
   {
    a1 = (125 * a1) % 17179869L;
    return ((REAL)a1 / 17179869L);
   }

 a2 = (32719 * a2 + 3) % 32749;
 return ((REAL)a2 / 32749);
}					/* daRandom */


/* must_malloc; malloc n bytes, exit if not successful */
void *
must_malloc (unsigned n, char *s)
{
 void *p = malloc (n);

 if (p == NULL && n)
   {
    fprintf (stderr, "***ERROR must_malloc: \"%s\": not enough memory\n", s);
    exit (1);
   }

 return p;
}					/* must_malloc */


/* must_realloc; realloc n bytes, exit if not successful */
void *
must_realloc (void *p, unsigned n, char *s)
{
 void *q = (p == NULL) ? malloc (n) : realloc (p, n);

 if (q == NULL && n)
   {
    fprintf (stderr, "***ERROR must_realloc: \"%s\": not enough memory\n", s);
    exit (1);
   }

 return q;
}					/* must_realloc */


/* ************** SETUP phase starts here ******************** */

PRIV char *head = "  index      value    derivative ";
PRIV int rankg;
PRIV SHORT lbg, ubg;
PRIV PTR_ROOT *rootg;
PRIV ITER_SET *iters;
PRIV KRYLOV *krys;
PRIV VAR_DATA *db;
PRIV REAL rtolg, atolg, h0;

PRIV struct table
{
 char *name;
 BOOL (*vfunc)();
} comtab[] = {				/* inputfile must be the first */
	      {"inputfile", set_inputfile},
	      {"data", set_database},	/* database must be the second */
	      {"option", arSet_option},
	      {"print", set_print},
	      {"debug", set_debug},
		  {"rank", set_rank},
	      {NULL, NULL}
	     };

PRIV REALvar
  REALopttab[] = {
		  {RVAR, "jacrate", 0.6, 0.0, 1.0, NULL},
		  {RVAR, "nonblank", 0.2, 0.0, 1.0, NULL},
		  {RVAR, "damps", 1.0, 1.e-3, 1.0, NULL},
		  {RVAR, "dampi", 1.0, 1.e-3, 1.0, NULL},
		  {RVAR, "tstop", HUGE, -HUGE, HUGE, NULL},
		  {RVAR, "rtol", 1.e-8, 0.0, HUGE, &rtolg},
		  {RVAR, "atol", 1.e-10, 0.0, HUGE, &atolg},
		  {RVAR, "stepsize", 0.0, -HUGE, HUGE, &h0},
		  {RVAR, "maxstep", 0.0, 0.0, HUGE, NULL},
		  {RVAR, "convtol", 0.33, 1.e-3, 1.0, NULL},
		  {RVAR, "litol", 0.05, 0.0, HUGE, NULL},
		  {0, NULL, 0., 0., 0., NULL}
		 };

PRIV SHORTvar
  SHORTopttab[] = {
		   {SVAR, "maxorder", 5, 1, MAX_ORDER, NULL},
		   {SVAR, "lband", -1, -1, MAX_SHORT, &lbg},
		   {SVAR, "uband", -1, -1, MAX_SHORT, &ubg},
		   {SVAR, "maxl", 5, 1, MAX_LI, NULL},
		   {SVAR, "kmp", 5, 1, MAX_LI, NULL},
		   {SVAR, "maxrest", 5, 1, MAX_SHORT, NULL},
		   {SVAR, "sparsethr", 20, 2, MAX_SHORT, NULL},
		   {SVAR, "maxconvfail", 10, 0, MAX_SHORT, NULL},
		   {SVAR, "maxerrorfail", 3, 0, MAX_SHORT, NULL},
		   {SVAR, "maxsingular", 3, 0, MAX_SHORT, NULL},
		   {SVAR, "maxnewton", 4, 1, MAX_SHORT, NULL},
		   {SVAR, "maxjacfix", 5, 1, MAX_SHORT, NULL},
		   {SVAR, "maxlen", 10000, 2, MAX_SHORT, NULL},
		   {0, NULL, 0, 0, 0, NULL}
		  };

PRIV BOOLvar
  BOOLopttab[] = {
		  {BVAR, "factor", TRUE, 0, 1, NULL},
		  {BVAR, "iststop", FALSE, 0, 1, NULL},
		  {BVAR, "istall", FALSE, 0, 1, NULL},
		  {BVAR, "stol", TRUE, 0, 1, NULL},
		  {BVAR, "nonneg", FALSE, 0, 1, NULL},
		  {BVAR, "restart", TRUE, 0, 1, NULL},
		  {0, NULL, 0, 0, 0, NULL}
		 };

PRIV FUNCvar
  FUNCopttab[] = {
		  {FUNC, "mtype", set_mtype},
		  {FUNC, "linearmode", set_linearmode},
		  {FUNC, "sparsemode", set_sparsemode},
		  {FUNC, "savefile", set_savefile},
		  {FUNC, "inputpert", set_pertfile},
		  {FUNC, "savepert", set_savepert},
		  {0, NULL, NULL}
		 };


/* daSetup; make all initialization */
BOOL
daSetup (char *inputfile, PTR_ROOT *root, DASSLC_RES *residuals, int rank, REAL t, REAL *y,
		 REAL *yp, int *index, DATABASE *problem, DATABASE **sub_prob)
{
 FILE *fp = NULL;
 REAL t0 = tused ();

 if (!inputfile) fp = stdin;
 else if (*inputfile != '?' && !(fp = fopen (inputfile, "r")))
	    {
	     fprintf (stderr, "***ERROR daSetup: Cannot open inputfile \"%s\"\n",
		          inputfile);
	     return ERROR;
	    }

 if (!rank)
   {
    if (*inputfile == '?')
	  {
	   fprintf (stderr, "***ERROR daSetup: Missing rank information\n");
	   return ERROR;
	  }

    if (fp == stdin)
      {
       fprintf (stdout, "rank = "); scanf (" %d", &rank);
      }
    else
      {
       char str[STRLEN], name[ALFALEN];
       int j;

       if (arGet_card (fp, str))
	     {
	      j = strlen (str) - 1;
	      if (str[j] == EOL) str[j] = EOS;

	      j = 0;
	      if (arGet_word (str, &j, name) != EOS && !(strcmp (name, "rank")) &&
	          arIs_integer (str, j))
	        rank = (int)arCtoi (str, &j);
	      else
	        {
	         fprintf (stderr, "***ERROR daSetup: The rank information must be the first entry in \"%s\"\n", str);
	         return ERROR;
	        }

	      if (!arEmpty_str (str, &j))
	        {
	         fprintf (stderr, "***ERROR daSetup: The rank entry must be alone in \"%s\"\n", str);
  	         return ERROR;
	        }
	     }
       else
	     {
	      fprintf (stderr, "***ERROR daSetup: Missing rank information\n");
	      return ERROR;
	     }
      }
   }

 if (rank < 1)
   {
    fprintf (stderr, "***ERROR daSetup: number of variables = %d < 1\n", rank);
    return ERROR;
   }

 if (!root)
   {
    fprintf (stderr, "***ERROR daSetup: root structure not defined!\n");
    return ERROR;
   }

 if (!residuals)
   {
    fprintf (stderr, "***ERROR daSetup: residuals function not defined!\n");
    return ERROR;
   }

 root -> rank = rank;
 root -> t = t;
 root -> y = y;
 root -> yp = yp;
 root -> problem = problem;
 root -> sub_prob = sub_prob;

 root -> alloc = (y ? 0 : 0x01);
 if (!yp) root -> alloc |= 0x02;
 if (!problem) root -> alloc |= 0x04;
 if (!sub_prob) root -> alloc |= 0x08;

 init_globals (root);

 if (!inputfile) inputfile = "stdin";
 else if (*inputfile == '?')
        {
		 if (user_init && (*user_init) (root))
		   {
	        fprintf (stderr, "***ERROR daSetup: Error in user init function.\n");
            return ERROR;
		   }
		}
      else if (get_information (fp, inputfile))
	      {
	       fprintf (stderr, "***ERROR daSetup: Error in input file.\n");
           return ERROR;
	      }

 if (!root -> y)
   {
    fprintf (stderr, "***ERROR daSetup: Missing initial condition or initial guess.\n");
    return ERROR;
   }

 if (!root -> yp)
   {
    root -> yp = NEW (REAL, rank, "YP");
    memset (root -> yp, 0, rank * sizeof (REAL));
   }

 root -> bdf.h = h0;
 set_priority ();

 if (root -> jac.mtype == BAND_MODE || root -> jac.mtype == USER_BAND)
   {
    root -> jac.lband = lbg;
    root -> jac.uband = ubg;
   }

 if (root -> jac.mtype <= DENSE_MODE && root -> jac.mtype > NONE)
   {
    root -> jac.spl = NEW (SPARSE_LIST, rankg, "SPL");
    sparse_structure (residuals);
   }

 update_root ();

 if (!index)
   {
	root -> index = NULL;
	root -> iter.dae_index = -1; /* index not given (0 or 1) */
   }
 else
  {
   FAST int i;
   SHORT dae_index = 0;

   for (i = 0; i < rank; i++)
	  if (index[i] > dae_index)
	    {
		 if (index[i] > rank)
		   {
			fprintf (stderr, "***ERROR daSetup: Differential index of variable %d = %d > %d.\n",
				     i, index[i], rank);
            return ERROR;
		   }
	     dae_index = index[i];
		}
	  else if (index[i] < 0)
		     {
			  fprintf (stderr, "***ERROR daSetup: Differential index of variable %d = %d < 0.\n",
				       i, index[i]);
              return ERROR;
		     }

   root -> iter.dae_index = dae_index;

   if (dae_index < 2) root -> index = NULL;
   else root -> index = index;
  }

 if (root -> filename)
   {
    daPrint_header (root -> savefile, inputfile);
    daPrint_run (root -> savefile, root);
   }
					/* start with direct mode in BOTH_MODE */
 if (root -> iter.linearmode == BOTH_MODE)
   root -> iter.linearmode = DIR_MODE;

 free (db);				/* free database (index vector is ok) */

 root -> debug.t_info.t_setup = tused () - t0;

 return STAT_OK;
}					/* daSetup */


/* init_globals; initialize option variables */
PRIV void
init_globals (PTR_ROOT *root)
{
 FAST REALvar *rp = REALopttab;
 FAST SHORTvar *sp = SHORTopttab;
 FAST BOOLvar *bp = BOOLopttab;
 FAST int i;

 rootg = root;
 iters = &root -> iter;
 krys = &root -> kry;
 rankg = root -> rank;

 rp[0].varp = &iters->jacrate;
 rp[1].varp = &iters->nonblank;
 rp[2].varp = &iters->damps;
 rp[3].varp = &iters->dampi;
 rp[4].varp = &iters->tstop;
 rp[8].varp = &iters->hmax;
 rp[9].varp = &iters->convtol;
 rp[10].varp = &krys->litol;

 sp[0].varp = &iters->maxorder;
 sp[3].varp = &krys->maxl;
 sp[4].varp = &krys->kmp;
 sp[5].varp = &krys->maxrest;
 sp[6].varp = &iters->sparsethresh;
 sp[7].varp = &iters->maxconvfail;
 sp[8].varp = &iters->maxerrorfail;
 sp[9].varp = &iters->maxsingular;
 sp[10].varp = &iters->maxnewton;
 sp[11].varp = &iters->maxjacfix;
 sp[12].varp = &iters->maxlen;

 bp[0].varp = &iters->factor;
 bp[1].varp = &iters->iststop;
 bp[2].varp = &iters->istall;
 bp[3].varp = &iters->stol;
 bp[4].varp = &iters->nonneg;
 bp[5].varp = &krys->restart;

					/* lband and uband max values */
 sp[1].vmax = sp[2].vmax = rankg - 1;

 for (; rp -> key != 0; rp++) *rp -> varp = rp -> deflt;
 for (; sp -> key != 0; sp++) *sp -> varp = sp -> deflt;
 for (; bp -> key != 0; bp++) *bp -> varp = bp -> deflt;

 if (krys -> maxl > rankg) krys -> maxl = rankg;
 if (krys -> kmp > krys -> maxl) krys -> kmp = krys -> maxl;

 root -> kry.q = NULL;             /* used as flag to indicate BOTH_MODE */
 root -> mode = NONE;

 root -> debug.nr = FALSE;
 root -> debug.bdf = FALSE;
 root -> debug.matrix = FALSE;
 root -> debug.conv = FALSE;
 memset ((void *)&root -> debug.t_info.t_save, 0, 8 * sizeof (REAL));

 iters -> linearmode = DIR_MODE;
 iters -> sparsemode = EVAL_SPARSE;
 iters -> rtol = NULL;
 iters -> wt = NEW (REAL, rankg, "WT");
 iters -> roundoff = daRoundoff ();
 memset ((void *)&iters -> tottpnt, 0, 8 * sizeof (long));

 db = NEW (VAR_DATA, rankg, "DB");
 for (i = 0; i < rankg; i++) db[i].set = 0x00;

 root -> print = FALSE;
 root -> savefile = stdout;
 root -> filename = NULL;
 root -> pertfile = NULL;
 root -> idxprint = NULL;

 root -> jac.mtype = DENSE_MODE;
 root -> jac.pivot = NULL;
 root -> jac.matrix = NULL;
 root -> bdf.alpha = NULL;
 root -> scratch = NEW (REAL, 3*rankg, "SCRATCH");
}					/* init_globals */


/*
 * set_priority; set flags and values according to the priorities.
 */
PRIV void
set_priority (void)
{
 BOOL nonneg = TRUE, stol = TRUE;
 FAST int i;
 int j[8], rank = rankg;

 memset ((void *)j, 0, 8 * sizeof (int));

 for (i = 0; i < rank; i++)
    {
     if (db[i].set & 0x01) j[0]++;      /* print */

     if (nonneg && (db[i].set & 0x02))  /* nonneg: lower priority (FALSE) */
       {
	j[1]++;
	if (!db[i].nonneg) nonneg = FALSE;
       }
					/* stol: higher priority (FALSE) */
     if (stol && (db[i].set & 0x08)) stol = FALSE;
    }

 if (j[0])				/* set print */
   if (j[0] < rank)
     {
      FAST int *idxp = rootg -> idxprint = NEW (int, j[0] + 1, "IDXP");

      for (i = 0; i < rank; i++) if (db[i].set & 0x01) *idxp++ = i;

      *idxp = -1;
     }
   else rootg -> print = TRUE;

 if (j[1])				/* set nonnegativity */
   iters -> nonneg = (nonneg ? (j[1] == rank ? TRUE : iters -> nonneg) : FALSE);

 if (stol) stol = iters -> stol;    	/* set tolerances */

 if (stol) rank = 1;
 iters -> rtol = NEW (REAL, rank + rank, "TOL");
 iters -> atol = iters -> rtol + rank;

 if (stol)
   {
    iters -> rtol[0] = rtolg;
    iters -> atol[0] = atolg;
   }
 else
   {
    for (i = 0; i < rank; i++)
       {
	if (db[i].set & 0x08)
	  {
	   iters -> rtol[i] = db[i].rtol;
	   iters -> atol[i] = db[i].atol;
	  }
	else
	  {
	   iters -> rtol[i] = rtolg;
	   iters -> atol[i] = atolg;
	  }
       }
   }
}					/* set_priority */


/*
 * sparse_structure; generates sparse row-vector structure of iteration
 * matrix.
 */
PRIV void
sparse_structure (DASSLC_RES *residuals)
{
 JACOBIAN *jac = &rootg -> jac;
 FAST int i, *index;
 int n1 = rankg - 1;
 FAST SPARSE_LIST *spl = jac -> spl;
 REAL t0 = tused ();

 if (iters -> sparsemode == NO_SPARSE)
   {
    if (jac -> mtype == SPARSE_MODE && rankg < iters -> sparsethresh)
      jac -> mtype = DENSE_MODE;

    index = NEW (int, rankg, "IDX");

    if (jac -> mtype != BAND_MODE)      /* DENSE and SPARSE modes */
      for (i = 0; i < rankg; i++)
		{
		 spl[i].size = rankg;
		 spl[i].index = index;
		 index[i] = i;
		}
    else if (set_band (spl, FALSE, n1, n1, index))
	   free (index);
   }
 else
   {
    GRAPH *gr, *grp;
    BOOL flag;
    FAST int j, k, m, *global;
	FAST REAL tsize;
    int *psize;
    VAR_DATA *dbj;

    if (iters -> sparsemode == INFILE_SPARSE) flag = FALSE;
    else
      {
       gr = Make_graph (residuals);
       flag = TRUE;
      }

    for (j = 0; j < rankg; j++)
       {
	if (flag)
	  {
	   grp = gr + j;
	   global = grp -> index;

	   free (grp -> dep);
	  }

	dbj = db + j;
	if (dbj -> set & 0x04)
	  {
	   i = dbj -> nonblanks;
	   if (flag)
	     {
	      index = dbj -> index;
	      for (k = grp -> size; k > 0 && i; k--, global++)
		 for (m = 0; m < i; )
		    if (index[m] == *global) index[m] = index[--i];
		    else m++;

	      if (i)
			{
			 m = grp -> size;
			 global = RENEW (int, grp -> index, i + m, "GRI");

			 for (k = 0; k < i; k++)
				{
				 global[m+k] = index[k];

				 if (rootg -> debug.matrix)
				   fprintf (stdout, "***DEBUG sparse_structure: variable %d added by user in equation %d\n",
							index[k], j);
				}
			}
	      else global = grp -> index;

	      i += grp -> size;

	      free (index);
	     }
	   else global = dbj -> index;
	  }
	else i = (flag ? grp -> size : 0);

	spl[j].size = i;		/* set sparse list */

	if (i) spl[j].index = global;
	else
	  {
	   fprintf (stderr, "***ERROR sparse_structure: Equation %d does not contain any variable\n", j);
	   exit (1);
	  }
   }

    if (flag) free (gr);		/* free graph structure */

    for (tsize = i = 0; i < rankg; i++)
       tsize += spl[i].size;		/* total matrix area */

    if (jac -> mtype == SPARSE_MODE && (rankg < iters -> sparsethresh ||
	tsize > (REAL)rankg * rankg * iters -> nonblank)) jac -> mtype = DENSE_MODE;

    psize = NEW (int, rankg, "IDX");

    if (tsize == (REAL)rankg * rankg)
      {
       if (rootg -> debug.matrix)
		 fprintf (stdout, "***DEBUG sparse_structure: full-rank matrix!\n");

       if (jac -> mtype != BAND_MODE)
		 for (i = 0; i < rankg; i++, spl++)
			{
			 spl -> size = rankg;
			 free (spl -> index);
			 spl -> index = psize;
			 psize[i] = i;
			}
       else
		{
		 for (i = 0; i < rankg; i++) free (spl[i].index);

		 if (set_band (spl, FALSE, n1, n1, psize))
			free (psize);
		}
      }
    else                                /* not full matrix */
      {					/* + rank to avoid extra checking */
       FAST LINK *lk = NEW (LINK, (unsigned int)tsize + rankg, "LK");
       LINK *linkr = lk, **linkp = NEW (LINK *, rankg, "LKP");
       int lb = 0, ub = 0;
       REAL nonblk = tsize * 100. / ((REAL)rankg * rankg);

       for (i = 0; i < rankg; i++)
		{
		 psize[i] = 0; linkp[i] = lk++;
		}
					/* sparse list by column vectors */
       for (i = 0; i < rankg; i++, spl++)
		{
		 index = spl -> index;
		 for (j = spl -> size; j > 0; j--)
	      {
	       k = *index++; psize[k]++;
	       linkp[k] -> index = i;
	       linkp[k] = linkp[k] -> next = lk++;
	      }

		 free (spl -> index);
		}
					/* reset sparse list pointers */
       for (spl = jac -> spl, flag = TRUE, i = 0; i < rankg; i++, spl++)
		{
		 j = spl -> size = psize[i];
		 if (j == rankg)		/* common full column vector */
		 {
	      spl -> index = psize;
	      if (flag) flag = FALSE;
					/* no. of diagonals below main diagonal */
	      if (n1 - i > lb) lb = n1 - i;
	     }
	   else
	     {
	      if (!j)
			{
			 fprintf (stderr, "***ERROR sparse_structure: Variable %d does not appear in any equation\n",
					  i);
			 exit (1);
			}

	      index = spl -> index = NEW (int, j, "SPX");
	      lk = linkr + i;
	      for (; j > 0 ; j--, NEXT(lk)) *index++ = lk -> index;

	     if (spl -> index[spl -> size - 1] - i > lb)
	       lb = spl -> index[spl -> size - 1] - i;
	     }

	   psize[i] = i;
					/* no. of diagonals above main diagonal */
	   if (i - spl -> index[0] > ub) ub = i - spl -> index[0];
	  }

       if (jac -> mtype == BAND_MODE)
		 flag = set_band (jac -> spl, TRUE, lb, ub, psize);

       if (flag) free (psize);

       free (linkr); free(linkp);	/* free auxiliary area */

       if (rootg -> debug.matrix)
		{
		 fprintf (stdout, "***DEBUG sparse_structure: %g %%-full matrix!\n",
				nonblk);
		 fprintf (stdout, "***DEBUG sparse_structure: matrix with lband = %d and uband = %d.\n",
				lb, ub);
		}
      }
   }

 rootg -> debug.t_info.t_sparse += tused () - t0;
}					/* sparse_structure */


/*
 * Make_graph; build the differential-algebraic dependency matrix
 * making perturbation on values and derivatives.
 */
PRIV GRAPH *
Make_graph (DASSLC_RES *residuals)
{
 FILE *fp;
 GRAPH *gr;
 FAST GRAPH *grp;
 char *dep;
 int nerr, size;
 FAST int i, j, k;
 REAL *wt, h, atol, rtol, bound, sround, *ys, *yps, del, *y, *yp,
      t0 = tused ();
					/* read structure from inputfile */
 if (rootg -> pertfile && rootg -> pertfile[0] == 'r')
   {
    if (!(fp = fopen (rootg -> pertfile + 1, "rb")))
      {
       fprintf (stderr, "***ERROR Make_graph: Cannot open inputpert file \"%s\"\n",
		rootg -> pertfile + 1);
       exit (1);
      }
					/* allocate space to graph */
    grp = gr = NEW (GRAPH, rankg, "GR");

    for (i = 0; i < rankg; i++, grp++)
	{
	  if (fread ((char *)&size, sizeof (int), 1, fp) < 1) EXIT;

	  if (size < 0 || size > rankg)
		{
		 fprintf (stderr, "***ERROR Make_graph: Size (%d) not correct in inputpert file \"%s\"\n",
				size, rootg -> pertfile + 1);
		 exit (1);
		}

		grp -> size = size;
		grp -> dep = (char *)must_malloc (size, "GRD");
		if (fread (grp -> dep, 1, size, fp) < (unsigned)size) EXIT;
		grp -> index = NEW (int, size, "GRI");
		if (fread ((char *)grp -> index, sizeof (int), size, fp) < (unsigned)size) EXIT;
	}

    fclose (fp);

    rootg -> debug.t_info.t_perturb += tused () - t0;

    return gr;
   }

 bound = pow ((double)iters -> roundoff, (double).3333);
 wt = iters -> wt;
 h = (h0 ? h0 : 1.e-3 * (rootg -> t ? rootg -> t : 1));

 size = rankg * sizeof (REAL);

 grp = gr = NEW (GRAPH, rankg, "GR");
 dep = (char *)must_malloc (rankg, "DEP");
 ys = (REAL *)must_malloc (size + size, "YS");

 yps = ys + rankg;
 y = rootg -> y;
 yp = rootg -> yp;
 BCOPY (ys, y, size);			/* copy values to backup */
 BCOPY (yps, yp, size);

 for (i = 0; i < rankg; i++)            /* set weights vector */
    {
     gr[i].size = 0;
	 gr[i].dep = NULL;
	 gr[i].index = NULL;

     if (db[i].set & 0x08)
       {
		rtol = db[i].rtol;
		atol = db[i].atol;

		if (rtol == 0.) rtol = (rtolg ? rtolg : bound);
       }
     else
       {
		rtol = (rtolg ? rtolg : bound);
		atol = atolg;
       }

     wt[i] = rtol * ABS(y[i]) + atol;
     if (!wt[i]) wt[i] = bound;
    }

 j = nerr = 0;
 sround = sqrt ((double)iters -> roundoff);
 bound = pow ((double)sround, (double).8);

 rootg -> mode = MAKE_GRAPH;		/* additional information to users */

 do					/* make a global perturbation */
   {
    for (i = 0; i < rankg; i++)
       {
		del = h * yp[i];
		del = (y[i] + SGNP(del) * sround * MAX(ABS(y[i]), MAX(wt[i], ABS(del))))
			 - y[i];

		if (ABS(del) < bound) del = SGNP(del) * bound;

		del *= (1. + daRandom (12345));	/* make a quasi-random perturbation */

		y[i] += del;
		yp[i] += del / h;
       }

	for (; j < rankg; j++)
		if (res_dep (gr, residuals, wt, dep, j, h, bound, sround)) break;
		else
		  {
		   if (nerr) nerr = 0;

		   for (i = 0; i < rankg; i++)
			  if (dep[i])
			    {
				 grp = gr + i;
				 k = grp -> size;
				 grp -> dep = RENEW(char, grp -> dep, k, "GRD");
			     grp -> index = RENEW(int, grp -> index, k, "GRI");
				 grp -> dep[k-1] = dep[i];
				 grp -> index[k-1] = j;
				}			  
		  }

    if (j == rankg) break;

    h *= .25;				/* reduce stepsize and try again */
    BCOPY (y, ys, size);		/* copy saved values */
    BCOPY (yp, yps, size);
   } while (++nerr < iters -> maxerrorfail);

 BCOPY (y, ys, size);		/* recover original values */
 BCOPY (yp, yps, size);

 if (j < rankg)
   {
    fprintf (stderr, "***ERROR Make_graph: Error in residual function was not able to be bypassed within %d retries\n",
	     nerr);
    exit (1);
   }

 for (i = 0; i < rankg; i++)
    {
     if (!gr[i].size)
       {
		fprintf (stderr, "***ERROR Make_graph: row number %d has only zero-elements\n", i);
		exit (1);
       }
    }

 free (ys); free (dep);
					/* save structure to savefile */
 if (rootg -> pertfile && rootg -> pertfile[0] == 'w')
   {
    if (!(fp = fopen (rootg -> pertfile + 1, "w+b")))
      {
       fprintf (stderr, "***ERROR Make_graph: Cannot open savepert file \"%s\"\n",
		rootg -> pertfile + 1);
       exit (1);
      }

    for (grp = gr, i = 0; i < rankg; i++, grp++)
       {
		size = grp -> size;
		if (fwrite ((char *)&size, sizeof (int), 1, fp) < 1) EXIT;
		if (fwrite (grp -> dep, 1, size, fp) < (unsigned)size) EXIT;
		if (fwrite ((char *)grp -> index, sizeof (int), size, fp) < (unsigned)size) EXIT;
       }

    fclose (fp);
   }

 rootg -> debug.t_info.t_perturb += tused () - t0;

 return gr;
}					/* Make_graph */


/*
 * res_dep; build the matrix dependency data using the
 * perturbation in each variable and its derivative.
 * The differential-algebraic dependency matrix 'dep' (transpose)
 * is set according to:
 *      0 - if the equation does not involve the current 'y' and 'yp'
 *      1 - if the equation involves the current 'y' but not 'yp'
 *      2 - if the equation involves the current 'yp' but not 'y'
 *      3 - if the equation involves both current 'y' and 'yp'
 * returns TRUE if occured error in residuals.
 */
PRIV BOOL
res_dep (GRAPH *gr, DASSLC_RES *residuals, REAL *wt, char *dep, int j, REAL h, REAL bound,
		 REAL sround)
{
 BOOL err, dummy = FALSE;
 FAST int i;
 int nerr;
 REAL ys, yps, hs, del, *res = rootg -> scratch, *resd = res + rankg,
      t, *y = rootg -> y, *yp = rootg -> yp;

 t = rootg -> t + h;

 ys = y[j];				/* save original values */
 yps = yp[j];

 if ((*residuals)(rootg, t, y, yp, resd, &dummy))
   return TRUE;				/* try another global perturbation */

 hs = h;
 nerr = 0;
 do
   {
    del = hs * yps;			/* make a perturbation on y */
    del = (ys + SGNP(del) * sround * MAX(ABS(ys), MAX(wt[j], ABS(del)))) - ys;

    if (ABS(del) < bound) del = SGNP(del) * bound;

    y[j] = ys + del;

    err = (*residuals)(rootg, t, y, yp, res, &dummy);

    if (!err) break;

    hs *= .25;
   } while (++nerr < iters -> maxerrorfail);

 y[j] = ys;				/* restore original value */

 if (!err)
   {
    for (i = 0; i < rankg; i++)
       dep[i] = (res[i] != resd[i]);   	/* equation involves y[j] or not */

    nerr = 0;
    do
      {
       yp[j] = yps + del / hs;

       err = (*residuals)(rootg, t, y, yp, res, &dummy);

       if (!err) break;

       hs *= .25;
       del = hs * yps;
       del = (ys + SGNP(del) * sround * MAX(ABS(ys), MAX(wt[j], ABS(del))))
	     - ys;

       if (ABS(del) < bound) del = SGNP(del) * bound;
      } while (++nerr < iters -> maxerrorfail);
   }

 if (err)
   {
    fprintf (stderr, "***ERROR res_dep: Error (%d) in the residual function was not able to be bypassed within %d retries\n",
	     err, nerr);
    exit (1);
   }

 for (i = 0; i < rankg; i++)
    {                                   /* equation involves yp[j] */
     if (res[i] != resd[i]) dep[i] += 2;
     if (dep[i]) gr[i].size++;
    }

 yp[j] = yps;				/* restore original value */

 return FALSE;
}					/* res_dep */


/*
 * set_band; build band structure for iteration matrix. If the
 * variable sparse = FALSE, then there is no sparse structure.
 * Return TRUE if psize must be deallocated, and FALSE otherwise.
 */
PRIV BOOL
set_band (FAST SPARSE_LIST *spl, BOOL sparse, int lb, int ub, int *psize)
{
 BOOL flag = TRUE;
 FAST int i, j, *index;
 int i1, size;

 if (lbg < 0)
   {
    fprintf (stderr, "***WARNING set_band: lband (-1) set to %d.\n",
	     lb);
    rootg -> jac.lband = lb;
   }
 else if (lbg != lb)
	{
	 fprintf (stderr, "***WARNING set_band: lband (%d) != %d (found).\n",
		  lbg, lb);
	 lbg = lb;
	 lb = rootg -> jac.lband;
	}

 if (ubg < 0)
   {
    fprintf (stderr, "***WARNING set_band: uband (-1) set to %d.\n",
	     ub);
    rootg -> jac.uband = ub;
   }
 else if (ubg != ub)
	{
	 fprintf (stderr, "***WARNING set_band: uband (%d) != %d (found).\n",
		  ubg, ub);
	 ubg = ub;
	 ub = rootg -> jac.uband;
	}

 if (sparse)
   for (j = 0; j < rankg; j++, spl++)
      {
       i = MAX(0,j-ub);
       i1 = MIN(rankg,j+lb+1);
       size = i1 - i;

       if (size == rankg)               /* keep using psize */
		{
		 if (flag && spl -> size == rankg) flag = FALSE;
		}
       else
		{
		 if (spl -> size == rankg)     /* do not keep using psize */
		 {
	      spl -> size = size;
	      spl -> index = index = NEW (int, size, "IDX");
	      for (; i < i1; i++) *index++ = i;
		 }
	     else
			if (spl -> index[0] < i || spl -> index[spl -> size - 1] >= i1)
			{
	         int *paux;

			 size = spl -> size;
			 paux = index = spl -> index;
			 while (*index < i)       /* cut exceeding rows */
				{
				 index++; size--;
				}
			 while (index[size-1] >= i1) size--;

			 spl -> size = size;
			 spl -> index = NEW (int, size, "IDX");
			 memcpy (spl -> index, index, size * sizeof (int));
			 free (paux);
			}
		}
      }
 else
   for (j = 0; j < rankg; j++, spl++)
      {
       psize[j] = j;

       i = MAX(0,j-ub);
       i1 = MIN(rankg,j+lb+1);
       spl -> size = i1 - i;

       if (spl -> size == rankg)
		{
		 spl -> index = psize;
		 if (flag) flag = FALSE;
		}
       else
		{
		 spl -> index = index = NEW (int, spl -> size, "IDX");
		 for (; i < i1; i++) *index++ = i;
		}
      }

 if (2 * lb + ub >= rankg)
   fprintf (stderr, "***WARNING set_band: better to use dense matrix!\n");

 return flag;
}					/* set_band */


/* update_root; allocates spaces and initializes remaining variables */
PRIV void
update_root (void)
{
 JACOBIAN *jac = &rootg -> jac;
 BDF_DATA *bdf = &rootg -> bdf;
 FAST int k;

 jac -> rank = 0;			/* clean structure */

 switch (jac -> mtype)
       {
	case USER_DENSE:
	case DENSE_MODE:
	     jac -> pivot = NEW (int, rankg, "PIVOT");
	     jac -> matrix = (char *)(NEW (REAL, rankg * rankg, "ITERATION MATRIX"));
	     break;

	case USER_BAND:
	     if (lbg < 0 || ubg < 0)
	       {
			fprintf (stderr, "***ERROR update_root: invalid %s = -1 in userband mode.\n",
					(lbg < 0 ? "lband" : "uband"));
			exit (1);
	       }

	case BAND_MODE:
	     jac -> pivot = NEW (int, rankg, "PIVOT");
	     jac -> matrix = (char *)(NEW (REAL, (2 * jac -> lband +
			     jac -> uband + 1) * rankg, "ITERATION MATRIX"));
	     break;

#ifdef SPARSE
	case USER_SPARSE:
	case SPARSE_MODE: jac -> matrix = daSparse_matrix (rankg);
	     break;
#endif /* SPARSE */

	case USER_ALGEBRA: /* nothing to do */
	     break;

	case NONE: jac -> matrix = NULL; break;

	default: fprintf (stderr, "***ERROR update_root: type of iteration matrix evaluation = %d is unknown.\n",
			  jac -> mtype);
		 exit (1);
       }

 k = iters -> maxorder + 1;		/* current BDF variables */
 bdf -> alpha = NEW (REAL, (rankg + 5) * k, "BDF");
 bdf -> beta = bdf -> alpha + k;
 bdf -> gamma = bdf -> beta + k;
 bdf -> sigma = bdf -> gamma + k;
 bdf -> psi = bdf -> sigma + k;
 bdf -> phi = bdf -> psi + k;

 if (iters -> linearmode == ITER_MODE || iters -> linearmode == BOTH_MODE)
   {
    if (krys -> kmp > krys -> maxl)
      {
       fprintf (stderr, "***WARNING update_root: kmp (%d) > maxl (%d) set to maxl.\n",
		krys -> kmp, krys -> maxl);

       krys -> kmp = krys -> maxl;
      }

    krys -> decomp = FALSE;
    krys -> ncfl = 0;
    krys -> nli = 0;
    krys -> nps = 0;
    krys -> perf = 0;
    krys -> sqrtn = sqrt ((double)rankg);
    krys -> rsqrtn = 1.0 / krys -> sqrtn;
    krys -> lintol = krys -> litol * iters -> convtol;

    k = krys -> maxl + 1;
    krys -> q = NEW (REAL, k * (rankg + 2) + rankg - 1, "QR,V Krylov");
    krys -> r = krys -> q + 2 * krys -> maxl;
    krys -> v = krys -> r + rankg + 1;
    krys -> z = NEW (REAL, rankg * (k + 4), "z,dl,res,yp,H Krylov");
    krys -> dl = krys -> z + rankg;
    krys -> yptem = krys -> dl + rankg;
    krys -> res = krys -> yptem + rankg;
    krys -> hes = krys -> res + rankg;
   }
}					/* update_root */


/*
 * is_command; find name in comtab and return a pointer to the corresponding
 * struct.  If not found return NULL.
 */
PRIV struct table *
is_command (char *name)
{
 FAST int i;

 for (i = 0; comtab[i].name; i++)
    if (!strcmp (name, comtab[i].name)) return (&comtab[i]);

 return NULL;
}					/* is_command */


/* get_information; read inputfile and setup data base, return false on error */
PRIV BOOL
get_information (FILE *fp, char *inputfile)
{
 struct table *tab;
 BOOL error = FALSE;
 char str[STRLEN], name[ALFALEN];
 int len, i;

 if (fp == stdin) fprintf (stdout, "Enter all information by keyboard or \nPut inputfile name fellowing the word 'inputfile' any time:\n");

 while (arGet_card (fp, str))		/* scan through inputfile */
      {
       len = strlen (str) - 1;
       if (str[len] == EOL) str[len] = EOS;

       i = 0;
       if (arGet_word (str, &i, name) != EOS && (tab = is_command (name), tab))
		 if (tab == &comtab[0]) error += tab -> vfunc (inputfile, str, i);
		 else if (tab == &comtab[1]) error += tab -> vfunc (fp, str, i);
			  else if (tab == &comtab[2])
					 error += tab -> vfunc (str, i, REALopttab, SHORTopttab, BOOLopttab, FUNCopttab);
				   else error += tab -> vfunc (str, i);
       else
		{
		 fprintf (stderr, "***ERROR get_information: Unknown command in \"%s\"\n", str);
		 error = TRUE;
		}
      }

 fclose (fp);

 return error;
}					/* get_information */


/* set_inputfile; set name of the inputfile */
PRIV BOOL
set_inputfile (char *inputfile, char *s, int i)
{
 FILE *fp;
 char name[ALFALEN];

 if (arGet_word (s, &i, name) == EOS)
   {
    fprintf (stderr, "***ERROR set_inputfile: Missing inputfile name in \"%s\"\n", s);
    return TRUE;
   }

 SWITCH (inputfile) return FALSE;
 CASE ("stdin") fp = stdin;
 DEFAULT
     if (!(fp = fopen (name, "r")))
       {
		fprintf (stderr, "***ERROR set_inputfile: Cannot open inputfile \"%s\"\n", name);
		return TRUE;
       }

 return (get_information (fp, name));
}					/* set_inputfile */


/* set_print; set printing flags */
PRIV BOOL
set_print (char *s, int i)
{
 BOOL error = FALSE;
 char name[ALFALEN];
 FAST int oldindex = -1;
 int index, j;

 if (arEmpty_str (s, &i))
   {
    rootg -> print = TRUE;
    return error;
   }

 while (j = i, !arEmpty_str (s, &i))
      if (arGet_word (s, &i, name) == EOS)
		{
		 fprintf (stderr, "***ERROR set_print: Unknown print flag in \"%s\"\n", s);
		 error = TRUE;
		}
      else
		{
		 if (!arIs_integer (name, 0))
		 {
		  fprintf (stderr, "***ERROR set_print: Unknown print option \"%s\" in \"%s\"\n",
				name, s);
		  error = TRUE;
	      continue;
		 }

		 i = j;

		 if (arIs_integer (s, i))
		   do
			{
			 if (((index = (int)arCtoi (s, &i)) < 0 && index > - oldindex) ||
				ABS(index) >= rankg)
			 {
			  fprintf (stderr, "***ERROR set_print: Illegal index %d in \"%s\"\n",
					index, s);
			  error = TRUE;
			 }
	         else
			 {
			  if (index < 0)
				{
				 index = - index;
				 for (++oldindex; oldindex < index; oldindex++)
					db[oldindex].set |= 0x01;
				}
			  else oldindex = index;

			  db[index].set |= 0x01;
			 }
			} while (arIs_integer (s, i));

		 else rootg -> print = TRUE;
		}

 return error;
}					/* set_print */


/* set_debug; set debug flags */
PRIV BOOL
set_debug (char *s, int i)
{
 DEBUG_SET *debug = &rootg -> debug;
 BOOL error = FALSE;
 char name[ALFALEN];

 if (arEmpty_str(s, &i))
   {
    debug -> nr = TRUE;
    debug -> bdf = TRUE;
    debug -> matrix = TRUE;
    debug -> conv = TRUE;
    return error;
   }

 while (!arEmpty_str (s, &i))
      if (arGet_word (s, &i, name) == EOS)
		{
		 fprintf (stderr, "***ERROR set_debug: Unknown debug flag in \"%s\"\n", s);
		 error = TRUE;
		}
      else
		{
		 SWITCH ("newton") debug -> nr = TRUE;
		 CASE ("bdf") debug -> bdf = TRUE;
		 CASE ("matrix") debug -> matrix = TRUE;
		 CASE ("conv") debug -> conv = TRUE;
		 DEFAULT
		 {
	      fprintf (stderr, "***ERROR set_debug: Unknown debug option \"%s\" in \"%s\"\n",
		       name, s);
	      error = TRUE;
	     }
	}
 return error;
}					/* set_debug */


/* set_rank; verify rank information */
PRIV BOOL
set_rank (char *s, int i)
{
 int rank;

 if (arIs_integer (s, i))
   {
    rank = (int)arCtoi (s, &i);
	if (rank != rootg -> rank)
	  {
       fprintf (stderr, "***ERROR set_rank: The rank value (%d) is different the one provided in daSetup (%d)\n", rank, rootg -> rank);
       return TRUE;
	  }
   }
 else
   {
    fprintf (stderr, "***ERROR set_rank: The rank value is missing in \"%s\"\n", s);
    return TRUE;
   }

 if (!arEmpty_str (s, &i))
   {
    fprintf (stderr, "***ERROR set_rank: The rank entry must be alone in \"%s\"\n", s);
    return TRUE;
   }
 
 return FALSE;
}					/* set_rank */

/*
 * is_option; find name in option tables and return a pointer to the
 * corresponding struct. If not found return NULL.
 */
PRIV char *
is_option (char *name, REALvar *rv, SHORTvar *sv, BOOLvar *bv, FUNCvar *fv)
{
 FAST int i;

 for (i = 0; rv[i].key; i++)
    if (!strcmp (name, rv[i].name))
      return ((char *)&rv[i]);

 for (i = 0; sv[i].key; i++)
    if (!strcmp (name, sv[i].name))
      return ((char *)&sv[i]);

 for (i = 0; bv[i].key; i++)
    if (!strcmp (name, bv[i].name))
      return ((char *)&bv[i]);

 for (i = 0; fv[i].key; i++)
    if (!strcmp (name, fv[i].name))
      return ((char *)&fv[i]);

 return (NULL);
}					/* is_option */


BOOL
arSetOption (PTR_ROOT *root, char *s)
{
 int i = 0;

 if (root)
 {
  FAST REALvar *rp = REALopttab;
  FAST SHORTvar *sp = SHORTopttab;
  FAST BOOLvar *bp = BOOLopttab;

  rootg = root;
  iters = &root -> iter;
  krys = &root -> kry;

  rp[0].varp = &iters->jacrate;
  rp[1].varp = &iters->nonblank;
  rp[2].varp = &iters->damps;
  rp[3].varp = &iters->dampi;
  rp[4].varp = &iters->tstop;
 /* rp[5].varp = iters->rtol; */
 /* rp[6].varp = iters->atol; */
 /* rp[7].varp = &root->bdf.h; */
  *iters->rtol = rtolg;
  *iters->atol = atolg;
  root->bdf.h = h0;
  rp[8].varp = &iters->hmax;
  rp[9].varp = &iters->convtol;
  rp[10].varp = &krys->litol;

  sp[0].varp = &iters->maxorder;
  sp[3].varp = &krys->maxl;
  sp[4].varp = &krys->kmp;
  sp[5].varp = &krys->maxrest;
  sp[6].varp = &iters->sparsethresh;
  sp[7].varp = &iters->maxconvfail;
  sp[8].varp = &iters->maxerrorfail;
  sp[9].varp = &iters->maxsingular;
  sp[10].varp = &iters->maxnewton;
  sp[11].varp = &iters->maxjacfix;
  sp[12].varp = &iters->maxlen;

  bp[0].varp = &iters->factor;
  bp[1].varp = &iters->iststop;
  bp[2].varp = &iters->istall;
  bp[3].varp = &iters->stol;

  bp[4].varp = &iters->nonneg;
  bp[5].varp = &krys->restart;
 }

 return arSet_option (s, i, REALopttab, SHORTopttab, BOOLopttab, FUNCopttab);
}					/* arSetOption */


/* arSet_option; set option flags and data */
BOOL
arSet_option (char *s, int i, REALvar *rv, SHORTvar *sv, BOOLvar *bv, FUNCvar *fv)
{
 REALvar *rvp;
 SHORTvar *svp;
 BOOLvar *bvp;
 FUNCvar *funcp;
 SET status;
 BOOL seenbang, error = FALSE;
 char name[ALFALEN], *vp;
 REAL r;

 if (arEmpty_str (s, &i))
   {
    fprintf (stderr, "***ERROR arSet_option: Missing option in \"%s\"\n", s);
    return TRUE;
   }

 while (!arEmpty_str (s, &i))		/* Scan rest of line for options */
      {
       if (s[i] == BANG)		/* Look for '!' prefix */
	   {
		seenbang = TRUE;
		i++;
	   }
       else seenbang = FALSE;

       if (arGet_word (s, &i, name) == EOS)
		{
		 fprintf (stderr, "***ERROR arSet_option: Missing option in \"%s\"\n", s);
		 error = TRUE;
		 continue;
		}

       if (!(vp = is_option (name, rv, sv, bv, fv)))
		{
		 fprintf (stderr, "***ERROR arSet_option: Unknown option \"%s\"\n", name);
		 error = TRUE;
		 continue;
		}

       status = STAT_OK;		/* reset value range status flag */

       if (*vp != BVAR && seenbang)	/* If seenbang, require the option to be
					 * boolean else signal an error.  If not
					 * seenbang use table to get value.
					 */
		{
		 fprintf (stderr, "***ERROR arSet_option: Option \"%s\" may not be preceeded by '%c'\n",
				name, BANG);
		 status = STAT_ERR;		/* Set status, cont and eat parameter */
		}

       switch (*vp)
	     {
	      case RVAR:
		   rvp = (REALvar *)vp;
		   r = *rvp -> varp = arGet_float (s, &i, rvp -> vmin,
				      rvp -> vmax, &status);
		   break;

	      case SVAR:
		   svp = (SHORTvar *)vp;
		   r = *svp -> varp = (SHORT)arGet_int (s, &i, (int)svp -> vmin,
				      (int)svp -> vmax, &status);
		   break;

	      case BVAR:
		   bvp = (BOOLvar *)vp;
		   r = *bvp -> varp = (BOOL)arGet_int (s, &i, (int) bvp -> vmin,
				      (int)bvp -> vmax, &status);
		   if (status == STAT_NO_VAL)
		     {
		      status = STAT_OK;
		      *bvp -> varp = !seenbang;
		     }
		   else if (seenbang)
			  {
			   fprintf (stderr, "***ERROR arSet_option: Don't mix \"!\" and parameter for option \"%s\"\n",
				    name);
			   status = STAT_ERR;
			  }
		   break;

	      case FUNC:
		   funcp = (FUNCvar *)vp;
		   status = (*funcp -> vfunc) (s, &i);
		   break;

	      default: fprintf (stderr, "***ERROR arSet_option: Internal error, unknown option class \"%s\"\n",
				name);
		       error = TRUE;
	     }

       switch (status)
	     {
	      case STAT_OK:
		   break;

	      case STAT_UP_RANGE:
	      case STAT_LOW_RANGE:
		   fprintf (stderr, "***ERROR arSet_option: Value out of %s range (%g) for parameter \"%s\" in \"%s\"\n",
			    (status == STAT_UP_RANGE ? "upper" : "lower"), r, name, s);
		   error = TRUE;
		   break;

	      case STAT_NO_VAL:
		   fprintf (stderr, "***ERROR arSet_option: No value for parameter \"%s\" in \"%s\"\n",
			    name, s);

	      case STAT_ERR:
	      default: error = TRUE;
	     }
      }
 return error;
}					/* arSet_option */


/* set_database; set database and get all additional data */
PRIV BOOL
set_database (FILE *fp, char *s, int i)
{
 BOOL error = FALSE;
 SET type;
 char name[ALFALEN];

 if (!arEmpty_str (s, &i))
   {
    if (arGet_word (s, &i, name) == EOS)
      {
       fprintf (stderr, "***ERROR set_database: Unknown data flag in \"%s\"\n", s);
       error = TRUE;
      }
    else
      {
       type = 0;

       SWITCH ("value") type = 1;	/* read individual information */

       CASE ("residual") type = 2;	/* read residual functions data */

       CASE ("database")		/* read global database */
	   {
	    int global = 0;

	    while (!(error += get_residual (s, &i, &global, -1)) &&
		   no_end_of_data (fp, s, &i));
	   }

       CASE ("initial")			/* read initial condition */
	   {
	    int *global = NEW (int, rankg, "GLB");

	    while (!(error += get_initial (s, &i, global)) &&
		   no_end_of_data (fp, s, &i));

	    free (global);
	   }

       CASE ("sparse")			/* read user-defined sparse structure */
	   {
	    int *global = NEW (int, 3 * rankg, "GLB");

	    while (!(error += get_sparse (s, &i, global)) &&
		   no_end_of_data (fp, s, &i));

	    free (global);
	   }

       DEFAULT
	   {
	    fprintf (stderr, "***ERROR set_database: Unknown data option \"%s\" in \"%s\"\n",
		     name, s);
	    error = TRUE;
	   }

       if (type)
	 {
	  int *global = NEW (int, rankg, "GLB"), rank = rankg;

	  if (!(error += arGet_list (s, &i, global, &rank)) && rank)
	    if (type == 1)
	      while (!(error += get_value (s, &i, global, rank)) &&
		     no_end_of_data (fp, s, &i));
	    else
	      while (!(error += get_residual (s, &i, global, rank)) &&
		     no_end_of_data (fp, s, &i));
	  else error = TRUE;

	  free (global);
	 }
    }
   }
 else
   {
    fprintf (stderr, "***ERROR set_database: Missing database in \"%s\"\n", s);
    error = TRUE;
   }

 if (error)				/* skip invalid database */
   while (no_end_of_data (fp, s, &i));

 return error;
}					/* set_database */


/*
 * no_end_of_data; returns FALSE if end of database was reached,
 * otherwise it returns TRUE and a new string.
 */
PRIV BOOL
no_end_of_data (FILE *fp, char *str, int *i)
{
 char name[ALFALEN];

 while (!arEmpty_str (str, i))
      {
       arGet_word (str, i, name);
       if (!strcmp (name, "endata")) return FALSE;
      }

 if (arGet_card (fp, str))
   {
    int len = strlen (str) - 1;

    if (str[len] == EOL) str[len] = EOS;

    *i = 0;
    arGet_word (str, i, name);
    if (strcmp (name, "endata"))
      {
       *i = 0;
       return TRUE;
      }
   }

 return FALSE;
}					/* no_end_of_data */


/* get_value; get all individual information about the variables */
PRIV BOOL
get_value (char *s, int *i, int *global, int rank)
{
 BOOL error = FALSE, seenbang;
 SET status;
 char name[ALFALEN];
 FAST int j;
 REAL r;

 while (!arEmpty_str (s, i))
      {
       status = STAT_OK;		/* reset value range status flag */

       if (s[*i] == BANG)		/* look for '!' prefix */
		{
		 seenbang = TRUE; (*i)++;
		}
       else seenbang = FALSE;

       j = *i;
       if (arGet_word (s, i, name) == EOS)
		{
		 fprintf (stderr, "***ERROR get_value: Missing information in \"%s\"\n", s);
		 error = TRUE;
		 continue;
		}

       SWITCH ("nonneg")		/* nonnegative solution */
	     {
	      BOOL val = (BOOL)arGet_int (s, i, 0, 1, &status);

	      if (status == STAT_NO_VAL)
			{
			 status = STAT_OK;
			 val = !seenbang;
			}
	      else if (seenbang)
		     {
		      fprintf (stderr, "***ERROR get_value: Don't mix '%c' and parameter for option \"%s\"\n",
			       BANG, name);
		      status = STAT_ERR;
		     }
	      for (r = val, j = 0; j < rank; j++)
			{
			 db[global[j]].nonneg = val;
			 db[global[j]].set |= 0x02;
			}
	     }

       else if (seenbang)
	      {
	       fprintf (stderr, "***ERROR get_value: Option \"%s\" may not be preceeded by '%c'\n",
			name, BANG);
	       status = STAT_ERR;
	      }

       CASE ("rtol")			/* relative tolerance */
	   {
	    r = arGet_float (s, i, 0., HUGE, &status);

	    if (status != STAT_NO_VAL)
	      for (j = 0; j < rank; j++)
			{
			 db[global[j]].rtol = r;
			 if (!(db[global[j]].set & 0x08)) db[global[j]].atol = 0.;
			 db[global[j]].set |= 0x08;
			}
	   }

       CASE ("atol")			/* absolute tolerance */
	   {
	    r = arGet_float (s, i, 0., HUGE, &status);

	    if (status != STAT_NO_VAL)
	      for (j = 0; j < rank; j++)
			{
			 db[global[j]].atol = r;
			 if (!(db[global[j]].set & 0x08)) db[global[j]].rtol = 0.;
			 db[global[j]].set |= 0x08;
			}
	   }

       CASE ("endata")
	   {
	    *i = j; break;
	   }

       DEFAULT
	   {
	    fprintf (stderr, "***ERROR get_value: Unknown data \"%s\" in \"%s\"\n",
		     name, s);
	    error = TRUE;
	   }

       switch (status)
	     {
	      case STAT_OK:
		   break;

	      case STAT_UP_RANGE:
	      case STAT_LOW_RANGE:
		   fprintf (stderr, "***ERROR get_value: Value out of %s range (%g) for parameter \"%s\" in \"%s\"\n",
			    (status == STAT_UP_RANGE ? "upper" : "lower"), r, name, s);
		   error = TRUE;
		   break;

	      case STAT_NO_VAL:
		   fprintf (stderr, "***ERROR get_value: No value for parameter \"%s\" in \"%s\"\n",
			    name, s);

	      case STAT_ERR:
	      default: error = TRUE;
	     }
      }

 return error;
}					/* get_value */


/*
 * get_residual; get database to each residual function according to
 * DATABASE structure. After partitioning the block-structures will
 * hold their respective databases.  Also, get global database when
 * rank is set to -1.
 */
PRIV BOOL
get_residual (char *s, int *i, int *global, int rank)
{
 DATABASE *prob, **probv;
 BOOL error = FALSE;
 char name[ALFALEN];
 FAST int j;
 int ntype, *k;

 if (rank > 0)
   {
    if (!rootg -> sub_prob)
      {
       rootg -> sub_prob = NEW (DATABASE *, rankg, "SUB");
       memset (rootg -> sub_prob, 0, rankg * sizeof (DATABASE *));
      }
    probv = rootg -> sub_prob;
   }
 else
   {
    rank = 1;
    probv = &rootg -> problem;
   }

 k = NEW (int, rank, "K");

 while (!arEmpty_str (s, i))
      {
       j = *i;
       if (arGet_word (s, i, name) == EOS)
		{
		 fprintf (stderr, "***ERROR get_residual: Missing information in \"%s\"\n", s);
		 error = TRUE;
		 continue;
		}

       if (arIs_integer (s, *i) && (ntype = (int)arCtoi (s, i)) > 0)
		{
		 SWITCH ("char")
			{
			 int len;

			 for (j = 0; j < rank; j++)
				{
				 if (!probv[global[j]])
				 {
				  prob = probv[global[j]] = NEW (DATABASE, 1, "DB");
				  memset (prob, 0, sizeof (DATABASE));
				 }
				 else prob = probv[global[j]];
				 k[j] = prob -> nchar;
				 prob -> nchar += ntype;
				 prob -> pchar = RENEW (char, prob -> pchar, prob -> nchar, "CHAR");
				}

			 while (ntype && !arEmpty_str (s, i) &&
				  !(arGet_char (s, i, name) == EOS && name[1] == EOL))
				{
				 len = strlen (name) + (name[0] == EOS);

				 if (ntype < len) len = ntype;

				 for (ntype--, j = 0; j < rank; j++)
					{
					 BCOPY (probv[global[j]] -> pchar + k[j], name, len);
					 k[j] += len;
					}
				}
			}

	  CASE ("short")
	      {
	       short d;

	       for (j = 0; j < rank; j++)
			{
			 if (!probv[global[j]])
			 {
		      prob = probv[global[j]] = NEW (DATABASE, 1, "DB");
		      memset (prob, 0, sizeof (DATABASE));
		     }
		 	 else prob = probv[global[j]];
			 k[j] = prob -> nshort;
			 prob -> nshort += ntype;
			 prob -> pshort = RENEW (short, prob -> pshort, prob -> nshort, "SHORT");
			}

	       while (ntype && arIs_integer (s, *i))
		    for (ntype--, d = (short)arCtoi (s, i), j = 0; j < rank; j++)
		       probv[global[j]] -> pshort[k[j]++] = d;
	      }

	  CASE ("int")
	      {
	       int d;

	       for (j = 0; j < rank; j++)
			{
			 if (!probv[global[j]])
			 {
		      prob = probv[global[j]] = NEW (DATABASE, 1, "DB");
		      memset (prob, 0, sizeof (DATABASE));
		     }
		     else prob = probv[global[j]];
		     k[j] = prob -> nint;
		     prob -> nint += ntype;
		     prob -> pint = RENEW (int, prob -> pint, prob -> nint, "INT");
			}

	       while (ntype && arIs_integer (s, *i))
		    for (ntype--, d = (int)arCtoi (s, i), j = 0; j < rank; j++)
		       probv[global[j]] -> pint[k[j]++] = d;
	      }

	  CASE ("long")
	      {
	       long d;

	       for (j = 0; j < rank; j++)
			{
			 if (!probv[global[j]])
		     {
		      prob = probv[global[j]] = NEW (DATABASE, 1, "DB");
		      memset (prob, 0, sizeof (DATABASE));
		     }
		     else prob = probv[global[j]];
		     k[j] = prob -> nlong;
		     prob -> nlong += ntype;
		     prob -> plong = RENEW (long, prob -> plong, prob -> nlong, "LONG");
			}

	       while (ntype && arIs_integer (s, *i))
		    for (ntype--, d = arCtoi (s, i), j = 0; j < rank; j++)
		       probv[global[j]] -> plong[k[j]++] = d;
	      }

	  CASE ("float")
	      {
	       float d;

	       for (j = 0; j < rank; j++)
			{
			 if (!probv[global[j]])
		     {
		      prob = probv[global[j]] = NEW (DATABASE, 1, "DB");
		      memset (prob, 0, sizeof (DATABASE));
		     }
		     else prob = probv[global[j]];
		     k[j] = prob -> nfloat;
		     prob -> nfloat += ntype;
		     prob -> pfloat = RENEW (float, prob -> pfloat, prob -> nfloat, "FLOAT");
			}

	       while (ntype && arIs_number (s, *i))
		    for (ntype--, d = (float)arCtof (s, i), j = 0; j < rank; j++)
		       probv[global[j]] -> pfloat[k[j]++] = d;
	      }

	  CASE ("double")
	      {
	       double d;

	       for (j = 0; j < rank; j++)
			{
			 if (!probv[global[j]])
		     {
		      prob = probv[global[j]] = NEW (DATABASE, 1, "DB");
		      memset (prob, 0, sizeof (DATABASE));
		     }
		     else prob = probv[global[j]];
		     k[j] = prob -> ndouble;
		     prob -> ndouble += ntype;
		     prob -> pdouble = RENEW (double, prob -> pdouble, prob -> ndouble, "DOUBLE");
			}

	       while (ntype && arIs_number (s, *i))
		    for (ntype--, d = arCtof (s, i), j = 0; j < rank; j++)
		       probv[global[j]] -> pdouble[k[j]++] = d;
	      }

	  DEFAULT
	      {
	       fprintf (stderr, "***ERROR get_residual: Unknown data \"%s\" in \"%s\"\n",
			name, s);
	       error = TRUE;
	       continue;
	      }

	  if (ntype)
	    {
	     fprintf (stderr, "***ERROR get_residual: Number of data not enough in \"%s\"\n", s);
	     error = TRUE;
	    }
	 }

       CASE ("endata")
	   {
	    *i = j; break;
	   }

       DEFAULT
	   {
	    fprintf (stderr, "***ERROR get_residual: Missing or illegal number of data type \"%s\" in \"%s\"\n",
		     name, s);
	    error = TRUE;
	   }
    }

 free (k);
 return error;
}					/* get_residual */


/* get_initial; read initial condition from input file */
PRIV BOOL
get_initial (char *s, int *i, int *global)
{
 BOOL error = FALSE;
 char name[ALFALEN];
 FAST int j;

 if (!rootg -> y || !rootg -> yp)
   {
    int size = rankg * sizeof (REAL);

    if (!rootg -> y) rootg -> y = (REAL *)must_malloc (size, "GLOBAL_Y");
    if (!rootg -> yp) rootg -> yp = (REAL *)must_malloc (size, "GLOBAL_YP");

    rootg -> t = 0.;			/* zero as default value */
    memset (rootg -> y, 0, size);
    memset (rootg -> yp, 0, size);
   }

 while (!arEmpty_str (s, i))
      {
       j = *i;
       if (arGet_word (s, i, name) == EOS)
		{
		 fprintf (stderr, "***ERROR get_initial: Unknown data \"%s\" in \"%s\"\n",
			  name, s);
		 error = TRUE;
		 continue;
		}

       if (arIs_integer (name, 0))	/* get value and derivative */
		{
		 int rank = rankg;
		 REAL y, yp;

		 *i = j;
		 error += arGet_list (s, i, global, &rank);

		 if (!error && rank && arIs_symbol (s, i, SEPARATOR) &&
	       arIs_number (s, *i))
		 {
	      y = (REAL)arCtof (s, i);
	      if (arIs_number (s, *i)) yp = (REAL)arCtof (s, i);
	      else yp = 0.;

	      for (j = 0; j < rank; j++)
			{
			 rootg -> y[global[j]] = y;
			 rootg -> yp[global[j]] = yp;
			}
		 }
	     else
		 {
	      fprintf (stderr, "***ERROR get_initial: No index or missing separator '%c' or value in \"%s\"\n",
		      SEPARATOR, s);
	      error = TRUE;
		 }
		}

       CASE ("time")			/* get initial time */
	   {
	    if (arIs_number (s, *i)) rootg -> t = (REAL)arCtof (s, i);
	    else
	      {
	       fprintf (stderr, "***ERROR get_initial: Missing initial time in \"%s\"\n", s);
	       error = TRUE;
	      }
	   }

       CASE ("endata")
	   {
	    *i = j; break;
	   }

       DEFAULT
	   {
	    fprintf (stderr, "***ERROR get_initial: Unknown data \"%s\" in \"%s\"\n",
		     name, s);
	    error = TRUE;
	   }
      }

 return error;
}					/* get_initial */


/* copy_index; copies index vector into VAR_DATA index */
PRIV void
copy_index (int i, int rank, int *index)
{
 int size = rank * sizeof (int), oldsize;

 if (db[i].set & 0x04)
   {
    oldsize = db[i].nonblanks;
    db[i].nonblanks += rank;
    db[i].index = RENEW (int, db[i].index, db[i].nonblanks, "DBX");
    BCOPY (db[i].index + oldsize, index, size);
   }
 else
   {
    db[i].nonblanks = rank;
    db[i].index = NEW (int, rank, "DBX");
    db[i].set |= 0x04;
    BCOPY (db[i].index, index, size);
   }
}                                       /* copy_index */


/* get_sparse; get the user-defined sparse structure */
PRIV BOOL
get_sparse (char *s, int *i, int *global)
{
 BOOL error = FALSE;
 char name[ALFALEN];
 FAST int j;

 while (!arEmpty_str (s, i))
      {
       j = *i;
       if (arGet_word (s, i, name) == EOS)
        {
		 fprintf (stderr, "***ERROR get_sparse: Unknown data \"%s\" in \"%s\"\n",
			 name, s);
		 error = TRUE;
		 continue;
		}

       if (arIs_integer (name, 0))	/* get non-blank index */
		{
		 int rank = rankg, *index, rankn = 0;

		 *i = j;
		 error += arGet_list (s, i, global, &rank);

		 if (!error && rank && arIs_symbol (s, i, SEPARATOR) &&
	        arIs_integer (s, *i))
		 {
	      rankn = rank; rank = rankg;
	      index = global + rankn;

	      if (!(error += arGet_list (s, i, index, &rank)) && rank)
	       for (j = 0; j < rankn; j++) copy_index (global[j], rank, index);
		 }

	     if (error || !rank)
		 {
	      fprintf (stderr, "***ERROR get_sparse: Missing separator '%c' or illegal indices in \"%s\"\n",
		      SEPARATOR, s);
	      error = TRUE;
		 }
		}

       CASE ("i")
	   {
	    FAST int m, n, p;
	    int rank = rankg, *index, *op, rankn = 0, sign, k;
	    char sp = '=';

	    if (arIs_symbol (s, i, sp))
	      {
	       sp = SEPARATOR;
	       error += arGet_list (s, i, global, &rank);

	       if (!error && rank && arIs_symbol (s, i, sp) && arIs_integer (s, *i))
			{
			 rankn = rank; rank = rankg;
			 op = global + rankn;
			 index = op + rank;

			 for (; *i < STRLEN && isspace (s[*i]); (*i)++);

			 sign = (s[*i] == '-' ? -1 : 1);

			 if (s[*i] == '+' || s[*i] == '-') (*i)++;

			 if (!(error += arGet_list (s, i, op, &rank)) && rank)
			 {
		      for (j = 0; j < rankn; j++)
				{
				 k = global[j];
				 for (n = m = 0; m < rank; m++)
					{
					 p = k + op[m] * sign;
					 if (p >= 0 && p < rankg) index[n++] = p;
					}

				 copy_index (k, n, index);
				}
			 }
			}
            else error = TRUE;
	      }
	    else error = TRUE;

	    if (error || !rank)
	      {
	       fprintf (stderr, "***ERROR get_sparse: Missing separator '%c' or illegal indices in \"%s\"\n",
			sp, s);
	       error = TRUE;
	      }
	   }

       CASE ("endata")
	   {
	    *i = j; break;
	   }

       DEFAULT
	   {
	    fprintf (stderr, "***ERROR get_sparse: Unknown data \"%s\" in \"%s\"\n",
		     name, s);
	    error = TRUE;
	   }
      }

 return error;
}					/* get_sparse */


/* set_mtype; set iteration matrix type */
PRIV SET
set_mtype (char *s, int *i)
{
 char name[ALFALEN];

 if (arGet_word (s, i, name) == EOS)
   {
    fprintf (stderr, "***ERROR set_mtype: Missing iteration matrix type in \"%s\"\n", s);
    return STAT_ERR;
   }

 SWITCH ("userdense") rootg -> jac.mtype = USER_DENSE;
 CASE ("dense") rootg -> jac.mtype = DENSE_MODE;
 CASE ("usersparse") rootg -> jac.mtype = USER_SPARSE;
 CASE ("sparse") rootg -> jac.mtype = SPARSE_MODE;
 CASE ("userband") rootg -> jac.mtype = USER_BAND;
 CASE ("band") rootg -> jac.mtype = BAND_MODE;
 CASE ("useralgebra") rootg -> jac.mtype = USER_ALGEBRA;
 CASE ("none") rootg -> jac.mtype = NONE;
 DEFAULT
     {
      fprintf (stderr, "***ERROR set_mtype: Unknown iteration matrix type \"%s\" in \"%s\"\n",
	       name, s);
      return STAT_ERR;
     }

 return STAT_OK;
}					/* set_mtype */


/* set_linearmode; set linear solver mode */
PRIV SET
set_linearmode (char *s, int *i)
{
 char name[ALFALEN];

 if (arGet_word(s, i, name) == EOS)
   {
    fprintf (stderr, "***ERROR set_linearmode: Missing linear mode in \"%s\"\n", s);
    return STAT_ERR;
   }

 SWITCH ("direct") iters -> linearmode = DIR_MODE;
 CASE ("iterative") iters -> linearmode = ITER_MODE;
 CASE ("both") iters -> linearmode = BOTH_MODE;
 DEFAULT
     {
      fprintf (stderr, "***ERROR set_linearmode: Unknown linear mode \"%s\" in \"%s\"\n",
	       name, s);
      return STAT_ERR;
     }

 return STAT_OK;
}					/* set_linearmode */


/* set_sparsemode; set sparse structure mode */
PRIV SET
set_sparsemode (char *s, int *i)
{
 char name[ALFALEN];

 if (arGet_word(s, i, name) == EOS)
   {
    fprintf (stderr, "***ERROR set_sparsemode: Missing sparse mode in \"%s\"\n", s);
    return STAT_ERR;
   }

 SWITCH ("infile") iters -> sparsemode = INFILE_SPARSE;
 CASE ("none") iters -> sparsemode = NO_SPARSE;
 CASE ("eval") iters -> sparsemode = EVAL_SPARSE;
 DEFAULT
     {
      fprintf (stderr, "***ERROR set_sparsemode: Unknown sparse mode \"%s\" in \"%s\"\n",
	       name, s);
      return STAT_ERR;
     }

 return STAT_OK;
}					/* set_sparsemode */


/* set_savefile; set name of the savefile */
PRIV SET
set_savefile (char *s, int *i)
{
 char name[ALFALEN];

 if (arGet_word (s, i, name) == EOS)
   {
    fprintf (stderr, "***ERROR set_savefile: Missing savefile name in \"%s\"\n", s);
    return STAT_ERR;
   }

 rootg -> filename = arStr_save (name);
 rootg -> alloc |= 0x10;

 SWITCH ("stdout") rootg -> savefile = stdout;
 CASE ("stderr") rootg -> savefile = stderr;
 DEFAULT if (!(rootg -> savefile = fopen (rootg -> filename, "w+")))
	   {
	    fprintf (stderr, "***ERROR set_savefile: Cannot open savefile \"%s\"\n",
		     rootg -> filename);
	    return STAT_ERR;
	   }

 return STAT_OK;
}					/* set_savefile */


/* set_pertfile; set name of the perturbation matrix inputfile */
PRIV SET
set_pertfile (char *s, int *i)
{
 char name[ALFALEN + 1];

 if (arGet_word (s, i, name + 1) == EOS)
   {
    fprintf (stderr, "***ERROR set_pertfile: Missing pertfile name in \"%s\"\n", s);
    return STAT_ERR;
   }

 name[0] = 'r';				/* read status */
 rootg -> pertfile = arStr_save (name);
 rootg -> alloc |= 0x20;

 return STAT_OK;
}					/* set_pertfile */


/* set_savepert; set name of the perturbation matrix savefile */
PRIV SET
set_savepert (char *s, int *i)
{
 if (set_pertfile (s, i) == STAT_OK)
   {
    rootg -> pertfile[0] = 'w';		/* write status */
    return STAT_OK;
   }
 else return STAT_ERR;
}					/* set_savepert */


/* daPrint_point; write present timepoint on file */
void
daPrint_point (FILE *fp, PTR_ROOT *root, REAL t)
{
 char *b = "               ";
 FAST int i = 0;
 int rank = root -> rank;
 REAL *y = root -> y, *yp = root -> yp, t0 = tused ();

 if (root -> mode != STEADY_STATE)
   if (rank > 1)
     fprintf (fp, "t = %10.3e%s%s\n", (double)t, head, head);
   else
     fprintf (fp, "t = %10.3e%s\n", (double)t, head);
 else
   if (rank > 1)
     fprintf (fp, "t =  infinity %s%s\n", head, head);
   else
     fprintf (fp, "t =  infinity %s\n", head);

 if (root -> print)
   {
    for (; i < rank; i += 2)
       if (rank - i > 1)
	 fprintf (fp, "%s(%5d) %12.5e %10.3e  (%5d) %12.5e %10.3e\n", b,
		  i, (double)y[i], (double)yp[i], i+1, (double)y[i+1], (double)yp[i+1]);
       else
	 fprintf (fp, "%s(%5d) %12.5e %10.3e\n", b, i, (double)y[i], (double)yp[i]);
   }
 else
   {
    int *idx = root -> idxprint, j, k;

    while ((j = idx[i], j) >= 0)
	 {
	  if ((k = idx[i+1], k) < 0)
	    {
	     fprintf (fp, "%s(%5d) %12.5e %10.3e\n", b, j, (double)y[j], (double)yp[j]);
	     break;
	    }
	  else
	    fprintf (fp, "%s(%5d) %12.5e %10.3e  (%5d) %12.5e %10.3e\n", b,
		     j, (double)y[j], (double)yp[j], k, (double)y[k], (double)yp[k]);
	  i += 2;
	 }
   }

 fprintf (fp, "\n");
 fflush (fp);

 root -> debug.t_info.t_save += tused () - t0;
}					/* daPrint_point */


#define PHI(i,j)	(*(phi + (i) * rank + j))

/* daPrint_bdf; print bdf data */
void
daPrint_bdf (FILE *fp, PTR_ROOT *root)
{
 BDF_DATA *bdf = &root -> bdf;
 SHORT orderold = bdf -> orderold;
 FAST int i, j;
 int rank = root -> rank;
 REAL *alpha = bdf -> alpha, *beta = bdf -> beta, *gamma = bdf -> gamma,
      *sigma = bdf -> sigma, *psi = bdf -> psi, *phi = bdf -> phi;

 fprintf (fp, "***DEBUG daPrint_bdf: BDF data:\n");
 fprintf (fp, "old steplength = %10.3e, steplength = %10.3e\n",
	  (double)bdf -> hold, (double)bdf -> h);
 fprintf (fp, "old bdf order = %d, bdf order = %d, order update = %d\n",
	  orderold, bdf -> order, bdf -> updateorder);
 fprintf (fp, "last time point = %10.3e\n\n", (double)bdf -> tfar);
 fprintf (fp, "matrix of divided differences:\n");

 for (i = 0; i <= orderold; i += 7)
    {
     for (j = 0; j < rank; j++)
	 switch (orderold - i + 1)
	      {
	       case 1: fprintf (fp, "%9.3e\n", (double)phi[j]);
		       break;

	       case 2: fprintf (fp, "%9.3e %9.3e\n", (double)phi[j], (double)PHI(1,j));
		       break;

	       case 3: fprintf (fp, "%9.3e %9.3e %9.3e\n", (double)phi[j],
				(double)PHI(1,j), (double)PHI(2,j));
		       break;

	       case 4: fprintf (fp, "%9.3e %9.3e %9.3e %9.3e\n", (double)phi[j],
				(double)PHI(1,j), (double)PHI(2,j), (double)PHI(3,j));
		       break;

	       case 5: fprintf (fp, "%9.3e %9.3e %9.3e %9.3e %9.3e\n",
				(double)phi[j],	(double)PHI(1,j), (double)PHI(2,j),
				(double)PHI(3,j), (double)PHI(4,j));
		       break;

	       case 6: fprintf (fp, "%9.3e %9.3e %9.3e %9.3e %9.3e %9.3e\n",
				(double)phi[j], (double)PHI(1,j), (double)PHI(2,j),
				(double) PHI(3,j), (double)PHI(4,j), (double)PHI(5,j));
		       break;

	       default: fprintf (fp, "%9.3e %9.3e %9.3e %9.3e %9.3e %9.3e %9.3e\n",
				 (double)phi[j], (double)PHI(1,j), (double)PHI(2,j),
				 (double)PHI(3,j), (double)PHI(4,j),
				 (double)PHI(5,j), (double)PHI(6,j));
	      }
     fprintf (fp, "\n");
    }

 fprintf (fp, "bdf coefficients:\n");
 fprintf (fp, "   alpha        beta        gamma      sigma       psi\n");
 for (i = 0; i <= orderold; i++)
    fprintf (fp, "%10.3e  %10.3e  %10.3e  %10.3e %10.3e\n", (double)alpha[i],
	     (double)beta[i], (double)gamma[i], (double)sigma[i], (double)psi[i]);

 fprintf (fp, "\n");
 fflush (fp);
}					/* daPrint_bdf */


/* Print_no_convergence; print no convergent time-point and bdf data */
PRIV void
Print_no_convergence (FILE *fp, PTR_ROOT *root, REAL t)
{
 fprintf (fp, "***DEBUG Print_no_convergence: No convergence at t = %10.3e\n",
	  (double)t);

 daPrint_point (fp, root, t);

 daPrint_bdf (fp, root);
}					/* Print_no_convergence */


/* daPrint_matrix; print the iteration matrix */
void
daPrint_matrix (FILE *fp, PTR_ROOT *root, REAL t)
{
 JACOBIAN *jac = &root -> jac;
 FAST int i, j;
 int rank = root -> rank;
 REAL value;

 fprintf (fp, "***DEBUG daPrint_matrix: Iteration Matrix at t = %10.3e:\n",
	  (double)t);

 fprintf (fp, "                       rank = %d\n\n", rank);
 fprintf (fp, "  row   column        value\n");
 switch (jac -> mtype)
       {
		case USER_DENSE:
		case DENSE_MODE:
	    {
	     REAL *matrix = (REAL *)jac -> matrix;

	     for (i = 0; i < rank; i++, matrix += rank)
			for (j = 0; j < rank; j++)
			{
		     value = *(matrix + j);
		     if (value)
		       fprintf (fp, "%4d     %4d      % e\n", i, j, (double)value);
			}
	     break;
	    }

		case USER_BAND:
		case BAND_MODE:
            {
             REAL *matrix;
			 int lb = jac -> lband, ub = jac -> uband, m = lb + ub, j0, j1;

             for (i = 0; i < rank; i++)
			 {
			  matrix = (REAL *)jac -> matrix + (m + MIN(lb,i)) * rank;
			  j0 = MAX(0,i-lb);
			  j1 = MIN(rank,i+ub+1);
                 for (j = j0; j < j1; j++, matrix -= rank)
                    {
                     value = *(matrix + j);
                     if (value)
                       fprintf (fp, "%4d     %4d      % e\n", i, j, (double)value);
                    }
			 }
             break;
            }

#ifdef SPARSE
		case USER_SPARSE:
		case SPARSE_MODE:
	    {
	     char *matrix = jac -> matrix;

	     for (i = 0; i < rank; i++)
			for (j = 0; j < rank; j++)
			{
		     value = daSparse_value (matrix, i, j);
		     if (value)
		      fprintf (fp, "%4d     %4d      % e\n",
			       i, j, (double)value);
			}
	     break;
	    }
#endif /* SPARSE */
	    case USER_ALGEBRA:
			{
			 if (ujacPrint) (*ujacPrint) (root);
			 break;
			}
       }
 fprintf (fp, "\n");
 fflush (fp);
}					/* daPrint_matrix */


/* Print_header; print header information */
void
daPrint_header (FILE *fp, char *in)
{
 char date[ALFALEN], time[ALFALEN];

 fprintf (fp, "*** DASSLC - Differential/Algebraic System Solver in C ***\n");
 fprintf (fp, "Copyright (C) 1992,2007 Argimiro R. Secchi, UFRGS - Version 3.2\n\n");
 fprintf (fp, "Input file: %s   Date: %s    Time: %s\n\n", (in ? in : "stdin"),
	  arDate (date), arTime (time));
 fflush (fp);
}					/* Print_header */


/* Print_run; print parameter settings */
void
daPrint_run (FILE *fp, PTR_ROOT *root)
{
 ITER_SET *iter = &root -> iter;
 char *tf[2], *mtype, *linear, *sparse;

 tf[0] = "FALSE"; tf[1] = "TRUE";

 switch (root -> jac.mtype)
       {
		case USER_DENSE: mtype = "userdense"; break;
		case DENSE_MODE: mtype = "dense"; break;
		case USER_SPARSE: mtype = "usersparse"; break;
		case SPARSE_MODE: mtype = "sparse"; break;
		case USER_BAND: mtype = "userband"; break;
		case BAND_MODE: mtype = "band"; break;
		case USER_ALGEBRA: mtype = "useralgebra"; break;
		case NONE: mtype = "none";
       }

 switch (iter -> linearmode)
       {
		case DIR_MODE: linear = "direct"; break;
		case ITER_MODE: linear = "iterative"; break;
		case BOTH_MODE: linear = "both";
       }

 switch (iter -> sparsemode)
       {
		case INFILE_SPARSE: sparse = "infile"; break;
		case NO_SPARSE: sparse = "none"; break;
		case EVAL_SPARSE: sparse = "eval";
       }

 fprintf (fp, "Iteration parameters:\n");
 fprintf (fp, " mtype %s\n", mtype);
 if (root -> jac.mtype == USER_BAND || root -> jac.mtype == BAND_MODE)
   {
    if (lbg == root -> jac.lband) fprintf (fp, " lband %d\n", lbg);
    else if (lbg < 0) fprintf (fp, " lband -1 --> set to %d\n", root -> jac.lband);
	 else fprintf (fp, " lband %d (found %d)\n", root -> jac.lband, lbg);

    if (ubg == root -> jac.uband) fprintf (fp, " uband %d\n", ubg);
    else if (ubg < 0) fprintf (fp, " uband -1 --> set to %d\n", root -> jac.uband);
	 else fprintf (fp, " uband %d (found %d)\n", root -> jac.uband, ubg);
   }
 fprintf (fp, " linearmode %s\n", linear);
 fprintf (fp, " sparsemode %s\n", sparse);
 fprintf (fp, " sparsethr %d\n", iter -> sparsethresh);
 fprintf (fp, " nonblank %g\n", (double)iter -> nonblank);
 fprintf (fp, " maxnewton %d\n", iter -> maxnewton);
 fprintf (fp, " maxjacfix %d\n", iter -> maxjacfix);
 fprintf (fp, " maxlen %d\n\n", iter -> maxlen);

 fprintf (fp, "Convergence parameters:\n");
 fprintf (fp, " maxorder %d\n", iter -> maxorder);
 fprintf (fp, " stepsize %g\n", (double)root -> bdf.h);
 fprintf (fp, " maxstep %g\n", (double)iter -> hmax);
 if (iter -> linearmode == ITER_MODE || iter -> linearmode == BOTH_MODE)
   {
    fprintf (fp, " maxl %d\n", root -> kry.maxl);
    fprintf (fp, " kmp %d\n", root -> kry.kmp);
    fprintf (fp, " maxrest %d\n", root -> kry.maxrest);
    fprintf (fp, " litol %g\n", (double)root -> kry.litol);
    fprintf (fp, " lintol %g\n", (double)root -> kry.lintol);
   }
 fprintf (fp, " factor %s\n", tf[iter -> factor]);
 fprintf (fp, " damps %g\n", (double)iter -> damps);
 fprintf (fp, " dampi %g\n", (double)iter -> dampi);
 fprintf (fp, " convtol %g\n", (double)iter -> convtol);
 fprintf (fp, " stol %s\n", tf[iter -> stol]);
 if (iter -> stol)
 {
    fprintf (fp, " rtol %g\n", (double)(*iter -> rtol));
    fprintf (fp, " atol %g\n", (double)(*iter -> atol));
   }
 else
   {
    fprintf (fp, " rtol (default) %g\n", (double)rtolg);
    fprintf (fp, " atol (default) %g\n", (double)atolg);
   }
 fprintf (fp, " nonneg %s\n", tf[iter -> nonneg]);
 fprintf (fp, " maxconvfail %d\n", iter -> maxconvfail);
 fprintf (fp, " maxerrorfail %d\n", iter -> maxerrorfail);
 fprintf (fp, " jacrate %g\n", (double)iter -> jacrate);
 fprintf (fp, " maxsingular %d\n\n", iter -> maxsingular);

 fprintf (fp, "Transient analysis parameters:\n");
 if (iter -> dae_index < 0) fprintf (fp, " differential index 0 or 1\n");
 else fprintf (fp, " differential index %d\n", iter -> dae_index);
 fprintf (fp, " iststop %s\n", tf[iter -> iststop]);
 if (iter -> iststop) fprintf (fp, " tstop %g\n", (double)iter -> tstop);
 fprintf (fp, " istall %s\n\n", tf[iter -> istall]);

 fprintf (fp, "Debugging parameters:\n");
 fprintf (fp, " print %s\n", tf[root -> print]);
 fprintf (fp, " newton %s\n", tf[root -> debug.nr]);
 fprintf (fp, " bdf %s\n", tf[root -> debug.bdf]);
 fprintf (fp, " conv %s\n", tf[root -> debug.conv]);
 fprintf (fp, " matrix %s\n\n", tf[root -> debug.matrix]);
 fflush (fp);
}					/* Print_run */


/* daStat; print statistic data */
void
daStat (FILE *fp, PTR_ROOT *root)
{
 TIMERTAB *t_info = &root -> debug.t_info;
 ITER_SET *iter = &root -> iter;
 char *mode;
 REAL t0 = tused ();

 switch (root -> mode)
       {
		case SETUP: mode = "Setup"; break;
		case STEADY_STATE: mode = "Steady-state"; break;
		case INITIAL_COND: mode = "Initial condition"; break;
		case TRANSIENT: mode = "Transient"; break;
		case ERROR: mode = "Error"; break;
		default: mode = "Unknown";
       }

 fprintf (fp, "Timing information (in seconds): last mode %s\n", mode);
 fprintf (fp, "    %-17s %10.2f\n", "setup", (double)t_info -> t_setup);
 fprintf (fp, "    %-17s %10.2f\n", "steady state", (double)t_info -> t_steady);
 fprintf (fp, "    %-17s %10.2f\n", "initial condition", (double)t_info -> t_initial);
 fprintf (fp, "    %-17s %10.2f\n", "transient", (double)t_info -> t_trans);
 fprintf (fp, "    %-17s %10.2f\n", "sparse structure", (double)t_info -> t_sparse);
 fprintf (fp, "    %-17s %10.2f\n", "perturb. matrix", (double)t_info -> t_perturb);
 fprintf (fp, "    %-17s %10.2f\n", "saving timepoints", (double)t_info -> t_save);
 t_info -> t_total = t_info -> t_setup + t_info -> t_steady + t_info -> t_initial +
		     t_info -> t_trans + tused () - t0;
 fprintf (fp, "    %-17s %10.2f\n\n", "total time", (double)t_info -> t_total);

 fprintf (fp, "Total number of time-points: %ld\n", iter -> tottpnt);
 fprintf (fp, "Total number of evaluation of residuals: %ld\n", iter -> reseval);
 fprintf (fp, "Total number of evaluation of jacobian: %ld\n", iter -> jaceval);
 fprintf (fp, "Total number of Newton-Raphson iterations: %ld\n", iter -> newton);
 if (root -> kry.q)
   {
    fprintf (fp, "Total number of preconditionnings: %ld\n", root -> kry.nps);
    fprintf (fp, "Total number of linear iterations: %ld\n", root -> kry.nli);
    fprintf (fp, "Total number of linear conv. failures: %ld\n", root -> kry.ncfl);
    fprintf (fp, "Total number of poor conv. performances: %ld\n", root -> kry.perf);
    if (iter -> newton)
      fprintf (fp, "Average Krylov subspace dimension: %g\n",
	       root -> kry.nli / (double)iter -> newton);
   }
 fprintf (fp, "Total number of error test failures: %ld\n", iter -> errorfail);
 fprintf (fp, "Total number of convergence test failures: %ld\n", iter -> convfail);
 fprintf (fp, "Total number of rejected time-points: %ld\n", iter -> rejtpnt);
 fprintf (fp, "Total number of rejected Newton-Raphson iterations: %ld\n\n",
	  iter -> rejnewton);

 fprintf (fp, "Roundoff: %20.18e\n", (double)iter -> roundoff);
 fprintf (fp, "Total CPU-time (Setup + Simulation): %.2f sec\n", (double)t_info -> t_total);
 fprintf (fp, "Simulation time: %.2f sec\n", (double)(t_info -> t_trans +
	  t_info -> t_initial + t_info -> t_steady));
 fprintf (fp, "\n");
 fflush (fp);
}					/* daStat */


/* daFree; free dynamic allocated area */
void
daFree (PTR_ROOT *root)
{
 if (root -> mode >= SETUP && root -> mode <= TRANSIENT)
   {
    int rank = root -> rank;
    FAST int i;

    pfree (root -> idxprint);

    switch (root -> jac.mtype)
       {
		case USER_DENSE:
		case DENSE_MODE:
		case USER_BAND:
		case BAND_MODE:
			pfree (root -> jac.pivot);
			pfree (root -> jac.matrix);
			break;

#ifdef SPARSE
		case USER_SPARSE:
		case SPARSE_MODE: spDestroy (root -> jac.matrix);
			break;
#endif /* SPARSE */

		case USER_ALGEBRA: if (ujacFree) (*ujacFree) (root);
			break;
		case NONE: break;

		default: fprintf (stderr, "***ERROR daFree: type of iteration matrix evaluation = %d is unknown.\n",
						  root -> jac.mtype);
				 exit (1);
       }

    if (root -> jac.mtype <= DENSE_MODE && root -> jac.mtype > NONE)
      {
       SPARSE_LIST *spl = root -> jac.spl;
       BOOL flag = TRUE;

       if (rank > 1 && (root -> iter.sparsemode != NO_SPARSE || root -> jac.mtype == BAND_MODE))
		{
	     for (i = 0; i < rank; i++)
	        if (spl[i].size < rank) {pfree (spl[i].index);}
	        else if (flag)
				   {
				    pfree (spl[i].index);
				    flag = FALSE;
			       }
		}
       else pfree (spl -> index);

       pfree (spl);
      }

    pfree (root -> bdf.alpha);
    pfree (root -> iter.rtol);
    pfree (root -> iter.wt);

    if (root -> kry.q)
      {
       pfree (root -> kry.q);
       pfree (root -> kry.z);
      }

    if (root -> alloc & 0x01) pfree (root -> y);
    if (root -> alloc & 0x02) pfree (root -> yp);
    if (root -> alloc & 0x04 && root -> problem) free_prob (root -> problem);

    if (root -> alloc & 0x08 && root -> sub_prob)
      {
       for (i = 0; i < rank; i++)
	      if (root -> sub_prob[i]) free_prob (root -> sub_prob[i]);

       pfree (root -> sub_prob);
      }

    pfree (root -> scratch);

    root -> mode = NONE;

    if (root -> filename) fclose (root -> savefile);
    
	if (root -> alloc & 0x10) pfree (root -> filename);
	if (root -> alloc & 0x20) pfree (root -> pertfile);
   }
}					/* daFree */


/* free_prob; free database space */
PRIV void
free_prob (DATABASE *prob)
{
 pfree (prob -> pchar);
 pfree (prob -> pshort);
 pfree (prob -> pint);
 pfree (prob -> plong);
 pfree (prob -> pfloat);
 pfree (prob -> pdouble);

 free (prob);
}					/* free_prob */


/* tused; return cpu-time in seconds */
REAL
tused (void)
{
#ifdef CRAY
 return (1.e-6 * (REAL)(clock ()));

#elif defined(UNIX)
 struct rusage ru;

 if (getrusage (RUSAGE_SELF, &ru) < 0)
   {
    fprintf (stderr, "***ERROR tused: Fail calling getrusage\n");
    exit (1);
   }

 return (ru.ru_utime.tv_sec + ru.ru_stime.tv_sec +
	 (ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) * 1.e-6);
#else
/* return (REAL)(clock () / CLK_TCK); */
 return (REAL)(clock () / CLOCKS_PER_SEC);
#endif /* CRAY */
}					/* tused */


/* ******* STRING functions to load inputfile start here ******* */

/* arDate; return date of day */
char *
arDate (char *word)
{
 struct tm *localtime(), *t;
 time_t time(), tloc;

#if ALFALEN < 10
   #error "***ERROR arDate: ALFALEN has to be >= 10\n"
#endif /* ALFALEN */

 tloc = time (&tloc);
 t = localtime (&tloc);
 sprintf (word, "%02d-%02d-%04d", t -> tm_mday, t -> tm_mon + 1, t -> tm_year + 1900);

#if ALFALEN > 10
 word[10] = EOS;
#endif /* ALFALEN */

 return (word);
}					/* arDate */


/* arTime; return time of day */
char *
arTime (char *word)
{
 struct tm *localtime(), *t;
 time_t time(), tloc;

#if ALFALEN < 10
   #error "***ERROR arTime: ALFALEN has to be >= 10\n"
#endif /* ALFALEN */

 tloc = time (&tloc);
 t = localtime (&tloc);
 sprintf (word, " %2d:%02d:%02d ", t -> tm_hour, t -> tm_min, t -> tm_sec);

#if ALFALEN > 10
 word[10] = EOS;
#endif /* ALFALEN */

 return (word);
}					/* arTime */


/* arAlimit; limit x between low an up, real version */
REAL
arAlimit (REAL x, REAL low, REAL up, SET *status)
{
 if (up < x)
   {
    *status = STAT_UP_RANGE;
    return (up);
   }

 if (x < low)
   {
    *status = STAT_LOW_RANGE;
    return (low);
   }

 return (x);
}					/* arAlimit */


/* arIlimit; limit x between low an up, integer version */
int
arIlimit (int x, int low, int up, SET *status)
{
 if (up < x)
   {
    *status = STAT_UP_RANGE;
    return (up);
   }

 if (x < low)
   {
    *status = STAT_LOW_RANGE;
    return (low);
   }

 return (x);
}					/* arIlimit */


/*
 * arEmpty_str; return true if string is empty after position i, i is
 * incremented.
 */
char
arEmpty_str (char *s, FAST int *i)
{
 for (; *i < STRLEN && s[*i] != EOS && isspace (s[*i]); (*i)++);

 return (*i == STRLEN || s[*i] == EOL || s[*i] == EOS);
}					/* arEmpty_str */


/* arStr_save; return a pointer to a string holding a copy of s */
char *
arStr_save (char *s)
{
 char *p =(char *)must_malloc ((unsigned)strlen (s) + 1, "str_save");

 return strcpy (p, s);
}					/* arStr_save */


/* arIs_number; true if at s[i] next nonblank is a number */
char
arIs_number (char *s, FAST int i)
{
 for (; i < STRLEN && IS_SIGNAL (s[i]); i++);

 return (isdigit (s[i]) || (s[i] == '.' && isdigit (s[i+1])));
}					/* arIs_number */


/* arIs_integer; true if at s[i] next nonblank is an integer */
char
arIs_integer (char *s, FAST int i)
{
 for (; i < STRLEN && IS_SIGNAL (s[i]); i++);

 return (isdigit (s[i]) != 0);
}					/* arIs_integer */


/*
 * arIs_symbol; true and increment i if at s[i] next nonblank is a symbol 'sym'
 */
char
arIs_symbol (char *s, FAST int *i, char sym)
{
 if (!arEmpty_str (s, i) && s[*i] == sym)
   {
    (*i)++;
    return TRUE;
   }

 return FALSE;
}


/* arCtoi; convert string at s[i] to higher integer (long), increment i */
long
arCtoi (char *s, FAST int *i)
{
 long n, sign;

 for (; *i < STRLEN && isspace (s[*i]); (*i)++);

 sign = (s[*i] == '-' ? -1 : 1);

 if (s[*i] == '+' || s[*i] == '-') (*i)++;

 for (; *i < STRLEN && isspace (s[*i]); (*i)++);

 for (n = 0; *i < STRLEN && isdigit (s[*i]); (*i)++) n = 10 * n + s[*i] - '0';

 return (sign * n);
}					/* arCtoi */


/* arCtof; convert string at s[i] to higher real (double), increment i */
double
arCtof (char *s, FAST int *i)
{
 double n, mant, sign;

 for (; *i < STRLEN && isspace (s[*i]); (*i)++);

 sign = (s[*i] == '-' ? -1 : 1);

 if (s[*i] == '+' || s[*i] == '-') (*i)++;

 for (; *i < STRLEN && isspace (s[*i]); (*i)++);

 for (n = 0; *i < STRLEN && isdigit (s[*i]); (*i)++) n = 10 * n + s[*i] - '0';

 if (s[*i] == '.') (*i)++;

 for (mant = 1; *i < STRLEN && isdigit (s[*i]); (*i)++)
		{
		 n = 10 * n + s[*i] - '0';
		 mant = mant * 10;
		}
 n = sign * n / mant;			/* mantissa is done */

 if (s[*i] == 'e' || s[*i] == 'E')
   {
    (*i)++;
    sign = (s[*i] == '-' ? -1 : 1);

    if (s[*i] == '+' || s[*i] == '-') (*i)++;

    for (mant = 0; *i < STRLEN && isdigit (s[*i]); (*i)++)
       mant = 10 * mant + s[*i] - '0';

    n *= exp (sign * mant * log ((double) 10.));
   }

 return (n);
}					/* arCtof */


/*
 * arGet_word; get one word from string at s[i], increment i, return first
 * character, and skip invalid token.
 */
char
arGet_word (char *s, FAST int *i, char *p)
{
 FAST int j = 0;
 PRIV int am1 = ALFALEN - 1;
 char c;

 for (; s[*i] != EOS && isspace (s[*i]); (*i)++);

 if (IS_QUOTE (s[*i]))			/* skip quote marks */
   {
    (*i)++;
    for (; s[*i] != EOS && isspace (s[*i]); (*i)++);
   }

 c = s[*i];
 if (isgraph (c))
   for (; j < am1 && IS_WORD (c); j++)
      {
       p[j] = c;
       c = s[++(*i)];
      }

 if (IS_QUOTE (c)) (*i)++;		/* skip quote marks */

 if (j == am1)
   {
    fprintf (stderr, "***ERROR arGet_word: Exceeded maximum string length (%d) in \"%s\"\n",
	     am1, s);
    j--;
   }

 p[j] = EOS;

 if (!isgraph (*p))			/* skip token */
   {
    while (*i < STRLEN && s[*i] != EOS && !isspace (s[*i])) (*i)++;

    return EOS;
   }

 return (*p);
}					/* arGet_word */


/*
 * arGet_char; get either a one-byte integer or one bounded string (by
 * quote-marks or spaces) from the string at s[i], increment i, return
 * first character, and skip invalid token.
 */
char
arGet_char (char *s, FAST int *i, char *p)
{
 PRIV int am1 = ALFALEN - 1;
 int j = 0;
 char c, quote = FALSE;

 for (; s[*i] != EOS && isspace (s[*i]); (*i)++);

 if (IS_QUOTE (s[*i]))			/* skip quote marks */
   {
    (*i)++;
    for (; s[*i] != EOS && isspace (s[*i]); (*i)++);
    quote = TRUE;
   }

 c = s[*i];
 if (isgraph (c))
   for (; j < am1 && (isalnum (c) || (ispunct (c) && !IS_QUOTE (c))
	  || (quote && isspace (c))); j++)
      {
       p[j] = c;
       c = s[++(*i)];
      }

 if (IS_QUOTE (c)) (*i)++;		/* skip quote marks */

 if (j == am1)
   {
    fprintf (stderr, "***ERROR arGet_char: Exceeded maximum string length (%d) in \"%s\"\n",
	     am1, s);
    j--;
   }

 p[j] = EOS;

 if (!isgraph (*p))			/* skip token */
   {
    p[1] = EOL;
    while (*i < STRLEN && s[*i] != EOS && !isspace (s[*i])) (*i)++;

    return EOS;
   }

 j = 0;
 if (!quote && arIs_integer (p, j))
   {
    *p = (char)arCtoi (p, &j);
    p[1] = EOS;
   }

 return (*p);
}					/* arGet_char */


/* arGet_sel; select one character in s by stdin */
char
arGet_sel (char *s, int nel)
{
 FAST int i, c;

 do
   {
    c = getchar (); c = tolower (c);

    for (i = 0; i < nel; i++) if (c == tolower (s[i])) break;
   } while (i == nel);

 if (isupper (s[i])) return ((char) toupper (c));

 return ((char)c);
}					/* arGet_sel */


/*
 * get_text; get one command string from inputfile and put it on string s
 * beggining at s[i], and returns the stopping character as integer.
 */
PRIV int
get_text (FILE *fp, char *s, int *i)
{
 FAST char *p = &s[*i];
 FAST int c, j = *i;
 PRIV int len2 = STRLEN - 2;

 for (; j < len2 && (c = getc (fp), c) && !IS_SPECIAL (c); j++) *p++ = c;

 if (j == len2 && (c = getc (fp), c) && !IS_SPECIAL (c))
   {
    s[STRLEN/2] = EOS;
    fprintf (stderr, "***ERROR get_text: Line \"%s ...\" greater than %d characters\n",
	     s, len2);
    exit (1);
   }

 if (c == EOL || c == COMMENT)
   {
    int k;

    if (c == COMMENT) READLN (fp, k);

    k = *i;
    s[j] = EOL;
    s[j+1] = EOS;

    if (arEmpty_str (s, &k)) return get_text (fp, s, i);

    *i = j;
   }
 else
   {
    *i = j;

    if (c == CONT)
      {
       READLN (fp, j);
       return get_text (fp, s, i);
      }
   }

 return c;
}					/* get_text */


/*
 * arGet_card; read a statment from file, and returns TRUE if we are not at EOF
 */
char
arGet_card (FILE *fp, char *s)
{
 int i = 0, c;

 c = get_text (fp, s, &i);
 s[i++] = EOL;
 s[i] = EOS;

 i = 0;
 if (c != EOF && c != EOL && arEmpty_str (s, &i)) return (arGet_card (fp, s));
 else return (c != EOF);
}					/* arGet_card */


/*
 * arGet_assign; get parameter assignment at s[i], increment i. Assignment is
 * of type: varname = value.
 */
char
arGet_assign (char *s, FAST int *i, char *name, REAL *val)
{
 int j = *i;

 if (arIs_symbol (s, i, '='))
   {
    for (; isspace (s[*i]); (*i)++);	/* skip spaces */

    if (arIs_number (s, *i))
      {
       *val = (REAL)arCtof (s, i);
       return (FALSE);			/* legal assignment */
      }
   }

 fprintf (stderr, "***ERROR arGet_assign: Illegal assignment to \"%s %s\"\n",
	  name, s + j);

 return (TRUE);				/* '=' or value is missing */
}					/* arGet_assign */


/*
 * arGet_list; get a list of different indices at s[i] and put in list vector
 * (rank-size), increment i.  A empty list means all elements in the list,
 * 'rank' returns the number of entries.  Format: first[-last,step]
 */
char
arGet_list (char *s, int *i, int *list, int *rank)
{
 char error = FALSE;
 FAST int index, oldindex = -1, j = 0, k, m;
 int p;

 if (*rank < 1) return TRUE;

 if (arIs_integer (s, *i))
   do
     {
      if (((index = (int)arCtoi (s, i)) < 0 && index > - oldindex) ||
		 ABS(index) >= *rank)
		{
		 fprintf (stderr, "***ERROR arGet_list: Illegal index %d in \"%s\"\n",
			index, s);
		 error = TRUE;
		}
      else
		{
		 p = j;
		 if (index < 0)			/* implicit list */
		 {
	      index = - index;
	      if (arIs_symbol (s, i, SEP2) && arIs_integer (s, *i))
	        k = (int)arCtoi (s, i);
	      else
	      k = 1;

	      for (oldindex += k; oldindex < index; oldindex += k)
	       {
			for (m = 0; m < p; m++) if (list[m] == oldindex) break;
			if (m < p) continue;
			else list[j++] = oldindex;
	       }
		 }
		 else oldindex = index;		/* single element */

		 for (m = 0; m < p; m++) if (list[m] == index) break;
		 if (m == p) list[j++] = index;
		}
     } while (arIs_integer (s, *i));
					/* all elements in the list */
 else for (; j < *rank; j++) list[j] = j;

 *rank = j;				/* number of entries */

 return error;
}					/* arGet_list */


/*
 * arGet_float; get float at s[*i]. Limit float between low and high or
 * return -HUGE if no float is found. If number is out of range or on error,
 * set return status.
 */
REAL
arGet_float (char *s, int *i, REAL low, REAL high, SET *status)
{
 if (!arIs_number (s, *i))
   {
    *status = STAT_NO_VAL;
    return (-HUGE);
   }

 return (arAlimit ((REAL)arCtof (s, i), low, high, status));
}					/* arGet_float */


/*
 * arGet_int; get int at s[*i]. Limit int between low and high or returns -1
 * if no int is found. If number is out of range or on error, set return
 * status.
 */
int
arGet_int (char *s, int *i, int low, int high, SET *status)
{
 if (!arIs_number (s, *i))
   {
    *status = STAT_NO_VAL;
    return (-1);
   }

 return (arIlimit ((int)arCtoi (s, i), low, high, status));
}					/* arGet_int */
