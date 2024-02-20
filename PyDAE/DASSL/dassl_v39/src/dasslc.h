/*
 * $Log:        dasslc.h,v $
 * Revision 3.8  12/06/04  01:45 {arge}
 * DASSL C version
 *
 * Revision 1.0  92/09/29  11:10 {arge}
 * - extract code from DAWRS package
 * Revision 1.1  96/06/21  01:15 {arge}
 * - set arSet_option as external function
 * Revision 1.2  96/07/24  09:19 {arge}
 * - change type BOOL and SET to signed char
 *   because this is not default for CRAY compilers
 * Revision 1.3  97/01/29  21:25 {arge, fabiola} (based on DASPK)
 * - set daWeights as BOOL
 * Revision 2.0  97/05/22  12:25 {arge, fabiola} (based on DASPK)
 * - added a void 'user' pointer in 'root' structure to user exchange
 *   data between routines like 'residuals', 'jacobian', and 'psolver'
 *   in addition to the 'problem' and 'sub_prob' database structures
 * - parameter convtol added to iter structure
 * - created the iteration matrix types NONE, BAND_MODE, and USER_BAND
 *   and changed the order of the types
 * - current residual vector added in 'root' structure to use in 'jacobian'
 *   and/or 'psolver' routines
 * - 'residuals' routine added to 'jacobian' routine as argument
 * - preconditioner added to dasslc's arguments as an user's routine
 * - created iterative method to solve linear system (using Krylov method)
 * Revision 2.4  99/12/04  22:10 {arge}
 * - move here sparse modes from dasslc.c
 * Revision 2.5 05/07/07  22:10 {arge}
 * - included user supplied linear algebra for iteration matrix (USER_ALGEBRA)
 * - daSetup() now returns BOOL
 * - created the prototypes DASSLC_RES, DASSLC_JAC, and DASSLC_PSOL
 * Revision 3.0  07/07/12  08:30 {arge}
 * - included the differential index vector of the variables in daSetup() arguments to
 *   treat high-index problems, index=NULL can be used for index-0 and index-1 DAE systems
 * - included the differential index vector of the variables in PTR_ROOT
 * - included the differential index of the DAE system in ITER_SET for high-index problems
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
 * Revision 3.8  12/06/04  01:45 {arge, tiago}
 * - compatibility with MacOS.
 */

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

/* #define SPARSE */

#define UNIX
#undef CRAY
#define CRAY


typedef double REAL;            /* real precision for arithmetic */
typedef short SHORT;            /* short definition, general case */
typedef signed char SET;        /* short sets */
/* typedef signed char BOOL; */      /* Boolean type */
typedef int BOOL;

/* analysis type */
#define NONE            (SET)0
#define SETUP           (SET)1
#define STEADY_STATE    (SET)2
#define INITIAL_COND    (SET)3
#define TRANSIENT       (SET)4
#define MAKE_GRAPH      (SET)5

/* conditions */
#define INTERMED        (SET)1          /* succ. step at intermediate point */
#define EXACT           (SET)2          /* successful step at stopping point */
#define INTERPOL        (SET)3          /* succ. step by interpol. at tout */
#define UNKNOWN         (SET)(-1)       /* unknown error. Same as ERROR */
#define MAX_TIMEPNT     (SET)(-2)       /* number of step is too big */
#define INV_TOL         (SET)(-3)       /* error tolerance is too small */
#define INV_WT          (SET)(-4)       /* element of wt is or become <= zero */
#define ERROR_FAIL      (SET)(-5)       /* error test failed repeatedly */
#define CONV_FAIL       (SET)(-6)       /* corrector could not converge */
#define SINGULAR        (SET)(-7)       /* iteration matrix is singular */
#define MAX_ERR_FAIL    (SET)(-8)       /* nonConv. due repeated errTestFails */
#define FAIL            (SET)(-9)       /* nonConv. due rep. err in residuals */
#define CALL_ERROR      (SET)(-10)      /* incorrigible error in residuals */
#define INV_MTYPE       (SET)(-15)      /* invalid matrix type to linear solver */
#define INV_TOUT        (SET)(-33)      /* tin = tout */
#define INV_RUN         (SET)(-34)      /* lastStep was interrupted by an err */
#define INV_RTOL        (SET)(-35)      /* some element of rtol is <= zero */
#define INV_ATOL        (SET)(-36)      /* some element of atol is <= zero */
#define ZERO_TOL        (SET)(-37)      /* all elements of rtol & atol are 0 */
#define TOO_CLOSE       (SET)(-38)      /* tout too close to tin to start */
#define TOUT_BEH_T      (SET)(-39)      /* tout behind tin */
#define TSTOP_BEH_T     (SET)(-40)      /* tstop behind tin */
#define TSTOP_BEH_TOUT  (SET)(-41)      /* tstop behind tout */

#define PRIV static             /* Hidden functions and variables */

#ifndef NOREGI                  /* use of char-short-int-long-pointer regs */
#  define FAST          register
#else
#  define FAST
#endif /* NOREGI */

#ifndef NOREGR                  /* selective use of float-double registers */
#  define FASTER        register
#else
#  define FASTER
#endif /* NOREGR */

#ifndef NOREGS                  /* selective use of struct registers */
#  define FASTEST       register
#else
#  define FASTEST
#endif /* NOREGS */

#ifdef ERROR
#undef ERROR
#endif

#define FALSE           0
#define TRUE            1
#define INITIAL         0       /* first call to DASSL */
#define GLOBAL          1       /* is not first call */
#define ERROR    (SET)(-1)

#define READLN(f,c)     while ((c = getc(f)) != EOL && c != EOF)
#define IS_QUOTE(c)     ((c) == QUOTE0 || (c) == QUOTE1 || (c) == QUOTE2)
#define IS_SIGNAL(c)    ((c) != EOS && \
			(isspace (c) || (c) == '+' || (c) == '-'))
#define IS_SPECIAL(c)   ((c) == EOF || (c) == EOS || (c) == CONT || \
			(c) == SEP1 || (c) == COMMENT || (c) == EOL)
#define IS_WORD(c)      (isalnum (c) || \
			(ispunct (c) && !IS_QUOTE (c) && (c) != '='))

/* data type */
#define SEPARATOR       ':'             /* separator symbol */
#define BANG            '!'             /* NOT logical */
#define RVAR            1               /* REAL variable */
#define SVAR            2               /* SHORT variable */
#define BVAR            3               /* BOOL variable */
#define FUNC            4               /* Function */

/* switch macro */
#define SWITCH(s)       if (!strcmp (name, s))
#define CASE(s)         else if (!strcmp (name, s))
#define DEFAULT         else

/* string sizes */
#define STRLEN          512
#define ALFALEN         80

/* separators */
#define QUOTE0          '`'
#define QUOTE1          '\''
#define QUOTE2          '\"'
#define COMMENT         '#'
#define SEP1            ';'
#define SEP2            ','
#define CONT            '\\'
#define EOS             '\0'
#define EOL             '\n'

/* status */
#define STAT_OK         0
#define STAT_UP_RANGE   1
#define STAT_LOW_RANGE  2
#define STAT_NO_VAL     3
#define STAT_ERR        4

/*
 * type of iteration matrix evaluation. User-defined types must be placed
 * above USER_DENSE, and automatic modes must be placed below DENSE. So,
 * the numbers must be adjusted according to this rule. The number 0 is
 * reserved to no type (NONE).
 */
#define BAND_MODE       1       /* band finite-difference iteration matrix */
#define SPARSE_MODE     2       /* sparse finite-difference iteration matrix */
#define DENSE_MODE      3       /* dense finite-difference iteration matrix */
#define USER_DENSE      4       /* user-defined dense iteration matrix */
#define USER_SPARSE     5       /* user-defined sparse iteration matrix */
#define USER_BAND       6       /* user-defined band iteration matrix */
#define USER_ALGEBRA    7       /* user-given algebra package for iteration matrix */

#define DIR_MODE        0       /* direct method to solve linear system */
#define ITER_MODE       1       /* iterative method to solve linear system */
#define BOTH_MODE       2       /* direct-iterative methods to solve linear system */

/* sparse modes */
#define INFILE_SPARSE   0		  /* user-defined sparse structure */
#define NO_SPARSE			1       /* no sparse mode */
#define EVAL_SPARSE		2       /* dependency matrix structure */

/* iteration matrix status */
#define JAC             (SET)(-1)       /* iteration matrix must be computed */
#define NO_JAC          (SET)1          /* iteration matrix is not current */
#define JAC_DONE        (SET)0          /* iteration matrix was computed */

#define MAX_ORDER       20              /* default = 5 */
#define MAX_LI          20              /* default = MIN(5,rank) */

/* counter-bound factors */
#define INIT_FACTOR     3
#define STEADY_FACTOR   5

/* allocation macros */
#define NEW(a,n,s)      (a *)must_malloc ((unsigned)(n) * sizeof (a), s)
#define RENEW(a,p,n,s)  (a *)must_realloc (p, (unsigned)(n) * sizeof (a), s)

/* other macros */
#define BCOPY(to,from,s) memcpy ((void *)(to), (void *)(from), (int)s)
#define NEXT(a)         ((a) = (a) -> next)

/* math macros */
#define ABS(a)          ((a) < 0 ? -(a) : (a))
#define MAX(a, b)       ((a) > (b) ? (a) : (b))
#define MIN(a, b)       ((a) < (b) ? (a) : (b))
#define SQR(a)          ((a) * (a))
#define SGN(a)          ((a) > 0 ? 1 : ((a) < 0 ? -1 : 0))
#define SGNP(a)         ((a) < 0 ? -1 : 1)

/* machine limits */
#include <limits.h>

#ifdef BITSPERBYTE
#  undef BITSPERBYTE
#endif /* BITSPERBYTE */

#if gcos                                /* according to values.h */
#  define BITSPERBYTE   9
#else
#  define BITSPERBYTE   8
#endif /* gcos */

#ifdef BITS
#  undef BITS
#endif /* BITS */
#define BITS(type)      (BITSPERBYTE * (int)sizeof (type))

#ifdef OLD_VERSION
#  define MAX_TYPE(tp)  ((tp)~((tp)(((tp)1 << BITS(tp)) - 1)))
#  define MAX_SHORT     MAX_TYPE(SHORT)
#  define MAX_long      MAX_TYPE(long)
#  define MAX_int       MAX_TYPE(int)
#else
#  define MAX_SHORT     SHRT_MAX
#  define MAX_long      LONG_MAX
#  define MAX_int       INT_MAX
#endif /* OLD_VERSION */

typedef struct jacobian         /* Structure of iteration matrix */
{
 SET mtype;                     /* type of iteration matrix evaluation */
 char *matrix;                  /* sparse/full matrix of partial derivatives */
 SHORT lband;					/* no. of diagonals below main diagonal */
 SHORT uband;					/* no. of diagonals above main diagonal */
 int *pivot;                    /* array of pivoting index */
 int rank;                      /* dimension of the local vector */
 int *index;                    /* index of the active residuals */
 struct sparse_list *spl;       /* list of data dependency */
} JACOBIAN;                     /* jacobian */

typedef struct krylov           /* Krylov's variables and parameters */
{
 BOOL decomp;					/* TRUE = do iteration matrix fatorization in 'psolver' */
 BOOL restart;                  /* restarting active (TRUE) or unactive (FALSE) */
 SHORT maxl;                    /* max. number of iterations before restart */
 SHORT kmp;                     /* number of orthogonalized vectors, <= maxl */
 SHORT maxrest;                 /* max. number of restarts */
 long ncfl;                     /* number of Krylov-convergence test failures */
 long nli;                      /* number of linear iterations (Krylov) */
 long nps;                      /* number of preconditionnings */
 long perf;                     /* number of poor convergence performances */
 REAL sqrtn;                    /* square root of rank */
 REAL rsqrtn;                   /* reciprocal of sqrtn */
 REAL litol;                    /* linear iterations tolerance */
 REAL lintol;                   /* litol * convtol, tolerance for r - A * z */
 REAL *v;                       /* matrix V(maxl+1,rank) */
 REAL *r;                       /* right hand side vector r[rank] of A * z = r */
 REAL *hes;                     /* Hessenberg matrix HES(maxl,maxl+1) */
 REAL *q;                       /* vector q[2*maxl] of Givens rotations */
 REAL *dl;                      /* scaled precond. residual vector dl[rank] */
 REAL *z;                       /* solution vector z[rank] of A * z = r */
 REAL *yptem;                   /* temporary vector yptem[rank] for yp */
 REAL *res;                     /* current NLS residual vector res[rank] */
} KRYLOV;                       /* krylov */

typedef struct sparse_list      /* Structure of sparse data dependency */
{
 int size;                      /* row size */
 int *index;                    /* column index vector */
} SPARSE_LIST;                  /* sparse */

typedef struct bdf_data         /* Bdf variables and coefficients */
{
 BOOL phase;                    /* set when order is constant, or maximum */
 SHORT orderold;                /* bdf order used on the last step */
 SHORT order;                   /* bdf order to be attempted on the next step */
 SHORT updateorder;             /* order changes */
 REAL ratefactor;               /* convergence rate factor */
 REAL hold;                     /* stepsize on the last successful step */
 REAL h;                        /* initial and current stepsize */
 REAL *psi;                     /* array of past stepsize history */
 REAL *phi;                     /* matrix of divided differences */
 REAL *alpha;                   /* array of leading coefficients */
 REAL *beta;                    /* array of update coefficients */
 REAL *gamma;                   /* array of derivatives coefficients */
 REAL *sigma;                   /* array of leading error coeficients */
 REAL tfar;                     /* current value of independent variable,
				 * will be different from t only when
				 * interpolation has been performed
				 */
} BDF_DATA;                     /* bdf_data */

typedef struct iter_set         /* Iteration sets for global information */
{
 SET linearmode;                /* linear solver mode */
 SET sparsemode;                /* sparse structure mode */
 BOOL factor;                   /* use (1) or not (0) conv. accelerate factor */
 BOOL iststop;                  /* set when stopping point must be checked */
 BOOL istall;                   /* set when intermediate point is required */
 BOOL stol;                     /* set when both error tolerances are scalars */
 BOOL nonneg;                   /* set when the solutions are nonnegative */
 BOOL jac;                      /* control of frequence of iter. matrix eval. */
 SHORT sparsethresh;            /* Threshold for sparse matrix usage */
 SHORT maxconvfail;             /* maximum number of conv. test failures */
 SHORT maxerrorfail;            /* maximum number of error test failures */
 SHORT maxsingular;             /* maximum number of singular iter. matrix */
 SHORT maxnewton;               /* maximum number of newton iteration */
 SHORT maxjacfix;               /* maximum number of iter. to update jacobian */
 SHORT maxlen;                  /* maximum number of time-points */
 SHORT maxorder;                /* maximum bdf order */
 SHORT dae_index;				/* differential index of the DAE system */
 long timepnt;                  /* number of time-points in current interval */
 long tottpnt;                  /* total number of time-points */
 long rejtpnt;                  /* number of rejected time-points */
 long rejnewton;                /* number of rejected N-R iterations */
 long errorfail;                /* number of error test failures */
 long convfail;                 /* number of convergence test failures */
 long reseval;                  /* number of evaluation of residuals */
 long newton;                   /* number of N-R iterations */
 long jaceval;                  /* number of evaluation of iteration matrix */
 REAL cj;                       /* leading coeff. of current iter. matrix */
 REAL cjold;                    /* leading coeff. on the last iter. matrix */
 REAL *wt;                      /* array of error weights */
 REAL *rtol;                    /* array of relative error tolerances */
 REAL *atol;                    /* array of absolute error tolerances */
 REAL hmin;                     /* minimum stepsize */
 REAL hmax;                     /* maximum stepsize */
 REAL jacrate;                  /* threshold for constant iteration matrix */
 REAL nonblank;                 /* factor of non blanks in iteration matrix */
 REAL damps;                    /* S.S. dampping factor for NR iteration */
 REAL dampi;                    /* I.C. dampping factor for NR iteration */
 REAL tstop;                    /* stopping point */
 REAL roundoff;                 /* unit roundoff */
 REAL convtol;                  /* NR convergence tolerance (0,1] in w-norm */
} ITER_SET;                     /* iter_set */

typedef struct timertab         /* CPU-time informatios */
{
 REAL t_save;                   /* Time for saving timepoints */
 REAL t_total;                  /* Total CPU-time */
 REAL t_setup;                  /* Setup time */
 REAL t_steady;                 /* Time to get steady state */
 REAL t_initial;                /* Time to get initial condition */
 REAL t_trans;                  /* Time for transient analysis */
 REAL t_sparse;                 /* Time to build sparse structures */
 REAL t_perturb;                /* Time to evaluate perturbation matrix */
} TIMERTAB;                     /* timertab */

typedef struct debug_set        /* Debuging control parameters */
{
 BOOL nr;                       /* N-R information */
 BOOL bdf;                      /* Print bdf-lists at beginning of window */
 BOOL matrix;                   /* Print new iteration matrix */
 BOOL conv;                     /* Print conv. data after each interval sol. */
 TIMERTAB t_info;               /* Collect CPU-time information */
} DEBUG_SET;                    /* debug_set */

typedef struct ptr_root         /* Root for all global pointers */
{
 FILE *savefile;                /* output file */
 char *filename;                /* output file name */
 char *pertfile;                /* perturbation matrix read/write file name */
 char alloc;                    /* bit set to allocated vars: y,yp,prob,sub,filename,pertfile */
 SET mode;                      /* analysis mode */
 BOOL print;                    /* save and print solution if set */
 int *idxprint;                 /* index of saved and printed variables */
 int rank;                      /* number of dependent variables */
 int *index;					/* differential index of dependent variables */
 REAL t;                        /* independent variable (time) */
 REAL *y;                       /* vector of unknown variables */
 REAL *yp;                      /* vector of time derivatives */
 REAL *res;                     /* current residual vector used in iter. matrix */
 REAL *scratch;                 /* scratch area of size = 3 * rank */
 void *user;                    /* pointer to user exchange data between routines */
 struct database *problem;      /* database problem pointer */
 struct database **sub_prob;    /* array of database sub-problem pointers */
 struct jacobian jac;           /* jacobian matrix structure */
 struct krylov kry;             /* krylov structure */
 struct bdf_data bdf;           /* BDF variables */
 struct iter_set iter;          /* iteration control parameters */
 struct debug_set debug;        /* debugging mode flag */
} PTR_ROOT;                     /* ptr_root */

typedef struct database         /* lowest-level database structure */
{
 int nchar;                     /* number of characters */
 int nshort;
 int nint;
 int nlong;
 int nfloat;
 int ndouble;
 char *pchar;                   /* pointer to characters */
 short *pshort;
 int *pint;
 long *plong;
 float *pfloat;
 double *pdouble;
} DATABASE;                     /* database */

typedef struct bool_var         /* BOOL variable table structure */
{
 char key;
 char *name;
 BOOL deflt, vmin, vmax;
 BOOL *varp;
} BOOLvar;

typedef struct short_var       /* SHORT variable table structure */
{
 char key;
 char *name;
 SHORT deflt, vmin, vmax;
 SHORT *varp;
} SHORTvar;

typedef struct real_var         /* REAL variable table structure */
{
 char key;
 char *name;
 REAL deflt, vmin, vmax;
 REAL *varp;
} REALvar;

typedef struct func_var         /* function table structure */
{
 char key;
 char *name;
 SET (*vfunc)();
} FUNCvar;

#ifdef __cplusplus
extern "C" {
#endif

#ifdef SPARSE
/* some external declaration to use the Sparse package */
extern void spClear (char *);
extern void spDestroy (char *);
extern void spSolve (char *, REAL *, REAL *);
extern char *spCreate (int, int, int *);
extern int spFactor (char *);
extern int spOrderAndFactor (char *, REAL *, REAL, REAL, int);
extern REAL *spGetElement (char *, int, int);

/* the used sparse package starts counting from 1 to rank */
#  define daSparse_clear(m)     spClear (m)
#  define daSparse_value(m,i,j) *(spGetElement (m, i + 1, j + 1))
#  define daSparse_set(m,i,j,v) *(spGetElement (m, i + 1, j + 1)) = v
#  define daSparse_LU(m)        spFactor (m)
#  define daSparse_solve(m,x)   spSolve (m, x, x)
#endif /* SPARSE */

/* types */
typedef BOOL (DASSLC_RES)(PTR_ROOT *, REAL, REAL *, REAL *, REAL *, BOOL *);
typedef BOOL (DASSLC_JAC)(PTR_ROOT *, REAL, REAL *, REAL *, REAL, void *, DASSLC_RES *);
typedef BOOL (DASSLC_PSOL)(PTR_ROOT *, REAL *, DASSLC_RES *);

/* function declarations */
extern BOOL dasslc (SET, PTR_ROOT *, DASSLC_RES *, REAL *, REAL, DASSLC_JAC *, DASSLC_PSOL *);
extern BOOL daIteration_matrix (PTR_ROOT *, DASSLC_RES *, REAL, REAL *, REAL *,
				REAL *, REAL *, DASSLC_JAC *);
extern BOOL daJacobian_matrix (PTR_ROOT *, DASSLC_RES *, REAL, REAL *,
			       REAL *, REAL *, REAL *, DASSLC_JAC *);
extern BOOL daSolve_krylov (PTR_ROOT *, REAL *, BOOL *, DASSLC_RES *, DASSLC_PSOL *);
extern BOOL daWeights (int, BOOL, FAST REAL *, FAST REAL *, FAST REAL *, FAST REAL *);
extern REAL daNorm (int, REAL *, REAL *);
extern REAL daNorm2 (int, REAL *, REAL *, REAL *);
extern REAL tused (void);
extern REAL daRoundoff (void);
extern REAL daRandom (unsigned);
extern BOOL daSetup (char *, PTR_ROOT *, DASSLC_RES *, int, REAL, REAL *, REAL *,
					int *, DATABASE *, DATABASE **);
extern void daStat (FILE *, PTR_ROOT *);
extern void daFree (PTR_ROOT *);
extern void daInterpol (int, BDF_DATA *, REAL, REAL *, REAL *);
extern void daLUsolve (PTR_ROOT *, REAL *);
extern void daSolve (int, int, FAST int *, REAL *, REAL *);
extern void daSolveb (int, int, int, int, FAST int *, REAL *, REAL *);
extern void daPrint_header (FILE *, char *);
extern void daPrint_run (FILE *, PTR_ROOT *);
extern void daPrint_point (FILE *, PTR_ROOT *, REAL);
extern void daPrint_bdf (FILE *, PTR_ROOT *);
extern void daPrint_matrix (FILE *, PTR_ROOT *, REAL);
extern void *must_malloc (unsigned, char *);
extern void *must_realloc (void *, unsigned, char *);
extern int daLU (int, FAST int *, REAL *);
extern int daLUb (int, int, int, int *, REAL *);
extern int daLUback (int, int, REAL *, REAL *);
extern int daLUbackb (int, int, int, int, REAL *, REAL *);

/* string functions */
extern BOOL arSet_option (char *, int, REALvar *, SHORTvar *, BOOLvar *, FUNCvar *);
extern BOOL arSetOption (PTR_ROOT *, char *);
extern char *arDate (char *);
extern char *arTime (char *);
extern char arEmpty_str (char *, FAST int *);
extern char *arStr_save (char *);
extern char arIs_number (char *, FAST int);
extern char arIs_integer (char *, FAST int);
extern char arIs_symbol (char *, FAST int *, char);
extern char arGet_word (char *, FAST int *, char *);
extern char arGet_char (char *, FAST int *, char *);
extern char arGet_sel (char *, int);
extern char arGet_card (FILE *, char *);
extern char arGet_assign (char *, FAST int *, char *, REAL *);
extern char arGet_list (char *, int *, int *, int *);
extern int arIlimit (int, int, int, SET *);
extern int arGet_int (char *, int *, int, int, SET *);
extern long arCtoi (char *, FAST int *);
extern double arCtof (char *, FAST int *);
extern REAL arAlimit (REAL, REAL, REAL, SET *);
extern REAL arGet_float (char *, int *, REAL, REAL, SET *);

/* user provided functions */
#ifndef DASSLC_H
#  define DASSLC_H

BOOL (*ujacFactor) (PTR_ROOT *);
BOOL (*ujacSolve) (PTR_ROOT *, REAL *, REAL *);
void (*ujacPrint) (PTR_ROOT *);
void (*ujacFree) (PTR_ROOT *);
BOOL (*user_init) (PTR_ROOT *);
// extern BOOL (*ujacFactor) (PTR_ROOT *);
// extern BOOL (*ujacSolve) (PTR_ROOT *, REAL *, REAL *);
// extern void (*ujacPrint) (PTR_ROOT *);
// extern void (*ujacFree) (PTR_ROOT *);
// extern BOOL (*user_init) (PTR_ROOT *);
#endif /* DASSLC_H */

#ifdef __cplusplus
}
#endif
