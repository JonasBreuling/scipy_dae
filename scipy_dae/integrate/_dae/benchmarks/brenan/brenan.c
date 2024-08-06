#include <time.h>
#include <ida/ida.h> /* prototypes for IDA fcts., consts.    */
#include <math.h>
#include <nvector/nvector_serial.h> /* access to serial N_Vector            */
#include <stdio.h>
#include <sundials/sundials_math.h> /* defs. of SUNRabs, SUNRexp, etc.      */
#include <sundials/sundials_types.h> /* defs. of sunrealtype, sunindextype      */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */
#include <sunnonlinsol/sunnonlinsol_newton.h> /* access to Newton SUNNonlinearSolver  */

#if defined(SUNDIALS_EXTENDED_PRECISION)
#define GSYM "Lg"
#define ESYM "Le"
#define FSYM "Lf"
#else
#define GSYM "g"
#define ESYM "e"
#define FSYM "f"
#endif

/* Problem Constants */
#define NEQ  2

/* Prototypes of functions called by IDA */
int res(sunrealtype tres, N_Vector yy, N_Vector yp, N_Vector resval,
        void* user_data);

/* Prototypes of private functions */
static int check_retval(void* returnvalue, const char* funcname, int opt);

/*
 *--------------------------------------------------------------------
 * Main Program
 *--------------------------------------------------------------------
 */

int main(void)
{
  void* mem;
  N_Vector yy, yp, avtol;
  sunrealtype rtol, atol, *yval, *ypval, *atval;
  sunrealtype t0, t1, tout, tret;
  int iout, retval, retvalr;
  SUNMatrix A;
  SUNLinearSolver LS;
  SUNNonlinearSolver NLS;
  SUNContext ctx;
  FILE* FID;

  mem = NULL;
  yy = yp = avtol = NULL;
  yval = ypval = atval = NULL;
  A                    = NULL;
  LS                   = NULL;
  NLS                  = NULL;

  double m_max = 45.0;
  for (double m=0.0; m<m_max; m++) {

    /* Create SUNDIALS context */
    retval = SUNContext_Create(SUN_COMM_NULL, &ctx);
    if (check_retval(&retval, "SUNContext_Create", 1)) { return (1); }

    /* Allocate N-vectors. */
    yy = N_VNew_Serial(NEQ, ctx);
    if (check_retval((void*)yy, "N_VNew_Serial", 0)) { return (1); }
    yp = N_VClone(yy);
    if (check_retval((void*)yp, "N_VNew_Serial", 0)) { return (1); }
    avtol = N_VClone(yy);
    if (check_retval((void*)avtol, "N_VNew_Serial", 0)) { return (1); }

    /* Create and initialize  y, y', and absolute tolerance vectors. */
    yval    = N_VGetArrayPointer(yy);
    yval[0] = SUN_RCONST(1.0);
    yval[1] = SUN_RCONST(0.0);

    ypval    = N_VGetArrayPointer(yp);
    ypval[0] = SUN_RCONST(-1.0);
    ypval[1] = SUN_RCONST(1.0);
    
    /* define tolerances */
    rtol = pow(10, -(1 + m / 4));

    atval    = N_VGetArrayPointer(avtol);
    atval[0] = rtol;
    atval[1] = rtol;
    atval[2] = rtol;

    int mxsteps = 1e8;

    /* Integration limits */
    t0 = SUN_RCONST(0.0);
    t1 = SUN_RCONST(10.0);

    /* Call IDACreate and IDAInit to initialize IDA memory */
    mem = IDACreate(ctx);
    if (check_retval((void*)mem, "IDACreate", 0)) { return (1); }
    retval = IDAInit(mem, res, t0, yy, yp);
    if (check_retval(&retval, "IDAInit", 1)) { return (1); }
  
    /* Call IDASVtolerances to set tolerances */
    retval = IDASVtolerances(mem, rtol, avtol);
    if (check_retval(&retval, "IDASVtolerances", 1)) { return (1); }

    /* Create dense SUNMatrix for use in linear solves */
    A = SUNDenseMatrix(NEQ, NEQ, ctx);
    if (check_retval((void*)A, "SUNDenseMatrix", 0)) { return (1); }

    /* Create dense SUNLinearSolver object */
    LS = SUNLinSol_Dense(yy, A, ctx);
    if (check_retval((void*)LS, "SUNLinSol_Dense", 0)) { return (1); }

    /* Attach the matrix and linear solver */
    retval = IDASetLinearSolver(mem, LS, A);
    if (check_retval(&retval, "IDASetLinearSolver", 1)) { return (1); }

    /* Create Newton SUNNonlinearSolver object. IDA uses a
    * Newton SUNNonlinearSolver by default, so it is unecessary
    * to create it and attach it. It is done in this example code
    * solely for demonstration purposes. */
    NLS = SUNNonlinSol_Newton(yy, ctx);
    if (check_retval((void*)NLS, "SUNNonlinSol_Newton", 0)) { return (1); }

    /* Attach the nonlinear solver */
    retval = IDASetNonlinearSolver(mem, NLS);
    if (check_retval(&retval, "IDASetNonlinearSolver", 1)) { return (1); }

    /* Maximum number of steps */
    IDASetMaxNumSteps(mem, mxsteps);

    /* Call IDASolve */
    clock_t start = clock();
    retval = IDASolve(mem, t1, &tret, yy, yp, IDA_NORMAL);
    clock_t end = clock();
    double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;

    /* Print elapsed time and error to a file in CSV format */
    yval  = N_VGetArrayPointer(yy);

    double diff_y1 = yval[0] - (exp(-t1) + t1 * sin(t1));
    double diff_y2 = yval[1] - sin(t1);
    double error = sqrt(diff_y1 * diff_y1 + diff_y2 * diff_y2);

    FID = fopen("brenan_errors_IDA.csv", "a");
    fprintf(FID, "%17.17e, %17.17e\n", error, elapsed_time);
    fclose(FID);

    /* Print rtol, elapsed time and error to a file in CSV format */
    printf("rtol: %e, elapsed time: %e, error: %e\n", rtol, elapsed_time, error);

  }

  /* Free memory */
  IDAFree(&mem);
  SUNNonlinSolFree(NLS);
  SUNLinSolFree(LS);
  SUNMatDestroy(A);
  N_VDestroy(avtol);
  N_VDestroy(yy);
  N_VDestroy(yp);
  SUNContext_Free(&ctx);

  return (retval);
}

/*
 *--------------------------------------------------------------------
 * Functions called by IDA
 *--------------------------------------------------------------------
 */

/*
 * Define the system residual function.
 */
int res(sunrealtype tres, N_Vector yy, N_Vector yp, N_Vector rr,
           void* user_data)
{
  sunrealtype *yval, *ypval, *rval;

  yval  = N_VGetArrayPointer(yy);
  ypval = N_VGetArrayPointer(yp);
  rval  = N_VGetArrayPointer(rr);

  rval[0] = ypval[0] - tres * ypval[1] + yval[0] - (1 + tres) * yval[1];
  rval[1] = yval[1] - sin(tres);

  return (0);
}

/*
 *--------------------------------------------------------------------
 * Private functions
 *--------------------------------------------------------------------
 */

/*
 * Check function return value...
 *   opt == 0 means SUNDIALS function allocates memory so check if
 *            returned NULL pointer
 *   opt == 1 means SUNDIALS function returns an integer value so check if
 *            retval < 0
 *   opt == 2 means function allocates memory so check if returned
 *            NULL pointer
 */

static int check_retval(void* returnvalue, const char* funcname, int opt)
{
  int* retval;
  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && returnvalue == NULL)
  {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return (1);
  }
  else if (opt == 1)
  {
    /* Check if retval < 0 */
    retval = (int*)returnvalue;
    if (*retval < 0)
    {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with retval = %d\n\n",
              funcname, *retval);
      return (1);
    }
  }
  else if (opt == 2 && returnvalue == NULL)
  {
    /* Check if function returned NULL pointer - no memory allocated */
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return (1);
  }

  return (0);
}
