/** 
* @file daspk.h
*
* Contains the function prototype for exposing the Fortran differential 
* equation solver DDASPK3.1 to C/C++.
*/

#ifdef __cplusplus
extern "C" {
#endif 

/**
 * Typedef for residual functions.
 */
typedef void (*residual_function)(double* t, double* y, double* yprime, double* cj, double* delta, int* ires, double* rpar, int* ipar, double* senpar);

/**
 * Typedef for Jacobian functions.
 */
typedef void (*jacobian_function)(double* t, double* y, double* yprime, double* pd, double* cj, double* rpar, int* ipar, double* senpar, int* ijac);

/**
 * Typedef for Psol functions.
 */
typedef void (*psol_function)(int* neq, double* wp, int* iwp, double* b, int* ier, double* rpar, int* ipar);


/**
 * Typedef for G_res functions.
 */
typedef void (*g_res_function)(double* t, double* y, double* yprime, double* pd, double* cj, double* rpar, int* ipar, double* senpar, int* ijac);

/**
 * Exposes the Fortran differential equation solver DASPK to C/C++.
 */
int ddaspk_(
    residual_function res,        /** The residual function that defines the ODE/DAE system */
    int* neq,                           /** The number of equations to be solved: including state variables and sensitivity variables */
    double* t,                          /** The current value of the independent variable */
    double* y,                          /** The current values of the dependent variables: solution (and sensitivity) components at T */
    double* yprime,                     /** The current values of the first derivatives of the dependent variables */
    double* tout,                       /** The value of the independent variable at which a solution is desired */
    int* info,                          /** Parameters controlling how the integration is performed */
    double* rtol,                       /** The relative error tolerance(s), either as a scalar or a vector of length NEQ */
    double* atol,                       /** The absolute error tolerance(s), either as a scalar or a vector of length NEQ */
    int* idid,                          /** Report of the solver actions, used to control subsequent calls */
    double* rwork,                      /** Work space for double-precision values */
    int* lrw,                           /** The length of the double-precision workspace */
    int* iwork,                         /** Work space for integer values */
    int* liw,                           /** The length of the integer workspace */
    double* rpar,                       /** Double-precision parameters to pass to the residual and Jacobian functions */
    int* ipar,                          /** Integer parameters to pass to the residual and Jacobian functions */
    jacobian_function jac,               /** The Jacobian function */
    psol_function psol,                  /** Function for the preconditioner P for linear systems if Krylov method is selected */
    double* senpar,                      /** Vector of sensitivity parameters that appear in res routine */
    g_res_function g_res                /** Optional adifor routine for evaluation of sensitivity equations */
);

#ifdef __cplusplus
}
#endif 
