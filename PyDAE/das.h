/** 
* @file ddassl.h
*
* Contains the function prototype for exposing the Fortran differential 
* equation solver DASSL to C/C++.
*/

#ifdef __cplusplus
extern "C" {
#endif 

/**
 * Typedef for residual functions.
 */
typedef void (*residual_function)(double* t, double* y, double* yprime, double* delta, int* ires, double* rpar, int* ipar);

/**
 * Typedef for Jacobian functions.
 */
typedef void (*jacobian_function)(double* t, double* y, double* yprime, double* pd, double* cj, double* rpar, int* ipar);

/**
 * Exposes the Fortran differential equation solver DASSL to C/C++.
 */
int ddassl_(
    residual_function res,              /** The residual function that defines the ODE/DAE system */
    int* neq,                           /** The number of equations to be solved */
    double* t,                          /** The current value of the independent variable */
    double* y,                          /** The current values of the dependent variables */
    double* yprime,                     /** The current values of the first derivatives of the dependent variables */
    double* tout,                       /** The value of the independent variable at which a solution is desired */
    int* info,                          /** Parameters controlling how the integration is performed */
    double* rtol,                       /** The relative error tolerance(s), either as a scalar or a vector */
    double* atol,                       /** The absolute error tolerance(s), either as a scalar or a vector */
    int* idid,                          /** Report of the solver actions, used to control subsequent calls */
    double* rwork,                      /** Work space for double-precision values */
    int* lrw,                           /** The length of the double-precision workspace */
    int* iwork,                         /** Work space for integer values */
    int* liw,                           /** The length of the integer workspace */
    double* rpar,                       /** Double-precision parameters to pass to the residual and Jacobian functions */
    int* ipar,                          /** Integer parameters to pass to the residual and Jacobian functions */
    jacobian_function jac               /** The Jacobian function */
);

#ifdef __cplusplus
}
#endif 
