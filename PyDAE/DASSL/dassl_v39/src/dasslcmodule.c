/*##################################################################
#                                                                  #
#   This is the source code of the python wrapper for the          #
#   "Differential-Algebraic System Solver  in C" by A. R. Secchi   #
#                                                                  #
#   Author: Ataide Neto                                            #
#   email: ataide@peq.coppe.ufrj.br                                #
#   Universidade Federal do Rio de Janeiro                         #
#   Version: 0.1.6                                                 #
#                                                                  #
#   Author: Jonas Harsch                                           #
#   email: harsch@inm.uni-stuttgart.de                             #
#   University of Stuttgart                                        #
#   Version: 0.1.7                                                 #
#                                                                  #
##################################################################*/

/* Change log: 
 *
 * v0.1-0 First working version in python2
 * v0.1-1 Added: python3 compatibility
 * v0.1-2 Added: inputfile and user-defined jacobian support
 * v0.1-3 Added: steady state and sparse algebra support
 * v0.1-4 Added: user-defined jacobian support (sparse)
 * v0.1-5 Fixed: sparse algebra compilation
 * v0.1-6 Added: display optional argument
 * v0.1-7 WIP:   scipy style interface and cleanup
 */

// set this macro before including python header files, see https://docs.python.org/3/c-api/arg.html
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION // disables numpy warnings
#include <numpy/arrayobject.h>
#include "dasslc.h"

// The function's prototypes */
static PyObject* dasslc_solve(PyObject *self, PyObject *args, PyObject *kwargs);

// TODO: rewrite the documentation!
PyDoc_STRVAR(dasslc_solve_doc,
"solve_dae(fun, t_span, y0, y0p=None, rtol=1.0e-8, atol=1e-10, indexList=None, jac=None) -> tuple(t_sol, y_sol, yp_sol)\n"
"\n"
"Solve an implicit dae problem of the form F(t, x, x') = 0.\n"
"\n"
"Parameters\n"
"----------\n"
"res : callable of the type res(t, y, yp), where\n"
"    t is the current integration time,\n"
"    y is the current state vector,\n"
"    yp is the derivative of the current state vector.\n"
"    It should return a vector F(t, x, x') where F(t, x, x').shape[0] == len(x).\n"
"t_span : ndarray\n"
"    Vector of time steps which should be computed and returned with nt = len(t_span).\n"
"y0 : ndarray\n"
"    Initial value of the state vector with ny = y0.shape[0].\n"
"yp0 : ndarray\n"
"    Initial value for the first derivative of the state vector with nyp = yp0.shape[0] == ny.\n"
"rtol : float\n"
"    Relative tolerance with default value is 1.0e-8.\n"
"    The tolerances are used by the code in a local error test at each step which requires roughly\n"
"    that abs(local_error) <= rtol * abs(y) + atol for each vector component. (More specifically, \n"
"    a root-mean-square norm is used to measure the size of vectors, and the error test uses the \n"
"    magnitude of the solution at the beginning of the step.)\n"
"atol : float\n"
"    Absolute tolerance with default value is 1.0e-10.\n"
"    The tolerances are used by the code in a local error test at each step which requires roughly\n"
"    that abs(local_error) <= rtol * abs(y) + atol for each vector component. (More specifically, \n"
"    a root-mean-square norm is used to measure the size of vectors, and the error test uses the \n"
"    magnitude of the solution at the beginning of the step.)\n"
"indexList: ndarray\n"
"    'Differential index of dependent variables'. Depending on the integer number used here the\n"
"    computation of the weighted root-mean-square norm dealing with high-index DAEs 'daNorm2'found\n"
"    in the source code 'dasslc.c' is modified. The weighted norm is computed as\n"
"    `||y||_w = \\sqrt{\\frac{1}{N}} \\sum_{i=1}^N \\left(y_i w_i h_i\\right)` with \n"
"    the weight `w_i` and `h_i = dt^{\\mathrm{max}(0, \\mathrm{idxedList}_i)}`.\n"
"    This ensures convergence for higher indices when the step length decreases, but the constraints\n"
"    won't be fulfilled anymore.\n"
"jac : callable of the type jac(t, y, yp), where\n"
"    t is the current integration time,\n"
"    y is the current state vector,\n"
"    yp is the derivative of the current state vector.\n"
"    It should return a matrix `\\pd{F}{y} + c \\pd{F}{y'}` as a tuple, i.e. 'return pd,'.\n"
"\n"
"Returns\n"
"-------\n"
"x : tuple(t_sol, y_sol, yp_sol) comtaining the dense output at the given time discretization of t_span.\n"
"    t_sol: time steps with length nt,\n"
"    y_sol: solution of the state vector at the time steps t_sol with shape (nt, ny),\n"
"    yp_sol: solution of the first derivative of the state vector at the time steps t_sol with shape (nt, ny).\n"
"\n"
"Notes\n"
"-----\n"
"TODO: Note literatur, e.g. Hairer and other BDF resources. Resources for the weighted norm would be great.\n"
"\n"
"Examples\n"
"--------\n"
"TODO: Write examples here."
);

static PyMethodDef dasslcMethods[] = {
    {"solve_dae", (PyCFunction)dasslc_solve, METH_VARARGS | METH_KEYWORDS, dasslc_solve_doc},
    {NULL, NULL, 0, NULL}
};

static DASSLC_RES residual;   // C residual function
static DASSLC_JAC jacobian;    // C jacobian function
static PyObject *py_fun = NULL; // python residual function
static PyObject *py_jac = NULL; // python jacobian function

// module definition structure PY3
// see https://docs.python.org/3/c-api/module.html#c.PyModuleDef
static struct PyModuleDef dasslcmodule = {
    PyModuleDef_HEAD_INIT,
    "dasslc", // module name
    "Python wrapper for DASSLC (Differential-Algebraic System Solver in C)", // module documentation
    4096, // size of per-interpreter state of the module,
          // or -1 if the module keeps state in global variables
    dasslcMethods // methods
};

// module initialization function PY3
PyMODINIT_FUNC PyInit_dasslc(void){
    PyObject *m = PyModule_Create(&dasslcmodule);
    if (m == NULL)
        return NULL;
    import_array();
    return m;
}

// function declaration
static PyObject* dasslc_solve(PyObject *self, PyObject *args, PyObject *kwargs) {

    // TODO: we want to have an interface that fits to the scipy.integrate.solve_ivp style,
    // see https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
    // scipy.integrate.solve_ivp(fun, t_span, y0, method='RK45', t_eval=None, dense_output=False, 
    //                           events=None, vectorized=False, args=None, **options)

    // parse inputs
    // TODO: t_eval is never used, I think we do all of this in the t_span argument
    PyObject *fun_obj = NULL;
    PyObject *t_span_obj = NULL;
    PyObject *y0_obj = NULL;
    PyObject *t_eval_obj = NULL;
    PyObject *yp0_obj = NULL;
    PyObject *idx_obj = NULL;
    PyObject *jac_obj = NULL;
    double rtol = 1.0e-3;
    double atol = 1.0e-6;
    static char *kwlist[] = {"fun", "t_span", "y0", // mandatory arguments
                             "t_eval", "yp0", "rtol", "atol", "indexList", "jac", NULL}; // optional arguments and NULL termination
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|OOddOO", kwlist, 
                                     &fun_obj, &t_span_obj, &y0_obj, // positional arguments
                                     &t_eval_obj, &yp0_obj, &rtol, &atol, &idx_obj, &jac_obj)) // optional arguments
        return NULL;

    // interpret the input objects as numpy arrays, 
    // see https://docs.scipy.org/doc/numpy/reference/c-api.array.html#from-other-objects
	PyArrayObject* t_span = (PyArrayObject *) PyArray_FROM_OTF(t_span_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (t_span == NULL) {
        PyErr_SetString(PyExc_TypeError, "t_span object can't be interpreted as 1D-ndarray");
        return NULL;
    }

	PyArrayObject* y0 = (PyArrayObject *) PyArray_FROM_OTF(y0_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (y0 == NULL) {
        Py_DECREF(t_span);
        PyErr_SetString(PyExc_TypeError, "y0 object can't be interpreted as 1D-ndarray");
        return NULL;
    }

    // if the yp vector is given interpret it as numpy array
    PyArrayObject *yp0 = NULL;
    if (yp0_obj && yp0_obj != Py_None) {
	    yp0 = (PyArrayObject *) PyArray_FROM_OTF(yp0_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        if (yp0 == NULL) {
            Py_DECREF(t_span);
            Py_DECREF(y0);
            PyErr_SetString(PyExc_TypeError, "yp0 object can't be interpreted as 1D-ndarray");
            return NULL;
        }
    }

    // if the index vector is given interpret it as numpy array
    PyArrayObject *idx = NULL;
    if (idx_obj && idx_obj != Py_None) {
	    idx = (PyArrayObject *) PyArray_FROM_OTF(idx_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
        if (idx == NULL) {
            Py_DECREF(t_span);
            Py_DECREF(y0);
            Py_XDECREF(yp0); // check for NULL when decrementing optional arguments
            PyErr_SetString(PyExc_TypeError, "indexList object can't be interpreted as 1D-ndarray of type NPY_INT32");
            return NULL;
        }
    }

    // get dimensions (number of dense time-steps and number of equations)
    int ntp = PyArray_NDIM(t_span) == 0 ? 1 : (int) PyArray_DIM(t_span, 0);
    int neq = PyArray_NDIM(y0) == 0 ? 1 : (int) PyArray_DIM(y0, 0);

    /***************************************/
    // check if residual function is callable
    /***************************************/
    if (!PyCallable_Check(fun_obj)) {
        Py_DECREF(t_span);
        Py_DECREF(y0);
        Py_XDECREF(yp0); // check for NULL when decrementing optional arguments
        Py_XDECREF(idx); // check for NULL when decrementing optional arguments
        if(PyErr_Occurred() == NULL) {
            PyErr_SetString(PyExc_TypeError, "Provided residual function is not callable.");
        }
        return NULL;
    }
    Py_XINCREF(fun_obj); // add a reference to new residual callback
    Py_XDECREF(py_fun);  // dispose of previous residual callback
    py_fun = fun_obj;    // remember new residual callback

    // dummy call to the residual function
    // note: we have to use the y0_obj here instead of yp0_obj because yp0_obj might be NULL here!
	PyObject *fun_value = PyObject_CallFunction(py_fun, "dOO", *(double*)PyArray_GETPTR1(t_span, 0), y0_obj, y0_obj);
    if (fun_value == NULL) {
        Py_DECREF(t_span);
        Py_DECREF(y0);
        Py_XDECREF(yp0); // check for NULL when decrementing optional arguments
        Py_XDECREF(idx); // check for NULL when decrementing optional arguments
        if(PyErr_Occurred() == NULL) {
            PyErr_SetString(PyExc_TypeError, "The residual function returned NULL but no error was raised.");
        }
        return NULL;
    }

    // PyArrayObject *fun_value_vec = (PyArrayObject*) PyArray_FROMANY(fun_value, NPY_DOUBLE, 0, 1, REQS);
    PyArrayObject *fun_value_vec = (PyArrayObject*) PyArray_FROM_OTF(fun_value, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (fun_value_vec == NULL) {
        Py_DECREF(t_span);
        Py_DECREF(y0);
        Py_XDECREF(yp0); // check for NULL when decrementing optional arguments
        Py_XDECREF(idx); // check for NULL when decrementing optional arguments
        Py_DECREF(fun_value);
        if(PyErr_Occurred() == NULL) {
            PyErr_SetString(PyExc_TypeError, "result of the residual function can't be interpreted as a PyArrayObject.");
        }
        return NULL;
    }

    // number of dimensions of the residual array (also works for the case of a single variable)
    int fun_length = PyArray_NDIM(fun_value_vec) == 0 ? 1 : (int) PyArray_DIM(fun_value_vec, 0);
    if (fun_length != neq) {
        Py_DECREF(t_span);
        Py_DECREF(y0);
        Py_XDECREF(yp0); // check for NULL when decrementing optional arguments
        Py_XDECREF(idx); // check for NULL when decrementing optional arguments
        Py_DECREF(fun_value);
        Py_DECREF(fun_value_vec);

        if(PyErr_Occurred() == NULL) {
            char *errorMessage = (char*)malloc(128 * sizeof(char));
            sprintf(errorMessage, "Residual function must return a vector of length %d, but a vector of length %d was given.", neq, fun_length);
            PyErr_SetString(PyExc_TypeError, errorMessage);
            free(errorMessage);
        }
        return NULL;
    }

    // if everything was fine we decrement the reference counter of the result vector and the result of the residual
    Py_DECREF(fun_value_vec);
    Py_DECREF(fun_value_vec);

    /***************************************/
    // check if jacobian function is callable
    /***************************************/
    if (jac_obj && jac_obj != Py_None) {
        if (!PyCallable_Check(jac_obj)) {
            Py_DECREF(t_span);
            Py_DECREF(y0);
            Py_XDECREF(yp0); // check for NULL when decrementing optional arguments
            Py_XDECREF(idx); // check for NULL when decrementing optional arguments
            if(PyErr_Occurred() == NULL) {
                PyErr_SetString(PyExc_TypeError, "Provided jacobian function is not callable.");
            }
            return NULL;
        }
        Py_XINCREF(jac_obj); // add a reference to new jacobian callback
        Py_XDECREF(py_jac);  // dispose of previous jacobian callback
        py_jac = jac_obj;    // remember new jacobian callback

	    PyObject *jac_value = PyObject_CallFunction(py_jac, "dOOd", 3.1415, y0_obj, y0_obj, 1.4142);
        if (jac_value == NULL) {
            Py_DECREF(t_span);
            Py_DECREF(y0);
            Py_XDECREF(yp0); // check for NULL when decrementing optional arguments
            Py_XDECREF(idx); // check for NULL when decrementing optional arguments
            if(PyErr_Occurred() == NULL) {
                PyErr_SetString(PyExc_TypeError, "The jacobian function returned NULL but no error was raised.");
            }
            return NULL;
        }

        // extract numpy array object from jacobian object
        PyArrayObject *jac = (PyArrayObject *) PyArray_FROM_OTF(jac_value, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        if (jac == NULL) {
            Py_DECREF(t_span);
            Py_DECREF(y0);
            Py_XDECREF(yp0); // check for NULL when decrementing optional arguments
            Py_XDECREF(idx); // check for NULL when decrementing optional arguments
            if(PyErr_Occurred() == NULL) {
                PyErr_SetString(PyExc_TypeError, "First tuple entry of jacobian function's result can't be interpreted as a PyArrayObject.");
            }
            return NULL;
        }

        int jac_rows = PyArray_NDIM(jac) == 0 ? 1 : (int)PyArray_DIM(jac, 0);
        int jac_cols = PyArray_NDIM(jac) < 2  ? 1 : (int)PyArray_DIM(jac, 1);
        if (jac_rows != jac_cols || jac_rows * jac_cols != neq * neq || PyArray_NDIM(jac) == 1) { 
            Py_DECREF(t_span);
            Py_DECREF(y0);
            Py_XDECREF(yp0); // check for NULL when decrementing optional arguments
            Py_XDECREF(idx); // check for NULL when decrementing optional arguments
            Py_DECREF(jac);
            
            if(PyErr_Occurred() == NULL) {
                char *errorMessage = (char*)malloc(128 * sizeof(char));
                sprintf(errorMessage, "Jacobian function must return a %d-by-%d matrix, but %d x %d was given.", neq, neq, jac_rows, jac_cols);
                PyErr_SetString(PyExc_TypeError, errorMessage);
                free(errorMessage);
            }
            return NULL;
        }

        // decrement all ref counter for the jacobian test
        Py_DECREF(jac_value);
        Py_DECREF(jac);
    }

    /***************************************************************/
    // get pointers to the data as C-types for y0, yp0 and indexList.
    /***************************************************************/
    double *y = (double*) malloc(neq * sizeof(double));
	memcpy(y, (double *)PyArray_DATA(y0), neq * sizeof(double));

    int ndr = -1;
    double *yp = NULL;
    if (yp0) {
        ndr = PyArray_NDIM(yp0) == 0 ? 1 : (int)PyArray_DIM(yp0, 0);
        if (ndr >= 0 && ndr != neq) { // throw an exception
            Py_DECREF(t_span);
            Py_DECREF(y0);
            Py_DECREF(yp0);
            Py_XDECREF(idx); // check for NULL when decrementing optional arguments

            if(PyErr_Occurred() == NULL) {
                char *errorMessage = (char*)malloc(128 * sizeof(char));
                sprintf(errorMessage, "yp0 has length %d. but it should have the same length as y0, i.e. %d.", ndr, neq);
                PyErr_SetString(PyExc_TypeError, errorMessage);
                free(errorMessage);
            }
            return NULL;
        }
        yp = (double*) malloc(neq*sizeof(double));
	    memcpy(yp, (double *)PyArray_DATA(yp0), neq * sizeof(double));
    }

    int idxnum = -1;
    int *index = NULL;
    if (idx){
        idxnum = PyArray_NDIM(idx) == 0 ? 1 : (int)PyArray_DIM(idx, 0);
        if (idxnum >= 0 && idxnum != neq) { // throw an exception
            Py_DECREF(t_span);
            Py_DECREF(y0);
            Py_XDECREF(yp0); // check for NULL when decrementing optional arguments
            Py_DECREF(idx);

            if(PyErr_Occurred() == NULL) {
                char *errorMessage = (char*)malloc(128 * sizeof(char));
                sprintf(errorMessage, "indexList has length %d. but it should have the same length as y0, i.e. %d.", idxnum, neq);
                PyErr_SetString(PyExc_TypeError, errorMessage);
                free(errorMessage);
            }
            return NULL;
        }
        index = (int*) malloc(neq * sizeof(int));
	    memcpy(index, (int *)PyArray_DATA(idx), neq * sizeof(int));
        // note: keep this workaround for reference!
        // for (j = 0; j < neq; j++)
        //     index[j] = (int) *(double*)PyArray_GETPTR1(idx,j);
        //     //Workaround: Problem creating an int array directly in python
        //     //            So, create a double array then convert it to int here
    }

    /****************************************/
	// prepare memory for call to DASSL solver
    /****************************************/
    // create the solution vector
    int ntp2 = ntp > 2 ? ntp : 100; // TODO: can we explain this behaviour?
    npy_intp dims1[1] = {ntp2};
    npy_intp dims2[2] = {ntp2, neq};
    
    // TODO: benchmark this: Is empty or zeros faster?
    // TODO: rename these arrays to t, y, yp and rename raw data pointer to t_data, y_data and yp_data!
    PyArrayObject *t_sol = (PyArrayObject*)PyArray_EMPTY(1, dims1, NPY_DOUBLE, 0);
    PyArrayObject *y_sol = (PyArrayObject*)PyArray_EMPTY(2, dims2, NPY_DOUBLE, 0);
    PyArrayObject *yp_sol = (PyArrayObject*)PyArray_EMPTY(2, dims2, NPY_DOUBLE, 0);
    // PyArrayObject *t_sol = (PyArrayObject*) PyArray_ZEROS(1, dims1, NPY_DOUBLE, 0);
    // PyArrayObject *y_sol = (PyArrayObject*) PyArray_ZEROS(2, dims2, NPY_DOUBLE, 0);
    // PyArrayObject *yp_sol = (PyArrayObject*) PyArray_ZEROS(2, dims2, NPY_DOUBLE, 0);

    // do not use input file, see dasslc documentation p. 34 for details
    char *inputfile = "?";
    
    /*
     * call the daSetup function,
     * see dasslc documentation p. 34 for details
     */
    double t0 = ntp == 1 ? 0 : *(double*)PyArray_GETPTR1(t_span, 0);
    PTR_ROOT root;
    BOOL err = daSetup(inputfile, &root, residual, neq, t0, y, yp, index, NULL, NULL);
    if (err) {
        Py_DECREF(t_span);
        Py_DECREF(y0);
        Py_XDECREF(yp0); // check for NULL when decrementing optional arguments
        Py_XDECREF(idx); // check for NULL when decrementing optional arguments
        Py_DECREF(t_sol);
        Py_DECREF(y_sol);
        Py_DECREF(yp_sol);
        
        if(PyErr_Occurred() == NULL) {
            char buff[128] = "Setup error: ";
            sprintf(buff, "%d", err);
            PyErr_SetString(PyExc_RuntimeError, buff);
        }
        return NULL;
    }

    // TODO: set some good parameters here that are not the default DASSL parameters
    // root.iter.maxconvfail = 20;
    // root.iter.maxerrorfail = 20;

    // TODO: we have to enable usage of a sparse jacobian here
    if (jac_obj && jac_obj != Py_None){
        root.jac.mtype = USER_DENSE;
    }
    
    /*************************/
    // configure root structure
    /*************************/
    // we use scalar tolerances, but dassl is able to use a individual tolerance for all components
    root.iter.stol = 1;
    root.iter.atol[0] = atol;
    root.iter.rtol[0] = rtol;

    // find initial derivatives if not given
    // TODO: document this behaviour: it has to do with the possible
    //       steady state solution and a single time step
    // TODO: here we have to use the t_eval instead of t_span
    double tf = 0;
    double dt = 0;
    if (ntp == 1 && t_span_obj != Py_None) {
        dt = (double) *(double*)PyArray_GETPTR1(t_span, 0) / (ntp2 - 1);
        tf = t0 + dt;
    } else if (ntp == 2) {
        dt = (double) (*(double*)PyArray_GETPTR1(t_span, 1) - *(double*)PyArray_GETPTR1(t_span, 0)) / (ntp2 - 1);
        tf = t0 + dt;
    } else if (t_span_obj == Py_None) {
        tf = 1;
    } else {
        tf = *(double*)PyArray_GETPTR1(t_span, 1);
    }

    if (yp == NULL && t_span_obj != Py_None) {
        // call to dasslc C function
        err = dasslc(INITIAL_COND, &root, residual, &t0, tf, py_jac ? jacobian : NULL, NULL);
        if (err < 0) {
            Py_DECREF(t_span);
            Py_DECREF(y0);
            Py_XDECREF(yp0); // check for NULL when decrementing optional arguments
            Py_XDECREF(idx); // check for NULL when decrementing optional arguments
            Py_DECREF(t_sol);
            Py_DECREF(y_sol);
            Py_DECREF(yp_sol);

            if(PyErr_Occurred() == NULL) {
                char buff[128] = "Failed in finding consistent initial condition. Error: ";
                sprintf(buff,"%d",err);
                PyErr_SetString(PyExc_RuntimeError, buff);
            }
            return NULL;
        }
    }

    // update soluton vector
    if (t_span_obj != Py_None) {
        // copy initial t, y0 and yp0
        *(double*)PyArray_GETPTR1(t_sol, 0) = root.t;
	    memcpy((double*)PyArray_GETPTR2(y_sol, 0, 0), root.y, neq * sizeof(REAL));
	    memcpy((double*)PyArray_GETPTR2(yp_sol, 0, 0), root.yp, neq * sizeof(REAL));
        
        // put initial data in first solution slot in order to work on that data in the first time step
	    memcpy((double*)PyArray_GETPTR2(y_sol, 1, 0), root.y, neq * sizeof(REAL));
	    memcpy((double*)PyArray_GETPTR2(yp_sol, 1, 0), root.yp, neq * sizeof(REAL));

        // Call the dasslc function for all tspan
        // TODO: here we have to use the t_eval instead of t_span
        for (int i = 1; i < ntp2; i++) {
            // update root pointers
            root.y = (double*)PyArray_GETPTR2(y_sol, i, 0);
            root.yp = (double*)PyArray_GETPTR2(yp_sol, i, 0);

            tf = ntp > 2 ? *(double*)PyArray_GETPTR1(t_span,i) : t0 + dt;
            err = dasslc(TRANSIENT, &root, residual, &t0, tf, py_jac ? jacobian : NULL, NULL);
            if (err < 0) {
                Py_DECREF(t_span);
                Py_DECREF(y0);
                Py_XDECREF(yp0); // check for NULL when decrementing optional arguments
                Py_XDECREF(idx); // check for NULL when decrementing optional arguments
                Py_DECREF(t_sol);
                Py_DECREF(y_sol);
                Py_DECREF(yp_sol);

                if(PyErr_Occurred() == NULL) {
                    char buff[128] = "Error during integration: ";
                    sprintf(buff,"%d",err);
                    PyErr_SetString(PyExc_RuntimeError, buff);
                }
                return NULL;
            }

            // store current time step t
            *(double*)PyArray_GETPTR1(t_sol, i) = root.t;            
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "STEADY_STATE is implemented but not tested yet. Uncomment these two lines and verify correctness.");
        return NULL;

        // dims1[0] = 1;
        // npy_intp dims0[1] = {neq};
        // t_sol = (PyArrayObject*)PyArray_EMPTY(1, dims1, NPY_DOUBLE, 0);
        // y_sol = (PyArrayObject*)PyArray_EMPTY(1, dims0, NPY_DOUBLE, 0);
        // yp_sol = (PyArrayObject*)PyArray_EMPTY(1, dims0, NPY_DOUBLE, 0);
        // err = dasslc(STEADY_STATE, &root, residual, &t0, tf, py_jac ? jacobian : NULL, NULL);
        // if (err < 0) {
        //     Py_DECREF(t_span);
        //     Py_DECREF(y0);
        //     Py_XDECREF(yp0); // check for NULL when decrementing optional arguments
        //     Py_XDECREF(idx); // check for NULL when decrementing optional arguments
        //     Py_DECREF(t_sol);
        //     Py_DECREF(y_sol);
        //     Py_DECREF(yp_sol);

        //     if(PyErr_Occurred() == NULL) {
        //         char buff[128] = "Error in finding steady state: ";
        //         sprintf(buff,"%d",err);
        //         PyErr_SetString(PyExc_RuntimeError, buff);
        //         return NULL;
        //     }
        // }

        // // store current time step t, y, yp (copies are made!)
        // *(double*)PyArray_GETPTR1(t_sol, 0) = 0;
        // memcpy((double*)PyArray_DATA(y_sol), root.y, neq * sizeof(REAL));
        // memcpy((double*)PyArray_DATA(yp_sol), root.y, neq * sizeof(REAL));
    }

    // clean up
    Py_DECREF(t_span);
    Py_DECREF(y0);
    Py_XDECREF(yp0); // check for NULL when decrementing optional arguments
    Py_XDECREF(idx); // check for NULL when decrementing optional arguments

    // Build the output tuple
    return Py_BuildValue("NNN", t_sol, y_sol, yp_sol);

    // // build dummy output and return a tuple
    // PyObject* dummyString = Py_BuildValue("s", "hello world!");
    // PyObject* dummyFloat = Py_BuildValue("f", 3.1415);
    // PyObject* dummyInt = Py_BuildValue("i", 42);
    // return Py_BuildValue("OOO", dummyString, dummyFloat, dummyInt);
}

static BOOL residual(PTR_ROOT *root, REAL t, REAL *y, REAL *yp, REAL *res, BOOL *jac) {
    // build numpy arrays on y and yp pointers (without copy)
    int neq = root -> rank;
    npy_intp dims[1] = {neq};
	PyObject *y_array = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, y);
	PyObject *yp_array = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, yp);

    // Call the python function and decrement reference counter of y and yp numpy arrays after that (they are not used anymore)
    PyObject *result = PyObject_CallFunction(py_fun, "dOO", t, y_array, yp_array);
    Py_XDECREF(y_array);  // Check for NULL when decrementing 
    Py_XDECREF(yp_array); // check for NULL when decrementing
	if(result == NULL) {
		if(PyErr_Occurred() == NULL) {
			PyErr_SetString(PyExc_RuntimeError, "The RHS function must return a value");
		}
		return -1;
	}
    
    // intepret residual object as numpy array
	PyArrayObject *res_array = (PyArrayObject *) PyArray_FROM_OTF(result, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	if(res_array == NULL) {
		if(PyErr_Occurred() == NULL) {
			PyErr_SetString(PyExc_RuntimeError, "The RHS function must return a ndarray");
		}
		return -1;
	}

    // copy data from residual vector to the C-pointer and decrement unused python objects after that
	memcpy(res, (double *)PyArray_DATA(res_array), neq * sizeof(REAL));
    Py_DECREF(result);
    Py_DECREF(res_array);

    return 0; // ires = 0: no error occurred
}

#define PD(i,j) (*(pd + neq * (i) + j))

static BOOL jacobian(PTR_ROOT *root, REAL t, REAL *y, REAL *yp, REAL cj, void *ja, DASSLC_RES *residual) {
    // build numpy arrays on y and yp pointers (without copy)
    int neq = root -> rank;
    npy_intp dims[1] = {neq};
	PyObject *y_array = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, y);
	PyObject *yp_array = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, yp);

    // Call the python function and decrement reference counter of y and yp numpy arrays after that (they are not used anymore)
    PyObject *pd_obj = PyObject_CallFunction(py_jac, "dOOd", t, y_array, yp_array, cj);
    Py_XDECREF(y_array);  // check for NULL when decrementing
    Py_XDECREF(yp_array); // check for NULL when decrementing
	if(pd_obj == NULL) {
		if(PyErr_Occurred() == NULL) {
			PyErr_SetString(PyExc_RuntimeError, "The jacobian function must return a value");
		}
		return -1;
	}

    // PyArrayObject *pd_array = (PyArrayObject *)PyArray_FROMANY(pd_obj, NPY_DOUBLE, 0, 2, REQS);
    PyArrayObject *pd_array = (PyArrayObject *) PyArray_FROM_OTF(pd_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	if(pd_array == NULL) {
		if(PyErr_Occurred() == NULL) {
			PyErr_SetString(PyExc_RuntimeError, "The jacobian function must return a ndarray");
		}
		return -1;
	}
    
    int ires = 0;

    // copy data from jacobian matrix to the C-pointer and decrement unused python objects after that
    switch (root->jac.mtype) {
        case USER_DENSE: {
            REAL *pd = (REAL *)ja;
            for (int i = 0; i < neq; i++)
                for (int j = 0; j < neq; j++)
                    PD(i, j) = *(double*)PyArray_GETPTR2(pd_array, i, j);
            break;
        }
        
	    default: {
            printf("currently only USER_DENSE jacobians are supported, you should not arrive here...\n"); 
            ires = -1;
            break;
        }
//         case USER_BAND:{
//             REAL *pd = (REAL *)ja;
//             int m = root->jac.lband + root->jac.uband, k = 0;
//             for (i = 0; i < neq; i++)
//                 for (j = 0; j < neq; j++){
//                     k = i - j + m;
//                     PD(k,j) = *(double*)PyArray_GETPTR2(pd_array,i,j);
//                 }
//             break;
//         }
// #ifdef SPARSE
//         case USER_SPARSE:{
//             if (!i_obj || !j_obj){
//                 ires = 1;
//                 break;
//             }
//             i_list = (PyArrayObject*)PyArray_FROMANY(i_obj, NPY_DOUBLE, 0, 0, REQS);
//             j_list = (PyArrayObject*)PyArray_FROMANY(j_obj, NPY_DOUBLE, 0, 0, REQS);
//             int iN = (int)PyArray_DIM(i_list, 0);
//             int jN = (int)PyArray_DIM(j_list, 0);
//             if (iN != jN){
//                 ires = 1;
//                 Py_XDECREF(i_list);
//                 Py_XDECREF(j_list);
//                 break;
//             }
//             char *pd = (char*)ja;
//             for (k = 0; k < iN; k++){
//                 i = (int) *(double*)PyArray_GETPTR1(i_list,k);
//                 j = (int) *(double*)PyArray_GETPTR1(j_list,k);
//                 daSparse_value(pd,i,j) = *(double*)PyArray_GETPTR2(pd_array,i,j);
//             }
//             Py_XDECREF(i_list);
//             Py_XDECREF(j_list);
//             break;
//         }
// #endif
    }

    // clean up
    Py_DECREF(pd_obj);
    Py_DECREF(pd_array);

    return ires;
}
