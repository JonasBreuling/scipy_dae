// main.cpp for Projectile problem

#include "StiffIntegratorT.h"
#include "NonStiffIntegratorT.h"

using std::cout;
using std::endl;

int main ()
{
	// dimension of problem
	const int n = 2;
	// initial values for y
	double y[n] = {2.0, 0.0};
	// initial value for x
	double xbeg = 0.0;
	// final value for x
	const double xend = 11.0;
	// interval of x for printing output
	double dx = 1.0;
	// rtoler and atoler are scalars
	int itoler = 0;
	// relative tolerance
	double *rtoler = new double(1.0e-10);
	// absolute tolerance
	double *atoler = new double(1.0e-10);
	// use SolutionOutput routine
	const int iout = 1;
	// initial step size
	double hinit = 0.0;
	// analytical Jacobian function provided
	const int ijac = 1;
	// number of non-zero rows below main diagonal of Jacobian
	int mljac = n;
	// number of non-zero rows above main diagonal of Jacobian
	int mujac = n;
	// Mass matrix routine is identity
	const int imas = 0;
	int mlmas = 0;
	int mumas = 0;
	
	// Use default values (see header files) for these parameters:
	double hmax(0.0);
	int nmax(0);
	double uround(0.0), safe(0.0), facl(0.0), facr(0.0);
	int nit(0);
	bool startn(false);
	int nind1(0), nind2(0), nind3(0), npred(0), m1(0), m2(0);
	bool hess(false);
	double fnewt(0.0), quot1(0.0), quot2(0.0), thet(0.0);

	StiffIntegratorT stiffT(n, y, xbeg, xend, dx, itoler, rtoler, atoler,
		iout, hinit, hmax, nmax, uround, safe, facl, facr, ijac, mljac,
		mujac, imas, mlmas, mumas, nit, startn, nind1, nind2, nind3, npred,
		m1, m2, hess, fnewt, quot1, quot2, thet);

	cout << "\n\n*******Problem integrated with RADAU5*******\n\n";
	
	stiffT.Integrate();
	
	// print statistics
	cout << "fcn = " << stiffT.NumFunction() <<
		    " jac = " << stiffT.NumJacobian() <<
			" step = " << stiffT.NumStep() <<
			" accpt = " << stiffT.NumAccept() <<
			" rejct = " << stiffT.NumReject() <<
			" dec = " << stiffT.NumDecomp() <<
			" sol = " << stiffT.NumSol() << endl;

	cout << "\n\n*******Problem integrated with DOPRI5*******\n\n";

	double y2[n] = {2.0, 0.0};
	const int iout2 = 2;
	hinit = 0.0;
	double beta = 0.0;
	int nstiff = 0;
	int nrdens = n;
	unsigned *icont = NULL;

	NonStiffIntegratorT nonstiffT(n, y2, xbeg, xend, dx, nrdens, itoler, rtoler,
		atoler, iout2, hinit, hmax, nmax, uround, safe, facl, facr, beta,
		nstiff, icont);

	nonstiffT.Integrate();

	// print statistics
	cout << "fcn = " << nonstiffT.NumFunction() <<
		    " step = " << nonstiffT.NumStep() <<
			" accpt = " << nonstiffT.NumAccept() <<
			" rejct = " << nonstiffT.NumReject() << endl;

	delete rtoler;
	delete atoler;

	return (0);
}

void Function(double x, double *y, double *f)
{
	double prod, eps = 1.0e-6;

	f[0] = y[1];
	prod = 1.0 - y[0]*y[0];
	f[1] = (prod*y[1] - y[0])/eps;

	return;

} // fEval

void Jacobian(double x, double *y, double **J)
{
	double eps = 1.0e-6;

	J[0][0] = 0.0;
	J[0][1] = 1.0;
	J[1][0] = (-2.0*y[0]*y[1] - 1.0)/eps;
	J[1][1] = (1.0 - y[0]*y[0])/eps;

	return;

} // Jacobian

void Mass(double **M)
{

} // Mass
