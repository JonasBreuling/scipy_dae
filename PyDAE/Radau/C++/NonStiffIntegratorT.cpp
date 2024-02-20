/***************************************************************************
                          NonStiffIntegratorT.cpp
                             -------------------
    written by:          : Blake Ashby
    last modified        : Nov 15, 2002
    email                : bmashby@stanford.edu
 ***************************************************************************/

#include "NonStiffIntegratorT.h"
#include <cstring>

using std::cout;
using std::endl;
using std::setw;
using std::setprecision;
using std::memcpy;

// Constructors

NonStiffIntegratorT::NonStiffIntegratorT(const int nin, double yin[], double xin,
	double xendin, double dxin, int nrdensin, int itolerin, double *rtolerin,
	double *atolerin, const int ioutin, double hin, double hmaxin, int nmaxin,
	double uroundin, double safein, double faclin, double facrin, double betain,
	int nstiffin, unsigned *icontin) :
	IntegratorT(nin, yin, xin, xendin, dxin, itolerin, rtolerin, atolerin,
		ioutin, hin, hmaxin, nmaxin, uroundin, safein, faclin, facrin),
	nrdens(nrdensin), beta(betain), meth(1), nstiff(nstiffin), icont(icontin)
{
	// iout, switch for calling SolutionOutput
	if ((iout < 0) || (iout > 2)) {
		cout << "Wrong input, iout = " << iout << endl;
		throw -1;
	}
	
	// facl, facr, parameters for step size selection
	if (facl == 0.0) facl = 0.2;
	if (facr == 0.0) facr = 10.0;

	// beta for step control stabilization
	if (beta == 0.0)
		beta = 0.04;
	else if (beta < 0.0)
		beta = 0.0;
	else if (beta > 0.2) {
		cout << "Curious input for beta : beta = " << beta << endl;
		throw -1;
	}
	
	// nstiff, parameter for stiffness detection
	if (nstiff == 0) nstiff = 1000;
	else if (nstiff < 0) nstiff = nmax + 10;

	indir = NULL;
	rcont1 = rcont2 = rcont3 = rcont4 = rcont5 = NULL;
	
	// nrdens, number of dense output components
	if (nrdens > n) {
		cout << "Curious input, nrdens = " << nrdens << endl;
		throw -1;
	}
	else if (nrdens != 0) {
		rcont1 = new double[nrdens];
		rcont2 = new double[nrdens];
		rcont3 = new double[nrdens];
		rcont4 = new double[nrdens];
		rcont5 = new double[nrdens];
		if (nrdens < n) indir = new int[n];

		// control of length of icont
		if (nrdens == n) {
			if (icont) cout << "Warning : when nrdens = n there is no need " <<
				"allocating memory for icont\r\n";
		}
		else {
			if (iout < 2) cout << "Warning : put iout = 2 for dense output\r\n";
			for (int i = 0; i < n; i++) indir[i] = UINT_MAX;
			for (int i = 0; i < nrdens; i++) indir[icont[i]] = i;
		}
	}

	f = new double[n];
	yy1 = new double[n];
	k1 = new double[n];
	k2 = new double[n];
	k3 = new double[n];
	k4 = new double[n];
	k5 = new double[n];
	k6 = new double[n];
	ysti = new double[n];
	
}  // Constructor

NonStiffIntegratorT::~NonStiffIntegratorT()
{
	if (icont) delete [] icont;
	if (indir) delete [] indir;
	delete [] f;
	delete [] yy1;
	delete [] k1;
	delete [] k2;
	delete [] k3;
	delete [] k4;
	delete [] k5;
	delete [] k6;
	delete [] ysti;
	if (rcont1) {
		delete [] rcont1;
		delete [] rcont2;
		delete [] rcont3;
		delete [] rcont4;
		delete [] rcont5;
	}
}

void NonStiffIntegratorT::Integrate()
{

	int idid = CoreIntegrator();

	if (idid < 0) {
		cout << " Computation failed " << endl;
		return;
	}

	// print final solution
	if (iout != 0) {
		cout << "Step " << naccpt << ": t = " << setw(5) <<
				setprecision(2) << xend << "  y = ";
		for (int i = 0; i < n; i++)
			cout << setw(10) << setprecision(8) << y[i] << "  ";
		cout << endl;
	}

	return;

} // Integrate


double sign(double a, double b)
{
	return (b > 0.0) ? fabs(a) : -fabs(a);
} // sign

// calculates initial value for step length, h
double NonStiffIntegratorT::hinit()
{
	int iord = 5;
	double posneg = sign(1.0, xend-x);
	double sk, sqr;
	double dnf = 0.0;
	double dny = 0.0;
	double atoli = atoler[0];
	double rtoli = rtoler[0];

	if (itoler == 0)
		for (int i = 0; i < n; i++) {
			sk = atoli + rtoli * fabs(y[i]);
			sqr = k1[i]/sk;
			dnf += sqr*sqr;
			sqr = y[i]/sk;
			dny += sqr*sqr;
		}
	else
		for (int i = 0; i < n; i++) {
			sk = atoler[i] + rtoler[i] * fabs(y[i]);
			sqr = k1[i]/sk;
			dnf += sqr*sqr;
			sqr = y[i]/sk;
			dny += sqr*sqr;
		}

	if ((dnf <= 1.0e-10) || (dny <= 1.0e-10)) h = 1.0e-6;
	else h = sqrt(dny/dnf)*0.01;

	h = min(h, hmax);
	h = sign(h, posneg);

	// perform an explicit Euler step
	for (int i = 0; i < n; i++) k3[i] = y[i] + h * k1[i];

	Function(x+h, k3, k2);

	// estimate the second derivative of the solution
	double der2 = 0.0;
	if (itoler == 0)
		for (int i = 0; i < n; i++) {
			sk = atoli + rtoli * fabs(y[i]);
			sqr = (k2[i] - k1[i])/sk;
			der2 += sqr*sqr;
		}
	else
		for (int i = 0; i < n; i++) {
			sk = atoler[i] + rtoler[i] * fabs(y[i]);
			sqr = (k2[i] - k1[i])/sk;
			der2 += sqr*sqr;
		}

	der2 = sqrt(der2)/h;

	// step size is computed such that
	// h**iord * max(norm(k1), norm(der2)) = 0.01
	double der12 = max(fabs(der2), sqrt(dnf));

	double h1;
	if (der12 <= 1.0e-15) h1 = max(1.0e-6, fabs(h)*1.0e-3);
	else h1 = pow(0.01/der12, 1.0/(double)iord);

	h = min(100.0*h, min(h1, hmax));

	return sign(h, posneg);

} // hinit


// core integrator

// return value for dopcor:
//	 1 : computation successful,
//	 2 : computation successful interrupted by SolutionOutput,
//	-1 : input is not consistent,
//	-2 : larger nmax is needed,
//	-3 : step size becomes too small,
//	-4 : the problem is probably stff (interrupted).

int NonStiffIntegratorT::CoreIntegrator()
{
	double c2, c3, c4, c5, e1, e3, e4, e5, e6, e7, d1, d3, d4, d5, d6, d7;
	double a21, a31, a32, a41, a42, a43, a51, a52, a53, a54;
	double a61, a62, a63, a64, a65, a71, a73, a74, a75, a76;

	// initialisations
	switch (meth)
	{
		case 1:

			c2=0.2, c3=0.3, c4=0.8, c5=8.0/9.0;
			a21=0.2, a31=3.0/40.0, a32=9.0/40.0;
			a41=44.0/45.0, a42=-56.0/15.0; a43=32.0/9.0;
			a51=19372.0/6561.0, a52=-25360.0/2187.0;
			a53=64448.0/6561.0, a54=-212.0/729.0;
			a61=9017.0/3168.0, a62=-355.0/33.0, a63=46732.0/5247.0;
			a64=49.0/176.0, a65=-5103.0/18656.0;
			a71=35.0/384.0, a73=500.0/1113.0, a74=125.0/192.0;
			a75=-2187.0/6784.0, a76=11.0/84.0;
			e1=71.0/57600.0, e3=-71.0/16695.0, e4=71.0/1920.0;
			e5=-17253.0/339200.0, e6=22.0/525.0, e7=-1.0/40.0;
			d1=-12715105075.0/11282082432.0, d3=87487479700.0/32700410799.0;
			d4=-10690763975.0/1880347072.0, d5=701980252875.0/199316789632.0;
			d6=-1453857185.0/822651844.0, d7=69997945.0/29380423.0;

			break;
	}

	double posneg = sign(1.0, xend-x);
	double facold = 1.0e-4;
	double expo1 = 0.2 - beta*0.75;
	double facc1 = 1.0/facl;
	double facc2 = 1.0/facr;

	// initial preparations
	double atoli = atoler[0];
	double rtoli = rtoler[0];

	bool last = false;
	double hlamb = 0.0;
	int iasti = 0;

	Function(x, y, k1);

	hmax = fabs(hmax);
	if (h == 0.0) h = hinit();

	nfcn += 2;
	bool reject = false;
	int nonsti;

	if (iout != 0) {
		hold = h;
		int irtrn = SolutionOutput();
		if (irtrn < 0) {
			cout << "Exit of dopri5 at x = " << x << endl;
			return 2;
		}
	}

	// basic integration step
	while (true) {
		if (nstep > nmax) {
			cout << "Exit of dopri5 at x = " << x << ", more than nmax = "
				<< nmax << " are needed." << endl;
			hold = h;
			return -2;
		}
		if (0.1*fabs(h) <= fabs(x)*uround) {
			cout << "Exit of dopri5 at x = " << x << ", step size too small h = "
				<< h << endl;
			hold = h;
			return -3;
		}
		if ((x + 1.01*h - xend)*posneg > 0.0) {
			h = xend - x;
			last = true;
		}
		nstep++;

		// the first 6 stages
		for (int i = 0; i < n; i++)
			yy1[i] = y[i] + h*a21*k1[i];

		Function(x+c2*h, yy1, k2);

		for (int i = 0; i < n; i++)
			yy1[i] = y[i] + h*(a31*k1[i] + a32*k2[i]);

		Function(x+c3*h, yy1, k3);

		for (int i = 0; i < n; i++)
			yy1[i] = y[i] + h*(a41*k1[i] + a42*k2[i] + a43*k3[i]);

		Function(x+c4*h, yy1, k4);

		for (int i = 0; i <n; i++)
			yy1[i] = y[i] + h*(a51*k1[i] + a52*k2[i] + a53*k3[i] + a54*k4[i]);

		Function(x+c5*h, yy1, k5);

		for (int i = 0; i < n; i++)
			ysti[i] = y[i] + h*(a61*k1[i] + a62*k2[i] + a63*k3[i] +
				a64*k4[i] + a65*k5[i]);

		Function(x+h, ysti, k6);

		for (int i = 0; i < n; i++)
			yy1[i] = y[i] + h*(a71*k1[i] + a73*k3[i] + a74*k4[i] +
				a75*k5[i] + a76*k6[i]);

		Function(x+h, yy1, k2);

		if (iout == 2) {
			if (nrdens == n) {
				for (int i = 0; i < n; i++) {
					rcont5[i] = h*(d1*k1[i] + d3*k3[i] + d4*k4[i] +
						d5*k5[i] + d6*k6[i] + d7*k2[i]);
				}
			}
			else {
				for (int j = 0; j < nrdens; j++) {
					unsigned i = icont[j];
					rcont5[j] = h*(d1*k1[i] + d3*k3[i] + d4*k4[i] +
						d5*k5[i] + d6*k6[i] + d7*k2[i]);
				}
			}
		}

		for (int i = 0; i < n; i++)
			k4[i] = h*(e1*k1[i] + e3*k3[i] + e4*k4[i] + e5*k5[i] +
				e6*k6[i] + e7*k2[i]);
		nfcn += 6;

		// error estimation
		double err = 0.0, sk, sqr;
		if (!itoler)
			for (int i = 0; i < n; i++) {
				sk = atoli + rtoli*max(fabs(y[i]), fabs(yy1[i]));
				sqr = k4[i]/sk;
				err += sqr*sqr;
			}
		else
			for (int i = 0; i < n; i++) {
				sk = atoler[i] + rtoler[i]*max(fabs(y[i]), fabs(yy1[i]));
				sqr = k4[i]/sk;
				err += sqr*sqr;
			}

		err = sqrt(err/(double)n);

		// computation of hnew
		double fac11 = pow(err, expo1);
		// Lund-stabilization
		double fac = fac11/pow(facold,beta);
		// we require facl <= hnew/h <= facr
		fac = max(facc2, min(facc1, fac/safe));
		double hnew = h/fac;

		if (err <= 1.0) {
		// step accepted
			facold = max(err, 1.0e-4);
			naccpt++;

			// stiffness detection
			if (!(naccpt % nstiff) || (iasti > 0)) {
				double stnum = 0.0, stden = 0.0;
				for (int i = 0; i < n; i++) {
					sqr = k2[i] - k6[i];
					stnum += sqr*sqr;
					sqr = yy1[i] - ysti[i];
					stden += sqr*sqr;
				}
				if (stden > 0.0) hlamb = h*sqrt(stnum/stden);
				if (hlamb > 3.25) {
					nonsti = 0;
					iasti++;
					if (iasti == 15) {
						cout << "The problem seems to become stiff at x = " << x << endl;
						hold = h;
						return -4;
					}
				}
				else {
					nonsti++;
					if (nonsti == 6) iasti = 0;
				}
			}

			if (iout == 2) {
				double yd0, ydiff, bspl;
				if (nrdens == n)
					for (int i = 0; i < n; i++) {
						yd0 = y[i];
						ydiff = yy1[i] - yd0;
						bspl = h*k1[i] - ydiff;
						rcont1[i] = y[i];
						rcont2[i] = ydiff;
						rcont3[i] = bspl;
						rcont4[i] = -h*k2[i] + ydiff - bspl;
					}
				else
					for (int j = 0; j < nrdens; j++) {
						unsigned i = icont[j];
						yd0 = y[i];
						ydiff = yy1[i] - yd0;
						bspl = h * k1[i] - ydiff;
						rcont1[j] = y[i];
						rcont2[j] = ydiff;
						rcont3[j] = bspl;
						rcont4[j] = -h * k2[i] + ydiff - bspl;
					}
			}

			memcpy(k1, k2, n*sizeof(double));
			memcpy(y, yy1, n*sizeof(double));
			xold = x;
			x += h;

			if (iout) {
				hold = h;
				int irtrn = SolutionOutput();
				if (irtrn < 0) {
					cout << "Exit of dopri5 at x = " << x << endl;
					return 2;
				}
			}

			// normal exit
			if (last) {
				hold = hnew;
				return 1;
			}

			if (fabs(hnew) > hmax) hnew = posneg*hmax;
			if (reject) hnew = posneg*min(fabs(hnew), fabs(h));
			reject = false;
		}
		else {
			// step rejected
			hnew = h/min(facc1, fac11/safe);
			reject = true;
			if (naccpt >= 1) nrejct++;
			last = false;
		}
		h = hnew;
	}

} // CoreIntegrator


// dense output function
double NonStiffIntegratorT::ContinuousOutput(unsigned i)
{
	unsigned ii = UINT_MAX;

	if (!indir) ii = i;
	else ii = indir[i];

	if (ii == UINT_MAX) {
		cout << "No dense output available for %uth component" << i << endl;
		return 0.0;
	}

	double theta = (xd - xold)/hold;
	double theta1 = 1.0 - theta;

	return rcont1[ii] + theta*(rcont2[ii] +
		theta1*(rcont3[ii] + theta*(rcont4[ii] + theta1*rcont5[ii])));

} // contd5




