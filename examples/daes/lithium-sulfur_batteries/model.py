import numpy as np

"""
References:
-----------
https://pubs.rsc.org/en/content/articlelanding/2016/cp/c5cp05755h#!divAbstract
"""
def gen_F(params, I):
    def F(t, y, yp):
        # Define useful local variables
        a = (params.R * params.T) / (4 * params.F)
        b = 1 / (2 * a)
        c = params.kp / (params.rhoS * params.vol)
        d = (2 * params.mm) / (4 * params.F)
        d2 = 2 * d
        d4 = 4 * d

        cm = -c
        m2a = -2 * a

        p1 = 1
        n1 = -1

        # Express system of equations as 0 = F(t,y,y'), with y = w, y' = wd
        return np.array([
            -yp[0] - d4 * y[5] - params.ks * y[0],
            -yp[1] + d4 * y[5] + params.ks * y[0] - d2 * y[6],
            -yp[2] + d * y[6],
            -yp[3] + d * y[6] - c * y[4] * (y[3] - params.Ksp),
            -yp[4] + c * y[4] * (y[3] - params.Ksp),
            -I + y[5] + y[6],
            y[5] + 2 * params.iHr * np.sinh(y[8] * b),
            y[6] + 2 * params.iLr * np.sinh(y[9] * b),
            -y[8]  + y[7] - y[10],
            -y[9] + y[7] - y[11],
            -y[10] + params.EH0 + a * np.log(0.7296 * y[0] / y[1]**2),
            -y[11] + params.EL0 + a * np.log(0.0665 * y[1] / (y[3]**2 * y[2])),
        ])
    
    return F


        # %% Calculate mass matrix


        # m = e12; % default values: 12x12 matrix of zeroes

        # m(1,1) = n1;
        # m(2,2) = n1;
        # m(3,3) = n1;
        # m(4,4) = n1;
        # m(5,5) = n1;

        # %% Calculate Jacobian matrix

        # k=e12; % default values: 12x12 matrix of zeroes

        # k(1,1) = -params.ks;
        # k(1,6) = -d4;
        # k(2,1) = params.ks;
        # k(2,6) = d4;
        # k(2,7) = -d2;
        # k(3,7) = d;
        # k(4,4) = cm*y(5);
        # k(4,5) = cm*(y(4)-params.Ksp);
        # k(4,7) = d;
        # k(5,4) = c*y(5);
        # k(5,5) = c*(y(4)-params.Ksp);
        # k(6,6) = p1;
        # k(6,7) = p1;
        # k(7,6) = p1;
        # k(7,9) = 2*params.iHr*cosh(y(9) * b)*b;
        # k(8,7) = p1;
        # k(8,10) = 2*params.iLr*cosh(y(10) * b)*b;
        # k(9,8) = p1;
        # k(9,9) = n1;
        # k(9,11) = n1;
        # k(10,8) = p1;
        # k(10,10) = n1;
        # k(10,12) = n1;
        # k(11,1) = a./w(1);
        # k(11,2) = m2a./w(2);
        # k(11,11) = n1;
        # k(12,2) = a./w(2);
        # k(12,3) = -a./w(3);
        # k(12,4) = m2a./w(4);
        # k(12,12) = n1;
