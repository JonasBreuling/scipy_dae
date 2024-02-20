C***BEGIN PROLOGUE  DHEATILU
C***REFER TO  DDASKR
C***DATE WRITTEN   020813   (YYMMDD)
C***REVISION DATE  
C
C***DESCRIPTON
C
C-----------------------------------------------------------------------
C Example program for DDASKR.
C DAE system derived from the discretized heat equation on a square.
C
C This is the double precision version.
C-----------------------------------------------------------------------
C
C This program solves a DAE system that arises from the heat equation,
C   du/dt = u   + u
C            xx    yy
C posed on the 2-D unit square with zero Dirichlet boundary conditions.
C An M+2 by M+2 mesh is set on the square, with uniform spacing 1/(M+1).
C The spatial deriviatives are represented by standard central finite
C difference approximations.  At each interior point of the mesh,
C the discretized PDE becomes an ODE for the discrete value of u.
C At each point on the boundary, we pose the equation u = 0.  The
C discrete values of u form a vector U, ordered first by x, then by y.
C The result is a DAE system G(t,U,U') = 0 of size NEQ = (M+2)*(M+2).
C
C Initial conditions are posed as u = 16x(1-x)y(1-y) at t = 0.
C The problem is solved by DDASKR on the time interval t .le. 10.24.
C
C The root functions are R1(U) = max(u) - 0.1, R2(U) = max(u) - 0.01.
C
C The Krylov linear system solution method, with preconditioning, is
C selected.  The preconditioner is a sparse matrix with half-bandwidths
C equal to 1, i.e. a tridiagonal matrix.  (The true half-bandwidths
C are equal to M+2.)  This corresponds to ignoring the y-direction
C coupling in the ODEs, for purposes of preconditioning.  The extra
C iterations resulting from this approximation are offset by the lower
C storage and linear system solution costs for a tridiagonal matrix.  
C
C The routines DJACILU and DPSOLILU that generate and solve the sparse
C preconditioner are provided in a separate file for general use.
C
C The output times are t = .01 * 2**n (n = 0,...,10).  The maximum of
C abs(u) over the mesh, and various performance statistics, are printed.
C
C For details and test results on this problem, see the reference:
C   Peter N. Brown, Alan C. Hindmarsh, and Linda R. Petzold,
C   Using Krylov Methods in the Solution of Large-Scale Differential-
C   Algebraic Systems, SIAM J. Sci. Comput., 15 (1994), pp. 1467-1488.
C-----------------------------------------------------------------------
C
C***ROUTINES CALLED
C   UINIT, DDASKR
C
C***END PROLOGUE  DHEATILU
C
C Here are necessary declarations.  The dimension statements use a
C maximum value for the mesh parameter M.
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      EXTERNAL RESH, RTHEAT, DJACILU, DPSOLILU
C Load parameters for sparse preconditioner.
      PARAMETER (LENPFAC = 5)
      PARAMETER (LENPLUFAC = 5)
      PARAMETER (IPREMETH = 1)  ! =1 means ILUT preconditioner used
                                ! =2 means ILUTP preconditioner used
      PARAMETER (LFILILUT = 5)
      PARAMETER (IREORDER = 1)
      PARAMETER (ISRNORM = 1)
      PARAMETER (NORMTYPE = 2)
      PARAMETER (JACOUT = 0)
      PARAMETER (JSCALCOL = 1)
      PARAMETER (TOLILUT = 0.001)
      PARAMETER (PERMTOL = 0.01)
C Load workspace lengths.
      PARAMETER (MAXM = 10, MAXM2 = MAXM+2, MXNEQ = MAXM2*MAXM2)
      PARAMETER (LENWP = 2*LENPFAC*MXNEQ + LENPLUFAC*MXNEQ
     .                   + ISRNORM*MXNEQ + 2*(MXNEQ+1) )
      PARAMETER (LENIWP = 4*(MXNEQ+1) + 3*LENPFAC*MXNEQ
     .                    + 2*LENPLUFAC*MXNEQ + IREORDER*2*MXNEQ
     .                    + (IPREMETH-1)*2*MXNEQ )
      PARAMETER (LENRW = 107 + 18*MXNEQ, LENIW  = 40)
      DIMENSION U(MXNEQ),UPRIME(MXNEQ),
     *          RWORK(LENRW+LENWP),IWORK(LENIW+LENIWP)
      DIMENSION INFO(20), JROOT(2), RPAR(4), IPAR(34)

C Set LOUT, the unit number of the output device.
      LOUT = 6

C Open matrix output file if JACOUT .EQ. 1.
      IF (JACOUT .EQ. 1) THEN
         IPAR(29) = 1
         OPEN(UNIT=1, FILE='Heat_Test_Matrix.dat', STATUS='unknown')
      ENDIF
C
C Here set parameters for the problem being solved.  Use RPAR and IPAR
C to communicate these to the other routines.
C
      M = MAXM
      DX = 1.0D0/(M+1)
      NEQ = (M+2)*(M+2)
      COEFF = 1.0D0/(DX*DX)
      IPAR(33) = NEQ
      IPAR(34) = M
      RPAR(3) = DX
      RPAR(4) = COEFF
C
C Set NRT = number of root functions.
      NRT = 2
C
C Here set the lengths of the preconditioner work arrays WP and IWP,
C load them into IWORK, and set the total lengths of WORK and IWORK.
      IWORK(27) = LENWP
      IWORK(28) = LENIWP
      LRW = LENRW + LENWP
      LIW = LENIW + LENIWP
C Load values into IPAR and RPAR for sparse preconditioner.
      ML = 1
      MU = 1
      IPAR(1) = ML
      IPAR(2) = MU
      IPAR(3) = LENPFAC
      IPAR(4) = LENPLUFAC
      IPAR(5) = IPREMETH
      IPAR(6) = LFILILUT
      IPAR(7) = IREORDER
      IPAR(8) = ISRNORM
      IPAR(9) = NORMTYPE
      IPAR(10) = JACOUT
      IPAR(11) = JSCALCOL
      IPAR(30) = 0
      RPAR(1) = TOLILUT
      RPAR(2) = PERMTOL
C Check IPAR, RPAR, LENWP and LENIWP for illegal entries and long
C enough work array lengths.
      CALL DSPSETUP (NEQ, LENWP, LENIWP, RPAR, IPAR, IERR,
     .               LWPMIN, LIWPMIN)
      IF (IERR .NE. 0) THEN
         WRITE(LOUT,15) IERR
 15      FORMAT(' Error return from DSPSETUP: IERR = ',i5)
         IF (LWPMIN .GT. LENWP) THEN
            WRITE(LOUT,*) ' More WP work array length needed'
         ENDIF
         IF (LIWPMIN .GT. LENIWP) THEN
            WRITE(LOUT,*) ' More IWP work array length needed'
         ENDIF
         STOP
      ENDIF
C
C Call subroutine UINIT to initialize U and UPRIME.
C
      CALL UINIT (U, UPRIME, RPAR, IPAR)
C
C-----------------------------------------------------------------------
C Here we set up the INFO array, which describes the various options
C in the way we want DDASKR to solve the problem.
C In this case, we select the iterative preconditioned Krylov method,
C and we supply the sparse preconditioner routines DJACILU/DPSOLILU.
C
C We first initialize the entire INFO array to zero, then set select
C entries to nonzero values for desired solution options.
C
C To select the Krylov iterative method for the linear systems,
C we set INFO(12) = 1.
C
C Since we are using a preconditioner that involves approximate
C Jacobian elements requiring preprocessing, we have a JAC routine,
C namely subroutine DJACILU, and we must set INFO(15) = 1 to indicate
C this to DDASKR.
C
C No other entries of INFO need to be changed for this example.
C-----------------------------------------------------------------------
C
      DO 20 I = 1,20
 20     INFO(I) = 0
C
      INFO(12) = 1
      INFO(15) = 1
C
C Here we set tolerances for DDASKR to indicate how much accuracy 
C we want in the solution, in the sense of local error control.
C For this example, we ask for pure absolute error control with a
C tolerance of 1.0D-5.
      RTOL = 0.0D0
      ATOL = 1.0D-5
C
C Here we generate a heading with important parameter values.
C
      WRITE (LOUT,30) M,NEQ,INFO(12),ML,MU,IPREMETH,RTOL,ATOL
 30   FORMAT(' DHEATILU: Heat Equation Example Program for DDASKR'//
     1       '    M+2 by M+2 mesh, M =',I3,',  System size NEQ =',I4/
     1       '    Root functions are: R1 = max(u) - 0.1',
     1            ' and R2 = max(u) - 0.01'//
     1       '    Linear solver method flag INFO(12) =',I3,
     1       '    (0 = direct, 1 = Krylov)'/
     1       '    Preconditioner is a sparse approximation with ML =',
     1            I3,'  MU =',I3/
     1       '    Incomplete factorization option =',I2,
     1       '    (1 = ILUT, 2 = ILUTP)'/,
     1       '    Tolerances are RTOL =',E10.1,'   ATOL =',E10.1//)
      WRITE (LOUT,40)
 40   FORMAT(5X,'t',12X,'UMAX',8X,'NQ',8X,'H',8X,'STEPS',
     1       5X,'NNI',5X,'NLI'/)
C
C-----------------------------------------------------------------------
C Now we solve the problem.
C
C DDASKR will be called to compute 11 intermediate solutions from
C tout = 0.01 to tout = 10.24 by powers of 2.
C
C We pass to DDASKR the names DJACILU and DPSOLILU for the JAC and PSOL
C routines to do the preconditioning.
C
C At each output time, we compute and print the max-norm of the
C solution (which should decay exponentially in t).  We also print
C some relevant statistics -- the current method order and step size,
C the number of time steps so far, and the numbers of nonlinear and
C linear iterations so far.
C
C If a root was found, we flag this, and return to the DDASKR call.
C
C If DDASKR failed in any way (IDID .lt. 0) we print a message and
C stop the integration.
C-----------------------------------------------------------------------
C
      NOUT = 11
      T = 0.0D0
      TOUT = 0.01D0
      DO 70 IOUT = 1,NOUT
C
 45      CALL DDASKR (RESH, NEQ, T, U, UPRIME, TOUT, INFO, RTOL, ATOL,
     1     IDID, RWORK, LRW, IWORK, LIW, RPAR, IPAR, DJACILU, DPSOLILU,
     2     RTHEAT, NRT, JROOT)
C
         UMAX = 0.0D0
         DO 50 I = 1,NEQ
 50        UMAX = MAX (UMAX, ABS(U(I)) )
C
         HU = RWORK(7)
         NQU = IWORK(8)
         NST = IWORK(11)
         NNI = IWORK(19)
         NLI = IWORK(20)
         WRITE (LOUT,60) T,UMAX,NQU,HU,NST,NNI,NLI
 60      FORMAT(E15.5,E12.4,I5,E14.3,I7,I9,I8)
C
         IF (IDID .EQ. 5) THEN
           WRITE(6,61)JROOT(1),JROOT(2)
 61        FORMAT(20X,'*****   Root found, JROOT =',2I3)
           GO TO 45
           ENDIF
C
         IF (IDID .LT. 0) THEN
           WRITE(LOUT,65)T
 65        FORMAT(//' Final time reached =',E12.4//)
           GO TO 80
           ENDIF
C
         TOUT = TOUT*2.0D0
 70      CONTINUE
C
C Here we display some final statistics for the problem.
C The ratio of NLI to NNI is the average dimension of the Krylov
C subspace involved in the Krylov linear iterative method.
 80   CONTINUE
      NST = IWORK(11)
      NPE = IWORK(13)
      NRE = IWORK(12) + NPE*MBAND
      LIW = IWORK(17)
      LRW = IWORK(18)
      NNI = IWORK(19)
      NLI = IWORK(20)
      NPS = IWORK(21)
      IF (NNI .NE. 0) AVDIM = REAL(NLI)/REAL(NNI)
      NCFN = IWORK(15)
      NCFL = IWORK(16)
      NRTE = IWORK(36)
      WRITE (LOUT,90) LRW,LIW,NST,NRE,IPAR(30),NRTE,NPE,NPS,NNI,NLI,
     1                AVDIM,NCFN,NCFL
 90   FORMAT(//' Final statistics for this run..'/
     1   '   RWORK size =',I5,'   IWORK size =',I4/
     1   '   Number of time steps ................ =',I5/
     1   '   Number of residual evaluations ...... =',I5/
     1   '   Number of res. evals. for precond.    =',I5/
     1   '   Number of root function evaluations . =',I5/
     1   '   Number of preconditioner evaluations  =',I5/
     1   '   Number of preconditioner solves ..... =',I5/
     1   '   Number of nonlinear iterations ...... =',I5/
     1   '   Number of linear iterations ......... =',I5/
     1   '   Average Krylov subspace dimension =',F8.4/
     1   I5,' nonlinear conv. failures,',I5,' linear conv. failures')
      WRITE(LOUT,100) LWPMIN, LIWPMIN
 100  FORMAT(' Minimum lengths for work arrays WP and IWP: ',i7,1x,i7)
C
C------  End of main program for DHEATILU example program --------------
      STOP
      END

      SUBROUTINE UINIT (U, UPRIME, RPAR, IPAR)
C
C This routine computes and loads the vector of initial values.
C The initial U values are given by the polynomial u = 16x(1-x)y(1-y).
C The initial UPRIME values are set to zero.  (DDASKR corrects these
C during the first time step.)
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION U(*), UPRIME(*), RPAR(4), IPAR(34)
C
      NEQ = IPAR(33)
      M = IPAR(34)
      DX = RPAR(3)
C
      DO 20 K = 0,M+1
        YK = K*DX
        IOFF = (M+2)*K
        DO 10 J = 0,M+1
          XJ = J*DX
          I = IOFF + J + 1
          U(I) = 16.0D0*XJ*(1.0D0-XJ)*YK*(1.0D0-YK)
 10       CONTINUE
 20     CONTINUE
      DO 30 I = 1,NEQ
 30     UPRIME(I) = 0.0D0
      RETURN
C------------  End of Subroutine UINIT  --------------------------------
      END

      SUBROUTINE RESH (T, U, UPRIME, CJ, DELTA, IRES, RPAR, IPAR)
C
C This is the user-supplied RES subroutine for this example.
C It computes the residuals for the 2-D discretized heat equation,
C with zero boundary values.
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION U(*), UPRIME(*), DELTA(*), RPAR(4), IPAR(34)
C
C Set problem constants using IPAR and RPAR.
      NEQ = IPAR(33)
      M = IPAR(34)
      COEFF = RPAR(4)
      M2 = M + 2
C
C Load U into DELTA, in order to set boundary values.
      DO 10 I = 1,NEQ
 10     DELTA(I) = U(I)
C
C Loop over interior points, and load residual values.
      DO 30 K = 1,M
        IOFF = M2*K
        DO 20 J = 1,M
          I = IOFF + J + 1
          TEMX = U(I-1)  + U(I+1)
          TEMY = U(I-M2) + U(I+M2)
          DELTA(I) = UPRIME(I) - (TEMX + TEMY - 4.0D0*U(I))*COEFF
 20       CONTINUE
 30     CONTINUE
C
      RETURN
C------------  End of Subroutine RESH  ---------------------------------
      END

      SUBROUTINE RTHEAT(NEQ, T, U, UP, NRT, RVAL, RPAR, IPAR)
C
C This routine finds the max of U, and sets RVAL(1) = max(u) - 0.1,
C RVAL(2) = max(u) - 0.01.
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION U(NEQ), UP(NEQ), RVAL(2)
C
      UMAX = 0.0D0
      DO 10 I = 1,NEQ
 10     UMAX = MAX (UMAX, U(I) )
      RVAL(1) = UMAX - 0.1D0
      RVAL(2) = UMAX - 0.01D0
C
      RETURN
C------------  End of Subroutine RTHEAT  -------------------------------
      END
