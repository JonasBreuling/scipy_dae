C***BEGIN PROLOGUE  DWEBILU
C***REFER TO  DDASKR
C***DATE WRITTEN   020813   (YYMMDD)
C***REVISION DATE  021217   Added JROOT output value.
C
C***AUTHORS  A. C. Hindmarsh, P. N. Brown
C            Lawrence Livermore National Laboratory
C            Livermore, CA 94551, USA
C
C***DESCRIPTION
C
C-----------------------------------------------------------------------
C Example program for DDASKR.
C DAE system derived from ns-species interaction PDE in 2 dimensions.
C
C This is the double precision version.
C-----------------------------------------------------------------------
C
C This program solves a DAE system that arises from a system
C of partial differential equations.  The PDE system is a food web
C population model, with predator-prey interaction and diffusion on
C the unit square in two dimensions.  The dependent variable vector is
C
C         1   2        ns
C   c = (c , c , ..., c  )
C
C and the PDEs are as follows..
C
C     i               i      i
C   dc /dt  =  d(i)*(c    + c   )  +  R (x,y,c)  (i=1,...,ns/2)
C                     xx     yy        i
C
C                     i      i
C   0       =  d(i)*(c    + c   )  +  R (x,y,c)  (i=(ns/2)+1,...,ns)
C                     xx     yy        i
C
C where
C                  i          ns         j
C   R (x,y,c)  =  c *(b(i) + sum a(i,j)*c )
C    i                       j=1
C
C The number of species is ns = 2*np, with the first np being prey and
C the last np being predators.  The coefficients a(i,j), b(i), d(i) are
C
C   a(i,i) = -a  (all i)
C   a(i,j) = -g  (i .le. np, j .gt. np)
C   a(i,j) =  e  (i .gt. np, j .le. np)
C   b(i) =  b*(1 + alpha*x*y + beta*sin(4*pi*x)*sin(4*pi*y))  (i .le. np)
C   b(i) = -b*(1 + alpha*x*y + beta*sin(4*pi*x)*sin(4*pi*y))  (i .gt. np)
C   d(i) = dprey  (i .le. np)
C   d(i) = dpred  (i .gt. np)
C
C The various scalar parameters are set in subroutine SETPAR.
C
C The boundary conditions are.. normal derivative = 0.
C A polynomial in x and y is used to set the initial conditions.
C
C The PDEs are discretized by central differencing on a MX by MY mesh.
C
C The root function is R(Y) = average(c1) - 20.
C
C The DAE system is solved by DDASKR with a sparse approximate Jacobian
C matrix as the preconditioner for the preconditioned Krylov option of
C DDASKR.  The Jacobian is formed by using finite-differences of residual
C values, and is stored in sparse format (compressed sparse row).
C An incomplete factorization is then performed using the ILUT(P) routine
C of the SPARSKIT library of Yousef Saad at the University of Minnesota.
C
C The preconditioner is set up and solved in the three main 
C subroutines --  DSPSETUP, DJACILU and DPSOLILU.
C These routines are provided separately for use on general DAE
C problems.
C
C Two output files are written.. one with the problem description and
C performance statistics on unit LOUT = 9, and one with solution 
C profiles at selected output times on unit LCOUT = 10.
C-----------------------------------------------------------------------
C Note.. in addition to the main program and subroutines given below,
C this program requires the BLAS routine DAXPY.
C-----------------------------------------------------------------------
C References
C [1] Peter N. Brown and Alan C. Hindmarsh,
C     Reduced Storage Matrix Methods in Stiff ODE Systems,
C     J. Appl. Math. & Comp., 31 (1989), pp. 40-91.
C [2] Peter N. Brown, Alan C. Hindmarsh, and Linda R. Petzold,
C     Using Krylov Methods in the Solution of Large-Scale Differential-
C     Algebraic Systems, SIAM J. Sci. Comput., 15 (1994), pp. 1467-1488.
C [3] Peter N. Brown, Alan C. Hindmarsh, and Linda R. Petzold,
C     Consistent Initial Condition Calculation for Differential-
C     Algebraic Systems, LLNL Report UCRL-JC-122175, August 1995;
C     submitted to SIAM J. Sci. Comp.
C-----------------------------------------------------------------------
C***ROUTINES CALLED
C   SETPAR, CINIT, DDASKR, OUTWEB
C
C***END PROLOGUE  DWEBILU
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      EXTERNAL RESWEB, RTWEB, DJACILU, DPSOLILU
C Load parameters for sparse preconditioner.
      PARAMETER (LENPFAC = 10)
      PARAMETER (LENPLUFAC = 2)
      PARAMETER (IPREMETH = 2)  ! =1 means ILUT preconditioner used
                                ! =2 means ILUTP preconditioner used
      PARAMETER (LFILILUT = 10)
      PARAMETER (IREORDER = 1)
      PARAMETER (ISRNORM = 1)
      PARAMETER (NORMTYPE = 2)
      PARAMETER (JACOUT = 0)
      PARAMETER (JSCALCOL = 1)
      PARAMETER (TOLILUT = 0.001)
      PARAMETER (PERMTOL = 0.01)
C
C Dimension solution arrays and work arrays.
C
C When INFO(12) = 1, with INFO(5) = 0, INFO(6) = 1:
C    The length required for RWORK is 
C       101 + 3*NRT + 19*NEQ + LENWP,
C    where LENWP is given by
C       LENWP =  length of RWORK segment WP = 
C              2*LENPFAC*NEQ + LENPLUFAC*NEQ + ISRNORM*NEQ + 2*(NEQ+1)
C    The length required for IWORK is
C       40 + NEQ + LENIWP,
C    where LENIWP is given by
C       LENIWP = length of IWORK segment IWP =
C              4*(NEQ+1) + 3*LENPFAC*NEQ + 2*LENPLUFAC*NEQ
C                 + 2*IREORDER*NEQ + 2*(IPREMETH-1)*NEQ
C
C The dimensions for the various arrays are set below using parameters
C   MAXN    which must be .ge. NEQ = NS*MX*MY,
C   MAXS    which must be .ge. NS.
C
      PARAMETER (MAXS = 2, MAXN = 800)
      PARAMETER (LENWP = 2*LENPFAC*MAXN + LENPLUFAC*MAXN
     1                   + ISRNORM*MAXN + 2*(MAXN+1) )
      PARAMETER (LENIWP = 4*(MAXN+1) + 3*LENPFAC*MAXN
     1                    + 2*LENPLUFAC*MAXN + IREORDER*2*MAXN
     2                    + (IPREMETH-1)*2*MAXN )
      PARAMETER (LENRW = 104 + 19*MAXN, LENIW  = 40,
     1           LRW = LENRW + LENWP, LIW = LENIW + 2*MAXN + LENIWP)
C
      DIMENSION CC(MAXN), CCPRIME(MAXN), RWORK(LRW), IWORK(LIW),
     1          INFO(20), RPAR(2+MAXN), IPAR(30)
C
C The COMMON blocks /PPAR1/ and /PPAR2/ contain problem parameters.
C
      COMMON /PPAR1/ AA, EE, GG, BB, DPREY, DPRED
      COMMON /PPAR2/ NP, NS, AX, AY, ACOEF(MAXS,MAXS), BCOEF(MAXS),
     1               DX, DY, ALPH, BETA, FPI, DIFF(MAXS),
     2               COX(MAXS), COY(MAXS), MX, MY, MXNS
C
C Set output unit numbers for main output and tabulated solution.
      DATA LOUT/9/, LCOUT/10/
C
C Set unit number for DDASKR error message output.
      CALL XSETUN(LOUT)
C
C Open output files.
      OPEN(UNIT=LOUT,FILE='wdout',STATUS='unknown')
      OPEN(UNIT=LCOUT,FILE='wccout',STATUS='unknown')
      IF (JACOUT .EQ. 1) THEN
         IPAR(29) = 1
         OPEN(UNIT=1, FILE='Web_Test_Matrix.dat', STATUS='unknown')
      ENDIF
C
C Call SETPAR to set basic problem parameters.
      CALL SETPAR
C
C Set remaining problem parameters.
      NEQ = NS*MX*MY
      MXNS = MX*NS
      DX = AX/REAL(MX-1)
      DY = AY/REAL(MY-1)
      DO 10 I = 1,NS
        COX(I) = DIFF(I)/DX**2
 10     COY(I) = DIFF(I)/DY**2
C
C Set NRT = number of root functions.
      NRT = 1
C
      WRITE(LOUT,20)NS
 20   FORMAT(' DWEBILU: Example program for DDASKR package'//
     1   ' Food web problem with NS species, NS =',I4/
     2   ' Predator-prey interaction and diffusion on a 2-D square'/)
      WRITE(LOUT,25) AA,EE,GG,BB,DPREY,DPRED, ALPH,BETA
 25   FORMAT(' Matrix parameters..  a =',E12.4,'   e =',E12.4,
     1   '   g =',E12.4/21x,' b parameter =',E12.4//
     2   ' Diffusion coefficients.. dprey =',E12.4,'   dpred =',E12.4//
     3   ' Rate parameters alpha =',E12.4,' and beta =',E12.4/)
      WRITE(LOUT,30) MX,MY,NEQ
 30   FORMAT(' Mesh dimensions (MX,MY) =',2I4,
     1   5x,' Total system size is NEQ =',I7/)
      WRITE(LOUT,35)
 35   FORMAT(' Root function is R(Y) = average(c1) - 20'/)
C
C Here set the flat initial guess for the predators.
      PREDIC = 1.0D5
C 
C Load values into IPAR and RPAR for sparse preconditioner.
      ML = NS*MX + 1
      MU = ML
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
         WRITE(LOUT,45) IERR
 45      FORMAT(' Error return from DSPSETUP: IERR = ',i5)
         IF (LWPMIN .GT. LENWP) THEN
            WRITE(LOUT,*) ' More WP work array length needed'
         ENDIF
         IF (LIWPMIN .GT. LENIWP) THEN
            WRITE(LOUT,*) ' More IWP work array length needed'
         ENDIF
         STOP
      ENDIF
C
C Set remaining method parameters for DDASKR.
C These include the INFO array and tolerances.
C 
      DO 50 I = 1,20
 50     INFO(I) = 0
C
C Here set INFO(11) = 1, indicating I.C. calculation requested.
      INFO(11) = 1
C
C Here set INFO(12) = 1, indicating the Krylov linear system method.
      INFO(12) = 1
C
C Here set INFO(14) = 1 to get the computed initial values.
      INFO(14) = 1
C
C Here set INFO(15) = 1 to signal that a preconditioner setup routine
C is to be called in the Krylov case.
      INFO(15) = 1
C
C Here set INFO(16) = 1 to get alternative error test (on the
C differential variables only).
      INFO(16) = 1
C
C Here set the tolerances.
      RTOL = 1.0D-5
      ATOL = RTOL
C
      WRITE(LOUT,70)RTOL,ATOL,INFO(11),PREDIC,INFO(16)
 70   FORMAT(' Tolerance parameters.. RTOL =',E10.2,'   ATOL =',E10.2//
     1   ' Internal I.C. calculation flag INFO(11) =',I2,
     2   '   (0 = off, 1 = on)'/
     3   ' Predator I.C. guess =',E10.2//
     4   ' Alternate error test flag INFO(16) =',I2,
     5        '  (0 = off, 1 = on)'/)
C
C Set NOUT = number of output times.
      NOUT = 18
C
C Set and print various preconditioner parameters.
      IWORK(27) = LENWP 
      IWORK(28) = LENIWP
      WRITE(LOUT,100) IPREMETH
 100  FORMAT(' Linear solver is: Krylov with ILU preconditioner'/
     1       ' Preconditioner flag is IPREMETH =',I3,
     2       '  (1 = ILUT, 2 = ILUTP)')
C Here call SETID to set the IWORK segment ID  indicating the
C differential and algebraic components.
      CALL SETID (MX, MY, NS, NP, 40, IWORK)
C Set the initial T and TOUT, and call CINIT to set initial values.
      T = 0.0D0
      TOUT = 1.0D-8
      CALL CINIT (CC, CCPRIME, PREDIC, RPAR)
C
      NLI = 0
      NNI = 0
C
      WRITE(LOUT,140)
 140  FORMAT(//'   t',12X,'Ave.c1  NSTEP  NRE  NNI  NLI  NPE  NQ',4X,
     1       'H',10X,'AVLIN')
C
C Loop over output times and call DDASKR.  At each output time,
C print average c1 value and performance data.
C The first call, with IOUT = 0, is to calculate initial values only.
C After the first call, reset INFO(11) = 0 and the initial TOUT.
C If a root was found, we flag this, and return to the DDASKR call.
C
      DO 200 IOUT = 0,NOUT
C
 150    CALL DDASKR (RESWEB, NEQ, T, CC, CCPRIME, TOUT, INFO, RTOL,ATOL,
     1        IDID, RWORK,LRW, IWORK,LIW, RPAR, IPAR, DJACILU, DPSOLILU,
     2        RTWEB, NRT, JROOT)
C
        NST = IWORK(11)
        NRE = IWORK(12)
        NPE = IWORK(13)
        NNIDIF = IWORK(19) - NNI
        NNI = IWORK(19)
        NLIDIF = IWORK(20) - NLI
        NLI = IWORK(20)
        NQU = IWORK(8)
        HU = RWORK(7)
        AVLIN = 0.0D0
        IF (NNIDIF .GT. 0) AVLIN = REAL(NLIDIF)/REAL(NNIDIF)
C
        IMOD3 = IOUT - 3*(IOUT/3)
        IF (IMOD3 .EQ. 0) CALL OUTWEB (T, CC, NS, MX, MY, LCOUT)
C
        CALL AVC1 (CC, C1AVE)
        WRITE(LOUT,160)T,C1AVE,NST,NRE,NNI,NLI,NPE,NQU,HU,AVLIN
 160    FORMAT(E13.5,f10.5,I5,I6,3I5,I4,E11.2,F9.4)
        IF (IDID .EQ. 5) THEN
          WRITE(LOUT,165)JROOT
 165      FORMAT(15X,'*****   Root found, JROOT =',I3)
          GO TO 150
          ENDIF
C
        IF (IDID .LT. 0) THEN
          WRITE(LOUT,170)T
 170      FORMAT(//' Final time reached =',E12.4//)
          GO TO 210
          ENDIF
C
        IF (TOUT .GT. 0.9D0) TOUT = TOUT + 1.0D0
        IF (TOUT .LT. 0.9D0) TOUT = TOUT*10.0D0
        IF (IOUT .EQ. 0) THEN
          INFO(11) = 0
          TOUT = 1.0D-8
          NLI = 0
          NNI = 0
          ENDIF
 200    CONTINUE
C
 210  CONTINUE
      LNRW = IWORK(18)
      LNIW = IWORK(17)
      NST = IWORK(11)
      NRE = IWORK(12)
      NPE = IWORK(13)
      NNI = IWORK(19)
      NLI = IWORK(20)
      NPS = IWORK(21)
      IF (NNI .GT. 0) AVLIN = REAL(NLI)/REAL(NNI)
      NCFN = IWORK(15)
      NCFL = IWORK(16)
      NRTE = IWORK(36)
      WRITE(LOUT,220) LNRW,LNIW,NST,NRE,IPAR(30),NRTE,NPE,NPS,NNI,NLI,
     1                AVLIN,NCFN,NCFL
 220   FORMAT(//' Final statistics for this run..'/
     1   ' RWORK size =',I8,'   IWORK size =',I6/
     1   ' Number of time steps            =',I5/
     1   ' Number of residual evaluations  =',I5/
     1   ' Number of res. evals. for prec. =',I5/
     1   ' Number of root fn. evaluations  =',I5/
     1   ' Number of Jac. or prec. evals.  =',I5/
     1   ' Number of preconditioner solves =',I5/
     1   ' Number of nonlinear iterations  =',I5/
     1   ' Number of linear iterations     =',I5/
     1   ' Average Krylov subspace dimension =',F8.4/
     1   I3,' nonlinear conv. failures,',i5,' linear conv. failures')
      WRITE(LOUT,230) LWPMIN, LIWPMIN
 230  FORMAT(' Minimum lengths for work arrays WP and IWP: ',i7,1x,i7)
C
      STOP
C------  End of main program for DWEBILU example program ---------------
      END

      SUBROUTINE SETPAR
C-----------------------------------------------------------------------
C This routine sets the basic problem parameters, namely
C AX, AY, NS, MX, MY,  problem coefficients ACOEF, BCOEF, DIFF,
C ALPH, BETA, using parameters NP, AA, EE, GG, BB, DPREY, DPRED.
C-----------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (MAXS = 2)
      COMMON /PPAR1/ AA, EE, GG, BB, DPREY, DPRED
      COMMON /PPAR2/ NP, NS, AX, AY, ACOEF(MAXS,MAXS), BCOEF(MAXS),
     1               DX, DY, ALPH, BETA, FPI, DIFF(MAXS),
     2               COX(MAXS), COY(MAXS), MX, MY, MXNS
C
      AX = 1.0D0
      AY = 1.0D0
      NP = 1
      MX = 20
      MY = 20
      AA = 1.0D0
      EE = 1.0D4
      GG = 0.5D-6
      BB = 1.0D0
      DPREY = 1.0D0
      DPRED = 0.05D0
      ALPH = 50.0D0
      BETA = 100.0D0
      NS = 2*NP
      DO 20 J = 1,NP
        DO 10 I = 1,NP
          ACOEF(NP+I,J) = EE
          ACOEF(I,NP+J) = -GG
 10       CONTINUE
        ACOEF(J,J) = -AA
        ACOEF(NP+J,NP+J) = -AA
        BCOEF(J) = BB
        BCOEF(NP+J) = -BB
        DIFF(J) = DPREY
        DIFF(NP+J) = DPRED
 20     CONTINUE
      PI = 3.141592653589793D0
      FPI = 4.0D0*PI
C
      RETURN
C------------  End of Subroutine SETPAR  -------------------------------
      END

      SUBROUTINE SETID (MX, MY, NS, NSD, LID, IWORK)
C-----------------------------------------------------------------------
C This routine sets the ID array in IWORK, indicating which components
C are differential and which are algebraic.
C-----------------------------------------------------------------------
      DIMENSION IWORK(*)
C
      NSDP1 = NSD + 1
      DO 40 JY = 1,MY
        I00 = MX*NS*(JY-1) + LID
        DO 30 JX = 1,MX
          I0 = I00 + NS*(JX-1)  
          DO 10 I = 1,NSD
 10         IWORK(I0+I) = 1
          DO 20 I = NSDP1,NS
 20         IWORK(I0+I) = -1
 30       CONTINUE
 40     CONTINUE
C
      RETURN
C------------  End of Subroutine SETID  --------------------------------
      END

      SUBROUTINE CINIT (CC, CCPRIME, PREDIC, RPAR)
C-----------------------------------------------------------------------
C This routine computes and loads the vectors of initial values.
C-----------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION CC(*), CCPRIME(*), RPAR(*)
      PARAMETER (MAXS = 2)
      COMMON /PPAR2/ NP, NS, AX, AY, ACOEF(MAXS,MAXS), BCOEF(MAXS),
     1               DX, DY, ALPH, BETA, FPI, DIFF(MAXS),
     2               COX(MAXS), COY(MAXS), MX, MY, MXNS
C
C Load CC.
      NPP1 = NP + 1
      DO 30 JY = 1,MY
        Y = REAL(JY-1)*DY
        ARGY = 16.0D0*Y*Y*(AY-Y)*(AY-Y)
        IYOFF = MXNS*(JY-1)
        DO 20 JX = 1,MX
          X = REAL(JX-1)*DX
          ARGX = 16.0D0*X*X*(AX-X)*(AX-X)
          IOFF = IYOFF + NS*(JX-1)
          FAC = 1.0D0 + ALPH*X*Y + BETA*SIN(FPI*X)*SIN(FPI*Y)
          DO 10 I = 1,NP
 10         CC(IOFF + I) = 10.0D0 + REAL(I)*ARGX*ARGY
          DO 15 I = NPP1,NS
 15         CC(IOFF + I) = PREDIC
 20       CONTINUE
 30     CONTINUE
C
C Load CCPRIME.
      T = 0.0D0
      CALL FWEB (T, CC, CCPRIME, RPAR)
      DO 60 JY = 1,MY
        IYOFF = MXNS*(JY-1)
        DO 50 JX = 1,MX
          IOFF = IYOFF + NS*(JX-1)
          DO 40 I = NPP1,NS
 40         CCPRIME(IOFF+I) = 0.0D0
 50     CONTINUE
 60   CONTINUE
C
      RETURN
C------------  End of Subroutine CINIT  --------------------------------
      END

      SUBROUTINE OUTWEB (T, C, NS, MX, MY, LUN)
C-----------------------------------------------------------------------
C This routine prints the values of the individual species densities
C at the current time T, to logical unit LUN.
C-----------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION C(NS,MX,MY)
C
      WRITE(LUN,10) T
 10   FORMAT(/1X,79('-')/30X,'At time t = ',E16.8/1X,79('-') )
C
      DO 40 I = 1,NS
        WRITE(LUN,20) I
 20     FORMAT(' the species c(',i2,') values are ')
        DO 30 JY = MY,1,-1
          WRITE(LUN,25) (C(I,JX,JY),JX=1,MX)
 25       FORMAT(6(1X,G12.6))
 30       CONTINUE
        WRITE(LUN,35)
 35     FORMAT(1X,79('-'),/)
 40     CONTINUE
C
      RETURN
C------------  End of Subroutine OUTWEB  -------------------------------
      END

      SUBROUTINE RESWEB (T, U, UPRIME, CJ, DELTA, IRES, RPAR, IPAR)
C-----------------------------------------------------------------------
C This routine computes the residual vector, using Subroutine FWEB
C for the right-hand sides.
C-----------------------------------------------------------------------
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION U(*),UPRIME(*),DELTA(*),RPAR(*),IPAR(*)
      PARAMETER (MAXS = 2)
      COMMON /PPAR2/ NP, NS, AX, AY, ACOEF(MAXS,MAXS), BCOEF(MAXS),
     1               DX, DY, ALPH, BETA, FPI, DIFF(MAXS),
     2               COX(MAXS), COY(MAXS), MX, MY, MXNS
C
      CALL FWEB (T, U, DELTA, RPAR)
C
      DO 30 JY = 1,MY
         IYOFF = MXNS*(JY-1)
         DO 20 JX = 1,MX
            IC0 = IYOFF + NS*(JX-1)
            DO 10 I = 1,NS
               ICI = IC0 + I
               IF (I .GT. NP) THEN
                  DELTA(ICI) = -DELTA(ICI)
               ELSE
                  DELTA(ICI) = UPRIME(ICI) - DELTA(ICI)
               ENDIF
 10         CONTINUE
 20      CONTINUE
 30   CONTINUE
C
      RETURN
C------------  End of Subroutine RESWEB  -------------------------------
      END

      SUBROUTINE FWEB (T, CC, CRATE, RPAR)
C-----------------------------------------------------------------------
C This routine computes the right-hand sides of all the equations
C and returns them in the array CRATE.
C The interaction rates are computed by calls to WEBR, and these are
C saved in RPAR(1),...,RPAR(NEQ) for use later in preconditioning.
C-----------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION CC(*), CRATE(*), RPAR(*)
      PARAMETER (MAXS = 2)
      COMMON /PPAR2/ NP, NS, AX, AY, ACOEF(MAXS,MAXS), BCOEF(MAXS),
     1               DX, DY, ALPH, BETA, FPI, DIFF(MAXS),
     2               COX(MAXS), COY(MAXS), MX, MY, MXNS
C
      DO 60 JY = 1,MY
        IYOFF = MXNS*(JY-1)
        IDYU = MXNS
        IF (JY .EQ. MY) IDYU = -MXNS
        IDYL = MXNS
        IF (JY .EQ. 1) IDYL = -MXNS
        DO 40 JX = 1,MX
          IC = IYOFF + NS*(JX-1) + 1
C Get interaction rates at one point (X,Y).
          CALL WEBR (T, JX, JY, CC(IC), RPAR(2+IC))
          IDXU = NS
          IF (JX .EQ. MX) IDXU = -NS
          IDXL = NS
          IF (JX .EQ. 1) IDXL = -NS
          DO 20 I = 1,NS
            ICI = IC + I - 1
C Do differencing in Y.
            DCYLI = CC(ICI) - CC(ICI-IDYL)
            DCYUI = CC(ICI+IDYU) - CC(ICI)
C Do differencing in X.
            DCXLI = CC(ICI) - CC(ICI-IDXL)
            DCXUI = CC(ICI+IDXU) - CC(ICI)
C Collect terms and load CRATE elements.
            CRATE(ICI) = COY(I)*(DCYUI - DCYLI) + COX(I)*(DCXUI - DCXLI)
     1                  + RPAR(2+ICI)
 20         CONTINUE
 40       CONTINUE
 60    CONTINUE
      RETURN
C------------  End of Subroutine FWEB  ---------------------------------
      END

      SUBROUTINE WEBR (T, JX, JY, C, CRATE)
C-----------------------------------------------------------------------
C This routine computes one block of the interaction term R of the 
C system, namely block (JX,JY), for use in preconditioning.
C-----------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION C(*), CRATE(*)
      PARAMETER (MAXS = 2)
      COMMON /PPAR2/ NP, NS, AX, AY, ACOEF(MAXS,MAXS), BCOEF(MAXS),
     1               DX, DY, ALPH, BETA, FPI, DIFF(MAXS),
     2               COX(MAXS), COY(MAXS), MX, MY, MXNS
C
      Y = REAL(JY-1)*DY
      X = REAL(JX-1)*DX
      DO 10 I = 1,NS
 10     CRATE(I) = 0.0D0
      DO 15 J = 1,NS
        CALL DAXPY (NS, C(J), ACOEF(1,J), 1, CRATE, 1)
 15     CONTINUE
      FAC = 1.0D0 + ALPH*X*Y + BETA*SIN(FPI*X)*SIN(FPI*Y)
      DO 20 I = 1,NS
 20     CRATE(I) = C(I)*(BCOEF(I)*FAC + CRATE(I))
      RETURN
C------------  End of Subroutine WEBR  ---------------------------------
      END

      SUBROUTINE AVC1 (CC, C1AVE)
C-----------------------------------------------------------------------
C This routine computes the average value of c1.
C-----------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION CC(*)
      PARAMETER (MAXS = 2)
      COMMON /PPAR2/ NP, NS, AX, AY, ACOEF(MAXS,MAXS), BCOEF(MAXS),
     1               DX, DY, ALPH, BETA, FPI, DIFF(MAXS),
     2               COX(MAXS), COY(MAXS), MX, MY, MXNS
C
      SUM = 0.0D0
      NPP1 = NP + 1
      DO 30 JY = 1,MY
        IYOFF = MXNS*(JY-1)
        DO 20 JX = 1,MX
          IOFF = IYOFF + NS*(JX-1)
          SUM = SUM + CC(IOFF + 1)
 20       CONTINUE
 30     CONTINUE
C
      C1AVE = SUM/(MX*MY)
C
      RETURN
C------------  End of Subroutine AVC1  ---------------------------------
      END

      SUBROUTINE RTWEB(NEQ, T, CC, CP, NRT, RVAL, RPAR, IPAR)
C
C This routine sets RVAL = average(c1) - 20.0.
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION CC(NEQ), CP(NEQ)
C
      CALL AVC1 (CC, C1AVE)
      RVAL = C1AVE - 20.0D0
C
      RETURN
C------------  End of Subroutine RTWEB  --------------------------------
      END
