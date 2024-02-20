C***BEGIN PROLOGUE  DWEB
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
C The root function is R(T,Y,Y') = average(c1) - 20.
C
C The DAE system is solved by DDASKR with three different method options:
C (1) direct band method for the linear systems (internal Jacobian),
C (2) preconditioned Krylov method for the linear systems, without
C     block-grouping in the reaction-based factor, and
C (3) preconditioned Krylov method for the linear systems, with
C     block-grouping in the reaction-based factor.
C
C In the Krylov cases, the preconditioner is the product of two factors:
C (a) The spatial factor uses a fixed number of Gauss-Seidel iterations
C based on the diffusion terms only.
C (b) The reaction-based factor is a block-diagonal matrix based on
C the partial derivatives of the interaction terms R only.
C With block-grouping, only a subset of the ns by ns blocks are computed.
C An integer flag, JPRE, is set in the main program to specify whether
C the preconditioner is to use only one of the two factors or both,
C and in which order.
C
C The reaction-based preconditioner factor is set up and solved in
C seven subroutines -- 
C   DMSET2, DRBDJA, DRBDPS  in the case of no block-grouping, and
C   DGSET2, GSET1, DRBGJA, DRBGPS  in the case of block-grouping.
C These routines are provided separately for general use on problems
C arising from reaction-transport systems.
C
C Two output files are written.. one with the problem description and
C performance statistics on unit LOUT = 9, and one with solution 
C profiles at selected output times on unit LCOUT = 10.
C The solution file is written only in the case of the direct method.
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
C   SETPAR, DGSET2, CINIT, DDASKR, OUTWEB
C
C***END PROLOGUE  DWEB
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      EXTERNAL RESWEB, RTWEB, JACRS, PSOLRS
C
C Set output unit numbers for main output and tabulated solution.
      DATA LOUT/9/, LCOUT/10/
C
C Dimension solution arrays and work arrays.
C
C When INFO(12) = 0, with INFO(5) = 0, INFO(6) = 1:
C    The length required for RWORK is
C        60 + 3*NRT + (2*ML+MU+11)*NEQ + 2*(NEQ/(ML+MU+1) + 1) .
C    For MX = MY = (even number) and ML = MU = NS*MX, this length is
C        60 + 3*NRT + (3*NS*MX + 11)*NEQ + MY .
C    The length required for IWORK is  40 + 2*NEQ .
C
C When INFO(12) = 1:
C    The length required for RWORK is 
C        101 + 3*NRT + 19*NEQ + LENWP = 104 + 19*NEQ + NS*NS*NGRP .
C    The length required for IWORK is
C         40 + NEQ + LENIWP = 40 + NEQ + NS*NGRP .
C
C The dimensions for the various arrays are set below using parameters
C   MAXN    which must be .ge. NEQ = NS*MX*MY,
C   MAXS    which must be .ge. NS,
C   MAXM    which must be .ge. MAX(MX,MY).
C
      PARAMETER (MAXS = 2, MAXM = 20, MAXN = 800,
     1           LRW = 63 + (3*MAXS*MAXM + 11)*MAXN + MAXM,
     2           LIW = 40 + 2*MAXN)
C
      DIMENSION CC(MAXN), CCPRIME(MAXN), RWORK(LRW), IWORK(LIW),
     1          INFO(20), RPAR(MAXN), IPAR(2)
C
C The COMMON blocks /PPAR1/ and /PPAR2/ contain problem parameters.
C
      COMMON /PPAR1/ AA, EE, GG, BB, DPREY, DPRED
      COMMON /PPAR2/ NP, NS, AX, AY, ACOEF(MAXS,MAXS), BCOEF(MAXS),
     1               DX, DY, ALPH, BETA, FPI, DIFF(MAXS),
     2               COX(MAXS), COY(MAXS), MX, MY, MXNS
C
C Open output files.
      OPEN(UNIT=LOUT,FILE='wdout',STATUS='unknown')
      OPEN(UNIT=LCOUT,FILE='wccout',STATUS='unknown')
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
 20   FORMAT(' DWEB: Example program for DDASKR package'//
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
C Set remaining method parameters for DDASKR.
C These include the INFO array and tolerances.
C 
      DO 50 I = 1,20
 50     INFO(I) = 0
C
C Here set INFO(11) = 1, indicating I.C. calculation requested.
      INFO(11) = 1
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
     5        '  (0 = off, 1 = on)')
C
C Set NOUT = number of output times.
      NOUT = 18
C
C Loop over method options: 
C METH = 0 means use INFO(12) = 0 (direct)
C METH = 1 means use INFO(12) = 1 (Krylov) without block-grouping in
C          the reaction-based factor in the preconditioner.
C METH = 2 means use INFO(12) = 1 (Krylov) with block-grouping in
C          the reaction-based factor in the preconditioner.
C A block-grouping flag JBG, communicated through IPAR, is set to
C 0 (no block-grouping) or 1 (use block-grouping) with METH = 1 or 2.
C Reset INFO(1) = 0 and INFO(11) = 1.
C
      DO 300 METH = 0,2
      INFO(12) = MIN(METH,1)
      INFO(1) = 0
      INFO(11) = 1
      JBG = METH - 1
      IPAR(2) = JBG
C
      WRITE(LOUT,80)INFO(12)
 80   FORMAT(//80('.')//
     1   ' Linear solver method flag INFO(12) =',I2,
     2   '   (0 = direct, 1 = Krylov)'/)
C
C In the case of the direct method, set INFO(6) = 1 to signal a banded 
C Jacobian, set IWORK(1) = IWORK(2) = MX*NS, the half-bandwidth, and
C call SETID to set the IWORK segment ID indicating the differential
C and algebraic components.
      IF (INFO(12) .EQ. 0) THEN
        INFO(6) = 1
        IWORK(1) = MXNS
        IWORK(2) = MXNS
        CALL SETID (MX, MY, NS, NP, 40, IWORK)
        WRITE(LOUT,90)MXNS
 90     FORMAT(' Difference-quotient banded Jacobian,',
     1         ' half-bandwidths =',I4)
        ENDIF
C
C In the case of the Krylov method, set and print various
C preconditioner parameters.
      IF (INFO(12) .EQ. 1) THEN
C First set the preconditioner choice JPRE.
C  JPRE = 1 means reaction-only (block-diagonal) factor A_R
C  JPRE = 2 means spatial factor (Gauss-Seidel) A_S
C  JPRE = 3 means A_S * A_R
C  JPRE = 4 means A_R * A_S
C Use IPAR to communicate JPRE to the preconditioner solve routine.
        JPRE = 3
        IPAR(1) = JPRE
        WRITE(LOUT,100)JPRE
 100    FORMAT(' Preconditioner flag is JPRE =',I3/
     1      '  (1 = reaction factor A_R, 2 = spatial factor A_S,',
     2      ' 3 = A_S*A_R, 4 = A_R*A_S )'/)
C Here call DMSET2 if JBG = 0, or DGSET2 if JBG = 1, to set the 2-D
C mesh parameters and block-grouping data, and the IWORK segment ID
C indicating the differential and algebraic components.
        IF (JBG .EQ. 0) THEN
          CALL DMSET2 (MX, MY, NS, NP, 40, IWORK)
          WRITE(LOUT,110)
 110      FORMAT(' No block-grouping in reaction factor')
          ENDIF
        IF (JBG .EQ. 1) THEN
          NXG = 5
          NYG = 5
          NG = NXG*NYG
          CALL DGSET2 (MX, MY, NS, NP, NXG, NYG, 40, IWORK)
          WRITE(LOUT,120)NG,NXG,NYG
 120      FORMAT(' Block-grouping in reaction factor'/
     1           ' Number of groups =',I5,
     2           '   (NGX by NGY, NGX =',I3,',  NGY =',I3,')')
          ENDIF
        ENDIF
C
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
     1   'H',10X,'AVLIN')
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
     1            IDID, RWORK,LRW, IWORK,LIW, RPAR, IPAR, JACRS, PSOLRS,
     2            RTWEB, NRT, JROOT)
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
        IF (METH .EQ. 0) THEN
          IMOD3 = IOUT - 3*(IOUT/3)
          IF (IMOD3 .EQ. 0) CALL OUTWEB (T, CC, NS, MX, MY, LCOUT)
          ENDIF
C
        CALL AVC1 (CC, C1AVE)
        WRITE(LOUT,160)T,C1AVE,NST,NRE,NNI,NLI,NPE,NQU,HU,AVLIN
 160    FORMAT(E13.5,f10.5,I5,I6,3I5,I4,E11.2,F9.4)
C
        IF (IDID .EQ. 5) THEN
          WRITE(LOUT,165)JROOT
 165      FORMAT(15X,'*****   Root found, JROOT =',I3)
          GO TO 150
          ENDIF

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
      LENRW = IWORK(18)
      LENIW = IWORK(17)
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
      WRITE(LOUT,220) LENRW,LENIW,NST,NRE,NRTE,NPE,NPS,NNI,NLI,AVLIN,
     1                NCFN,NCFL
 220   FORMAT(//' Final statistics for this run..'/
     1   ' RWORK size =',I8,'   IWORK size =',I6/
     2   ' Number of time steps            =',I5/
     3   ' Number of residual evaluations  =',I5/
     3   ' Number of root fn. evaluations  =',I5/
     4   ' Number of Jac. or prec. evals.  =',I5/
     5   ' Number of preconditioner solves =',I5/
     6   ' Number of nonlinear iterations  =',I5/
     7   ' Number of linear iterations     =',I5/
     8   ' Average Krylov subspace dimension =',F8.4/
     9   I3,' nonlinear conv. failures,',i5,' linear conv. failures')
C
 300   CONTINUE

      STOP
C------  End of main program for DWEB example program ------------------
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
          CALL WEBR (T, JX, JY, CC(IC), RPAR(IC))
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
     1                  + RPAR(ICI)
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

      SUBROUTINE JACRS(RES, IRES, NEQ, T, CC, CCPRIME, REWT, SAVR, WK,
     1                  H, CJ, WP, IWP, IER, RPAR, IPAR)
C-----------------------------------------------------------------------
C This routine interfaces to Subroutine DRBDJA or Subroutine DRBGJA,
C depending on the flag JBG = IPAR(2), to generate and preprocess the
C block-diagonal Jacobian corresponding to the reaction term R.
C If JBG = 0, we call DRBDJA, with no block-grouping.
C If JBG = 1, we call DRBGJA, and use block-grouping.
C Array RPAR, containing the current R vector, is passed to DRBDJA and
C DRBGJA as argument R0, consistent with the loading of RPAR in FWEB.
C The external name WEBR is passed, as the routine which computes the
C individual blocks of the R vector. 
C-----------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      EXTERNAL WEBR
      DIMENSION CC(*), CCPRIME(*), REWT(*), SAVR(*), WK(*), WP(*),
     1          IWP(*), RPAR(*), IPAR(*)
C
      JBG = IPAR(2)
      IF (JBG .EQ. 0) THEN
          CALL DRBDJA (T, CC, RPAR, WEBR, WK, REWT, CJ, WP, IWP, IER)
        ELSE
          CALL DRBGJA (T, CC, RPAR, WEBR, WK, REWT, CJ, WP, IWP, IER)
        ENDIF
C
      RETURN
C------------  End of Subroutine JACRS  --------------------------------
      END

      SUBROUTINE PSOLRS (NEQ, T, CC, CCPRIME, SAVR, WK, CJ, WT, WP, IWP,
     1                   B, EPLIN, IER, RPAR, IPAR)
C-----------------------------------------------------------------------
C This routine applies the inverse of a product preconditioner matrix 
C to the vector in the array B.  Depending on the flag JPRE, this 
C involves a call to GS, for the inverse of the spatial factor, and/or
C a call to DRBDPS or DRBGPS for the inverse of the reaction-based
C factor (CJ*I_d - dR/dy).  The latter factor uses block-grouping
C (with a call to DRBGPS) if JBG = 1, and does not (with a call to
C DRBDPS) if JBG = 0.  JBG is communicated as IPAR(2).
C The array B is overwritten with the solution.
C-----------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION CC(*), CCPRIME(*), SAVR(*), WK(*), WP(*), IWP(*), B(*),
     1          RPAR(*), IPAR(*)
C
      JPRE = IPAR(1)
      IER = 0
      HL0 = 1.0D0/CJ
C
      JBG = IPAR(2)
C
      IF (JPRE .EQ. 2 .OR. JPRE .EQ. 3) CALL GS (NEQ, HL0, B, WK)
C
      IF (JPRE .NE. 2 ) THEN
        IF (JBG .EQ. 0) CALL DRBDPS (B, WP, IWP)
        IF (JBG .EQ. 1) CALL DRBGPS (B, WP, IWP)
        ENDIF
C
      IF (JPRE .EQ. 4) CALL GS (NEQ, HL0, B, WK)
C
      RETURN
C------------  End of Subroutine PSOLRS  -------------------------------
      END

      SUBROUTINE GS (N, HL0, Z, X)
C-----------------------------------------------------------------------
C This routine provides the inverse of the spatial factor for a 
C product preconditoner in an NS-species reaction-diffusion problem.
C It performs ITMAX = 5 Gauss-Seidel iterations to compute an
C approximation to (A_S)-inverse * Z, where A_S = I - hl0*Jd, and Jd 
C represents the diffusion contributions to the Jacobian.  
C The solution is vector returned in Z.
C-----------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION Z(*), X(*)
      PARAMETER (MAXS = 2)
      COMMON /PPAR2/ NP, NS, AX, AY, ACOEF(MAXS,MAXS), BCOEF(MAXS),
     1               DX, DY, ALPH, BETA, FPI, DIFF(MAXS),
     2               COX(MAXS), COY(MAXS), MX, MY, MXNS
      DATA ITMAX/5/
C
      DIMENSION BETA1(MAXS), GAMMA1(MAXS), BETA2(MAXS), GAMMA2(MAXS), 
     1          DINV(MAXS)
C
C-----------------------------------------------------------------------
C Write matrix as A = D - L - U.
C Load local arrays BETA, BETA2, GAMMA1, GAMMA2, and DINV.
C-----------------------------------------------------------------------
      DO 10 I = 1,NS
        ELAMDA = 1.0D0/(1.0D0 + 2.0D0*HL0*(COX(I) + COY(I)))
        BETA1(I) = HL0*COX(I)*ELAMDA
        BETA2(I) = 2.0D0*BETA1(I)
        GAMMA1(I) = HL0*COY(I)*ELAMDA
        GAMMA2(I) = 2.0D0*GAMMA1(I)
        DINV(I) = ELAMDA
 10     CONTINUE
C-----------------------------------------------------------------------
C Begin iteration loop.
C Load array X with (D-inverse)*Z for first iteration.
C-----------------------------------------------------------------------
      ITER = 1
C Zero X in all its components, since X is added to Z at the end.
      DO 15 II = 1,N
 15      X(II) = 0.0D0
C
      DO 50 JY = 1,MY
        IYOFF = MXNS*(JY-1)
        DO 40 JX = 1,MX
          IC = IYOFF + NS*(JX-1)
          DO 30 I = 1,NS
            ICI = IC + I
            X(ICI) = DINV(I)*Z(ICI)
            Z(ICI) = 0.0D0
 30         CONTINUE
 40       CONTINUE
 50     CONTINUE
      GO TO 160
C-----------------------------------------------------------------------
C Calculate (D-inverse)*U*X.
C-----------------------------------------------------------------------
 70   CONTINUE
      ITER = ITER + 1
      JY = 1
        JX = 1
        IC = NS*(JX-1)
        DO 75 I = 1,NS
          ICI = IC + I
 75       X(ICI) = BETA2(I)*X(ICI+NS) + GAMMA2(I)*X(ICI+MXNS)
        DO 85 JX = 2,MX-1
          IC = NS*(JX-1)
          DO 80 I = 1,NS
            ICI = IC + I
 80         X(ICI) = BETA1(I)*X(ICI+NS) + GAMMA2(I)*X(ICI+MXNS)
 85       CONTINUE
        JX = MX
        IC = NS*(JX-1)
        DO 90 I = 1,NS
          ICI = IC + I
 90       X(ICI) = GAMMA2(I)*X(ICI+MXNS)
      DO 115 JY = 2,MY-1
        IYOFF = MXNS*(JY-1)
          JX = 1
          IC = IYOFF
          DO 95 I = 1,NS
            ICI = IC + I
 95         X(ICI) = BETA2(I)*X(ICI+NS) + GAMMA1(I)*X(ICI+MXNS)
          DO 105 JX = 2,MX-1
            IC = IYOFF + NS*(JX-1)
            DO 100 I = 1,NS
              ICI = IC + I
 100          X(ICI) = BETA1(I)*X(ICI+NS) + GAMMA1(I)*X(ICI+MXNS)
 105        CONTINUE
          JX = MX
          IC = IYOFF + NS*(JX-1)
          DO 110 I = 1,NS
            ICI = IC + I
 110        X(ICI) = GAMMA1(I)*X(ICI+MXNS)
 115      CONTINUE
      JY = MY
      IYOFF = MXNS*(JY-1)
        JX = 1
        IC = IYOFF
        DO 120 I = 1,NS
          ICI = IC + I
 120      X(ICI) = BETA2(I)*X(ICI+NS)
        DO 130 JX = 2,MX-1
          IC = IYOFF + NS*(JX-1)
          DO 125 I = 1,NS
            ICI = IC + I
 125      X(ICI) = BETA1(I)*X(ICI+NS)
 130      CONTINUE
        JX = MX
        IC = IYOFF + NS*(JX-1)
        DO 135 I = 1,NS
          ICI = IC + I
 135      X(ICI) = 0.0D0
C-----------------------------------------------------------------------
C Calculate [(I - (D-inverse)*L)]-inverse * X.
C-----------------------------------------------------------------------
 160  CONTINUE
      JY = 1
        DO 175 JX = 2,MX-1
          IC = NS*(JX-1)
          DO 170 I = 1,NS
            ICI = IC + I
 170        X(ICI) = X(ICI) + BETA1(I)*X(ICI-NS)
 175      CONTINUE
        JX = MX
        IC = NS*(JX-1)
        DO 180 I = 1,NS
          ICI = IC + I
 180      X(ICI) = X(ICI) + BETA2(I)*X(ICI-NS)
      DO 210 JY = 2,MY-1
        IYOFF = MXNS*(JY-1)
          JX = 1
          IC = IYOFF
          DO 185 I = 1,NS
            ICI = IC + I
 185        X(ICI) = X(ICI) + GAMMA1(I)*X(ICI-MXNS)
          DO 200 JX = 2,MX-1
            IC = IYOFF + NS*(JX-1)
            DO 195 I = 1,NS
              ICI = IC + I
              X(ICI) = (X(ICI) + BETA1(I)*X(ICI-NS))
     1             + GAMMA1(I)*X(ICI-MXNS)
 195          CONTINUE
 200        CONTINUE
            JX = MX
            IC = IYOFF + NS*(JX-1)
            DO 205 I = 1,NS
              ICI = IC + I
              X(ICI) = (X(ICI) + BETA2(I)*X(ICI-NS))
     1             + GAMMA1(I)*X(ICI-MXNS)
 205          CONTINUE
 210        CONTINUE
      JY = MY
      IYOFF = MXNS*(JY-1)
        JX = 1
        IC = IYOFF
        DO 215 I = 1,NS
          ICI = IC + I
 215      X(ICI) = X(ICI) + GAMMA2(I)*X(ICI-MXNS)
        DO 225 JX = 2,MX-1
          IC = IYOFF + NS*(JX-1)
          DO 220 I = 1,NS
            ICI = IC + I
            X(ICI) = (X(ICI) + BETA1(I)*X(ICI-NS))
     1           + GAMMA2(I)*X(ICI-MXNS)
 220        CONTINUE
 225      CONTINUE
        JX = MX
        IC = IYOFF + NS*(JX-1)
        DO 230 I = 1,NS
          ICI = IC + I
          X(ICI) = (X(ICI) + BETA2(I)*X(ICI-NS))
     1         + GAMMA2(I)*X(ICI-MXNS)
 230      CONTINUE
C-----------------------------------------------------------------------
C Add increment X to Z.
C-----------------------------------------------------------------------
      DO 300 I = 1,N
 300    Z(I) = Z(I) + X(I)
C
      IF (ITER .LT. ITMAX) GO TO 70
      RETURN
C------------  End of Subroutine GS  -----------------------------------
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
