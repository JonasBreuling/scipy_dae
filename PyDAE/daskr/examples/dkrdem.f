C DKRDEM program: DDASKR demonstration program
C-----------------------------------------------------------------------
C
C***BEGIN PROLOGUE  DKRDEM
C***DATE WRITTEN   020813     (YYMMDD)
C***REVISION DATE  021217   Added JROOT output value in Problem 2.
C***AUTHORS  Linda R. Petzold and Alan C. Hindmarsh
C             LAWRENCE LIVERMORE NATIONAL LABORATORY
C             LIVERMORE, CA    94550
C***PURPOSE  Quick check for routine DDASKR.
c***DESCRIPTION
C       Demonstration program for DDASKR.
C       This version is in double precision.
C
C       DDASKR is used to solve two simple problems,
C       one nonstiff and one intermittently stiff.
C       If the errors are too large, or other difficulty occurs,
C       a warning message is printed.  All output is on unit LUN.
C
C       To run the demonstration problems with full printing,
C       set KPRINT = 3.
C
C***REFERENCES  (NONE)
C***ROUTINES CALLED  DDASKR,RES1,RT1,RES2,JAC2,RT2
C***END PROLOGUE DKRDEM
C
C-----------------------------------------------------------------------
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      EXTERNAL RES1, RT1, RES2, JAC2, RT2 
      INTEGER IDID, IOUT, IWORK, JROOT, INFO, JTYPE,
     *   KROOT, LENIW, LENRW, LIW, LRW, LUN, NEQ, NERR, NRT,
     *   NRE, NREA, NRTE, NJE, NST
      DOUBLE PRECISION ATOL, ER, ERO, ERRT, RTOL, RWORK,
     *   T, TOUT, TZERO, Y, YT
      DIMENSION Y(2), YPRIME(2), RTOL(2), ATOL(2),
     *   RWORK(100), IWORK(100), JROOT(2), INFO(20)
      COMMON /LOCAL/NEQ
C
      LUN = 6
      KPRINT = 3
      IPASS = 1
      NERR = 0
C-----------------------------------------------------------------------
C First problem.
C The initial value problem is..
C   dY/dT = ((2*LOG(Y) + 8)/T - 5)*Y,  Y(1) = 1,  1 .LE. T .LE. 6
C The solution is  Y(T) = EXP(-T**2 + 5*T - 4), YPRIME(1) = 3
C The two root functions are..
C   R1(T,Y,Y') = dY/dT  (with root at T = 2.5),
C   R2(T,Y,Y') = LOG(Y) - 2.2491  (with roots at T = 2.47 and 2.53)
C-----------------------------------------------------------------------
C Set all input parameters and print heading.
      DO 10 I = 1,20
 10      INFO(I) = 0
      NEQ = 1
      T = 1.0D0
      Y(1) = 1.0D0
      TOUT = 2.0D0
      RTOL(1) = 0.0D0
      ATOL(1) = 1.0D-6
      LRW = 100
      LIW = 100
      IDID = 0
C
C Set INFO(11) = 1 if DDASKR is to compute the initial YPRIME, and
C generate an initial guess for YPRIME.  Otherwise, set INFO(11) = 0
C and supply the correct initial value for YPRIME.
C
      INFO(11) = 0
      YPRIME(1) = 3.0D0
C
C Note: JTYPE indicates the Jacobian type:
C       JTYPE = 1 ==> Jacobian is dense and user-supplied
C       JTYPE = 2 ==> Jacobian is dense and computed internally
C
      JTYPE = 2
      INFO(5) = 2 - JTYPE
      NRT = 2
      IF (KPRINT .GE. 2) WRITE (LUN,110) RTOL(1),ATOL(1),JTYPE
 110  FORMAT(' DKRDEM: Demonstration Program for DDASKR'///
     1       ' Problem 1..'//
     2       ' Problem is  dY/dT = ((2*LOG(Y)+8)/T - 5)*Y,  Y(1) = 1'/
     3       ' Solution is  Y(T) = EXP(-T**2 + 5*T - 4)'/
     4       ' Root functions are..'/
     5       ' R1 = dY/dT  (root at T = 2.5)'/
     6       ' R2 = LOG(Y) - 2.2491  (roots at T = 2.47 and T = 2.53)'/
     7       ' RTOL =',E10.1,'   ATOL =',E10.1,'   JTYPE =',I3/)
     8      
C
C Call DDASKR in loop over TOUT values = 2, 3, 4, 5, 6.
      ERO = 0.0D0
      DO 180 IOUT = 1,5
C
 120    CALL DDASKR (RES1,NEQ,T,Y,YPRIME,TOUT,
     *               INFO,RTOL,ATOL,IDID,
     *               RWORK,LRW,IWORK,LIW,RPAR,IPAR,JDUM,PSDUM,
     *               RT1,NRT,JROOT)
C
C Print Y and error in Y, and print warning if error too large.
        YT = EXP(-T*T + 5.0D0*T - 4.0D0)
        ER = Y(1) - YT
        IF (KPRINT .GT. 2) WRITE (LUN,130) T,Y(1),ER
 130    FORMAT(' At t =',E15.7,5X,'y =',E15.7,5X,'error =',E12.4)
        IF (IDID .LT. 0) GO TO 185
        ER = ABS(ER)/ATOL(1)
        ERO = MAX(ERO,ER)
        IF (ER .LT. 1000.0D0) GO TO 140
        IF (KPRINT .GE. 2) THEN
          IPASS = 0
          WRITE (LUN,135)
        ENDIF
 135    FORMAT(//' WARNING.. Error exceeds 1000 * tolerance'//)
        NERR = NERR + 1
 140    CONTINUE
        IF (IDID .NE. 5) GO TO 175
C
C If a root was found, write results and check root location.
C Then return to DDASKR to continue the integration.
        IF (KPRINT .GT. 2) WRITE (LUN,150) T,JROOT(1),JROOT(2)
 150    FORMAT(/'      Root found at t =',E15.7,5X,'JROOT =',2I5)
        IF (JROOT(1) .NE. 0) ERRT = T - 2.5D0
        IF (JROOT(2) .NE. 0 .AND. T .LE. 2.5D0) ERRT = T - 2.47D0
        IF (JROOT(2) .NE. 0 .AND. T .GT. 2.5D0) ERRT = T - 2.53D0
        IF (KPRINT .GT. 2) WRITE (LUN,160) ERRT
 160    FORMAT('      Error in t location of root is',E12.4/)
        IF (ABS(ERRT) .LT. 1.0D-3) GO TO 170
        IF (KPRINT .GE. 2) THEN
           IPASS = 0
           WRITE (LUN,165)
        ENDIF
 165    FORMAT(//' WARNING.. Root error exceeds 1.0D-3'//)
        NERR = NERR + 1
 170    CONTINUE
        GO TO 120
C
C If no root found, increment TOUT and loop back.
 175    TOUT = TOUT + 1.0D0
 180    CONTINUE
C
C Problem complete.  Print final statistics.
 185  CONTINUE
      IF (IDID .LT. 0) NERR = NERR + 1
      NST = IWORK(11)
      NRE = IWORK(12)
      NJE = IWORK(13)
      NRTE = IWORK(36)
      LENRW = 0
      LENIW = 0
      NREA = NRE
      IF (JTYPE .EQ. 2) NRE = NRE + NEQ*NJE
C
      IF (KPRINT .GT. 2) WRITE (LUN,190) NST,NRE,NREA,NJE,NRTE,ERO
 190  FORMAT(/' Final statistics for this run..'/
     *       ' number of steps =',I5/
     *       ' number of Gs    =',I5/
     *       ' (excluding Js)  =',I5/
     *       ' number of Js    =',I5/
     *       ' number of Rs    =',I5/
     *       ' error overrun   =',E10.2)
C
C-----------------------------------------------------------------------
C Second problem (Van Der Pol oscillator).
C The initial value problem is..
C   dY1/dT = Y2,  dY2/dT = 100*(1 - Y1**2)*Y2 - Y1,
C   Y1(0) = 2,  Y2(0) = 0,  0 .LE. T .LE. 200
C   Y1PRIME(0) = 0, Y2PRIME(0) = -2
C The root function is  R(t,Y,Y') = Y1.
C An analytic solution is not known, but the zeros of Y1 are known
C to 15 figures for purposes of checking the accuracy.
C-----------------------------------------------------------------------
C
C Reset INFO array
C
      DO 195 I = 1,20
 195     INFO(I) = 0
C
C Set tolerance parameters and print heading.
C Note that INFO(2) is set to 1, indicating that RTOL and ATOL
C are arrays.  Each entry of RTOL and ATOL must then be defined.
C
      INFO(2) = 1
      RTOL(1) = 1.0D-6
      RTOL(2) = 1.0D-6
      ATOL(1) = 1.0D-6
      ATOL(2) = 1.0D-4
      IF (KPRINT .GE. 2) WRITE (LUN,200) RTOL(1),ATOL(1),ATOL(2)
 200  FORMAT(/80('-')//' Problem 2.. Van Der Pol oscillator'//
     *       ' Problem is dY1/dT = Y2,  dY2/dT = 100*(1-Y1**2)*Y2 - Y1'/
     *       '            Y1(0) = 2,  Y2(0) = 0'/
     *       ' Root function is  R(T,Y,YP) = Y1'/
     *       ' RTOL =',E10.1,'   ATOL =',2E10.1)
C
C Note: JTYPE indicates the Jacobian type:
C       JTYPE = 1 ==> Jacobian is dense and user-supplied
C       JTYPE = 2 ==> Jacobian is dense and computed internally
C
C Loop over JTYPE = 1, 2.  Set remaining parameters and print JTYPE.
      DO 290 JTYPE = 1,2
C
C     Set INFO(1) = 0 to indicate start of a new problem
C     Set INFO(5) = 2-JTYPE to tell DDASKR the Jacobian type.
C
      INFO(1) = 0
      INFO(5) = 2-JTYPE
      NEQ = 2
      T = 0.0D0
      Y(1) = 2.0D0
      Y(2) = 0.0D0
      YPRIME(1) = 0.D0
      YPRIME(2) = -2.0D0
      TOUT = 20.0D0
      NRT = 1
      IF (KPRINT .GT. 2) WRITE (LUN,210) JTYPE
 210  FORMAT(/80('.')//' Solution with JTYPE =',I2/)
C
C Call DDASKR in loop over TOUT values = 20, 40, ..., 200.
      DO 270 IOUT = 1,10
C
 220    CALL DDASKR (RES2,NEQ,T,Y,YPRIME,TOUT,
     *               INFO,RTOL,ATOL,IDID,
     *               RWORK,LRW,IWORK,LIW,RPAR,IPAR,JAC2,PSDUM,
     *               RT2,NRT,JROOT)
C
C Print Y1 and Y2.
        IF (KPRINT .GT. 2) WRITE (LUN,230) T,Y(1),Y(2)
 230    FORMAT(' At t =',E15.7,5X,'y1 =',E15.7,5X,'y2 =',E15.7)
        IF (IDID .LT. 0) GO TO 275
        IF (IDID .NE. 5) GO TO 265
C
C If a root was found, write results and check root location.
C Then return to DDASKR to continue the integration.
        IF (KPRINT .GT. 2) WRITE (LUN,240) T,JROOT(1)
 240    FORMAT(/'      Root found at t =',E15.7,'  JROOT =',I3)
        KROOT = INT(T/81.2D0 + 0.5D0)
        TZERO = 81.17237787055D0 + DFLOAT(KROOT-1)*81.41853556212D0
        ERRT = T - TZERO
        IF (KPRINT .GT. 2) WRITE (LUN,250) ERRT
 250    FORMAT('      Error in t location of root is',E12.4/)
        IF (ERRT .LT. 1.0D0) GO TO 260
        IF (KPRINT .GE. 2) THEN
          IPASS = 0
          WRITE (LUN,255)
        ENDIF
 255    FORMAT(//' WARNING.. Root error exceeds 1.0'//)
        NERR = NERR + 1
 260    CONTINUE
        GO TO 220
C
C If no root found, increment TOUT and loop back.
 265    TOUT = TOUT + 20.0D0
 270    CONTINUE
C
C Problem complete.  Print final statistics.
 275  CONTINUE
      IF (IDID .LT. 0) NERR = NERR + 1
      NST = IWORK(11)
      NRE = IWORK(12)
      NJE = IWORK(13)
      NRTE = IWORK(36)
      LENRW = 0
      LENIW = 0
      NREA = NRE
      IF (JTYPE .EQ. 2) NRE = NRE + NEQ*NJE
      IF (KPRINT .GE. 2) WRITE (LUN,280) NST,NRE,NREA,NJE,NRTE
 280  FORMAT(/' Final statistics for this run..'/
     *       ' number of steps =',I5/
     *       ' number of Gs    =',I5/
     *       ' (excluding Js)  =',I5/
     *       ' number of Js    =',I5/
     *       ' number of Rs    =',I5)
 290  CONTINUE
C
C
      IF (KPRINT .GE. 2) WRITE (LUN,300) NERR
 300  FORMAT(/80('-')//' Number of errors encountered =',I3)
C
      IF (NERR .GT. 0) IPASS = 0
      IF ((IPASS .EQ. 1) .AND. (KPRINT .GT. 1)) WRITE (LUN,700)
      IF ((IPASS .EQ. 0) .AND. (KPRINT .NE. 0)) WRITE (LUN,800)
 700  FORMAT (/,' **********DDASKR passed all tests**********')
 800  FORMAT (/,' **********DDASKR failed some tests*********')
      STOP
      END
C
      SUBROUTINE RES1(T,Y,YPRIME,CJ,DELTA,IRES,RPAR,IPAR)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      COMMON /LOCAL/NEQ
      DIMENSION Y(NEQ), YPRIME(NEQ), DELTA(NEQ)
C
C     Check Y to make sure that it is valid input.
C     If Y is less than or equal to zero, this is invalid input.
C
      IF (Y(1) .LE. 0.0D0) THEN
         IRES = -1
         RETURN
      ELSE
C
C        Call F1 to obtain F(T,Y)
C
         CALL F1(NEQ,T,Y,DELTA)
C
C        Form G = Y' - F(T,Y)
C
         DO 10 I = 1,NEQ
            DELTA(I) = YPRIME(I) - DELTA(I)
 10      CONTINUE
      ENDIF
C
      RETURN
      END
C
      SUBROUTINE F1 (NEQ, T, Y, YDOT)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER NEQ
      DOUBLE PRECISION T, Y, YDOT
      DIMENSION Y(1), YDOT(1)
      YDOT(1) = ((2.0D0*LOG(Y(1)) + 8.0D0)/T - 5.0D0)*Y(1)
      RETURN
      END
C
      SUBROUTINE RT1 (NEQ, T, Y, YP, NRT, RVAL, RPAR, IPAR)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER NEQ, NRT
      DOUBLE PRECISION T, Y, RVAL
      DIMENSION RVAL(2)
      RVAL(1) = YP
      RVAL(2) = LOG(Y) - 2.2491D0
      RETURN
      END
C
      SUBROUTINE RES2(T,Y,YPRIME,CJ,DELTA,IRES,RPAR,IPAR)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      COMMON /LOCAL/NEQ
      DIMENSION Y(NEQ), YPRIME(NEQ), DELTA(NEQ)
C
C     CALL F2 TO OBTAIN F(T,Y)
C
      CALL F2(NEQ,T,Y,DELTA)
C
C     FORM G = Y' - F(T,Y)
C
      DO 10 I = 1,NEQ
 10      DELTA(I) = YPRIME(I) - DELTA(I)
C
      RETURN
      END
C
      SUBROUTINE F2 (NEQ, T, Y, YDOT)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER NEQ
      DOUBLE PRECISION T, Y, YDOT
      DIMENSION Y(2), YDOT(2)
      YDOT(1) = Y(2)
      YDOT(2) = 100.0D0*(1.0D0 - Y(1)*Y(1))*Y(2) - Y(1)
      RETURN
      END
C
      SUBROUTINE JAC2 (T, Y, YPRIME, PD, CJ, RPAR, IPAR)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER NEQ,  NROWPD
      DOUBLE PRECISION T, Y, PD
      PARAMETER (NROWPD=2)
      DIMENSION Y(2), PD(NROWPD,2)
      COMMON /LOCAL/NEQ
C
C First define the Jacobian matrix for the right-hand side
C of the ODE: Y' = F(T,Y) , i.e. dF/dY.
C
      PD(1,1) = 0.0D0
      PD(1,2) = 1.0D0
      PD(2,1) = -200.0D0*Y(1)*Y(2) - 1.0D0
      PD(2,2) = 100.0D0*(1.0D0 - Y(1)*Y(1))
C
C Next update the Jacobian with the right-hand side to form the
C DAE Jacobian: CJ*dR/dY' + dR/dY = CJ*I - dF/dY.
C
      PD(1,1) = CJ - PD(1,1)
      PD(1,2) =    - PD(1,2)
      PD(2,1) =    - PD(2,1)
      PD(2,2) = CJ - PD(2,2)
C
      RETURN
      END
C
      SUBROUTINE RT2 (NEQ, T, Y, YP, NRT, RVAL, RPAR, IPAR)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER NEQ, NRT
      DOUBLE PRECISION T, Y, RVAL
      DIMENSION Y(2), YP(2), RVAL(1)
      RVAL(1) = Y(1)
      RETURN
      END
