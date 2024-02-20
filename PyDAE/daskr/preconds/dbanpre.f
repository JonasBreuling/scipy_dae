C-----------------------------------------------------------------------
C 
C             Preconditioner Routines for Banded Problems
C                          14 September 1995
C
C The following pair of subroutines -- DBANJA and DBANPS -- provides a
C general-purpose banded preconditioner matrix for use with the DDASPK
C solver, with the Krylov linear system method.  When using DDASPK to
C solve a problem G(t,y,y') = 0, whose iteration matrix (Jacobian)
C    J = dG/dy + c * dG/dy'  (c = scalar)
C is either banded or approximately equal to a banded matrix, these
C routines can be used to generate a banded approximation to J as the
C preconditioner and to solve the resulting banded linear system, in 
C conjunction with the Krylov method option (INFO(12) = 1) in DDASPK.
C
C Other than the user-supplied residual routine RES defining G(t,y,y'),
C the only other inputs required by these routines are the
C half-bandwidth parameters ML and MU of the approximate banded
C Jacobian.  If the system size is NEQ, the half-bandwidths are
C defined as integers between 0 and NEQ - 1 such that only elements
C with indices (i,j) satisfying
C    -ML .le. j - i .le. MU
C are to be retained in the preconditioner.  E.g., if ML = MU = 0, a
C diagonal matrix will be generated as the preconditioner.  The banded
C preconditioner is obtained by difference quotient approximations.  If
C the true problem Jacobian is not banded but is approximately equal to
C a matrix that is banded, the procedure used here will have the effect
C of lumping the elements outside of the band onto the elements within
C the band.
C
C To use these routines in conjunction with DDASPK, the user's calling
C program should include the following, in addition to setting the other
C DDASPK input parameters.
C
C (a) Dimension the array IPAR to have length at least 2, and load the
C     half-bandwidths into IPAR as
C       IPAR(1) = ML   and   IPAR(2) = MU
C     IPAR is used to communicate these parameters to DBANJA and DBANPS.
C     If the user program also uses IPAR for communication with RES,
C     that data should be located beyond the first 2 words of IPAR.
C
C (b) Include the names DBANJA and DBANPS in an EXTERNAL statement.
C     Set INFO(15) = 1 to indicate that a JAC routine exists.
C     Then in the call to DDASPK, pass the names DBANJA and DBANPS as
C     the arguments JAC and PSOL, respectively.
C
C (c) The DDASPK work arrays RWORK and IWORK must include segments WP
C     and IWP for use by DBANJA/DBANPS.  The lengths of these depend on
C     the problem size and half-bandwidths, as follows:
C       LWP =  length of RWORK segment WP = 
C                     (2*ML + MU + 1)*NEQ + 2*( (NEQ/(ML+MU+1)) + 1)
C       LIWP = length of IWORK segment IWP = NEQ
C     (Note the integer divide in LWP.)  Load these lengths in IWORK as
C       IWORK(27) = LWP 
C       IWORK(28) = LIWP
C     and include these values in the declared size of RWORK and IWORK.
C
C
C The DBANJA and DBANPS routines generate and solve the banded
C preconditioner matrix P within the preconditioned Krylov algorithm
C used by DDASPK when INFO(12) = 1.  P is generated and LU-factored 
C periodically during the integration, and the factors are used to 
C solve systems Px = b as needed.
C-----------------------------------------------------------------------


      SUBROUTINE DBANJA (RES, IRES, NEQ, T, Y, YPRIME, REWT, SAVR,
     *                  WK, H, CJ, WP, IWP, IER, RPAR, IPAR)
C
C***BEGIN PROLOGUE  DBANJA
C***DATE WRITTEN   891204   (YYMMDD)
C***REVISION DATE  900122
C***REVISION DATE  920929   CJ in RES call sequence
C***REVISION DATE  950914   Name change, minor revisions throughout
C***AUTHORS  L. R. Petzold, P. N. Brown, A. C. Hindmarsh, C. W. Ulrich
C            Numerical Mathematics Group
C            Lawrence Livermore National Laboratory
C            Livermore, CA 94551
C
C***DESCRIPTION
C
C Subroutine DBANJA generates a banded preconditioner matrix P that 
C approximates the DDASPK iteration matrix  J = dG/dy + CJ*dG/dy',
C where the DAE system is  G(t,y,y') = 0.  The band matrix P has
C half-bandwidths ML and MU.  It is computed by making (ML + MU + 1)
C calls to the user's RES routine and forming difference quotients,
C exactly as in the banded direct method option of DDASPK.
C DBANJA calls the LINPACK routine DGBFA to do an LU factorization of
C this matrix.
C
C The call sequence parameters have the following meanings.
C
C     RES      = External user-supplied subroutine to evaluate the
C                residuals.  See RES description in DDASPK prologue.
C     IRES     = Output flag set by RES.  See RES description in DDASPK.
C     NEQ      = Problem size.
C     T        = Independent variable t.
C     Y        = Array containing current dependent variables y.
C     YPRIME   = Array containing current derivative y'.
C     REWT     = Vector of reciprocal error weights, used here for 
C                computing increments.
C     SAVR     = Current residual evaluated at (T,Y,YPRIME).
C     WK       = Real work space of length NEQ.
C     H        = Current step size.
C     CJ       = Scalar proportional to 1/H.
C     WP       = Real work array for P etc.  On output, it contains
C                the LU decomposition of the banded approximation P.
C     IWP      = Integer work space for matrix pivot information.
C     IER      = Output flag, > 0 if P is singular, and 0 otherwise.
C     RPAR,IPAR= Real and integer arrays used for communication between
C                the calling program and external user routines.
C                IPAR(1) and IPAR(2) must contain ML and MU, resp.
C                RPAR is not used here.
C
C***ROUTINES CALLED
C   D1MACH, DGBFA, RES
C
C***END PROLOGUE  DBANJA
C
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
      EXTERNAL RES
      DIMENSION Y(*), YPRIME(*), SAVR(*), REWT(*), WK(*)
      DIMENSION WP(*), IWP(*), RPAR(*), IPAR(*)
C
C Set band parameters.
      ML = IPAR(1)
      MU = IPAR(2)
      MBAND = ML + MU + 1
      MBA = MIN(MBAND, NEQ)
      MEBAND = MBAND + ML
      MEB1 = MEBAND - 1
C
C Set the machine unit roundoff UROUND and SQRT(UROUND), used to 
C set increments in the difference quotient procedure. 
      UROUND = D1MACH(4)
      SQUR = SQRT(UROUND)
C
C Set pointers into WP.  LENP is the length of the segment for P.
C Following that are two segments of size (NEQ/MBAND), with offsets
C ISAVE and IPSAVE, for temporary storage of Y and YPRIME elements. 
      LENP = (2*ML+MU+1)*NEQ
      MSAVE = (NEQ/MBAND) + 1
      ISAVE = LENP
      IPSAVE = ISAVE + MSAVE
C
C Initialize error flags.
      IER = 0
      IRES = 0
C
C Generate the banded approximate iteration matrix P using
C difference quotients on the results of calls to RES.
C
      DO 40 J = 1,MBA
        DO 10 N = J,NEQ,MBAND
          K= (N-J)/MBAND + 1
          WP(ISAVE+K) = Y(N)
          WP(IPSAVE+K) = YPRIME(N)
          DEL = SQUR*MAX(ABS(Y(N)), ABS(H*YPRIME(N)), ABS(1.D0/REWT(N)))
          DEL = SIGN(DEL, H*YPRIME(N))
          DEL = (Y(N) + DEL) - Y(N)
          Y(N) = Y(N) + DEL
          YPRIME(N) = YPRIME(N) + CJ*DEL
 10       CONTINUE
        CALL RES (T, Y, YPRIME, CJ, WK, IRES, RPAR, IPAR)
        IF (IRES .LT. 0) RETURN
        DO 30 N = J,NEQ,MBAND
          K = (N-J)/MBAND + 1
          Y(N) = WP(ISAVE+K)
          YPRIME(N) = WP(IPSAVE+K)
          DEL = SQUR*MAX(ABS(Y(N)), ABS(H*YPRIME(N)), ABS(1.D0/REWT(N)))
          DEL = SIGN(DEL, H*YPRIME(N))
          DEL = (Y(N) + DEL) - Y(N)
          DELINV = 1.0D0/DEL
          I1 = MAX(1, N-MU)
          I2 = MIN(NEQ, N+ML)
          II = N*MEB1 - ML
          DO 20 I = I1,I2
 20         WP(II+I) = (WK(I) - SAVR(I))*DELINV
 30       CONTINUE
 40     CONTINUE
C
C Do LU decomposition of the band matrix P.
C
      CALL DGBFA (WP, MEBAND, NEQ, ML, MU, IWP, IER)
      RETURN
C
C------------  End of Subroutine DBANJA  -------------------------------
      END

      SUBROUTINE DBANPS (NEQ, T, Y, YPRIME, SAVR, WK, CJ, WGHT,
     *                   WP, IWP, B, EPLIN, IER, RPAR, IPAR)
C
C***BEGIN PROLOGUE  DBANPS
C***DATE WRITTEN   891204   (YYMMDD)
C***REVISION DATE  900110   (YYMMDD)
C***REVISION DATE  950914   Name change, minor revisions throughout
C***AUTHORS  L. R. Petzold, P. N. Brown, A. C. Hindmarsh, C. W. Ulrich
C            Numerical Mathematics Group
C            Lawrence Livermore National Laboratory
C            Livermore, CA 94551
C
C***DESCRIPTION
C
C Subroutine DBANPS uses the factors produced by DBANJA to solve linear
C systems P x = b for the banded preconditioner P and a given vector b.
C It calls the LINPACK routine SGBSL for this.
C
C The call sequence parameters have the following meanings.
C
C     NEQ      = Problem size.
C     T        = Independent variable t (not used).
C     Y        = Array containing current dependent vars. (not used).
C     YPRIME   = Array containing current derivative (not used).
C     SAVR     = Current residual evaluated at (T,Y,YPRIME) (not used).
C     WK       = Real work space of length NEQ (not used).
C     CJ       = Scalar proportional to 1/H (H = step size) (not used).
C     WGHT     = Vector of error weights for computing norms (not used).
C     WP       = Real work array containing the LU decomposition of P.
C     IWP      = Integer array containing matrix pivot information.
C     B        = Right-hand side vector on input; solution on output.
C     EPLIN    = tolerance on linear system solver (not used).
C     IER      = Output error flag (not used; assumed 0 on input). 
C     RPAR,IPAR= Real and integer arrays used for communication between
C                the calling program and external user routines.
C                IPAR(1) and IPAR(2) must contain ML and MU, resp.
C                RPAR is not used here.
C
C***ROUTINES CALLED
C   DGBSL
C
C***END PROLOGUE  DBANPS
C
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
      DIMENSION B(*),WP(*),IWP(*),RPAR(*),IPAR(*)
C
      ML = IPAR(1)
      MU = IPAR(2)
      MEBAND = 2*ML + MU + 1
      CALL DGBSL (WP, MEBAND, NEQ, ML, MU, IWP, B, 0)
      RETURN
C
C------------  End of Subroutine DBANPS  -------------------------------
      END
