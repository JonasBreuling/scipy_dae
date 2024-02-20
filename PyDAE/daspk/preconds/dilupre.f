C-----------------------------------------------------------------------
C 
C             Preconditioner Routines for Sparse Problems
C                          13 December 2000
C
C The following triple of subroutines -- DSPSETUP, DJACILU and
C DPSOLILU -- provides a general-purpose sparse incomplete LU (ILU)
C preconditioner matrix for use with the DDASPK solver, with the Krylov
C linear system method.  When using DDASPK to solve a problem 
C G(t,y,y') = 0, whose iteration matrix (Jacobian)
C    J = dG/dy + c * dG/dy'  (c = scalar)
C is a general sparse matrix, these routines can be used to generate
C an approximation to J as the preconditioner and to solve the resulting
C sparse linear system, in conjunction with the Krylov method option
C (INFO(12) = 1) in DDASPK.
C
C The incomplete LU factorization is achieved via one of two routines
C from the SPARSKIT library available from Yousef Saad at the
C University of Minnesota.  The two routines are ILUT and ILUTP.
C See below for detailed descriptions of these routines.
C
C Descriptions of the above routines are as follows:
C
C DSPSETUP   - Setup routine for the preconditioner.  This routine
C              checks the user input for illegal values, and calculates
C              the minimum length needed for preconditioner workspace
C              arrays in DDASPK.
C
C DJACILU   -  This routine is a version of JAC for DDASPK.
C              It uses finite-differences to calculate the Jacobian
C              matrix J in sparse format, and then performs an
C              incomplete LU factorization using either ILUT or ILUTP.
C              DJACILU must be declared EXTERNAL in the user's main
C              program and passed through to DDASPK.
C
C DPSOLILU   - This routine is a version of PSOL for DDASPK.
C              It uses the incomplete factorization calculated in
C              DJACILU.  DPSOLILU must be declared EXTERNAL in the
C              user's main program and passed through to DDASPK.
C
C
C The routines ILUT and ILUTP are part of the SPARSKIT library,
C and are contained in the file 'dsparsk.f'.  ILUT performs an
C incomplete LU factorization of a sparse matrix using a dual
C thresholding technique based on a drop tolerance (TOLILUT below)
C and a level of fill-in parameter (LFILILUT).  LFILILUT controls
C the amount of fill-in allowed in the factorization (limited to a
C maximum of 2*LFILILUT*NEQ, but normally much less).  Increasing
C LFILILUT will generally make the ILU factorization more accurate.
C TOLILUT also controls the accuracy of the ILU factorization via
C a drop tolerance based on element size.  Descreasing TOLILUT
C will increase the amount of fill-in and make for a more accurate
C factorization.  ILUTP is a variant of ILUT that in addition performs
C pivoting based on a tolerance ratio PERMTOL.
C
C An important aspect of using incomplete factorization techniques
C is that of reordering the rows and columns in the Jacobian matrix
C J before performing the ILU.  In this package, this is accomplished
C via the parameter IREORDER, which when equal to 1 performs
C a reverse Cuthill-McKee (RCM) reordering before performing the
C ILU factorization.  Based on the limited amount of testing done so
C far, RCM seems the best overall choice.  It is possible to include
C a different reordering technique if desired.
C
C To use these routines in conjunction with DDASPK, the user's calling
C program should include the following, in addition to setting the other
C DDASPK input parameters.
C
C (a) Dimension the array IPAR to have length at least 30, and load the
C     following parameters into IPAR as
C
C      IPAR(1)  = ML        - The lower bandwidth used in calculating
C                             the approximate Jacobian matrix.
C      IPAR(2)  = MU        - The upper bandwidth used in calculating
C                             the approximate Jacobian matrix.
C      IPAR(3)  = LENPFAC   - The average number of nonzeros in a
C                             row of the Jacobian matrix.  The
C                             maximum of nonzeros allowed in the
C                             Jacobian matrix is NNZMX = LENPFAC*NEQ.
C                             LENPFAC must be .GE. 2.
C      IPAR(4)  = LENPLUFAC - The average amount of fill-in per row
C                             in the factored Jacobian matrix.  The
C                             maximum number of nonzeros allowed
C                             in the factored Jacobian matrix is
C                             LENPLUMX = NNZMX + LENPLUFAC*NEQ.
C                             LENPLUFAC must be .GE. 2.
C      IPAR(5)  = IPREMETH  - Preconditioner type flag.
C                             =1 means ILUT preconditioner used
C                             =2 means ILUTP preconditioner used
C      IPAR(6)  = LFILILUT  - Fill-in parameter for ILUT and ILUTP.
C                             The largest LFILILUT elements per row
C                             of the L and U factors are kept.  Each
C                             row of L and U will have a maximum of
C                             LFILILUT elements in addition to
C                             their original number of nonzero
C                             elements.
C      IPAR(7)  = IREORDER  - Reordering flag.
C                             =0 means that no reordering of the
C                                rows and columns of the Jacobian
C                                matrix is performed before the
C                                incomplete factorization is performed.
C                             =1 means that a reverse Cuthill-McKee
C                                (RCM) reordering is performed.
C      IPAR(8)  = ISRNORM   - Row norm flag.
C                             =1 means that row norms of the Jacobian
C                                matrix are computed and used as
C                                row scalings when solving the
C                                preconditioner linear system P*x=b.
C                             =0 means no row norm scaling is used.
C      IPAR(9)  = NORMTYPE  - Type of row norm scaling for ISRNORM
C                             =0 means max-norm is used.
C                             =1 means 1-norm is used.
C                             =2 means 2-norm is used.
C      IPAR(10) = JACOUT    - Output Jacobian flag.
C                             =1 means write the calculated Jacobian
C                                matrix along with the initial value of
C                                the residual G to a file pointed to by 
C                                the logical unit number in IPAR(29).
C                                This is done after any reordering and
C                                scaling is performed.  The user must
C                                have attached the unit number to a
C                                file before calling DDASPK.  The
C                                integration is then halted by setting
C                                IRES = -2 (and a false message about
C                                failure to initialize is printed).
C                             =0 means no Jacobian matrix output.
C                             The matrix and initial residual G are
C                             written in Boeing-Harwell format.
C      IPAR(11) = JSCALCOL  - Flag for scaling columns of the
C                             Jacobian matrix by the inverses of the
C                             elements in the EWT array.
C                             =0 means no scaling.
C                             =1 means perform scaling.
C
C      IPAR(21:28)          - Used to hold pointer information.
C      IPAR(29)             - Logical unit number to write matrix output
C                             file on.  Only needed when JACOUT = 1.
C      IPAR(30)             - On return from DDASPK, this value
C                             holds the number of calls to the
C                             RES routine used in the preconditioner
C                             evaluations.
C
C (b) Dimension the array RPAR to have length at least 2, and load the
C     following parameters into RPAR as
C
C      RPAR(1)  = TOLILUT   - Drop tolerance for use by ILUT and ILUTP.
C                             TOLILUT must be .ge. 0.  Larger values
C                             of TOLILUT cause less fill-in.  Good 
C                             values range from 0.001 to 0.01.
C      RPAR(2)  = PERMTOL   - Tolerance ratio used in determining column
C                             pivoting by ILUTP.  PERMTOL must be
C                             .ge. 0.  Good values are from 0.1 to
C                             0.01.  Two columns are permuted only if
C                             A(i,j)*PERMTOL .GT. A(i,i).
C
C     The two paramaters TOLILUT and LFILILUT gives the user a great
C     deal of flexibility: one can use TOLILUT=0 to get a strategy
C     based on keeping the largest elements in each row of L and U.
C     Taking TOLILUT .NE. 0 but LFILILUT=NEQ will give the usual 
C     threshold strategy (however, fill-in is then unpredictable).
C
C (c) Include the names DJACILU and DPSOLILU in an EXTERNAL statement.
C     Set INFO(15) = 1 to indicate that a JAC routine exists.
C     Then in the call to DDASPK, pass the names DJACILU and DPSOLILU
C     as the arguments JAC and PSOL, respectively.
C
C (d) The DDASPK work arrays RWORK and IWORK must include segments WP
C     and IWP for use by DJACILU/DPSOLILU.  The lengths of these depend
C     on the problem size, half-bandwidths, and other parameters 
C     as follows:
C       LWP =  length of RWORK segment WP = 
C              2*LENPFAC*NEQ + LENPLUFAC*NEQ + ISRNORM*NEQ + NEQ
C       LIWP = length of IWORK segment IWP =
C              3*NEQ+1 + 3*LENPFAC*NEQ + 2*LENPLUFAC*NEQ
C                 + 2*IREORDER*NEQ + 2*(IPREMETH-1)*NEQ
C     Load these lengths in IWORK as
C       IWORK(27) = LWP 
C       IWORK(28) = LIWP
C     and include these values in the declared size of RWORK and IWORK.
C
C
C The DJACILU and DPSOLILU routines generate and solve the sparse
C preconditioner matrix P within the preconditioned Krylov algorithm
C used by DDASPK when INFO(12) = 1.  P is generated and ILU-factored 
C periodically during the integration, and the factors are used to 
C solve systems Px = b as needed.
C-----------------------------------------------------------------------

      SUBROUTINE DSPSETUP (NEQ, LWP, LIWP, RPAR, IPAR, IERR,
     .                     LWP_MIN, LIWP_MIN)

C ... Version of 12-12-00

C ... Calculate storage needed for ILU decomposition of the Jacobian
C     matrix for use as a preconditioner by the DDASPK solver.
C     Also, check for illegal input.

      IMPLICIT NONE

C ... Input arguments:
      INTEGER NEQ      ! total number of equations
      REAL*8  RPAR(*)  ! user real workspace
      INTEGER IPAR(*)  ! user integer workspace
      INTEGER LWP      ! current length of WP for DDASPK
      INTEGER LIWP     ! current length of IWP for DDASPK

C ... Output arguments:
      INTEGER IERR     ! error flag (0 means success, else failure)
                       ! IERR between 1 and 11, means there's an
                       ! illegal value for IPAR(IERR).
                       ! IERR = 12 means IPAR(29) is illegal.
                       ! IERR = 21 means RPAR(1) is illegal.
                       ! IERR = 22 means RPAR(2) is illegal.
                       ! IERR = 30 means more WP length is needed.
                       ! IERR = 31 means more IWP length is needed.
      INTEGER LWP_MIN  ! minimum WP length needed.
      INTEGER LIWP_MIN ! minimum IWP length needed.

C ... Local variables:
      INTEGER LBW, UBW, LENPLUMX, LJAC, LJACI, LJACJ,
     .        LROWNRMS, LRWK1, LIWK1
      INTEGER LENPFAC, LENPLUFAC, LFILILUT, IPREMETH, NEQP1, NNZMX
      INTEGER LPLU, LJU, LJLU, LPERM, LQPERM, LLEVELS, LMASK
      INTEGER ISRNORM  !=1 causes row normalization of JAC.
      INTEGER NORMTYPE !=0,1,2 for max-norm, 1-norm, or
                       ! 2-norm row scaling
      INTEGER IREORDER !=1 causes row and column reordering of JAC.
      INTEGER JACOUT   !=1 causes the Jacobian matrix and SAVR to
                       !   be written to a file and then exit with
                       !   ierr = 1 to signal a stop to DDASPK.
      INTEGER JSCALCOL !=1 causes the columns of the Jacobian matrix
                       !   to be scaled by EWT-inverse
      REAL*8 TOLILUT, PERMTOL

C ... Load values from IPAR and RPAR.  Check for illegal values.
      LBW = IPAR(1)          ! LBW must be .gt. 0
      IF (LBW .LE. 0) THEN
         IERR = 1
         RETURN
      ENDIF
      UBW = IPAR(2)          ! UBW must be .gt. 0
      IF (UBW .LE. 0) THEN
         IERR = 2
         RETURN
      ENDIF
      LENPFAC = IPAR(3)      ! LENPFAC must be .ge. 2
      IF (LENPFAC .LE. 1) THEN
         IERR = 3
         RETURN
      ENDIF
      LENPLUFAC = IPAR(4)    ! LENPLUFAC must be .ge. 2
      IF (LENPLUFAC .LE. 1) THEN
         IERR = 4
         RETURN
      ENDIF
      IPREMETH = IPAR(5)     ! IPREMETH must be .eq. 1 or 2 currently
      IF (IPREMETH .NE. 1 .AND. IPREMETH .NE. 2) THEN
         IERR = 5
         RETURN
      ENDIF
      LFILILUT = IPAR(6)     ! LFILILUT must be .ge. 0
      IF (LFILILUT .LT. 0) THEN
         IERR = 6
         RETURN
      ENDIF
      IREORDER = IPAR(7)     ! IREORDER must be 0 or 1
      IF ((IREORDER .LT. 0) .OR. (IREORDER .GT. 1)) THEN
         IERR = 7
         RETURN
      ENDIF
      ISRNORM = IPAR(8)      ! ISRNORM must be 0 or 1
      IF ((ISRNORM .LT. 0) .OR. (ISRNORM .GT. 1)) THEN
         IERR = 8
         RETURN
      ENDIF
      NORMTYPE = IPAR(9)     ! NORMTYPE must be 0, 1, or 2
      IF ((NORMTYPE .LT. 0) .OR. (NORMTYPE .GT. 2)) THEN
         IERR = 9
         RETURN
      ENDIF
      JACOUT = IPAR(10)      ! JACOUT must be 0 or 1
      IF ((JACOUT .LT. 0) .OR. (JACOUT .GT. 1)) THEN
         IERR = 10
         RETURN
      ENDIF
      JSCALCOL = IPAR(11)      ! JSCALCOL must be 0 or 1
      IF ((JSCALCOL .LT. 0) .OR. (JSCALCOL .GT. 1)) THEN
         IERR = 11
         RETURN
      ENDIF
      IF (JACOUT .EQ. 1) THEN ! IPAR(29) must be .gt. 0
         IF (IPAR(29) .LE. 0) THEN
            IERR = 12
            RETURN
         ENDIF
      ENDIF
      TOLILUT = RPAR(1)      ! TOLILUT must be .ge. 0.0
      IF (TOLILUT .LT. 0.) THEN
         IERR = 21
         RETURN
      ENDIF
      IF (IPREMETH .EQ. 2) THEN
         PERMTOL = RPAR(2)      ! PERMTOL must be .ge. 0.0
         IF (PERMTOL .LT. 0.) THEN
            IERR = 22
            RETURN
         ENDIF
      ENDIF

C ... Calculate minimum work lengths for WP and IWP arrays.
      NEQP1 = NEQ + 1
      NNZMX = LENPFAC*NEQ
      LENPLUMX = NNZMX + LENPLUFAC*NEQ
C ... Set up pointers into WP
      LJAC = 1
      LROWNRMS = NNZMX + LJAC
      IF (ISRNORM .EQ. 1) THEN
         LPLU = LROWNRMS + NEQ
      ELSE
         LPLU = LROWNRMS
      ENDIF
      LRWK1 = LPLU + LENPLUMX
      LWP_MIN = LRWK1 + NEQ - 1
      IF (LWP .LT. LWP_MIN) THEN
         IERR = 30    ! more WP length needed.
         RETURN
      ENDIF
C ... Set up pointers into IWP
      LJACI = 1
      LJACJ = LJACI + NEQP1
      LJU = LJACJ + NNZMX
      LJLU = LJU + MAX(LENPLUMX,NEQP1)
      IF (IREORDER .NE. 0) THEN
         LPERM = LJLU + LENPLUMX
         LQPERM = LPERM + NEQ
         LIWK1 = LQPERM + NEQ
         LLEVELS = LJLU + NNZMX   ! assumes that LENPLUFAC >= 2.
         LMASK = LLEVELS + NEQ
      ELSE
         LPERM = 0
         LQPERM = 0
         LLEVELS = 0
         LMASK = 0
         LIWK1 = LJLU + LENPLUMX
      ENDIF
      LIWP_MIN = LIWK1 + 2*NEQ - 1
      IF (IPREMETH .EQ. 2) LIWP_MIN = LIWP_MIN + 2*NEQ
      IF (LIWP .LT. LIWP_MIN) THEN
         IERR = 31   ! more IWP length needed.
         RETURN
      ENDIF

      IERR = 0
      RETURN
C------------  End of Subroutine DSPSETUP  -----------------------------
      END

      SUBROUTINE DJACILU (RES, IRES, NEQ, T, Y, YPRIME, REWT, SAVR,
     .                    WK, H, CJ, WP, IWP, IERR, RPAR, IPAR)

C ... Version of 12-12-00

C ... Calculate ILU decomposition of the Jacobian matrix
C     for use as a preconditioner by the DDASPK solver.

      IMPLICIT NONE

C ... Input arguments:
      INTEGER NEQ        ! total number of equations
      REAL*8 T           ! independent variable t
      REAL*8 Y(NEQ)      ! most recent iterate of solution vector y
      REAL*8 YPRIME(NEQ) ! most recent iterate of solution vector y'
      REAL*8 SAVR(NEQ)   ! current residual evaluated at (T,Y,YPRIME)
      REAL*8 REWT(NEQ)   ! scale factors for Y and YPRIME
      EXTERNAL RES       ! function that evaluates residuals
      INTEGER IRES       ! error flag for RES routine
      REAL*8 WK(NEQ)     ! work space available to this subroutine
      REAL*8 H           ! current step size
      REAL*8 CJ          ! scalar proportional to 1/H
      REAL*8 RPAR(*)     ! user real workspace
      INTEGER IPAR(*)    ! user integer workspace

C ... Output arguments:
      REAL*8 WP(*)       ! matrix elements of ILU
      INTEGER IWP(*)     ! array indices for elements of ILU
      INTEGER NRE        ! number of RES calls needed to evaluate
                         ! Jacobian NRE is returned in IPAR(30)
      INTEGER IERR       ! error flag (0 means success, else failure)

C ... Local variables:
      REAL*8 TOLILUT, PERMTOL, SQRTN
      INTEGER I, LBW, UBW, LENPLUMX, LJAC, LJACI, LJACJ, LIPERM,
     .        LROWNRMS, LRWK1, LIWK1, IFMT
      INTEGER LENPFAC, LENPLUFAC, LFILILUT, IPREMETH, NEQP1, NNZMX
      INTEGER LPLU, LJU, LJLU, LPERM, LQPERM, LLEVELS, LMASK
      INTEGER ISRNORM  !=1 causes row normalization of Jac.
      INTEGER NORMTYPE !=0,1,2 for max-norm, 1-norm, or
                       ! 2-norm row scaling
      INTEGER IREORDER !=1 causes row and column reordering of Jac.
      INTEGER JACOUT   !=1 causes the Jacobian matrix and SAVR to
                       !   be written to a file and then exit with
                       !   IRES = -2 to signal a stop to DDASPK.
      INTEGER IUNIT    ! logical unit number to use when JACOUT .EQ. 1
      INTEGER JSCALCOL !=1 causes the columns of the Jacobian matrix
                       !   to be scaled by REWT-inverse
      CHARACTER*8 PMETH(4), PREMETH
      CHARACTER*72 TITLE
      CHARACTER*80 MSG
      SAVE
      DATA PMETH/'ILUT','ILUTP','ILU0','MILU0'/

C ... Zero out NRE counter
      NRE = 0

C ... Load values from IPAR and RPAR.
      LBW = IPAR(1)
      UBW = IPAR(2)
      LENPFAC = IPAR(3)
      LENPLUFAC = IPAR(4)
      IPREMETH = IPAR(5)
      LFILILUT = IPAR(6)
      IREORDER = IPAR(7)
      ISRNORM = IPAR(8)
      NORMTYPE = IPAR(9)
      JACOUT = IPAR(10)
      JSCALCOL = IPAR(11)
      TOLILUT = RPAR(1)
      PERMTOL = RPAR(2)
      PREMETH = PMETH(IPREMETH)

C ... Set pointers into the WP and IWP arrays.
      NEQP1 = NEQ + 1
      NNZMX = LENPFAC*NEQ
      LENPLUMX = NNZMX + LENPLUFAC*NEQ
C ... Set up pointers into WP
      LJAC = 1
      LROWNRMS = NNZMX + LJAC
      IF (ISRNORM .EQ. 1) THEN
         LPLU = LROWNRMS + NEQ
      ELSE
         LPLU = LROWNRMS
      ENDIF
      LRWK1 = LPLU + LENPLUMX
C ... Set up pointers into IWP
      LJACI = 1
      LJACJ = LJACI + NEQP1
      LJU = LJACJ + NNZMX
      LJLU = LJU + LENPLUMX

C ... Calculate Jacobian matrix.
      IERR = 0
      CALL DJCALC (NEQ, T, Y, YPRIME, SAVR, LBW, UBW, WK, REWT, RES,
     .             H, CJ, NNZMX, WP(LJAC), IWP(LJACJ), IWP(LJACI),
     .             WP(LPLU), IWP(LJLU), IWP(LJU), IPAR, RPAR,
     .             IRES, NRE, IERR)
      IF (IRES .LT. 0) RETURN
      IF (IERR .NE. 0) RETURN

C ... Save NRE value for user output.
      IPAR(30) = IPAR(30) + NRE
      
C ... Modify pointers into IWP
      LJLU = LJU + NEQP1
      IF (IREORDER .NE. 0) THEN
         LPERM = LJLU + LENPLUMX
         LQPERM = LPERM + NEQ
         LIWK1 = LQPERM + NEQ
         LLEVELS = LJLU + NNZMX   ! assumes that LENPLUFAC >= 2.
         LMASK = LLEVELS + NEQ
      ELSE
         LPERM = 0
         LQPERM = 0
         LLEVELS = 0
         LMASK = 0
         LIWK1 = LJLU + LENPLUMX
      ENDIF
      IF (PREMETH .EQ. 'ILUTP') THEN
         LIPERM = LIWK1 + 2*NEQ
      ELSE
         LIPERM = LIWK1
      ENDIF
C ... Multiply Jacobian columns by inverse of scaling vector REWT.
C     In PSOLILU, the WGHT array equals REWT/SQRT(NEQ), so we must
C     be consistent here.
      IF (JSCALCOL .EQ. 1) THEN
         SQRTN = SQRT(REAL(NEQ))
         DO 10 I = 1, NEQ
            WK(I) = SQRTN / REWT(I)
 10      CONTINUE
         CALL AMUDIA (NEQ, 0, WP(LJAC), IWP(LJACJ), IWP(LJACI), WK,
     .                WP(LJAC), IWP(LJACJ), IWP(LJACI))
      ENDIF

C ... Normalize Jacobian rows, if desired.
      IF (ISRNORM .EQ. 1) THEN
         CALL ROSCAL (NEQ,0,NORMTYPE,WP(LJAC),IWP(LJACJ),IWP(LJACI),
     .             WP(LROWNRMS),WP(LJAC),IWP(LJACJ),IWP(LJACI),IERR)
         IF (IERR .NE. 0) RETURN
      ENDIF

C ... Reorder Jacobian rows and columns, if desired.
      IF (IREORDER .NE. 0) THEN
         CALL DJREORD (NEQ, NEQP1, NNZMX, PREMETH,
     .                 WP(LJAC), IWP(LJACJ), IWP(LJACI),
     .                 WP(LPLU), IWP(LJLU), IWP(LJU),
     .                 IWP(LPERM), IWP(LQPERM), IWP(LLEVELS), 
     .                 IWP(LMASK), IREORDER)
      ENDIF

C ... Write matrix JAC and scaled RES to file if JACOUT .eq. 1.
      IF (JACOUT .EQ. 1) THEN
         IUNIT = IPAR(29)
         IF (ISRNORM .EQ. 1) THEN
            DO 20 I = 1, NEQ
               SAVR(I) = SAVR(I) * WP(LROWNRMS+I-1)
 20         CONTINUE
         ENDIF
         IF (IREORDER .NE. 0) CALL DVPERM (NEQ, SAVR, IWP(LPERM))
         TITLE = ' DDASPK Test Matrix '
         IFMT = 15
         CALL PRTMT (NEQ,NEQ,WP(LJAC),IWP(LJACJ),IWP(LJACI),SAVR,
     .        'NN',TITLE,'SPARSKIT','RUA',IFMT,3,IUNIT)
         MSG = 'DJACILU -- Jacobian Matrix written to file.'
         CALL XERRWD(MSG,80,0,0,0,0,0,0,0.0,0.0)
         IERR = 1
         IRES = -2
         RETURN
      ENDIF

C ... Compute ILU decomposition.
      CALL DJILU (NEQ, NEQ+1, NNZMX, WP(LJAC), IWP(LJACJ), 
     .            IWP(LJACI), IWP(LJU), WP(LPLU), IWP(LJLU),
     .            WP(LRWK1), IWP(LIWK1), LENPLUMX, TOLILUT,
     .            LFILILUT, PERMTOL, PREMETH, IWP(LIPERM), IERR)
      IF ((IERR .EQ. -2) .OR. (IERR .EQ. -3)) THEN
         IRES = -2  ! Stop since more storage needed.
      ENDIF

C ... Save pointers for use in DPSOLILU into IPAR array.
      IPAR(21) = LPLU
      IPAR(22) = LJU
      IPAR(23) = LJLU
      IPAR(24) = LROWNRMS
      IPAR(25) = LPERM
      IPAR(26) = LQPERM

      RETURN
C------------  End of Subroutine DJACILU  ------------------------------
      END

      SUBROUTINE DJCALC (NEQ, T, Y, YPRIME, R0, ML, MU, R1, REWT, RES,
     .                   H, CJ,  NNZMX, JAC, JA, IA, RCOO, JCOO, ICOO,
     .                   IPAR, RPAR, IRES, NRE, IERR)

C ... Version of 10-6-95

C ... Calculate Jacobian matrix (derivatives with respect to each
C     dependent variable of the right-hand side of each rate equation).
C     Lower and upper bandwidths are used to select for computation
C     only those Jacobian elements that may be nonzero.
C     Estimates of Jacobian elements are computed by finite differences.
C     The Jacobian is stored in compressed sparse row format.

      IMPLICIT NONE

C ... Input arguments:
      INTEGER NEQ        ! total number of equations
      REAL*8 T           ! independent variable t
      REAL*8 Y(NEQ)      ! most recent iterate of solution vector y
      REAL*8 YPRIME(NEQ) ! most recent iterate of solution vector y'
      REAL*8 R0(NEQ)     ! current residual evaluated at (T,Y,YPRIME)
      REAL*8 REWT(NEQ)   ! array of scaling factors for Y and YPRIME
      EXTERNAL RES       ! function that evaluates residuals
      INTEGER IRES       ! error flag for RES routine
      INTEGER ML, MU     ! lower and upper bandwidths
      INTEGER NNZMX      ! maximum number of nonzeros in Jacobian
      REAL*8 H           ! current step size
      REAL*8 CJ          ! scalar proportional to 1/H
      REAL*8 RPAR(*)     ! user real workspace
      INTEGER IPAR(*)    ! user integer workspace

C ... Work-array argument:
      REAL*8 R1(NEQ)   ! work space available to this subroutine

C ... Output arguments:
      REAL*8 JAC(NNZMX)   ! nonzero Jacobian elements
      INTEGER JA(NNZMX)   ! col indices of nonzero Jacobian elements
      INTEGER IA(NEQ+1)   ! pointers to beginning of each row in JAC,JA
      INTEGER IERR

C ... Workspace for temporary storage of Jacobian elements:
      REAL*8 RCOO(NNZMX)   ! nonzero Jacobian elements
      INTEGER JCOO(NNZMX)  ! col indices of nonzero Jacobian elements
      INTEGER ICOO(NNZMX)  ! row indices of nonzero Jacobian elements

C ... Local variables:
      INTEGER NNZ, I, I1, I2, J, JJ, MBA, MEBAND, MEB1, MBAND, NRE
      REAL*8 JACELEM, UROUND, D1MACH, SQUR, DEL, DELINV
      CHARACTER*80 MSG

C ... Set band parameters.
      NNZ = 1
      MBAND = ML + MU + 1
      MBA = MIN0(MBAND,NEQ)
      MEBAND = MBAND + ML
      MEB1 = MEBAND - 1

C ... Set the machine unit roundoff UROUND and SQRT(UROUND), used to 
C ... set increments in the difference quotient procedure. 
      UROUND = D1MACH(4)
      SQUR = SQRT(UROUND)

C ... Initial error flags.
      IERR = 0
      IRES = 0

C ... Make MBA calls to RES to approximate the Jacobian.
C ... Here, R0(1),...,R0(neq) contains the base RES value, and 
C     R1(1),...,R1(NEQ) contains the perturbed values of RES.
      DO 40 J = 1,MBA
        DO 10 JJ = J,NEQ,MBAND
          JAC(JJ) = Y(JJ)
          JAC(JJ+NEQ) = YPRIME(JJ)
          DEL = SQUR*MAX(ABS(Y(JJ)),ABS(H*YPRIME(JJ)),ABS(1.0/REWT(JJ)))
          DEL = SIGN(DEL, H*YPRIME(JJ))
          DEL = (Y(JJ) + DEL) - Y(JJ)
          Y(JJ) = Y(JJ) + DEL
          YPRIME(JJ) = YPRIME(JJ) + CJ*DEL
 10       CONTINUE
        CALL RES (T, Y, YPRIME, CJ, R1, IRES, RPAR, IPAR)
        IF (IRES .LT. 0) RETURN
        NRE = NRE + 1
        DO 30 JJ = J,NEQ,MBAND
          Y(JJ) = JAC(JJ)
          YPRIME(JJ) = JAC(JJ+NEQ)
          DEL = SQUR*MAX(ABS(Y(JJ)),ABS(H*YPRIME(JJ)),ABS(1.0/REWT(JJ)))
          DEL = SIGN(DEL, H*YPRIME(JJ))
          DEL = (Y(JJ) + DEL) - Y(JJ)
          DELINV=1.0/DEL
          I1 = MAX(1,(JJ-MU))
          I2 = MIN(NEQ,(JJ+ML))
          DO 20 I = I1,I2
C ... Calculate possibly nonzero Jacobian elements for this variable,
C     and store nonzero elements in coordinate format.
            JACELEM = (R1(I) - R0(I))*DELINV
            IF (JACELEM .NE. 0.) THEN
               IF (NNZ .GT. NNZMX) THEN
                  MSG = 'DJCALC -- More storage needed for Jacobian.'
                  CALL XERRWD(MSG,80,0,0,0,0,0,0,0.0,0.0)
                  MSG = 'DJCALC -- Increase LENPFAC.'
                  CALL XERRWD(MSG,80,0,0,0,0,0,0,0.0,0.0)
                  MSG = 'DJCALC -- Storage exceeded at (I,J) = (I1,I2)'
                  CALL XERRWD(MSG,80,0,0,2,I,JJ,0,0.0,0.0)
                  IERR = 1
                  IRES = -2
	          RETURN
               ENDIF
               RCOO(NNZ) = JACELEM
               JCOO(NNZ) = JJ
               ICOO(NNZ) = I
               NNZ = NNZ + 1
            ENDIF
 20         CONTINUE
 30       CONTINUE
 40    CONTINUE
      NNZ = NNZ - 1
C
C ... Convert Jacobian from coordinate to compressed sparse row format.
      CALL COOCSR (NEQ, NNZ, RCOO, ICOO, JCOO, JAC, JA, IA)

      RETURN
C------------  End of Subroutine DJCALC  -------------------------------
      END

      SUBROUTINE DPSOLILU (NEQ, T, Y, YPRIME, R0, WK, CJ, WGHT, 
     .                     WP, IWP, BL, EPLIN, IERR, RPAR, IPAR)

C ... Version of 12-5-00

C ... Solve the linear system P*x=c, using elements of P loaded into
C     arrays WP and IWP.
C

      IMPLICIT NONE

C ... Input arguments:
      INTEGER NEQ        ! total number of equations
      REAL*8 T           ! independent variable t
      REAL*8 Y(NEQ)      ! most recent iterate of solution vector y
      REAL*8 YPRIME(NEQ) ! most recent iterate of solution vector y'
      REAL*8 R0(NEQ)     ! function value G(T,Y,YPRIME)
      REAL*8 WGHT(NEQ)   ! scaling array for Y and YPRIME
      REAL*8 CJ          ! scalar proportional to 1/H
      REAL*8 EPLIN       ! not used
      REAL*8 WP(*)       ! matrix elements of ILU
      INTEGER IWP(*)     ! array indices for elements of ILU
      INTEGER IPAR(*)    ! user workspace
      REAL*8 RPAR(*)     ! user workspace

C ... Work-array argument:
      REAL*8 WK(NEQ)     ! work space available to this subroutine

C ... In-out argument:
      REAL*8 BL(NEQ)     ! on input, c of P*x=c; on output, x

C ... Output arguments:
      INTEGER IERR     ! error flag (0 only possible value here)

C ... Local variables:
      INTEGER I, LPLU, LJU, LJLU, LROWNRMS, LPERM, LQPERM, IREORDER,
     .        ISRNORM, IPREMETH, JSCALCOL

C ... Load IPREMETH, IREORDER and ISRNORM values from IPAR.
      IPREMETH = IPAR(5)
      IREORDER = IPAR(7)
      ISRNORM = IPAR(8)
      JSCALCOL = IPAR(11)

C ... Load pointers into WP and iWP arrays.
      LPLU = IPAR(21)
      LJU  = IPAR(22)
      LJLU = IPAR(23)
      LROWNRMS = IPAR(24)
      LPERM = IPAR(25)
      LQPERM = IPAR(26)

C ... Scale c by multiplying by row-normalization factors, if used.
      IF (ISRNORM .EQ. 1) THEN
         DO 10 I = 1, NEQ
            BL(I) = BL(I) * WP(LROWNRMS+I-1)
 10      CONTINUE
      ENDIF
      
C ... Solve P*x=c for a preconditioner stored as a sparse matrix in
C     compressed sparse row format.
C     If rows and columns of P were reordered (permuted), permute c,
C     then use inverse permutation on x.
      IF (IPREMETH .EQ. 1 .OR. IPREMETH .EQ. 2) THEN
	 IF (IREORDER .EQ. 1) CALL DVPERM (NEQ, BL, IWP(LPERM))
         CALL LUSOL (NEQ, BL, WK, WP(LPLU), IWP(LJLU), IWP(LJU))
	 IF (IREORDER .EQ. 1) CALL DVPERM (NEQ, WK, IWP(LQPERM))
      ENDIF

C ... Unscale x by dividing by column scaling vector WGHT.
      IF (JSCALCOL .EQ. 1) THEN
         DO 40 I = 1,NEQ
 40         BL(I) = WK(I) / WGHT(I)
      ELSE
         DO 50 I = 1,NEQ
 50         BL(I) = WK(I)
      ENDIF

      IERR = 0
      RETURN
C------------  End of Subroutine DPSOLILU  -----------------------------
      END

      SUBROUTINE DJILU (NEQ, NEQP1, NNZMX, JAC, JA, IA, JU,
     .                  PLU, JLU, RWK1, IWK1, LENPLUMX, TOLILUT,
     .                  LFILILUT, PERMTOL, PREMETH, IPERM, IERR)

C ... Version of 12-12-00

C ... Compute ILU decomposition of Jacobian and return it in any one
C     of the storage formats.

      IMPLICIT NONE

C ... Input arguments:
      INTEGER NEQ        ! total number of equations
      INTEGER NEQP1      ! NEQ + 1
      INTEGER NNZMX      ! maximum number of nonzeros in Jacobian
      REAL*8 JAC(NNZMX)  ! nonzero Jacobian elements
      INTEGER JA(NNZMX)  ! col indices of nonzero Jacobian elements
      INTEGER IA(NEQP1)  ! pointers to beginning of each row in jac,ja
      CHARACTER*8 PREMETH
      REAL*8 TOLILUT
      REAL*8 PERMTOL
      INTEGER LENPLUMX
      INTEGER LFILILUT
      REAL*8 RWK1(NEQ)
      INTEGER IWK1(2*NEQ), IPERM(2*NEQ)

C ... Output arguments:
      REAL*8 PLU(LENPLUMX) ! matrix elements of ILU
      INTEGER JLU(LENPLUMX)! sizes and array indices for elements of ILU
      INTEGER JU(NEQ)      ! pointer to beginning of each row of U in
                           ! matrix PLU,JLU
      INTEGER IERR         ! error flag

C ... Local variables:
      CHARACTER*80 MSG
      LOGICAL ERROR

      ERROR = .FALSE.

      IF (PREMETH .EQ. 'ILUT') THEN

C ... Use incomplete factorization routine ILUT from SparsKit.
         CALL ILUT (NEQ,JAC,JA,IA,LFILILUT,TOLILUT,PLU,JLU,
     .              JU,LENPLUMX,RWK1,IWK1,IERR)
         IF (IERR .NE. 0) THEN
            MSG = 'DJILU -- Error return from ILUT: IERR = (I1)'
            CALL XERRWD(MSG,80,0,0,1,IERR,0,0,0.0,0.0)
            ERROR = .TRUE.
         ENDIF

      ELSEIF (PREMETH .EQ. 'ILUTP') then

C ... Use incomplete factorization routine ILUTP from SparsKit.
         CALL ILUTP (NEQ,JAC,JA,IA,LFILILUT,TOLILUT,PERMTOL,NEQ,
     .               PLU,JLU,JU,LENPLUMX,RWK1,IWK1,IPERM,IERR)
         IF (IERR .NE. 0) THEN
            MSG = 'DJILU -- Error return from ILUTP: IERR = (I1)'
            CALL XERRWD(MSG,80,0,0,1,IERR,0,0,0.0,0.0)
            ERROR = .TRUE.
         ENDIF

C ... Put in other options here for incomplete factorizations.
      ENDIF

      IF(ERROR) THEN
         MSG = 
     . 'DJILU -- IERR .NE. 0 means one of the following has occurred:'
         CALL XERRWD(MSG,80,0,0,0,0,0,0,0.0,0.0)
         MSG =
     . '    IERR >  0   --> Zero pivot encountered at step number IERR.'
         CALL XERRWD(MSG,80,0,0,0,0,0,0,0.0,0.0)
         MSG =
     . '    IERR = -1   --> Error. input matrix may be wrong.'
         CALL XERRWD(MSG,80,0,0,0,0,0,0,0.0,0.0)
         MSG =
     . '                     (The elimination process has generated a'
         CALL XERRWD(MSG,80,0,0,0,0,0,0,0.0,0.0)
         MSG =
     . '                     row in L or U with length > NEQ.)'
         CALL XERRWD(MSG,80,0,0,0,0,0,0,0.0,0.0)
         MSG ='    IERR = -2   --> Matrix L overflows.'
         CALL XERRWD(MSG,80,0,0,0,0,0,0,0.0,0.0)
         MSG ='    IERR = -3   --> Matrix U overflows.'
         CALL XERRWD(MSG,80,0,0,0,0,0,0,0.0,0.0)
         MSG ='    IERR = -4   --> Illegal value for LFILILUT.'
         CALL XERRWD(MSG,80,0,0,0,0,0,0,0.0,0.0)
         MSG ='    IERR = -5   --> Zero row encountered.'
         CALL XERRWD(MSG,80,0,0,0,0,0,0,0.0,0.0)
         MSG ='    '
         CALL XERRWD(MSG,80,0,0,0,0,0,0,0.0,0.0)
         MSG =
     . '    For IERR = -2 or -3, increase the value of LENPLUFAC or'
         CALL XERRWD(MSG,80,0,0,0,0,0,0,0.0,0.0)
         MSG =
     . '    decrease the value of LFILILUT if LENPLUFAC cannot be'
         CALL XERRWD(MSG,80,0,0,0,0,0,0,0.0,0.0)
         MSG ='    increased.'
         CALL XERRWD(MSG,80,0,0,0,0,0,0,0.0,0.0)
      ENDIF

      RETURN
C------------  End of Subroutine DJILU  --------------------------------
      END

      SUBROUTINE DJREORD (NEQ, NEQP1, NNZMX, PREMETH,
     .                    JAC, JA, IA, AWK, JWK, IWK,
     .                    PERM, QPERM, LEVELS, MASK, IREORDER)

C ... Version of 10-6-95

C ... If desired, reorder the Jacobian matrix.

      IMPLICIT NONE

C ... Input arguments:
      INTEGER NEQ        ! total number of equations
      INTEGER NEQP1      ! NEQ + 1
      INTEGER NNZMX      ! maximum number of nonzeroes in Jacobian
      REAL*8 JAC(NNZMX)  ! nonzero Jacobian elements
      INTEGER JA(NNZMX)  ! column indices of nonzero Jacobian elements
      INTEGER IA(NEQP1)  ! indices of 1st nonzero element in each row
      CHARACTER*8 PREMETH

C ... Work-array arguments:
      REAL*8 AWK(NNZMX)
      INTEGER JWK(NNZMX)
      INTEGER IWK(NEQP1)
      INTEGER PERM(NEQ)   ! Integer array containing the permutation
                          ! used in reordering the rows and columns of
			  ! the Jacobian matrix.
      INTEGER QPERM(NEQ)  ! Integer array holding the inverse of the
			  ! permutation in array perm.
      INTEGER LEVELS(NEQ) ! Work array used by the bfs reordering
			  ! subroutine.   See subroutine BFS for
			  ! more details.
      INTEGER MASK(NEQ)	  ! Work array used by the BFS reordering
			  ! subroutine.  See BFS subroutine.
      INTEGER IREORDER    ! Flag used to determine if a reordering
			  ! of the Jacobian matrix is desired.
			  ! = 1 means a reverse Cuthill-McKee
			  !     reordering of the rows and columns
			  !     of the Jacobian is done.
			  ! = 0 means no reordering.


C ... Local variables:
      INTEGER NLEV	  ! Number of levels in levels array.
			  ! See subroutine BFS for more details.
      INTEGER MASKVAL	  ! Scalar used with MASK.
      INTEGER I, NFIRST

      IF (IREORDER .EQ. 1) THEN

C ... Copy JAC, JA, and IA to AWK, JWK, and IWK.
         CALL ATOB (NEQ, JAC, JA, IA, AWK, JWK, IWK)

C ... Perform a Cuthill-McKee reordering of the Jacobian.
         NFIRST = 1
         PERM(1) = 0
         DO I = 1, NEQ
            MASK(I) = 1
         ENDDO
         MASKVAL = 1
         QPERM(1) = 1
         CALL BFS (NEQ,JWK,IWK,NFIRST,PERM,MASK,MASKVAL,QPERM,LEVELS,
     .             NLEV)

C ... Reverse the permutation to obtain the reverse Cuthill-McKee
C     reordering.
         CALL RVERSP (NEQ,QPERM)

C ... Calculate the inverse of QPERM and put it in PERM.
         DO I = 1, NEQ
            PERM(QPERM(I)) = I
         ENDDO

C ... Permute rows and columns of Jacobian using PERM.
         CALL DPERM (NEQ,AWK,JWK,IWK,JAC,JA,IA,PERM,PERM,1)

C ... End of If block
      ENDIF

      RETURN
C------------  End of Subroutine DJREORD  ------------------------------
      END
