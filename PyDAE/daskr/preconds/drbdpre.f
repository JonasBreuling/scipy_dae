C-----------------------------------------------------------------------
C
C    Preconditioner Tools for Reaction-Transport Problems
C    Part I: Block-Diagonal Reaction-Based Factor without Grouping
C                        14 September 1995
C
C The following three subroutines -- DMSET2, DRBDJA, DRBDPS --
C are provided to assist in the generation and solution of
C preconditioner matrices for problems arising from reaction-transport
C systems, as solved with DASPK.  More specifically, they are intended
C as tools for preconditioners that include a contribution from the
C reaction terms of the system.  These are intended as auxiliary
C routines for the user-supplied routines JAC and PSOL called by
C DASPK when the Krylov method is selected.
C
C These routines are intended for a DAE system obtained from a system
C of reaction-transport PDEs, in which some of the PDE variables obey
C evolution equations, and the rest obey algebraic (time-independent)
C equations.  See Ref. 2, Section 4.  It is assumed that the 
C right-hand sides of all the equations have the form of a sum of a
C reaction term R and a transport term S, that the transport term
C is discretized by finite differences, and that in the spatial
C discretization the PDE variables at each spatial point are kept
C together.  Thus the DAE system function, in terms of a dependent
C variable vector u, has the form
C     G(t,u,u') = I_d u' - R(t,u) - S(t,u) ,  where
C     I_d = identity matrix with zeros in the positions corresponding
C           to the algebraic components, ones in those for the
C           evolution (differential) components.
C     R(t,u) = the reaction terms (spatial coupling absent), and
C     S(t,u) = the spatial transport terms.
C
C As shown in [2], two possible preconditioners for such a system are:
C (a) P_R = c I_d - dR/du, based on the reaction term R alone, and
C (b) P_SR = (I - (1/c) dS/du) (c I_d - dR/du), the product of two
C     factors (in either order), one being P_R and the other being
C     based on the spatial term S alone.
C Here c is the scalar CJ that is input to the JAC and PSOL routines
C provided by the user (1/c is proportional to the step size H).
C
C The routines given here can be used to provide the reaction-based
C factor P_R.  More precisely, they provide an approximation A_R to 
C P_R.  The matrix P_R is block-diagonal, with each block corresponding
C to one spatial point.  In A_R, we compute each block by difference
C quotient approximations, by way of calls to a user-supplied routine,
C subroutine RBLOCK, that evaluates the reaction terms at a single
C spatial point.  A_R has one such block for each spatial point in
C the mesh.  (For a more economical approximation, see Part II,
C on block-grouping in A_R.)
C
C The routines given here are specialized to the case of a 2-D problem
C on a rectangular mesh in the x-y plane.  However, they can be easily
C modified for a different problem geometry.  It is also assumed
C that the PDE variables are ordered so that the differential
C variables appear first, followed by the algebraic variables.
C
C To make use of these routines in a DASPK solution, the user must
C provide:
C (a) a calling program that sets the DASPK input parameters, and calls
C     DMSET2 to set mesh data and mesh-related DASPK inputs;
C (b) a JAC routine, as prescribed by the DASPK instructions, which
C     calls DRBDJA, and does any other Jacobian-related preprocessing
C     needed for preconditioning; and
C (c) a PSOL routine, as prescribed by the DASPK instructions, which
C     calls DRBDPS for the solution of systems A_R x = b, and does
C     any other linear system solving required by the preconditioner.
C Detailed descriptions and instructions are given below.
C
C In addition, the use of these routines requires:
C  * the LINPACK routines DGEFA and DGESL for dense linear sytems, and
C  * the machine constant routine D1MACH for the machine unit roundoff.
C
C (a) The calling program.
C The calling program sets the DASPK inputs and makes calls to DASPK.
C Here the DASPK inputs include
C   INFO(12) = 1 [to signal the Krylov method]
C   INFO(15) = 1 [to signal the presence of a JAC routine]
C
C Also, the use of the DRBDJA/DRBDPS routines in conjunction with
C DASPK requires that certain mesh-related data be set.  This can be
C done with the call
C     CALL DMSET2 (MX, MY, NS, NSD, LID, IWORK)
C The input arguments to DMSET2 are:
C   MX and MY = the mesh dimensions.
C   NS  = number of PDE variables.
C   NSD = number of differential PDE variables.
C   LID = offset in IWORK for array showing the differential and
C         algebraic components on input to DASPK, required if either
C         INFO(11) = 1 or INFO(16) = 1.  Set LID = 0 otherwise.
C         If this array is required, set LID = 40 or 40 + NEQ,
C         depending on the value of the constraint option INFO(10).
C DMSET2 loads mesh data in a COMMON block /DRPRE1/ used by the
C DRBDJA/DRBDPS routines.
C
C DMSET2 also loads the preconditioner work lengths into
C IWORK(27) and IWORK(28), and if LID > 0 it sets the ID array
C in IWORK showing the differential and algebraic components.
C 
C (b) The JAC routine.
C The user-supplied JAC routine called by DASPK with the Krylov
C method specified, is to generate and preprocess Jacobian-related
C data as needed for later solution of the preconditioner system
C P x = b.  Assuming that P is to be an approximation of either P_R
C or P_SR, the JAC routine should call DRBDJA for the approximation
C A_R to P_R.  Subroutine DRBDJA generates A_R using difference
C quotients.  It then performs an LU decomposition of each block,
C using the LINPACK routine DGEFA.
C
C In terms of the arguments passed to JAC by DASPK, the call to
C DRBDJA should have the form
C     CALL DRBDJA (T, U, R0, RBLOCK, WK, REWT, CJ, WP, IWP, IER)
C where we use U instead of Y for the dependent variable array.
C The argument R0 is an array assumed to contain the current value
C of the R vector, at the current values (T,U).  This can be done, for
C example, by taking R0 to be RPAR, and loading RPAR with the
C vector R in the last call to the RES routine; in that case, the
C calling program must declare RPAR to have length at least NEQ.
C Alternatively, insert a call to RBLOCK (see below) within the
C loop over mesh points in DRBDJA.
C
C To use DRBDJA, the user must provide the following subroutine,
C which DRBDJA calls to obtain individual blocks of R:
C      SUBROUTINE RBLOCK (T, JX, JY, UXY, RXY)
C The input arguments to RBLOCK are:
C   T     = current time.
C   JX,JY = spatial indices in x- and y-directions.
C   UXY   = block of NS dependent variables at spatial point (JX,JY).
C RBLOCK is to load block (JX,JY) of R(t,u) into the array RXY.
C
C (c) The PSOL routine.
C The user-supplied PSOL routine must solve the linear system P x = b,
C where P is the preconditioner matrix.  For this, the PSOL routine
C should call DRBDPS for the solution of A_R.  Subroutine DRBDPS
C solves a linear system A_R x = b, using the LINPACK backsolve
C routine DGESL.  In terms of the arguments passed to PSOL by DASPK,
C the call to DRBDPS should have the form
C     CALL DRBDPS (B, WP, IWP)
C DRBDPS overwrites the B array (containing b) with the solution x.
C
C-----------------------------------------------------------------------
C
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


      SUBROUTINE DMSET2 (MX, MY, NS, NSD, LID, IWORK)
C***BEGIN PROLOGUE  DMSET2
C***DATE WRITTEN   950830   (YYMMDD)
C
C***AUTHORS  A. C. Hindmarsh
C            Lawrence Livermore National Laboratory
C            L-316, P.O. Box 808
C            Livermore, CA 94551
C
C***DESCRIPTION
C
C-----------------------------------------------------------------------
C This routine sets mesh parameters needed to use the routines
C DRBDJA and DRBDPS, assuming a 2-D rectangular problem.
C Given the mesh parameters, it loads the COMMON block /DRPRE1/,
C and the lengths LENWP and LENIWP in IWORK.
C Then if LID > 0, it also sets the ID array in IWORK, indicating
C which components are differential and which are algebraic.
C
C The variables in the COMMON block are defined as follows:
C   SRUR   = SQRT(unit roundoff), used in difference quotients.
C            UROUND = D1MACH(4) generates the unit roundoff.
C   MP     = NS = number of PDE variables, the size of each block in 
C            the block-diagonal preconditioner matrix P_R.
C   MPD    = NSD = number of differential PDE variables.  In the DAE
C            system, the first MPD variables at each spatial point have
C            time  derivatives, and the remaining (MP - MPD) do not.
C   MPSQ   = MP*MP.
C   MESHX  = MX = x mesh size.
C   MESHY  = MY = y mesh size (the mesh is MESHX by MESHY).
C-----------------------------------------------------------------------
C***ROUTINES CALLED
C   D1MACH
C
C***END PROLOGUE  DMSET2
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION IWORK(*)
      COMMON /DRPRE1/ SRUR, MP, MPD, MPSQ, MESHX, MESHY, MXMP
C
C Load the COMMON block.
      UROUND = D1MACH(4)
      SRUR = SQRT(UROUND)
      MP = NS
      MPD = NSD
      MPSQ = NS*NS
      MESHX = MX
      MESHY = MY
      MXMP = MESHX*MP
C
C Here set the sizes of the preconditioning storage space segments
C in RWORK and IWORK.
      IWORK(27) = MPSQ*MESHX*MESHY
      IWORK(28) = MP*MESHX*MESHY
C
C If LID .GT. 0, set the ID array in IWORK.
      IF (LID .EQ. 0) RETURN
      I0 = LID
      DO 40 JY = 1,MY
        DO 30 JX = 1,MX
          DO 10 I = 1,MPD
 10         IWORK(I0+I) = 1
          DO 20 I = MPD+1,MP
 20         IWORK(I0+I) = -1
          I0 = I0 + MP
 30       CONTINUE
 40     CONTINUE
C
      RETURN
C------------  End of Subroutine DMSET2  -------------------------------
      END

      SUBROUTINE DRBDJA (T, U, R0, RBLOCK, R1, REWT, CJ, BD, IPBD, IER)
C***BEGIN PROLOGUE  DRBDJA
C***DATE WRITTEN   950914   (YYMMDD)
C
C***AUTHORS  A. C. Hindmarsh
C            Lawrence Livermore National Laboratory
C            L-316, P.O. Box 808
C            Livermore, CA 94551
C
C***DESCRIPTION
C
C-----------------------------------------------------------------------
C This routine generates and preprocesses a block-diagonal
C preconditioner matrix, based on the part of the Jacobian corresponding
C to the reaction terms R of the problem.
C It generates a matrix of the form CJ * I_d - dR/du.
C It calls DGEFA to do LU decomposition of each diagonal block.
C The computation of the diagonal blocks uses the mesh information in
C the COMMON block /DRPRE1/.  One block per spatial point is computed.
C The Jacobian elements are generated by difference quotients.
C This routine calls a user routine of the form
C      SUBROUTINE RBLOCK (T, JX, JY, UXY, RXY)
C which is to set RXY to block (JX,JY) of R, as a function of the
C current time T and block UXY of current dependent variable vector U.
C The array R0 is assumed to contain the current value of R at (T,U).
C-----------------------------------------------------------------------
C On input:
C   T      = current value of independent variable.
C   U      = current dependent variable array.
C   R0  = array of current values of the vector R at (T,U)
C   RBLOCK = name of external routine that computes a single block of R.
C   R1     = array of length NEQ for work space.
C   REWT   = reciprocal error weights.
C   CJ     = scalar used in forming the system Jacobian.
C
C On output:
C   BD     = array containing the LU factors of the diagonal blocks.
C   IPBD   = integer array of pivots for the LU factorizations.
C   IER    = integer error flag.  If no error occurred, IER = 0.
C            If a zero pivot was found at stage k in one of the LU
C            factorizations, this routine returns IER = k > 0.
C Here BD is the RWORK segment WP, and IPBD is the IWORK segment IWP.
C-----------------------------------------------------------------------
C***ROUTINES CALLED
C   RBLOCK, DGEFA
C
C***END PROLOGUE  DRBDJA
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      EXTERNAL RBLOCK
      DIMENSION U(*), R0(*), R1(*), REWT(*), BD(*), IPBD(*)
C
      COMMON /DRPRE1/ SRUR, MP, MPD, MPSQ, MESHX, MESHY, MXMP
C
C Make MP calls to RBLOCK to approximate each diagonal block of dR/du. 
      DFAC = 1.0D-2
      IBD = 0
      J0 = 0
      DO 40 JY = 1,MESHY
        DO 30 JX = 1,MESHX
C If R0 has not been set previously as an array of length NEQ, it can
C be set here, as an array of length MP, with the call
C         CALL RBLOCK (T, JX, JY, U(J0+1), R0)
C In this case, change R0(J0+I) below to R0(I).
          DO 20 JS = 1,MP
            J = J0 + JS
            UJ = U(J)
            DEL = MAX(SRUR*ABS(UJ),DFAC/REWT(J))
            U(J) = U(J) + DEL
            FAC = -1.0D0/DEL
            CALL RBLOCK (T, JX, JY, U(J0+1), R1)
            DO 10 I = 1,MP
 10           BD(IBD+I) = (R1(I) - R0(J0+I))*FAC
            U(J) = UJ
            IBD = IBD + MP
 20         CONTINUE
          J0 = J0 + MP
 30       CONTINUE
 40     CONTINUE
C
C Add matrix CJ * I_d, and do LU decomposition on blocks. --------------
      IBD = 1
      IIP = 1
      DO 80 J = 1,MESHX*MESHY
        IDIAG = IBD
        DO 70 I = 1,MP
          IF (I .LE. MPD) BD(IDIAG) = BD(IDIAG) + CJ
 70       IDIAG = IDIAG + (MP + 1)
        CALL DGEFA (BD(IBD), MP, MP, IPBD(IIP), IER)
        IF (IER .NE. 0) GO TO 90
        IBD = IBD + MPSQ
        IIP = IIP + MP
 80     CONTINUE
 90   RETURN
C------------  End of Subroutine DRBDJA  -------------------------------
      END

      SUBROUTINE  DRBDPS (B, BD, IPBD)
C***BEGIN PROLOGUE  DRBDPS
C***DATE WRITTEN   950914   (YYMMDD)
C
C***AUTHORS  A. C. Hindmarsh
C            Lawrence Livermore National Laboratory
C            L-316, P.O. Box 808
C            Livermore, CA 94551
C
C***DESCRIPTION
C
C-----------------------------------------------------------------------
C This routine solves a linear system A_R x = b, using the LU factors
C of the diagonal blocks computed in DRBDJA, and mesh parameters
C in the COMMON block /DRPRE1/.
C Here BD is the RWORK segment WP, and IPBD is the IWORK segment IWP.
C The right-hand side vector b, contained in B on entry, is overwritten
C with the solution vector x on return.
C-----------------------------------------------------------------------
C***ROUTINES CALLED
C   DGESL
C
C***END PROLOGUE  DRBDPS
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION B(*), BD(*), IPBD(*)
C
      COMMON /DRPRE1/ SRUR, MP, MPD, MPSQ, MESHX, MESHY, MXMP
C
      IER = 0
      IB = 1
      IBD = 1
      DO 20 JY = 1,MESHY
        DO 10 JX = 1,MESHX
          CALL DGESL (BD(IBD), MP, MP, IPBD(IB), B(IB), 0)
          IB = IB + MP
          IBD = IBD + MPSQ
 10       CONTINUE
 20     CONTINUE
C
      RETURN
C------------  End of Subroutine DRBDPS  -------------------------------
      END
