C-----------------------------------------------------------------------
C
C    Preconditioner Tools for Reaction-Transport Problems
C    Part II: Block-Grouping in Block-Diagonal Reaction-Based Factor
C                        14 September 1995
C
C The following four subroutines -- DGSET2, GSET1, DRBGJA, DRBGPS --
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
C spatial point.  In addition, rather than evaluate a block at every
C spatial point in the mesh, we use a block-grouping scheme, described
C in Ref. 1.  In this scheme, the mesh points are grouped, as in domain
C decomposition, and only one block of dR/du is computed for each group;
C then in solving A_R x = b, the inverse of the representative block is
C applied to all the blocks of unknowns in the group.  Block-grouping
C greatly reduces the storage required for the preconditioner. 
C
C The routines given here are specialized to the case of a 2-D problem
C on a rectangular mesh in the x-y plane, and for a block-grouping
C arrangement that is rectangular (i.e. the Cartesian product of two
C 1-D groupings).  However, they can be easily modified for a different
C problem geometry or a different grouping arrangement.  It is also
C assumed that the PDE variables are ordered so that the differential
C variables appear first, followed by the algebraic variables.
C
C To make use of these routines in a DASPK solution, the user must
C provide:
C (a) a calling program that sets the DASPK input parameters, and calls
C     DGSET2 to set the mesh and block-grouping data needed later;
C (b) a JAC routine, as prescribed by the DASPK instructions, which
C     calls DRBGJA, and does any other Jacobian-related preprocessing
C     needed for preconditioning; and
C (c) a PSOL routine, as prescribed by the DASPK instructions, which
C     calls DRBGPS for the solution of systems A_R x = b, and does
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
C Also, the use of the DRBGJA/DRBGPS routines in conjunctin with
C DASPK requires that certain mesh-related and block-grouping data
C be set.  This can be done with the call
C     CALL DGSET2 (MX, MY, NS, NSD, NXG, NYG, LID, IWORK)
C The input arguments to DGSET2 are:
C   MX and MY = the mesh dimensions.
C   NS  = number of PDE variables.
C   NSD = number of differential PDE variables.
C   NXG = number of groups in the x direction.
C   NYG = number of groups in the y direction.
C   LID = offset in IWORK for array showing the differential and
C         algebraic components on input to DASPK, required if either
C         INFO(11) = 1 or INFO(16) = 1.  Set LID = 0 otherwise.
C         If this array is required, set LID = 40 or 40 + NEQ,
C         depending on the value of the constraint option INFO(10).
C
C DGSET2 loads mesh parameters and group data in two COMMON
C blocks, /DRPRE1/ and /RPRE2/, used by these routines.
C Note: the declaration of /RPRE2/ in DGSET2, DRBGJA, and DRBGPS
C uses a parameter MAXM that must be .GE. MAX (MX, MY);
C this must be altered for larger mesh sizes.
C
C DGSET2 also loads the preconditioner work lengths into
C IWORK(27) and IWORK(28), and if LID > 0 it sets the ID array
C in IWORK showing the differential and algebraic components.
C
C DGSET2 generates a rectangular grouping arrangement, with
C partitioning in each mesh direction being uniform (or as nearly
C uniform as possible), by calling the routine GSET1 twice.
C If a different grouping arrangement is desired, the user must
C alter or replace DGSET2 accordingly.
C 
C (b) The JAC routine.
C The user-supplied JAC routine called by DASPK with the Krylov
C method specified, is to generate and preprocess Jacobian-related
C data as needed for later solution of the preconditioner system
C P x = b.  Assuming that P is to be an approximation of either P_R
C or P_SR, the JAC routine should call DRBGJA for the approximation
C A_R to P_R.  Subroutine DRBGJA generates A_R using difference
C quotients and the block-grouping information.  It then performs an
C LU decomposition of each block, using the LINPACK routine DGEFA.
C
C In terms of the arguments passed to JAC by DASPK, the call to
C DRBGJA should have the form
C     CALL DRBGJA (T, U, R0, RBLOCK, WK, REWT, CJ, WP, IWP, IER)
C where we use U instead of Y for the dependent variable array.
C The argument R0 is an array assumed to contain the current value
C of the R vector, at the current values (T,U).  This can be done, for
C example, by taking R0 to be RPAR, and loading RPAR with the
C vector R in the last call to the RES routine; in that case, the
C calling program must declare RPAR to have length at least NEQ.
C Alternatively, insert a call to RBLOCK (see below) within the
C loop over mesh points in DRBGJA.
C
C To use DRBGJA, the user must provide the following subroutine,
C which DRBGJA calls to obtain individual blocks of R:
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
C should call DRBGPS for the solution of A_R.  Subroutine DRBGPS
C solves a linear system A_R x = b, using the LINPACK backsolve
C routine DGESL.  In terms of the arguments passed to PSOL by DASPK,
C the call to DRBGPS should have the form
C     CALL DRBGPS (B, WP, IWP)
C DRBGPS overwrites the B array (containing b) with the solution x.
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


      SUBROUTINE DGSET2 (MX, MY, NS, NSD, NXG, NYG, LID, IWORK)
C***BEGIN PROLOGUE  DGSET2
C***DATE WRITTEN   950828   (YYMMDD)
C
C***AUTHORS  A. C. Hindmarsh
C            Lawrence Livermore National Laboratory
C            L-316, P.O. Box 808
C            Livermore, CA 94551
C
C***DESCRIPTION
C
C-----------------------------------------------------------------------
C This routine sets mesh and block-grouping data needed to use the
C DRBGJA and DRBGPS routines, assuming a 2-D rectangular problem with
C uniform rectangular block-grouping.  Given the mesh and group
C parameters, it loads the COMMON blocks /DRPRE1/ and /RPRE2/, and the
C lengths LENWP and LENIWP in IWORK.  Then if LID > 0, it also sets the
C ID array in IWORK, indicating which components are differential and
C which are algebraic.
C
C The variables in the COMMON blocks are defined as follows:
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
C   NGX    = NXG = no. groups in x direction in block-grouping scheme.
C   NGY    = NYG = no. groups in y direction in block-grouping scheme.
C   NGRP   = total number of groups = NGX*NGY.
C   MXMP   = MESHX*MP.
C   JGX    = length NGX+1 array of group boundaries in x direction.
C            Group igx has x indices jx = JGX(igx),...,JGX(igx+1)-1.
C   JIGX   = length MESHX array of x group indices vs x node index.
C            x node index jx is in x group JIGX(jx).
C   JXR    = length NGX array of x indices representing the x groups.
C            The index for x group igx is jx = JXR(igx).
C   JGY, JIGY, JYR = analogous arrays for grouping in y direction.
C The COMMON block /RPRE2/ declared below has arrays whose minimum
C lengths depend on MX, MY, NXG, and NYG.  For simplicity, the
C declaration uses a single parameter MAXM, assumed to be .GE. all
C four of these numbers.  If this declaration is altered here, it must
C be altered consistently in subroutines DRBGJA and DRBGPS.
C-----------------------------------------------------------------------
C***ROUTINES CALLED
C   D1MACH, GSET1
C
C***END PROLOGUE  DGSET2
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION IWORK(*)
      PARAMETER (MAXM = 50)
      COMMON /DRPRE1/ SRUR, MP, MPD, MPSQ, MESHX, MESHY, MXMP
      COMMON /RPRE2/ NGX, NGY, NGRP, JGX(MAXM+1), JGY(MAXM+1),
     1               JIGX(MAXM), JIGY(MAXM), JXR(MAXM), JYR(MAXM)
C
C Load all the scalars in COMMON blocks.
      UROUND = D1MACH(4)
      SRUR = SQRT(UROUND)
      MP = NS
      MPD = NSD
      MPSQ = NS*NS
      MESHX = MX
      MESHY = MY
      MXMP = MESHX*MP
      NGX = NXG
      NGY = NYG
      NGRP = NGX*NGY
C
C Call GSET1 for each mesh direction to load grouping arrays.
      CALL GSET1 (MESHX, NGX, JGX, JIGX, JXR)
      CALL GSET1 (MESHY, NGY, JGY, JIGY, JYR)
C
C Here set the sizes of the preconditioning storage space segments
C in RWORK and IWORK.
      IWORK(27) = MPSQ*NGRP
      IWORK(28) = MP*NGRP
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
C------------  End of Subroutine DGSET2  -------------------------------
      END

      SUBROUTINE GSET1 (M, NG, JG, JIG, JR)
C***BEGIN PROLOGUE  GSET1
C***DATE WRITTEN   950828   (YYMMDD)
C
C***AUTHORS  A. C. Hindmarsh
C            Lawrence Livermore National Laboratory
C            L-316, P.O. Box 808
C            Livermore, CA 94551
C
C***DESCRIPTION
C
C-----------------------------------------------------------------------
C This routine sets arrays JG, JIG, and JR describing a uniform
C (or nearly uniform) partition of (1,2,...,M) into NG groups.
C-----------------------------------------------------------------------
C***ROUTINES CALLED
C   NONE
C
C***END PROLOGUE  GSET1
C
      DIMENSION JG(*), JIG(*), JR(*)
C
      MPER = M/NG
      DO 10 IG = 1,NG
 10     JG(IG) = 1 + (IG - 1)*MPER
      JG(NG+1) = M + 1
C
      NGM1 = NG - 1
      LEN1 = NGM1*MPER
      DO 20 J = 1,LEN1
 20     JIG(J) = 1 + (J-1)/MPER
      LEN1 = LEN1 + 1
      DO 25 J = LEN1,M
 25     JIG(J) = NG
C
      DO 30 IG = 1,NGM1
 30     JR(IG) = 0.5D0 + (REAL(IG) - 0.5D0)*REAL(MPER)
      JR(NG) = 0.5D0*REAL(1 + NGM1*MPER + M)
C
      RETURN
C------------  End of Subroutine GSET1  --------------------------------
      END

      SUBROUTINE DRBGJA (T, U, R0, RBLOCK, R1, REWT, CJ, BD, IPBD, IER)
C***BEGIN PROLOGUE  DRBGJA
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
C to the reaction terms R of the problem, using block-grouping.
C It generates a matrix of the form CJ * I_d - dR/du.
C It calls DGEFA to do LU decomposition of each diagonal block.
C The computation of the diagonal blocks uses the mesh and grouping
C information in the COMMON blocks /DRPRE1/ and /RPRE2/.  One block
C per group is computed.  The Jacobian elements are generated by
C difference quotients.
C This routine calls a user-supplied routine of the form
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
C***END PROLOGUE  DRBGJA
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      EXTERNAL RBLOCK
      DIMENSION U(*), R0(*), R1(*), REWT(*), BD(*), IPBD(*)
C
C The declaration of /RPRE2/ below must agree with that in DGSET2.
      PARAMETER (MAXM = 50)
      COMMON /DRPRE1/ SRUR, MP, MPD, MPSQ, MESHX, MESHY, MXMP
      COMMON /RPRE2/ NGX, NGY, NGRP, JGX(MAXM+1), JGY(MAXM+1),
     1               JIGX(MAXM), JIGY(MAXM), JXR(MAXM), JYR(MAXM)
C
C Make MP calls to RBLOCK to approximate each diagonal block of dR/du. 
      DFAC = 1.0D-2
      IBD = 0
      DO 40 IGY = 1,NGY
        JY = JYR(IGY)
        J00 = (JY - 1)*MXMP
        DO 30 IGX = 1,NGX
          JX = JXR(IGX)
          J0 = J00 + (JX - 1)*MP
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
 30       CONTINUE
 40     CONTINUE
C
C Add matrix CJ * I_d, and do LU decomposition on blocks. --------------
      IBD = 1
      IIP = 1
      DO 80 IG = 1,NGRP
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
C------------  End of Subroutine DRBGJA  -------------------------------
      END

      SUBROUTINE  DRBGPS (B, BD, IPBD)
C***BEGIN PROLOGUE  DRBGPS
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
C of the diagonal blocks computed in DRBGJA, and the mesh and
C block-grouping data in the COMMON blocks /DRPRE1/ and /RPRE2/.
C Here BD is the RWORK segment WP, and IPBD is the IWORK segment IWP.
C The right-hand side vector b, contained in B on entry, is overwritten
C with the solution vector x on return.
C-----------------------------------------------------------------------
C***ROUTINES CALLED
C   DGESL
C
C***END PROLOGUE  DRBGPS
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION B(*), BD(*), IPBD(*)
C
C The declaration of /RPRE2/ below must agree with that in DGSET2.
      PARAMETER (MAXM = 50)
      COMMON /DRPRE1/ SRUR, MP, MPD, MPSQ, MESHX, MESHY, MXMP
      COMMON /RPRE2/ NGX, NGY, NGRP, JGX(MAXM+1), JGY(MAXM+1),
     1               JIGX(MAXM), JIGY(MAXM), JXR(MAXM), JYR(MAXM)
C
      IER = 0
      IB = 1
      DO 20 JY = 1,MESHY
        IGY = JIGY(JY)
        IG0 = (IGY - 1)*NGX
        DO 10 JX = 1,MESHX
          IGX = JIGX(JX)
          IGM1 = IGX - 1 + IG0
          IBD = 1 + IGM1*MPSQ
          IIP = 1 + IGM1*MP
          CALL DGESL (BD(IBD), MP, MP, IPBD(IIP), B(IB), 0)
          IB = IB + MP
 10       CONTINUE
 20     CONTINUE
C
      RETURN
C------------  End of Subroutine DRBGPS  -------------------------------
      END
