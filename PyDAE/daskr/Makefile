################################################################################
#
#   Makefile for DASKR
#
################################################################################

F77=gfortran

CFLAGS=-fPIC -O3

DOBJ=solver/daux.o solver/ddaskr.o solver/dlinpk.o preconds/dbanpre.o preconds/dilupre.o preconds/drbdpre.o preconds/drbgpre.o preconds/dsparsk.o

SOBJ=solver/saux.o solver/sdaskr.o solver/slinpk.o preconds/sbanpre.o preconds/silupre.o preconds/srbdpre.o preconds/srbgpre.o preconds/ssparsk.o

DLIB=libddaskr.a

SLIB=libsdaskr.a

all: $(DLIB) $(SLIB)

$(DLIB): $(DOBJ)
	ar rcs $(DLIB) $(DOBJ)

$(SLIB): $(SOBJ)
	ar rcs $(SLIB) $(SOBJ)

%.o: %.f
	$(F77) $(CFLAGS) -c $< -o $@
	
clean:
	rm -f $(DLIB) $(DOBJ) $(SLIB) $(SOBJ)
