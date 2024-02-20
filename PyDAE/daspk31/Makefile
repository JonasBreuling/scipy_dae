################################################################################
#
#   Makefile for DASPK3.1
#
################################################################################

F77=gfortran

CFLAGS=-fPIC -O3

DOBJ=adf_dummy.o daux.o ddaskr.o ddaspk.o dlinpk.o dsensd.o mpi_dummy.o

DLIB=libddaspk31.a

all: $(DLIB)

$(DLIB): $(DOBJ)
	ar rcs $(DLIB) $(DOBJ)

%.o: %.f
	$(F77) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(DLIB) $(DOBJ)
