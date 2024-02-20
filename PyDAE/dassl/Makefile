################################################################################
#
#   Makefile for DASSL
#
################################################################################

F77=gfortran

CFLAGS=-fPIC -O3

DOBJ=daux.o ddassl.o dlinpk.o

DLIB=libddassl.a

all: $(DLIB)

$(DLIB): $(DOBJ)
	ar rcs $(DLIB) $(DOBJ)

%.o: %.f
	$(F77) $(CFLAGS) -c $< -o $@
	
clean:
	rm -f $(DLIB) $(DOBJ)
