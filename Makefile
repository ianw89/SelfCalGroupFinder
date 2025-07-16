
# ------------------------------------ #
# Setup Compiler and locate libraries
# ------------------------------------ #

# for my laptop
#hd = $(HOME)/cosmo/lib
#LIB = -lm -L${hd} -lcutil
#CC = gcc

# -- for sirocco/howdy/sirocco1
#CC = gcc
CC = g++
CFLAGS = -O3 -fopenmp -std=c++11	
CFLAGS += -Isrc/libs
#CFLAGS += -march=native
LIB = -lm -fopenmp -lgsl -lgslcblas
LIB += -L/mount/sirocco1/imw2293/GROUP_CAT/libs/gsl

# Ian's iMac
# For Apple Computers. Tested on an Intel iMac but should be OK with M based ones too

# YOU MUST RUN: brew install libomp argp-standalone
# CHECK THE PATH YOU INSTALLED THEM TO WITH brew ls libomp argp-standalone
# USE THOSE INCLUDE AND LIB PATHS IN THE CFLAGS AND LIB FLAGS

#CC = clang -Xclang -fopenmp # apple built-in clang
#CFLAGS = -I/usr/local/opt/libomp/include  -I/usr/local/Cellar/argp-standalone/1.3/include
#LIB = -L${LIB_DIR} -L/usr/local/opt/libomp/lib -L/usr/local/Cellar/argp-standalone/1.3/lib -lm -lomp -lcutil -largp

# ------------------------------------ #
# Define Files
# ------------------------------------ #

SRCDIR = src
ODIR = obj
BDIR = bin

_GF_OBJ =  main.o
GF_OBJ = $(patsubst %,$(ODIR)/%,$(_GF_OBJ))

_TESTS_OBJ = tests.o
TESTS_OBJ = $(patsubst %,$(ODIR)/%,$(_TESTS_OBJ))

_OBJ = kdGroupFinder_omp.o qromo.o midpnt.o polint.o sham.o spline.o splint.o \
	zbrent.o sort2.o kdtree.o fit_clustering_omp.o \
	group_center.o fof.o utils.o nrutil.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))


# ------------------------------------ #
# TARGETS
# ------------------------------------ #

# General entry point builds group finder and tests
main: $(BDIR)/kdGroupFinder_omp $(BDIR)/tests perf

perf: $(BDIR)/PerfGroupFinder $(BDIR)/tests

# Object file to c file dependencies
$(ODIR)/%.o: $(SRCDIR)/%.cpp
	$(CC) -c -o $@ $< $(CFLAGS)

# Build main program
$(BDIR)/kdGroupFinder_omp:	$(OBJ) $(GF_OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIB) -Wl,-rpath,/mount/sirocco1/imw2293/GROUP_CAT/libs/gsl

# Build main program for profiling
$(BDIR)/PerfGroupFinder:	$(OBJ) $(GF_OBJ)
	$(CC) -o $@ $^ $(CFLAGS) -g $(LIB) -Wl,-rpath,/mount/sirocco1/imw2293/GROUP_CAT/libs/gsl

# Build tests program
$(BDIR)/tests: $(OBJ) $(TESTS_OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIB) -Wl,-rpath,/mount/sirocco1/imw2293/GROUP_CAT/libs/gsl

clean:
	rm -f *.o $(ODIR)/*.o
	rm -f $(BDIR)/kdGroupFinder_omp
	rm -f $(BDIR)/PerfGroupFinder
	rm -f $(BDIR)/tests
