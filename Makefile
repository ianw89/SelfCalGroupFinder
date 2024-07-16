# for my laptop
#hd = $(HOME)/cosmo/lib
#LIB = -lm -L${hd} -lcutil
#CC = gcc

# -- for sirocco
hd = $(HOME)/lib
LIB = -lm -fopenmp -L${hd} -lcutil 
CC = gcc
CFLAGS = -O2 -fopenmp

# Ian
#hd = $(HOME)/lib
#LIB = -lm -fopenmp -L${hd} -lcutil 
#CC = gcc-11
#CFLAGS = -O2 -fopenmp

SRCDIR = src
ODIR = obj
BDIR = bin

_GF_OBJ =  main.o
GF_OBJ = $(patsubst %,$(ODIR)/%,$(_GF_OBJ))

_TESTS_OBJ = tests.o
TESTS_OBJ = $(patsubst %,$(ODIR)/%,$(_TESTS_OBJ))

_OBJ = kdGroupFinder_omp.o qromo.o midpnt.o polint.o sham.o spline.o splint.o \
	zbrent.o sort2.o kdtree.o fit_clustering_omp.o gasdev.o ran1.o \
	group_center.o fof.o utils.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

main: $(BDIR)/kdGroupFinder_omp $(BDIR)/tests

$(ODIR)/%.o: $(SRCDIR)/%.c 
	$(CC) -c -o $@ $< $(CFLAGS)

# Build main program
$(BDIR)/kdGroupFinder_omp:	$(OBJ) $(GF_OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIB)

# Build tests program
$(BDIR)/tests: $(OBJ) $(TESTS_OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIB)

clean:
	rm -f *.o $(ODIR)/*.o
	rm -f $(BDIR)/kdGroupFinder_omp
	rm -f $(BDIR)/tests
