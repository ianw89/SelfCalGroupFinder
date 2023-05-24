# for my laptop
#hd = $(HOME)/cosmo/lib
#LIB = -lm -L${hd} -lcutil
#CC = gcc

# -- for sirocco
#hd = $(HOME)/lib
#LIB = -lm -fopenmp -L${hd} -lcutil 
#CC = gcc
#CFLAGS = -O2 -fopenmp

# Ian
hd = $(HOME)/lib
LIB = -lm -fopenmp -L${hd} -lcutil 
CC = gcc-11
CFLAGS = -O2 -fopenmp
ODIR = obj
BDIR = bin

_OBJ = kdGroupFinder_omp.o qromo.o midpnt.o polint.o sham.o spline.o splint.o \
	zbrent.o sort2.o kdtree.o fit_clustering_omp.o gasdev.o ran1.o search.o \
	group_center.o fof.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

main: $(BDIR)/kdGroupFinder_omp

$(ODIR)/%.o: %.c 
	$(CC) -c -o $@ $< $(CFLAGS)

$(BDIR)/kdGroupFinder_omp:	$(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIB)
#	cp -f $@ $(HOME)/exec/$@

clean:
	rm -f *.o $(ODIR)/*.o
	rm -f $(BDIR)/kdGroupFinder_omp
