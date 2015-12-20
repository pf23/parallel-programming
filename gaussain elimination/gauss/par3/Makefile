all: gauss_par3

CC = gcc
CCFLAGS = -g -Wall -O3 -D_REENTRANT
LDFLAGS = -lm -lpthread

gauss_par3: gauss_par3.o
	$(CC) $(LDFLAGS) -o gauss_par3 gauss_par3.o

gauss_par3.o: gauss_par3.c
	$(CC) $(CCFLAGS) -c gauss_par3.c -o gauss_par3.o

clean:
	-rm *.o *~ gauss_par3
