all: gauss_par

CC = gcc
CCFLAGS = -g -Wall -O3 -D_REENTRANT
LDFLAGS = -lm -pthread

gauss_par: gauss_par.o
	$(CC) $(LDFLAGS) -o gauss_par gauss_par.o

gauss_par.o: gauss_par.c
	$(CC) $(CCFLAGS) -c gauss_par.c -o gauss_par.o

clean:
	-rm *.o *~ gauss_par
