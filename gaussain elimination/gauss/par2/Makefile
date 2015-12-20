all: gauss_par2

CC = gcc
CCFLAGS = -g -Wall -O3 -D_REENTRANT
LDFLAGS = -lm -pthread

gauss_par2: gauss_par2.o
	$(CC) $(LDFLAGS) -o gauss_par2 gauss_par2.o

gauss_par2.o: gauss_par2.c
	$(CC) $(CCFLAGS) -c gauss_par2.c -o gauss_par2.o

clean:
	-rm *.o *~ gauss_par2
