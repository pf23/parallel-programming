/* 
 * Original author:  UNKNOWN
 *
 * Modified:    Fei Pang (September.24 2015)
 */

#ifndef _REENTRANT
#define _REENTRANT      /* basic 3-lines for threads */
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>
#include <pthread.h>

//#define DEBUG

#define SWAP(a,b)       {double tmp; tmp = a; a = b; b = tmp;}

/* Solve the equation:
 *   matrix * X = R
 */

double **matrix, *X, *R;

/* Pre-set solution. */

double *X__;


int task_num = 1;
int nsize = 1;
//* barrier*//

pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
//* struct to pass parameters to threads
typedef struct para
    {
        int id;
        int start;
        int step;
    } PARA;

void
barrier (int expect)
{
    static int arrived = 0;

    pthread_mutex_lock (&mut);  //lock

    arrived++;
    if (arrived < expect)
        pthread_cond_wait (&cond, &mut);
    else {
        arrived = 0;        // reset the barrier before broadcast is important
        pthread_cond_broadcast (&cond);
    }

    pthread_mutex_unlock (&mut);    //unlock
}

void
errexit (const char *err_str)
{
    fprintf (stderr, "%s", err_str);
}

/* Initialize the matirx. */

int initMatrix(const char *fname)
{
    FILE *file;
    int l1, l2, l3;
    double d;
    int nsize;
    int i, j;
    double *tmp;
    char buffer[1024];

    if ((file = fopen(fname, "r")) == NULL) {
	fprintf(stderr, "The matrix file open error\n");
        exit(-1);
    }
    
    /* Parse the first line to get the matrix size. */
    fgets(buffer, 1024, file);
    sscanf(buffer, "%d %d %d", &l1, &l2, &l3);
    nsize = l1;
#ifdef DEBUG
    fprintf(stdout, "matrix size is %d\n", nsize);
#endif

    /* Initialize the space and set all elements to zero. */
    matrix = (double**)malloc(nsize*sizeof(double*));
    assert(matrix != NULL);
    tmp = (double*)malloc(nsize*nsize*sizeof(double));
    assert(tmp != NULL);    
    for (i = 0; i < nsize; i++) {
        matrix[i] = tmp;
        tmp = tmp + nsize;
    }
    for (i = 0; i < nsize; i++) {
        for (j = 0; j < nsize; j++) {
            matrix[i][j] = 0.0;
        }
    }

    /* Parse the rest of the input file to fill the matrix. */
    for (;;) {
	fgets(buffer, 1024, file);
	sscanf(buffer, "%d %d %lf", &l1, &l2, &d);
	if (l1 == 0) break;

	matrix[l1-1][l2-1] = d;
#ifdef DEBUG
	fprintf(stdout, "row %d column %d of matrix is %e\n", l1-1, l2-1, matrix[l1-1][l2-1]);
#endif
    }

    fclose(file);
    return nsize;
}

/* Initialize the right-hand-side following the pre-set solution. */

void initRHS(int nsize)
{
    int i, j;

    X__ = (double*)malloc(nsize * sizeof(double));
    assert(X__ != NULL);
    for (i = 0; i < nsize; i++) {
	X__[i] = i+1;
    }

    R = (double*)malloc(nsize * sizeof(double));
    assert(R != NULL);
    for (i = 0; i < nsize; i++) {
	R[i] = 0.0;
	for (j = 0; j < nsize; j++) {
	    R[i] += matrix[i][j] * X__[j];
	}
    }
}

/* Initialize the results. */

void initResult(int nsize)
{
    int i;

    X = (double*)malloc(nsize * sizeof(double));
    assert(X != NULL);
    for (i = 0; i < nsize; i++) {
	X[i] = 0.0;
    }
}

/* Get the pivot - the element on column with largest absolute value. */

void getPivot(int nsize, int currow)
{
    int i, pivotrow;

    pivotrow = currow;
    for (i = currow+1; i < nsize; i++) {
	if (fabs(matrix[i][currow]) > fabs(matrix[pivotrow][currow])) {
	    pivotrow = i;
	}
    }

    if (fabs(matrix[pivotrow][currow]) == 0.0) {
        fprintf(stderr, "The matrix is singular\n");
        exit(-1);
    }
    
    if (pivotrow != currow) {
#ifdef DEBUG
	fprintf(stdout, "pivot row at step %5d is %5d\n", currow, pivotrow);
#endif
        for (i = currow; i < nsize; i++) {
            SWAP(matrix[pivotrow][i],matrix[currow][i]);
        }
        SWAP(R[pivotrow],R[currow]);
    }
}

void *
work_thread (void *lp)
{
    int j,k;
    PARA *recv_para;
    recv_para = (PARA *)lp;

    // receive parameters from struct
    int task_id = (*recv_para).id;
    int i = (*recv_para).start;
    int stp = (*recv_para).step;
    double pivotval;

    int begin, end;
    /*get the divided task*/
    begin = task_id*stp + i + 1;
    end = (begin + stp - 1) > (nsize-1) ? (nsize-1) : (begin + stp -1);

    /*
    if(begin > end)
        printf("#%d# thread %d: skip\n", i, task_id);
    else
        printf("#%d# thread %d: begin %d, end %d\n", i, task_id, begin, end);
        */
    barrier (task_num);

    /* do the GE work */
    for (j = begin; j <= end; j++) {
            pivotval = matrix[j][i];
            matrix[j][i] = 0.0;
            for (k = i+1; k <= nsize-1; k++) {
                matrix[j][k] -= pivotval * matrix[i][k];
            }
            R[j] -= pivotval * R[i];
        }
    barrier (task_num);
    //printf("threads #%d finished\n", task_id);
}

/* For all the rows, get the pivot and eliminate all rows and columns
 * for that particular pivot row. */

void computeGauss(int nsize)
{
    int i, j, l, q;
    int stp;
    double pivotval;

    int *id;
    int ret;
    pthread_attr_t attr;
    pthread_t *tid;
    PARA *p = (PARA *)malloc(sizeof (PARA) * task_num);
    id = (int *) malloc (sizeof (int) * task_num);
    tid = (pthread_t *) malloc (sizeof (pthread_t) * task_num);

    for (l = 0; l < task_num; l++)
    {
        id[l] = l;
        p[l].id = id[l];
    }
    
    if (!id || !tid)
        errexit ("out of shared memory");
    ret = pthread_attr_init (&attr);
    if ( ret != 0 )
    {
        printf("Error pthread_attr_init()\n");
    }
    pthread_attr_setscope (&attr, PTHREAD_SCOPE_SYSTEM);


    for (i = 0; i < nsize; i++) {
	getPivot(nsize, i);
    
	/* Scale the main row. */
    pivotval = matrix[i][i];
	if (pivotval != 1.0) {
	    matrix[i][i] = 1.0;
	    for (j = i + 1; j < nsize; j++) {
		  matrix[i][j] /= pivotval;
	    }
	    R[i] /= pivotval;
	}
    
	/* Factorize the rest of the matrix. */
    //* using parallel *//
    //* initialize threads *//
    //printf("////////////////////////////////////////////////////////////////\n");
    if( (nsize-1-i)%task_num == 0)
        stp = (nsize-1-i)/task_num;
    else
        stp = (nsize-1-i)/task_num + 1;

    for (l = 0; l < task_num; l++)
    {
        p[l].start = i;
        p[l].step = stp;
        ret = pthread_create (&tid[l], &attr, work_thread, &(p[l]));
        if (ret != 0)
            printf("Error pthread_create()\n");
    }

    //* wait for all threads to finish *//
    for (q = 0; q < task_num; q++)
        pthread_join (tid[q], NULL);

    }

}

/* Solve the equation. */

void solveGauss(int nsize)
{
    int i, j;

    X[nsize-1] = R[nsize-1];
    for (i = nsize - 2; i >= 0; i --) {
        X[i] = R[i];
        for (j = nsize - 1; j > i; j--) {
            X[i] -= matrix[i][j] * X[j];
        }
    }

#ifdef DEBUG
    fprintf(stdout, "X = [");
    for (i = 0; i < nsize; i++) {
        fprintf(stdout, "%.6f ", X[i]);
    }
    fprintf(stdout, "];\n");
#endif
}

extern char *optarg;

int main(int argc, char *argv[])
{
    int i, c;
    struct timeval start, finish;
    double error;

    char *path = "./input_matrices/jpwh_991.dat";

    while ((c = getopt (argc, argv, "m:p:")) != -1)
        switch (c) {
        case 'm':
            path = optarg;
            break;
        case 'p':
            task_num = atoi (optarg);
            break;
        }

    nsize = initMatrix(path);
    initRHS(nsize);
    initResult(nsize);

    gettimeofday(&start, 0);
    computeGauss(nsize);
    gettimeofday(&finish, 0);

    solveGauss(nsize);
    
    fprintf(stdout, "Time:  %f seconds\n", (finish.tv_sec - start.tv_sec) + (finish.tv_usec - start.tv_usec)*0.000001);

    error = 0.0;
    for (i = 0; i < nsize; i++) {
	double error__ = (X__[i]==0.0) ? 1.0 : fabs((X[i]-X__[i])/X__[i]);
	if (error < error__) {
	    error = error__;
	}
    }
    fprintf(stdout, "Error: %e\n", error);

    return 0;
}
