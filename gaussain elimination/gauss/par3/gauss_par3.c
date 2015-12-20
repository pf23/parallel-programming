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

#define DEBUG

#define SWAP(a,b)       {double tmp; tmp = a; a = b; b = tmp;}

/* Solve the equation:
 *   matrix * X = R
 */

double **matrix, *X, *R;

/* Pre-set solution. */

double *X__;


int task_num = 1;
int nsize = 1;
int current_row = 0;
int *pivot;  //record the pivoting column number


pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

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

// modify the pivoting function to do column pivoting instead of row pivoting
// return the column number which is to be pivoted
/* Get the pivot - the element on column with largest absolute value. */
void getPivot(int nsize, int currow)
{
    int i, j, pivotcln;
    double pivotval = 1;

    // find out the column with largest absolute value on the current row
    // as pivot column
    pivotcln = currow;
    for (i = currow+1; i < nsize; i++) {
    	if (fabs(matrix[currow][i]) > fabs(matrix[currow][currow])) {
    	    pivotcln = i;
    	}
    }

    // record the column number for each iteration
    pivot[current_row] = pivotcln;
    //printf("pivot value row %d: %d\n",current_row, pivotcln );

    if (fabs(matrix[currow][pivotcln]) == 0.0) {
        fprintf(stderr, "The matrix is singular\n");
        exit(-1);
    }

    //scale the row
    pivotval = matrix[currow][pivotcln];
    if (pivotval != 1.0) 
    {
        for (j = currow; j < nsize; j++)
          matrix[currow][j] /= pivotval;
        R[currow] /= pivotval;
    }

    //printf("pivoted %d\n", currow);
}

//* barrier*//
// barrier also update the pivot and scale the main row
void
barrier (int expect)
{
    static int arrived = 0;

    pthread_mutex_lock (&mut);  //lock

    arrived++;
    if (arrived < expect)
        pthread_cond_wait (&cond, &mut);
    else 
    {
        // reset the barrier before broadcast is important
        arrived = 0;        

        // update the current row
        current_row++;
        
        pthread_cond_broadcast (&cond);
    }

    //printf("currow:%d\n", current_row);

    pthread_mutex_unlock (&mut);    //unlock
}


void reduce(int begin, int step)
{
    int j;
    for (j = begin; j <= nsize-1; j+=task_num) 
        { 
            pivotval = matrix[j][current_row];
            matrix[j][current_row] = 0.0;
            for (k = current_row + 1; k <= nsize-1; k++) 
            {
                matrix[j][k] -= pivotval * matrix[current_row][k];
            }
            R[j] -= pivotval * R[current_row];
            //printf("thread#%d row:%d\n", task_id, j);

            // if the row is the first row to reduce,
            // do column pivoting in itself after reduction
            if (j == current_row +1)
            {                
                getPivot(nsize, j);
            }
        }
}
// if the row is the first row to reduce,
// do column pivoting in itself after reduction
void *
work_thread (void *lp)
{
    int j,k;
    int task_id = *((int *) lp);

    double pivotval;

    //the first row to reduce
    int  begin = task_id + 1;

    for(;;)
    {
        if(current_row > nsize-1)
            break;
        if(begin > nsize-1)
        {
            barrier (task_num);
            continue;
        }

        // do column pivoting
        for (j = task_id; j<=nsize-1; j+=task_num)
        {
            SWAP(matrix[j][current_row],matrix[j][pivot[current_row]]);
        }

        /* do the GE work */
        reduce(begin, task_num);
        

        // wait fot other threads to synchronize and get next pivoting
        barrier (task_num);

        // calculate the next row to begin reduction
        if (current_row % task_num > task_id)
        {
            begin = (current_row/task_num + 1)*task_num + task_id + 1;
        }
        else
        {
            begin = (current_row/task_num)*task_num + task_id + 1;
        }
    }
    
}

/* For all the rows, get the pivot and eliminate all rows and columns
 * for that particular pivot row. */
void computeGauss(int nsize)
{
    int l, q;

    int *id;
    int ret;
    pthread_attr_t attr;
    pthread_t *tid;
    id = (int *) malloc (sizeof (int) * task_num);
    tid = (pthread_t *) malloc (sizeof (pthread_t) * task_num);
    
    if (!id || !tid)
        errexit ("out of shared memory");
    ret = pthread_attr_init (&attr);
    if ( ret != 0 )
    {
        printf("Error pthread_attr_init()\n");
    }
    pthread_attr_setscope (&attr, PTHREAD_SCOPE_SYSTEM);

    // pivot the first row
    getPivot(nsize, 0);

    /*
    // Scale the main row.
    pivotval = matrix[0][0];
    if (pivotval != 1.0) {
        matrix[0][0] = 1.0;
        for (j = 1; j < nsize; j++)
          matrix[0][j] /= pivotval;
        R[0] /= pivotval;
    }
    */

    // create threads
    for (l = 0; l < task_num; l++)
    {
        id[l] = l;
        ret = pthread_create (&tid[l], &attr, work_thread, &id[l]);
        if (ret != 0)
            printf("Error pthread_create()\n");
    }

    //* wait for all threads to finish *//
    for (q = 0; q < task_num; q++)
        pthread_join (tid[q], NULL);
}

// rearrange the solution
void rearrange(int nsize)
{
    int i;
    for (i = nsize-1; i>=0 ;i--)
    {
        SWAP(X[i], X[pivot[i]]);
        //printf("%d\n", pivot[i]);
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

    rearrange(nsize);

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

    //char *path = "./input_matrices/orsreg_1.dat";
    char *path = "./input_matrices/jpwh_991.dat";
    //char *path = "./input_matrices/saylr4.dat";

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
    pivot = (int *) malloc (sizeof(int) * nsize);

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
