#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
#define THREADS_PER_BLOCK 128

__global__ void histogram_gpu(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
    int i;
    for ( i = threadIdx.x ; i < nbr_bin; i += THREADS_PER_BLOCK){
        hist_out[i] = 0;
    }

    __syncthreads();

    for ( i = threadIdx.x; i < img_size; i += THREADS_PER_BLOCK){
        hist_out[img_in[i]] ++;
    }
}

__global__ void histogram_equalization_gpu(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin){
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, cdf, min, d;
    __shared__ int min_of_all = 0;

    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = threadIdx.x;
    while(min == 0){
        min = hist_in[i];
        i+=THREADS_PER_BLOCK;
    }

    __syncthreads();
    if(threadIdx.x == 0)
        for (int j=0;j<THREADS_PER_BLOCK;j++)
            if(min_of_all>min)
                min_of_all = min;
    __syncthreads();

    min = min_of_all;
    d = img_size - min;
    for(i = 0; i < nbr_bin; i++){
        cdf += hist_in[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }
        
    }
    __syncthreads();
    
    /* Get the result image */
    for(i = threadIdx.x; i < img_size; i+=THREADS_PER_BLOCK){
        if(lut[img_in[i]] > 255){
            img_out[i] = 255;
        }
        else{
            img_out[i] = (unsigned char)lut[img_in[i]];
        }
        
    }
}



