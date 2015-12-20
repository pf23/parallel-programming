#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
#define BLOCKS 1
#define THREADS_PER_BLOCK 256

void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
    int i;
    for ( i = 0; i < nbr_bin; i ++){
        hist_out[i] = 0;
    }

    for ( i = 0; i < img_size; i ++){
        hist_out[img_in[i]] ++;
    }

    //for debug
    /*
    for ( i = 0; i < nbr_bin; i ++){
        printf("%d ", hist_out[i]);
    }*/

}

void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin){
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, cdf, min, d;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    while(min == 0){
        min = hist_in[i++];
    }
    d = img_size - min;
    for(i = 0; i < nbr_bin; i ++){
        cdf += hist_in[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }
        
    }
    //for debug
    /*
    for ( i = 0; i < nbr_bin; i ++){
        printf("%d ", lut[i]);
    }*/
    /* Get the result image */
    for(i = 0; i < img_size; i ++){
        if(lut[img_in[i]] > 255){
            img_out[i] = 255;
        }
        else{
            img_out[i] = (unsigned char)lut[img_in[i]];
        }
        
    }
}

__global__ void histogram_gpu(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
    int i;
    //int hist_local[256];

    /*for ( i = 0 ; i < 256; i ++){
        hist_local[i] = 0;
    }*/

    if( threadIdx.x ==0 )
    {
        for ( i = 0 ; i < nbr_bin; i ++){
            hist_out[i] = 0;
        }
    }  
    __syncthreads();
    for ( i = threadIdx.x; i < img_size; i += THREADS_PER_BLOCK)
    {
        atomicAdd(&hist_out[img_in[i]],1);
    }

    //for debug
    /*
    __syncthreads();
    if(threadIdx.x==0)
    for( i=0;i<nbr_bin;i++)
    {
       printf("%d ", hist_out[i]);
    }*/
    
}

__global__ void histogram_equalization_gpu(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin){
    __shared__ int *lut;
    int i, cdf, min, d;

    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;

    if(threadIdx.x == 0)
    {
        lut = (int *)malloc(sizeof(int)*nbr_bin);
        while(min == 0){
                min = hist_in[i];
                i++;
            }

        d = img_size - min;

        for(i = 0; i < nbr_bin; i++){
            cdf += hist_in[i];
            //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
            lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
            if(lut[i] < 0){
                lut[i] = 0;
            }
            
        }
        // for debug
        /*
        for ( i = 0; i < nbr_bin; i ++){
            printf("%d ", lut[i]);
        }*/
    }
    __syncthreads();
    
    /* Get the result image */
    for(i = threadIdx.x; i < img_size; i+=THREADS_PER_BLOCK){

        //img_out[i] = (lut[img_in[i]] > 255)?255:(unsigned char)lut[img_in[i]];
        if(lut[img_in[i]] > 255){
            img_out[i] = 255;
        }
        else{
            img_out[i] = (unsigned char)lut[img_in[i]];
        }   
    }
}


