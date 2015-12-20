#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
#define BLOCKS 1
#define THREADS_PER_BLOCK 256

PGM_IMG contrast_enhancement_g(PGM_IMG img_in)
{
    PGM_IMG result;
    int hist[256];
    
    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    
    histogram(hist, img_in.img, img_in.h * img_in.w, 256);
    histogram_equalization(result.img,img_in.img,hist,result.w*result.h, 256);
    return result;
}

PGM_IMG contrast_enhancement_g_gpu(PGM_IMG img_in)
{
    PGM_IMG result;
    //int hist[256];
    
    result.w = img_in.w;
    result.h = img_in.h;
    int size = result.w * result.h * sizeof(unsigned char);
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    
    // allocate memory on gpu
    unsigned char *result_img;
    unsigned char *img_in_img;
    int *hist;
    int size2 = result.w * result.h;

    cudaMalloc( (void **)&hist, 256*sizeof(int));
    cudaMalloc( (void **)&img_in_img, size );
    cudaMalloc( (void **)&result_img, size );
    cudaMemcpy( img_in_img, img_in.img, size, cudaMemcpyHostToDevice);
    cudaMemcpy( result_img, result.img, size, cudaMemcpyHostToDevice);

    histogram_gpu<<<BLOCKS, THREADS_PER_BLOCK>>>(hist, img_in_img, size2, 256);
    histogram_equalization_gpu<<<BLOCKS, THREADS_PER_BLOCK>>>(result_img, img_in_img, hist, size2, 256); 

    cudaMemcpy( result.img, result_img, size, cudaMemcpyDeviceToHost );

    cudaFree(result_img);
    cudaFree(img_in_img);
    cudaFree(hist);

    return result;
}

PPM_IMG contrast_enhancement_c_rgb(PPM_IMG img_in)
{
    PPM_IMG result;
    int hist[256];
    
    result.w = img_in.w;
    result.h = img_in.h;
    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));  
    
    histogram(hist, img_in.img_r, img_in.h * img_in.w, 256);
    histogram_equalization(result.img_r,img_in.img_r,hist,result.w*result.h, 256);
    histogram(hist, img_in.img_g, img_in.h * img_in.w, 256);
    histogram_equalization(result.img_g,img_in.img_g,hist,result.w*result.h, 256);
    histogram(hist, img_in.img_b, img_in.h * img_in.w, 256);
    histogram_equalization(result.img_b,img_in.img_b,hist,result.w*result.h, 256);

    return result;
}

PPM_IMG contrast_enhancement_c_rgb_gpu(PPM_IMG img_in)
{
    PPM_IMG result;
    //int hist[256];
    
    result.w = img_in.w;
    result.h = img_in.h;
    int size = result.w * result.h * sizeof(unsigned char);
    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

    // allocate memory on gpu
    unsigned char *result_img_r;
    unsigned char *result_img_g;
    unsigned char *result_img_b;
    unsigned char *img_in_img_r;
    unsigned char *img_in_img_g;
    unsigned char *img_in_img_b;
    int *hist;
    int size2 = result.w * result.h;

    cudaMalloc( (void **)&hist, 256*sizeof(int));
    cudaMalloc( (void **)&img_in_img_r, size );
    cudaMalloc( (void **)&img_in_img_g, size );
    cudaMalloc( (void **)&img_in_img_b, size );
    cudaMalloc( (void **)&result_img_r, size );
    cudaMalloc( (void **)&result_img_g, size );
    cudaMalloc( (void **)&result_img_b, size );
    cudaMemcpy( img_in_img_r, img_in.img_r, size, cudaMemcpyHostToDevice);
    cudaMemcpy( img_in_img_g, img_in.img_g, size, cudaMemcpyHostToDevice);
    cudaMemcpy( img_in_img_b, img_in.img_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy( result_img_r, result.img_r, size, cudaMemcpyHostToDevice);
    cudaMemcpy( result_img_g, result.img_g, size, cudaMemcpyHostToDevice);
    cudaMemcpy( result_img_b, result.img_b, size, cudaMemcpyHostToDevice);
    
    histogram_gpu<<<BLOCKS,THREADS_PER_BLOCK>>>(hist, img_in.img_r, size2, 256);
    histogram_equalization_gpu<<<BLOCKS,THREADS_PER_BLOCK>>>(result.img_r,img_in.img_r,hist, size2, 256);
    histogram_gpu<<<BLOCKS,THREADS_PER_BLOCK>>>(hist, img_in.img_g, size2, 256);
    histogram_equalization_gpu<<<BLOCKS,THREADS_PER_BLOCK>>>(result.img_g,img_in.img_g,hist, size2, 256);
    histogram_gpu<<<BLOCKS,THREADS_PER_BLOCK>>>(hist, img_in.img_b, size2, 256);
    histogram_equalization_gpu<<<BLOCKS,THREADS_PER_BLOCK>>>(result.img_b,img_in.img_b,hist, size2, 256);

    cudaMemcpy( result.img_r, result_img_r, size, cudaMemcpyDeviceToHost );
    cudaMemcpy( result.img_g, result_img_g, size, cudaMemcpyDeviceToHost );
    cudaMemcpy( result.img_b, result_img_b, size, cudaMemcpyDeviceToHost );

    cudaFree(result_img_r);
    cudaFree(result_img_g);
    cudaFree(result_img_b);
    cudaFree(img_in_img_r);
    cudaFree(img_in_img_g);
    cudaFree(img_in_img_b);
    cudaFree(hist);

    return result;
}

PPM_IMG contrast_enhancement_c_yuv(PPM_IMG img_in)
{
    YUV_IMG yuv_med;
    PPM_IMG result;
    
    unsigned char * y_equ;
    int hist[256];
    
    yuv_med = rgb2yuv(img_in);
    y_equ = (unsigned char *)malloc(yuv_med.h*yuv_med.w*sizeof(unsigned char));
    
    histogram(hist, yuv_med.img_y, yuv_med.h * yuv_med.w, 256);
    histogram_equalization(y_equ,yuv_med.img_y,hist,yuv_med.h * yuv_med.w, 256);

    free(yuv_med.img_y);
    yuv_med.img_y = y_equ;
    
    result = yuv2rgb(yuv_med);
    free(yuv_med.img_y);
    free(yuv_med.img_u);
    free(yuv_med.img_v);
    
    return result;
}

PPM_IMG contrast_enhancement_c_yuv_gpu(PPM_IMG img_in)
{
    YUV_IMG yuv_med;
    PPM_IMG result;
    
    unsigned char * y_equ;
    //int hist[256];
    
    yuv_med = rgb2yuv_gpu(img_in);
    int size = yuv_med.h * yuv_med.w;
    y_equ = (unsigned char *)malloc(size*sizeof(unsigned char));
    
    // allocate memmory on gpu
    int *hist;
    unsigned char* yuv_med_y;
    unsigned char* y_equ_g;

    cudaMalloc( (void **)&hist, 256*sizeof(int));
    cudaMalloc( (void **)&yuv_med_y, size*sizeof(unsigned char) );
    cudaMalloc( (void **)&y_equ_g, size*sizeof(unsigned char) );

    cudaMemcpy( yuv_med_y, yuv_med.img_y, size*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy( y_equ_g, y_equ, size*sizeof(unsigned char), cudaMemcpyHostToDevice);

    histogram_gpu<<<BLOCKS, THREADS_PER_BLOCK>>>(hist, yuv_med_y, size, 256);
    histogram_equalization_gpu<<<BLOCKS, THREADS_PER_BLOCK>>>(y_equ_g,yuv_med_y,hist,size, 256);

    cudaMemcpy( y_equ, y_equ_g, size*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    free(yuv_med.img_y);
    yuv_med.img_y = y_equ;
    
    result = yuv2rgb_gpu(yuv_med);
    free(yuv_med.img_y);
    free(yuv_med.img_u);
    free(yuv_med.img_v);

    cudaFree(hist);
    cudaFree(yuv_med_y);
    cudaFree(y_equ_g);

    return result;
}

PPM_IMG contrast_enhancement_c_hsl(PPM_IMG img_in)
{
    HSL_IMG hsl_med;
    PPM_IMG result;
    
    unsigned char * l_equ;
    int hist[256];

    hsl_med = rgb2hsl(img_in);
    l_equ = (unsigned char *)malloc(hsl_med.height*hsl_med.width*sizeof(unsigned char));

    histogram(hist, hsl_med.l, hsl_med.height * hsl_med.width, 256);
    histogram_equalization(l_equ, hsl_med.l,hist,hsl_med.width*hsl_med.height, 256);
    
    free(hsl_med.l);
    hsl_med.l = l_equ;

    result = hsl2rgb(hsl_med);
    free(hsl_med.h);
    free(hsl_med.s);
    free(hsl_med.l);
    return result;
}

PPM_IMG contrast_enhancement_c_hsl_gpu(PPM_IMG img_in)
{

    HSL_IMG hsl_med;
    PPM_IMG result;
    
    unsigned char * l_equ;
    //int hist[256];
    
    hsl_med = rgb2hsl_gpu(img_in);
    int size = hsl_med.height * hsl_med.width;
    l_equ = (unsigned char *)malloc(size*sizeof(unsigned char));
    
    // allocate memmory on gpu
    int *hist;
    unsigned char* hsl_med_l;
    unsigned char* l_equ_g;

    cudaMalloc( (void **)&hist, 256*sizeof(int));
    cudaMalloc( (void **)&hsl_med_l, size*sizeof(unsigned char) );
    cudaMalloc( (void **)&l_equ_g, size*sizeof(unsigned char) );

    cudaMemcpy( hsl_med_l, hsl_med.l, size*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy( l_equ_g, l_equ, size*sizeof(unsigned char), cudaMemcpyHostToDevice);

    histogram_gpu<<<BLOCKS, THREADS_PER_BLOCK>>>(hist, hsl_med_l, size, 256);
    histogram_equalization_gpu<<<BLOCKS, THREADS_PER_BLOCK>>>(l_equ_g,hsl_med_l,hist,size, 256);

    cudaMemcpy( l_equ, l_equ_g, size*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    free(hsl_med.l);
    hsl_med.l = l_equ;
    
    result = hsl2rgb_gpu(hsl_med);
    free(hsl_med.h);
    free(hsl_med.s);
    free(hsl_med.l);

    cudaFree(hist);
    cudaFree(hsl_med_l);
    cudaFree(l_equ_g);

    return result;
}


//Convert RGB to HSL, assume R,G,B in [0, 255]
//Output H, S in [0.0, 1.0] and L in [0, 255]
HSL_IMG rgb2hsl(PPM_IMG img_in)
{
    int i;
    float H, S, L;
    HSL_IMG img_out;// = (HSL_IMG *)malloc(sizeof(HSL_IMG));
    img_out.width  = img_in.w;
    img_out.height = img_in.h;
    img_out.h = (float *)malloc(img_in.w * img_in.h * sizeof(float));
    img_out.s = (float *)malloc(img_in.w * img_in.h * sizeof(float));
    img_out.l = (unsigned char *)malloc(img_in.w * img_in.h * sizeof(unsigned char));
    
    for(i = 0; i < img_in.w*img_in.h; i ++){
        
        float var_r = ( (float)img_in.img_r[i]/255 );//Convert RGB to [0,1]
        float var_g = ( (float)img_in.img_g[i]/255 );
        float var_b = ( (float)img_in.img_b[i]/255 );
        float var_min = (var_r < var_g) ? var_r : var_g;
        var_min = (var_min < var_b) ? var_min : var_b;   //min. value of RGB
        float var_max = (var_r > var_g) ? var_r : var_g;
        var_max = (var_max > var_b) ? var_max : var_b;   //max. value of RGB
        float del_max = var_max - var_min;               //Delta RGB value
        
        L = ( var_max + var_min ) / 2;
        if ( del_max == 0 )//This is a gray, no chroma...
        {
            H = 0;         
            S = 0;    
        }
        else                                    //Chromatic data...
        {
            if ( L < 0.5 )
                S = del_max/(var_max+var_min);
            else
                S = del_max/(2-var_max-var_min );

            float del_r = (((var_max-var_r)/6)+(del_max/2))/del_max;
            float del_g = (((var_max-var_g)/6)+(del_max/2))/del_max;
            float del_b = (((var_max-var_b)/6)+(del_max/2))/del_max;
            if( var_r == var_max ){
                H = del_b - del_g;
            }
            else{       
                if( var_g == var_max ){
                    H = (1.0/3.0) + del_r - del_b;
                }
                else{
                        H = (2.0/3.0) + del_g - del_r;
                }   
            }
            
        }
        
        if ( H < 0 )
            H += 1;
        if ( H > 1 )
            H -= 1;

        img_out.h[i] = H;
        img_out.s[i] = S;
        img_out.l[i] = (unsigned char)(L*255);
    }
    
    return img_out;
}

// gpu version
HSL_IMG rgb2hsl_gpu(PPM_IMG img_in)
{
    //int i;
    //float H, S, L;
    HSL_IMG img_out;// = (HSL_IMG *)malloc(sizeof(HSL_IMG));
    img_out.width  = img_in.w;
    img_out.height = img_in.h;
    img_out.h = (float *)malloc(img_in.w * img_in.h * sizeof(float));
    img_out.s = (float *)malloc(img_in.w * img_in.h * sizeof(float));
    img_out.l = (unsigned char *)malloc(img_in.w * img_in.h * sizeof(unsigned char));
    
    // allocate gpu memory
    unsigned char *r, *g, *b;
    float *h, *s;
    unsigned char *l;
    int size = img_in.w * img_in.h;

    cudaMalloc((void **)&r, size*sizeof(unsigned char));
    cudaMalloc((void **)&g, size*sizeof(unsigned char));
    cudaMalloc((void **)&b, size*sizeof(unsigned char));
    cudaMalloc((void **)&h, size*sizeof(float));
    cudaMalloc((void **)&s, size*sizeof(float));
    cudaMalloc((void **)&l, size*sizeof(unsigned char));

    // copy memory from host to device
    cudaMemcpy(r, img_in.img_r, size*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(g, img_in.img_g, size*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(b, img_in.img_b, size*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(h, img_out.h, size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(s, img_out.s, size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l, img_out.l, size*sizeof(unsigned char), cudaMemcpyHostToDevice);

    rgb2hsl_call<<<BLOCKS, THREADS_PER_BLOCK>>>(r,g,b,h,s,l,img_in.w,img_in.h);   

    // copy memory from device to host
    cudaMemcpy(img_out.h, h, size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.s, s, size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.l, l, size*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // free gpu memory
    cudaFree(r);
    cudaFree(g);
    cudaFree(b);
    cudaFree(h);
    cudaFree(s);
    cudaFree(l);

    return img_out;
}

__global__ void rgb2hsl_call(   unsigned char* in_r, unsigned char* in_g, unsigned char* in_b,
                                float* out_h, float* out_s, unsigned char* out_l, int w, int h )
{
    int size = w*h;
    int i;
    float H, S, L;
    for(i = threadIdx.x; i < size; i += THREADS_PER_BLOCK)
    {
        
        float var_r = ( (float)in_r[i]/255 );//Convert RGB to [0,1]
        float var_g = ( (float)in_g[i]/255 );
        float var_b = ( (float)in_b[i]/255 );
        float var_min = (var_r < var_g) ? var_r : var_g;
        var_min = (var_min < var_b) ? var_min : var_b;   //min. value of RGB
        float var_max = (var_r > var_g) ? var_r : var_g;
        var_max = (var_max > var_b) ? var_max : var_b;   //max. value of RGB
        float del_max = var_max - var_min;               //Delta RGB value
        
        L = ( var_max + var_min ) / 2;
        if ( del_max == 0 )//This is a gray, no chroma...
        {
            H = 0;         
            S = 0;    
        }
        else                                    //Chromatic data...
        {
            if ( L < 0.5 )
                S = del_max/(var_max+var_min);
            else
                S = del_max/(2-var_max-var_min );

            float del_r = (((var_max-var_r)/6)+(del_max/2))/del_max;
            float del_g = (((var_max-var_g)/6)+(del_max/2))/del_max;
            float del_b = (((var_max-var_b)/6)+(del_max/2))/del_max;
            if( var_r == var_max ){
                H = del_b - del_g;
            }
            else{       
                if( var_g == var_max ){
                    H = (1.0/3.0) + del_r - del_b;
                }
                else{
                        H = (2.0/3.0) + del_g - del_r;
                }   
            }
            
        }
        
        if ( H < 0 )
            H += 1;
        if ( H > 1 )
            H -= 1;

        out_h[i] = H;
        out_s[i] = S;
        out_l[i] = (unsigned char)(L*255);
    }

}

float Hue_2_RGB( float v1, float v2, float vH )             //Function Hue_2_RGB
{
    if ( vH < 0 ) vH += 1;
    if ( vH > 1 ) vH -= 1;
    if ( ( 6 * vH ) < 1 ) return ( v1 + ( v2 - v1 ) * 6 * vH );
    if ( ( 2 * vH ) < 1 ) return ( v2 );
    if ( ( 3 * vH ) < 2 ) return ( v1 + ( v2 - v1 ) * ( ( 2.0f/3.0f ) - vH ) * 6 );
    return ( v1 );
}

//Convert HSL to RGB, assume H, S in [0.0, 1.0] and L in [0, 255]
//Output R,G,B in [0, 255]
PPM_IMG hsl2rgb(HSL_IMG img_in)
{
    int i;
    PPM_IMG result;
    
    result.w = img_in.width;
    result.h = img_in.height;
    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    
    for(i = 0; i < img_in.width*img_in.height; i ++){
        float H = img_in.h[i];
        float S = img_in.s[i];
        float L = img_in.l[i]/255.0f;
        float var_1, var_2;
        
        unsigned char r,g,b;
        
        if ( S == 0 )
        {
            r = L * 255;
            g = L * 255;
            b = L * 255;
        }
        else
        {
            
            if ( L < 0.5 )
                var_2 = L * ( 1 + S );
            else
                var_2 = ( L + S ) - ( S * L );

            var_1 = 2 * L - var_2;
            r = 255 * Hue_2_RGB( var_1, var_2, H + (1.0f/3.0f) );
            g = 255 * Hue_2_RGB( var_1, var_2, H );
            b = 255 * Hue_2_RGB( var_1, var_2, H - (1.0f/3.0f) );
        }
        result.img_r[i] = r;
        result.img_g[i] = g;
        result.img_b[i] = b;
    }

    return result;
}

// gpu version
PPM_IMG hsl2rgb_gpu(HSL_IMG img_in)
{
    //int i;
    PPM_IMG result;
    
    result.w = img_in.width;
    result.h = img_in.height;
    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    
    // allocate gpu memory
    float *h, *s;
    unsigned char *l;
    unsigned char *r, *g, *b;
    int size = result.w*result.h;

    cudaMalloc((void **)&r, size*sizeof(unsigned char));
    cudaMalloc((void **)&g, size*sizeof(unsigned char));
    cudaMalloc((void **)&b, size*sizeof(unsigned char));
    cudaMalloc((void **)&h, size*sizeof(float));
    cudaMalloc((void **)&s, size*sizeof(float));
    cudaMalloc((void **)&l, size*sizeof(unsigned char));

    // copy memory from host to device
    cudaMemcpy(r, result.img_r, size*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(g, result.img_g, size*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(b, result.img_b, size*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(h, img_in.h, size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(s, img_in.s, size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l, img_in.l, size*sizeof(unsigned char), cudaMemcpyHostToDevice);

    hsl2rgb_call<<<BLOCKS, THREADS_PER_BLOCK>>>(h,s,l,r,g,b,result.w,result.h);   

    // copy memory from device to host
    cudaMemcpy(result.img_r, r, size*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(result.img_g, g, size*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(result.img_b, b, size*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // free gpu memory
    cudaFree(r);
    cudaFree(g);
    cudaFree(b);
    cudaFree(h);
    cudaFree(s);
    cudaFree(l);

    return result;
}

__global__ void hsl2rgb_call(   float *in_h, float *in_s, unsigned char *in_l, 
                                unsigned char *out_r, unsigned char *out_g, unsigned char *out_b, int w, int h)
{
    int i;
    int size = w*h;
    for(i = threadIdx.x; i < size; i += THREADS_PER_BLOCK){
        float H = in_h[i];
        float S = in_s[i];
        float L = in_l[i]/255.0f;
        float var_1, var_2;
        
        unsigned char r,g,b;
        
        if ( S == 0 )
        {
            r = L * 255;
            g = L * 255;
            b = L * 255;
        }
        else
        {
            
            if ( L < 0.5 )
                var_2 = L * ( 1 + S );
            else
                var_2 = ( L + S ) - ( S * L );

            var_1 = 2 * L - var_2;

            float value[3];
            float vH;

            for (int j=0;j<3;j++)
            {
                vH = H - (j-1)*(1.0f/3.0f);
                if ( vH < 0 ) vH += 1;
                if ( vH > 1 ) vH -= 1;
                if ( ( 6 * vH ) < 1 ) 
                    value[j] = var_1 + ( var_2 - var_1 ) * 6 * vH ;
                else if ( ( 2 * vH ) < 1 ) 
                        value[j] = var_2 ;
                else if ( ( 3 * vH ) < 2 ) 
                        value[j] = var_1 + ( var_2 - var_1 ) * ( ( 2.0f/3.0f ) - vH ) * 6;
                else
                    value[j] =  var_1;
            }

            //r = 255 * Hue_2_RGB( var_1, var_2, H + (1.0f/3.0f) );
            //g = 255 * Hue_2_RGB( var_1, var_2, H );
            //b = 255 * Hue_2_RGB( var_1, var_2, H - (1.0f/3.0f) );
            r = 255 * value[0];
            g = 255 * value[1];
            b = 255 * value[2];
        }
        out_r[i] = r;
        out_g[i] = g;
        out_b[i] = b;
    }
}

//Convert RGB to YUV, all components in [0, 255]
YUV_IMG rgb2yuv(PPM_IMG img_in)
{
    YUV_IMG img_out;
    int i;//, j;
    unsigned char r, g, b;
    unsigned char y, cb, cr;
    
    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_y = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_u = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_v = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);

    for(i = 0; i < img_out.w*img_out.h; i ++){
        r = img_in.img_r[i];
        g = img_in.img_g[i];
        b = img_in.img_b[i];
        
        y  = (unsigned char)( 0.299*r + 0.587*g +  0.114*b);
        cb = (unsigned char)(-0.169*r - 0.331*g +  0.499*b + 128);
        cr = (unsigned char)( 0.499*r - 0.418*g - 0.0813*b + 128);
        
        img_out.img_y[i] = y;
        img_out.img_u[i] = cb;
        img_out.img_v[i] = cr;
    }
    
    return img_out;
}

// gpu version
YUV_IMG rgb2yuv_gpu(PPM_IMG img_in)
{
    YUV_IMG img_out;
    
    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_y = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_u = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_v = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);

    // allocate memory on gpu
    unsigned char *r, *g, *b, *y, *u, *v;
    int size = img_out.w*img_out.h;

    cudaMalloc((void **)&r, size*sizeof(unsigned char));
    cudaMalloc((void **)&g, size*sizeof(unsigned char));
    cudaMalloc((void **)&b, size*sizeof(unsigned char));
    cudaMalloc((void **)&y, size*sizeof(unsigned char));
    cudaMalloc((void **)&u, size*sizeof(unsigned char));
    cudaMalloc((void **)&v, size*sizeof(unsigned char));

    // copy memory from host to device
    cudaMemcpy(r, img_in.img_r, size*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(g, img_in.img_g, size*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(b, img_in.img_b, size*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(y, img_out.img_y, size*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(u, img_out.img_u, size*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(v, img_out.img_v, size*sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    // call gpu function
    rgb2yuv_call<<<BLOCKS, THREADS_PER_BLOCK>>>(r,g,b,y,u,v,img_out.w,img_out.h);

    // copy memory from device to host
    cudaMemcpy(img_out.img_y, y, size*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_u, u, size*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_v, v, size*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // free gpu memory
    cudaFree(r);
    cudaFree(g);
    cudaFree(b);
    cudaFree(y);
    cudaFree(u);
    cudaFree(v);

    return img_out;
}

__global__ void rgb2yuv_call(   unsigned char *in_r, unsigned char *in_g, unsigned char *in_b,
                                unsigned char *out_y, unsigned char *out_u, unsigned char *out_v, int w, int h)
{
    int i;
    unsigned char r, g, b;
    unsigned char y, cb, cr;
    int size = w*h;

    for(i = threadIdx.x; i < size; i += THREADS_PER_BLOCK){
        r = in_r[i];
        g = in_g[i];
        b = in_b[i];
        
        y  = (unsigned char)( 0.299*r + 0.587*g +  0.114*b);
        cb = (unsigned char)(-0.169*r - 0.331*g +  0.499*b + 128);
        cr = (unsigned char)( 0.499*r - 0.418*g - 0.0813*b + 128);
        
        out_y[i] = y;
        out_u[i] = cb;
        out_v[i] = cr;
    }
}

unsigned char clip_rgb(int x)
{
    if(x > 255)
        return 255;
    if(x < 0)
        return 0;

    return (unsigned char)x;
}

//Convert YUV to RGB, all components in [0, 255]
PPM_IMG yuv2rgb(YUV_IMG img_in)
{
    PPM_IMG img_out;
    int i;
    int  rt,gt,bt;
    int y, cb, cr;
    
    
    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_r = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_g = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_b = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);

    for(i = 0; i < img_out.w*img_out.h; i ++){
        y  = (int)img_in.img_y[i];
        cb = (int)img_in.img_u[i] - 128;
        cr = (int)img_in.img_v[i] - 128;
        
        rt  = (int)( y + 1.402*cr);
        gt  = (int)( y - 0.344*cb - 0.714*cr);
        bt  = (int)( y + 1.772*cb);

        img_out.img_r[i] = clip_rgb(rt);
        img_out.img_g[i] = clip_rgb(gt);
        img_out.img_b[i] = clip_rgb(bt);
    }
    
    return img_out;
}

// gpu version
PPM_IMG yuv2rgb_gpu(YUV_IMG img_in)
{
    PPM_IMG img_out;
    
    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_r = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_g = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_b = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);

    // allocate memory on gpu
    unsigned char *r, *g, *b, *y, *u, *v;
    int size = img_out.w*img_out.h;

    cudaMalloc((void **)&r, size*sizeof(unsigned char));
    cudaMalloc((void **)&g, size*sizeof(unsigned char));
    cudaMalloc((void **)&b, size*sizeof(unsigned char));
    cudaMalloc((void **)&y, size*sizeof(unsigned char));
    cudaMalloc((void **)&u, size*sizeof(unsigned char));
    cudaMalloc((void **)&v, size*sizeof(unsigned char));

    // copy memory from host to device
    cudaMemcpy(r, img_out.img_r, size*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(g, img_out.img_g, size*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(b, img_out.img_b, size*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(y, img_in.img_y, size*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(u, img_in.img_u, size*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(v, img_in.img_v, size*sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    // call gpu function
    yuv2rgb_call<<<BLOCKS, THREADS_PER_BLOCK>>>(y,u,v,r,g,b,img_out.w,img_out.h);

    // copy memory from device to host

    cudaMemcpy(img_out.img_r, r, size*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_g, g, size*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_b, b, size*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // free gpu memory
    cudaFree(r);
    cudaFree(g);
    cudaFree(b);
    cudaFree(y);
    cudaFree(u);
    cudaFree(v);

    return img_out;
}

__global__ void yuv2rgb_call(   unsigned char *in_y, unsigned char *in_u, unsigned char *in_v,
                                unsigned char *out_r, unsigned char *out_g, unsigned char *out_b, int w, int h)
{
    int i;
    int rt,gt,bt;
    int y, cb, cr;
    int size = w*h;

    for(i = threadIdx.x; i < size; i +=THREADS_PER_BLOCK ){
        y  = (int)in_y[i];
        cb = (int)in_u[i] - 128;
        cr = (int)in_v[i] - 128;
        
        rt  = (int)( y + 1.402*cr);
        gt  = (int)( y - 0.344*cb - 0.714*cr);
        bt  = (int)( y + 1.772*cb);

        //out_r[i] = clip_rgb(rt);
        //out_g[i] = clip_rgb(gt);
        //out_b[i] = clip_rgb(bt);
        //out_r[i] = (unsigned char)( (rt>255?255:rt)<0?0:rt );
        //out_g[i] = (unsigned char)( (gt>255?255:rt)<0?0:gt );
        //out_b[i] = (unsigned char)( (bt>255?255:rt)<0?0:bt );
        if(rt>255)
            out_r[i] = 255;
        else if(rt<0)
            out_r[i] = 0;
        else
            out_r[i] = (unsigned char)(rt);

        if(gt>255)
            out_g[i] = 255;
        else if(gt<0)
            out_g[i] = 0;
        else
            out_g[i] = (unsigned char)(gt);

        if(bt>255)
            out_b[i] = 255;
        else if(bt<0)
            out_b[i] = 0;
        else
            out_b[i] = (unsigned char)(bt);

    }

}






