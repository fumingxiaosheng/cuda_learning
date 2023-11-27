//https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/ tensor-cuda概述
//https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma 官方文档的编程接口
//几个知乎上的例子代码: https://zhuanlan.zhihu.com/p/353208013 https://zhuanlan.zhihu.com/p/620766588
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<iostream>
#include <mma.h>
#include <cuda_fp16.h>


#include <stdio.h>
#include <chrono>
#include <cstdlib>
#include<ctime>

//using namespace std;
using namespace nvcuda;

#define M 16
#define N 16
#define K 16
#define newType 1
__global__ void wmma_ker(half *a, half *b, float *c) {
    printf("this is test\n");
    
    // Declare the fragments
   wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
   wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

   // Initialize the output to zero
   wmma::fill_fragment(c_frag, 0.0f);

   // Load the inputs
   wmma::load_matrix_sync(a_frag, a, 16);
   wmma::load_matrix_sync(b_frag, b, 16);

   // Perform the matrix multiplication
   wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

   // Store the output
   wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
   c[1]=2;

}
__global__ void hxw(){
    printf("hxw");
}

/*
void matrix_init(half a[],half b[],float c[]){ //TODO:2023-11-26:目前测试结果来看是没有计算的，初始化一下再看看
    for(int i=0;i<M*K;i++){
        a[i]=static_cast<half>(1);
    }
    for(int i=0;i<N*K;i++){
        b[i]=static_cast<half>(1);
    }
    for(int i=0;i<M*N;i++){
        c[i]=1;
    }
}
*/
void matrix_init(half a[], half b[], float c[]) {
    for (int i = 0; i < M * K; i++) {
        a[i] = __float2half(1.0f);  // 使用 __float2half 将 float 转换为 half
    }
    for (int i = 0; i < N * K; i++) {
        b[i] = __float2half(1.0f);  // 使用 __float2half 将 float 转换为 half
    }
    for (int i = 0; i < M * N; i++) {
        c[i] = 1.0f;  // 使用 float 直接初始化
    }
}
void myprint(half a[],half b[],float c[]){
    //输出矩阵a
    printf("matrix a:\n");
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            printf("%f ",__half2float(a[i*M+j])); //debug:non-POD class type passed through ellipsis
        }
        printf("\n");
    }

    printf("matrix b:\n");
    for(int i=0;i<K;i++){
        for(int j=0;j<N;j++){
            printf("%f ",__half2float(b[i*K+j])); //debug:non-POD class type passed through ellipsis 注意这里不能直接输出b[i*K+j]，而首先需要转化为float类型
        }
        printf("\n");
    }

    printf("matrix c:\n");
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            printf("%f ",c[i*M+j]); //debug:non-POD class type passed through ellipsis
        }
        printf("\n");
    }
}


int main(){
    printf("in main\n");
    /*
    half *ha[M][K];
    half *hb[K][N];
    float *hc[M][N];
    */
#ifdef mallocType
    half *ha;
    half *hb;
    float *hc;
 
    ha = (half *)malloc(sizeof(half)*M*K);//这边不能使用ha[i]来访问数组中的元素
    hb = (half *)malloc(sizeof(half)*K*N);
    hc = (float *)malloc(sizeof(float)*M*N);
#endif

#ifdef newType
    half *ha = new half[M*K];
    half *hb = new half[K*N];
    float *hc = new float [M*N];
    matrix_init(ha,hb,hc);
    //ha[1]=0; //error: calling a __device__ function("operator=") from a __host__ function("main") is not allowed ->half类型在cuda中进行了定义，因此此时会引发类似于设备端函数调用的错误
#endif


    half *da;
    half *db;
    float *dc;
    cudaMalloc(&da,sizeof(half)*M*K);
    cudaMalloc(&db,sizeof(half)*N*K);
    cudaMalloc(&dc,sizeof(float)*M*N);


    //初始化device端的函数
    cudaMemcpy(da,ha,sizeof(half)*M*K,cudaMemcpyHostToDevice);
    cudaMemcpy(db,hb,sizeof(half)*N*K,cudaMemcpyHostToDevice);
    cudaMemcpy(dc,hc,sizeof(float)*M*N,cudaMemcpyHostToDevice);



    printf("before kernel\n");
    myprint(ha,hb,hc);
    //myprint(ha,hb,hc);

    dim3 gridDim(1, 1, 1);
    dim3 blockDim(32, 1, 1);

    wmma_ker<<<gridDim,blockDim>>>(da,db,dc);
    //hxw<<<1,32>>>(); //似乎在arch模式下，不会输出相应的值
    //下面的目标是通过具体的值来看kernel函数是否产生了效果

    cudaDeviceSynchronize();//等待设备端执行完成

    cudaMemcpy(ha,da,sizeof(half)*M*K,cudaMemcpyDeviceToHost);
    cudaMemcpy(hb,db,sizeof(half)*N*K,cudaMemcpyDeviceToHost);
    cudaMemcpy(hc,dc,sizeof(float)*M*N,cudaMemcpyDeviceToHost);

    printf("after kernel\n");
    //hc[1]=2;在这里是可以重新进行赋值的
    myprint(ha,hb,hc);

    //释放掉申请的空间
    free(ha);
    free(hb);
    free(hc);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    printf("exit\n");

}