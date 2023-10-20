#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<iostream>
#include <stdio.h>
#include <chrono>
#include <cstdlib>
#include<ctime>//debug:three header must be included
using namespace std;
#define N 16

/*
TODO:
the output is:
int parallel,min is 12,time consume is 0.027500 
in atomic,min is 12,time consume is 0.006800 
why the parallel is slower?!!!
*/
//#define test_atomic
//#define compare_atomic_with_reduction
/* 当每个线程调用atomicMin函数时，会将从共享内存中读取的最小值与当前的值进行比较，然后把较小值写回到对应的共享内存中（P130）
*/
/*function to see how atomicMin works*/
__global__ void test_for_atomic(){
    __shared__ int minValue;
    int id = threadIdx.x;
    atomicMin(&minValue,id); 

    __syncthreads();

    printf("in id%d :%d\n",id,minValue);
    /*the output is:
    in id0 :0
    in id1 :0
    in id2 :0
    in id3 :0
    in id4 :0
    in id5 :0
    in id6 :0
    in id7 :0
    in id8 :0
    in id9 :0
    */
}

__global__ void parallel_reduction(int *A){
    
    __shared__ int values[N];
    int id=threadIdx.x;
    //printf("id:%d %d\n",id,A[id]);
    values[id]=A[id];//copy data from global memory to shared memory
    //printf("id:%d %d\n",id,values[id]);
    for(int i=N/2;i>0;i=i/2){ //represent the num of threads needed
        if(id <i){
            //this thread nedd to do reduction
            //printf("id%d: cop(%d,%d)\n",id,id,id+i);
            if(values[id]>values[id+i]){
                values[id]=values[id+i];//here is id+i not id+1 ,be careful
            }
        }
        //because the number of threads is less than 32,so there is no need to use __syhcthreads here
        //__syncthreads();
    }
    //copy the last result values[0] to A[0]
    if(id==0) A[0]=values[0]; //TODO:if(id==0) is necessary?without it,is there any confliction?
    
}

__global__ void atomic_reduction(int *A){
    __shared__ int minValue;
    int id = threadIdx.x;
    atomicMin(&minValue,A[id]); 

    //__syncthreads();
    if(id==0) A[0]= minValue;
}

int main(){
    struct cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop,0);
    printf("sharedmemory: %u",device_prop.sharedMemPerBlock);
#ifdef test_atomic
    test_for_atomic<<<1,10>>>();
#endif

#ifdef compare_atomic_with_reduction:

    int as[N]={290,56,123,656,23,546,12,567,902,903,423,233,91,231,90,1213};
    int *array;
    array=(int *)malloc(sizeof(int)*N);
    for(int i=0;i<N;i++) array[i]=as[i];
    
    int *A;
    double parallel_time=0.0;
    //allocate device memoryint *A;
    cudaMalloc(&A,sizeof(int)*N);//there is no return pointer
    cudaMemcpy(A,array,sizeof(int)*N,cudaMemcpyHostToDevice);
    auto parallel_start = std::chrono::high_resolution_clock::now();
    parallel_reduction<<<1,N>>>(A);
    auto parallel_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> parallel_elapsed=parallel_end-parallel_start;
    parallel_time=parallel_elapsed.count();
    //why "died with status 0xC0000005 (ACCESS_VIOLATION) " here? ->because host can not access device memory
    //printf("%d \n" ,A[0]);//debug:wrong access to device memory from host
    cudaMemcpy(array,A,sizeof(int)*N,cudaMemcpyDeviceToHost);
    printf("int parallel,min is %d,time consume is %f \n",array[0],parallel_time);


    int *B;
    double atomic_time=0.0;
    //allocate device memoryint *A;
    cudaMalloc(&B,sizeof(int)*N);//there is no return pointer
    cudaMemcpy(B,array,sizeof(int)*N,cudaMemcpyHostToDevice);
    auto atomic_start = std::chrono::high_resolution_clock::now();
    atomic_reduction<<<1,N>>>(B);
    auto atomic_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> atomic_elapsed=atomic_end-atomic_start;
    atomic_time=atomic_elapsed.count();

    cudaMemcpy(array,B,sizeof(int)*N,cudaMemcpyDeviceToHost);
    printf("in atomic,min is %d,time consume is %f \n",array[0],atomic_time);


    
#endif
}