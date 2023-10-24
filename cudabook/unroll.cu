#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<iostream>
#include <stdio.h>
#include <chrono>
#include <cstdlib>
#include<ctime>//debug:three header must be included
using namespace std;

/*
todo:测试结果为:
unroll time consume is 0.027200
not unroll time consume is 0.009100
为什么会存在没有展开的时间是较小的情况呢？
*/

__global__ void unroll(int *p1, int *p2) {
#pragma unroll
for (int i = 0; i < 12; ++i)
  p1[i] += p2[i]*2;

}

__global__ void not_unroll(int *p1, int *p2) {
for (int i = 0; i < 12; ++i)
  p1[i] += p2[i]*2;

}

int main(){
    int h_p1[12]={0,1,2,3,4,5,6,7,8,9,10,11};
    int h_p2[12]={0,1,2,3,4,5,6,7,8,9,10,11};

    int *d_p1;
    int *d_p2;
    int *d_d1;
    int *d_d2;
    cudaMalloc(&d_p1,sizeof(int)*12);
    cudaMalloc(&d_p2,sizeof(int)*12);
    cudaMalloc(&d_d1,sizeof(int)*12);
    cudaMalloc(&d_d2,sizeof(int)*12);

    cudaMemcpy(d_p1,h_p1,sizeof(int)*12,cudaMemcpyHostToDevice);
    cudaMemcpy(d_p2,h_p2,sizeof(int)*12,cudaMemcpyHostToDevice);

    double parallel_time=0.0;
    auto parallel_start = std::chrono::high_resolution_clock::now();
    unroll<<<1,1>>>(d_d1,d_p1);
    auto parallel_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> parallel_elapsed=parallel_end-parallel_start;
    parallel_time=parallel_elapsed.count();
    printf("unroll time consume is %f\n",parallel_time);

    double atomic_time=0.0;
    auto atomic_start = std::chrono::high_resolution_clock::now();
    not_unroll<<<1,1>>>(d_d2,d_p2);
    auto atomic_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> atomic_elapsed=atomic_end-atomic_start;
    atomic_time=atomic_elapsed.count();
    printf("not unroll time consume is %f\n",atomic_time);

    cudaFree(d_p1);
    cudaFree(d_p2);
}

