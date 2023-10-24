//2023.10.24 14:15
//线程级原语的学习
/*
1.知乎参考博客:https://zhuanlan.zhihu.com/p/641937871
参数mask代表的是掩码,是一个无符号整数，具有32位，从右边起，对应线程束内的32个线程。掩码用于指定要参加计算的线程：当掩码中的二进制位为1时，表示对应的线程参与计算；当掩码中的二进制位为0时，表示忽略对应的线程，不参与计算。
最后一个参数width是可选的，默认数值为warpSize，在当前GPU架构中都是32。参数width 只能取2、4、8、16、32这5个整数中的一个。当width小于32时，相当于（逻辑上的）线程束大小是width，而非32。
2.nividia官网https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<iostream>
#include <stdio.h>
#include <chrono>
#include <cstdlib>
#include<ctime>//debug:three header must be included
using namespace std;
#define N 64
#define wrap_shuffle_test4
//#define vote_test
__global__ void vote_all(int *a, int *b, int n) //https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-vote-functions
{
    int tid = threadIdx.x;
    if (tid > n)
        return;
    int temp = a[tid];
    b[tid] = __all_sync(0xffffffff, temp); //只有当一个线程束中所有线程传入的值都非0时，才会返回1。
    printf("in id %d : %d\n",tid,b[tid]);
     
}

//在一个线程束中，每一个线程被称为一个lane,标号从lane0到lane32
//wrap_shuffle仅仅限制在线程束内的线程之间交换数据,且只能从执行交换数据当下正在活跃的线程中读取新的数据
__global__ void wrap_shuffle(int *a){ //https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions


}

/*
    描述:函数原型 T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize);
        返回线程srcLance中的变量var的值。
        TODO:这句话是什么意思？then each subsection of the warp behaves as a separate entity with a starting logical lane ID of 0. If srcLane is outside the range [0:width-1], the value returned corresponds to the value of var held by the srcLane modulo width (i.e. within the same subsection).
    
    功能:测试__shfl_sync的功能。
*/
__global__ void bcast(int arg) {
    int laneId = threadIdx.x & 0x1f;
    int value=laneId;
    //if (laneId == 0)        // Note unused variable for
    //    value = arg;        // all threads except lane 0
    int value1 = __shfl_sync(0xffffffff, value, (laneId+31)%32);   // Synchronize all threads in warp, and get "value" from lane 0
    printf("mask is 0xffffffff\n");
    printf("id %d:%d\n",laneId,value1);

    int value2 = __shfl_sync(0, value, (laneId+31)%32); //和第九行的语句相同
    printf("mask is 0\n");
    printf("id %d:%d\n",laneId,value2);

    int value3 = __shfl_sync(laneId , value, (laneId+31)%32); //和第九行的语句相同
    printf("mask is not the same\n");
    printf("id %d:%d\n",laneId,value3);
    // (value != arg)
    //    printf("Thread %d failed.\n", threadIdx.x);
}

/*
    描述:函数原型 T __shfl_up_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
        以width所指示的大小作为线程束的大小，返回当前laneID-delta线程中变量var的值，假如landID-delta小于0,则直接返回该线程对应的var值
    功能:测试delta为2时，width分别为32和8时对应的结果，来展示width对结果的影响，已经delta产生的类似于移位的效果

*/
__global__ void test_for_shfl_up_sync() {
    int laneId = threadIdx.x & 0x1f; //TODO:这是在mod 32
    // Seed sample starting value (inverse of lane ID)
    int value = laneId;
    printf("id is %d,laneId is %d,value is %d\n",threadIdx.x,laneId,value);
    
    printf("when wrap_size=32\n");
    int n = __shfl_up_sync(0xffffffff, value, 2, 32);//掩码0xffffffff代表所有的线程都参与
    printf("id is %d,n is %d\n",laneId,n);

    printf("when wrap_size=8\n");
    n=__shfl_up_sync(0xffffffff, value, 2, 8); //此时默认8个线程为一个线程束
    printf("id is %d,n is %d\n",laneId,n);

}

/*
    描述:函数原型 T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
        以width所指示的大小作为线程束的大小，返回当前laneID-delta线程中变量var的值，假如landID+delta超过width,则直接返回该线程对应的var值

*/
__global__ void test_for_shfl_down_sync() {
    int laneId = threadIdx.x & 0x1f; //TODO:这是在mod 32
    // Seed sample starting value (inverse of lane ID)
    int value = laneId;
    printf("id is %d,laneId is %d,value is %d\n",threadIdx.x,laneId,value);
    
    printf("when wrap_size=32\n");
    int n = __shfl_down_sync(0xffffffff, value, 2, 32);//掩码0xffffffff代表所有的线程都参与
    printf("id is %d,n is %d\n",laneId,n);

}

/*
    描述:函数原型T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=warpSize);
        标号为t的参与线程返回标号为t ^ laneMask的线程中变量val的值
*/
__global__ void test_shuf_xor_sync(){
    int val = threadIdx.x;
    val = __shfl_xor_sync(0Xffffffff, val, 1, 32);
    printf("id is %d,n is %d\n",threadIdx.x,val);
}
void init_as(int *as){
   int a_module[16]={290,56,123,656,23,546,12,567,902,903,423,233,91,231,90,1213};//前16个元素
   for(int i=0;i<16;i++) as[i]=a_module[i];
   for(int i=16;i<32;i++) as[i]=1;
   for(int i=32;i<64;i++) as[i]=0;
}

int main(){

#ifdef vote_test
    int as[N];
    init_as(as);
    int *h_a;
    h_a=(int *)malloc(sizeof(int)*N);
    //初始化h_a的值
    for(int i=0;i<N;i++) h_a[i]=as[i];

    int *d_a;
    int *d_b;
    cudaMalloc(&d_a,sizeof(int)*N);
    cudaMalloc(&d_b,sizeof(int)*N);
    cudaMemcpy(d_a,h_a,sizeof(int)*N,cudaMemcpyHostToDevice);
    vote_all<<<1,N>>>(d_a,d_b,N);

    //不要忘记free掉申请的内存空间
    free(h_a);
    cudaFree(d_a);
    cudaFree(d_b);
#endif

#ifdef wrap_shuffle_test1
    bcast<<< 1, 32 >>>(1234);
#endif

#ifdef wrap_shuffle_test2
    test_for_shfl_up_sync<<< 1, 32 >>>();
#endif

#ifdef wrap_shuffle_test3
    test_for_shfl_down_sync<<< 1, 32 >>>();
#endif

#ifdef wrap_shuffle_test4
    test_shuf_xor_sync<<< 1, 32 >>>();
#endif
}

