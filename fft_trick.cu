
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<iostream>
#include<ctime>// 用于测速的函数
using namespace std;

#include <chrono>
#include <cstdlib>
#define gpu_exp1
#define gpu_exp2
#define gpu_exp3
#define cpu_exp

/* every kernel use data from global memory*/
__global__ void gpu_v1(int *device_vector, int *w)
{
    // use idx of threads to determine which data to use
    int id=threadIdx.x; 
#ifdef print1
    printf("int the gpu %d:%d %d %d %d\n",id,device_vector[0],device_vector[1],device_vector[2],device_vector[3]);
#endif
    int layers_num=2;//TODO:how to pass the layers of the fft-trick to device,here use the value 2
    int q=17;//TODO:as layers_num,the same thing need done with q
    //do the first butterfly
#ifdef print1
    printf("int the gpu %d: %d +- %d * %d\n",id,device_vector[id],device_vector[id+2],w[1]);
#endif
    int t=(device_vector[id+2]*w[1])%q;//if id =0,then use 0 and 2,multiply w[0] ;else the use 1 and 3,multiply w[0]
    device_vector[id+2]=(device_vector[id]-t+q)%q;
    device_vector[id]=(device_vector[id]+t)%q;
#ifdef print1
    printf("int the gpu %d:%d %d %d %d\n",id,device_vector[0],device_vector[1],device_vector[2],device_vector[3]);
#endif
    //TODO:is there need any synchronization?
    //do the second butterfly
#ifdef print1
    printf("int the gpu %d: %d +- %d * %d\n",id,device_vector[id*2],device_vector[id*2+1],w[id+2]);
#endif
    t=((device_vector[id*2+1]*w[id+2]))%q;//if id =0,then use o and 1,multiply w[2];else then use 2 and 3,multiply w[3]
    device_vector[id*2+1]=(device_vector[id*2]-t+q)%q;
    device_vector[id*2]=(device_vector[id*2]+t)%q;
#ifdef print1
    printf("int the gpu %d:%d %d %d %d\n",id,device_vector[0],device_vector[1],device_vector[2],device_vector[3]);
#endif
}

/* int the same block,kernels use data from shared memory*/
__global__ void gpu_v2(int *device_vector, int *w)
{
    __shared__ int shared_device_vector[4] ; //declaration must be seprated fromdefinition
    shared_device_vector[0] = device_vector[0],shared_device_vector[1]=device_vector[1];
    shared_device_vector[2] = device_vector[2],shared_device_vector[3] =device_vector[3]; //debug:here after "__shared__" there must be a type,like "int"
    // use idx of threads to determine which data to use
    int id=threadIdx.x; 
#ifdef print1
    printf("int the gpu %d:%d %d %d %d\n",id,shared_device_vector[0],shared_device_vector[1],shared_device_vector[2],shared_device_vector[3]);
#endif
    int layers_num=2;//TODO:how to pass the layers of the fft-trick to device,here use the value 2
    int q=17;//TODO:as layers_num,the same thing need done with q
    //do the first butterfly
#ifdef print1
    printf("int the gpu %d: %d +- %d * %d\n",id,shared_device_vector[id],shared_device_vector[id+2],w[1]);
#endif
    int t=(shared_device_vector[id+2]*w[1])%q;//if id =0,then use 0 and 2,multiply w[0] ;else the use 1 and 3,multiply w[0]
    shared_device_vector[id+2]=(shared_device_vector[id]-t+q)%q;
    shared_device_vector[id]=(shared_device_vector[id]+t)%q;
#ifdef print1
    printf("int the gpu %d:%d %d %d %d\n",id,shared_device_vector[0],shared_device_vector[1],shared_device_vector[2],shared_device_vector[3]);
#endif
    //TODO:is there need any synchronization?
    //do the second butterfly
#ifdef print1
    printf("int the gpu %d: %d +- %d * %d\n",id,shared_device_vector[id*2],shared_device_vector[id*2+1],w[id+2]);
#endif
    t=((shared_device_vector[id*2+1]*w[id+2]))%q;//if id =0,then use o and 1,multiply w[2];else then use 2 and 3,multiply w[3]
    shared_device_vector[id*2+1]=(shared_device_vector[id*2]-t+q)%q;
    shared_device_vector[id*2]=(shared_device_vector[id*2]+t)%q;
#ifdef print1
    printf("int the gpu %d:%d %d %d %d\n",id,shared_device_vector[0],shared_device_vector[1],shared_device_vector[2],shared_device_vector[3]);
#endif

    //copy results to global memory
    device_vector[0]=shared_device_vector[0],device_vector[1]=shared_device_vector[1],device_vector[2]=shared_device_vector[2],device_vector[3]=shared_device_vector[3];
}

/* int the same block,kernels use data from registers*/
//conclusion:every kernel has its own register file ,so if just modify the data type of shared_device_vector from __shared to int ,the result will be wrong.So this code aims to use regiters must be redesigned.
__global__ void gpu_v3(int *device_vector, int *w)
{
    int shared_device_vector[4] ; 
    //load from global memory to registers
    shared_device_vector[0] = device_vector[0],shared_device_vector[1]=device_vector[1];
    shared_device_vector[2] = device_vector[2],shared_device_vector[3] =device_vector[3];
    // use idx of threads to determine which data to use
    int id=threadIdx.x; 
#ifdef print1
    printf("int the gpu %d:%d %d %d %d\n",id,shared_device_vector[0],shared_device_vector[1],shared_device_vector[2],shared_device_vector[3]);
#endif
    int layers_num=2;//TODO:how to pass the layers of the fft-trick to device,here use the value 2
    int q=17;//TODO:as layers_num,the same thing need done with q
    //do the first butterfly
#ifdef print1
    printf("int the gpu %d: %d +- %d * %d\n",id,shared_device_vector[id],shared_device_vector[id+2],w[1]);
#endif
    int t=(shared_device_vector[id+2]*w[1])%q;//if id =0,then use 0 and 2,multiply w[0] ;else the use 1 and 3,multiply w[0]
    shared_device_vector[id+2]=(shared_device_vector[id]-t+q)%q;
    shared_device_vector[id]=(shared_device_vector[id]+t)%q;
#ifdef print1
    printf("int the gpu %d:%d %d %d %d\n",id,shared_device_vector[0],shared_device_vector[1],shared_device_vector[2],shared_device_vector[3]);
#endif
    //TODO:is there need any synchronization?
    //do the second butterfly
#ifdef print1
    printf("int the gpu %d: %d +- %d * %d\n",id,shared_device_vector[id*2],shared_device_vector[id*2+1],w[id+2]);
#endif
    t=((shared_device_vector[id*2+1]*w[id+2]))%q;//if id =0,then use o and 1,multiply w[2];else then use 2 and 3,multiply w[3]
    shared_device_vector[id*2+1]=(shared_device_vector[id*2]-t+q)%q;
    shared_device_vector[id*2]=(shared_device_vector[id*2]+t)%q;
#ifdef print1
    printf("int the gpu %d:%d %d %d %d\n",id,shared_device_vector[0],shared_device_vector[1],shared_device_vector[2],shared_device_vector[3]);
#endif

    //copy results to global memory
    device_vector[0]=shared_device_vector[0],device_vector[1]=shared_device_vector[1],device_vector[2]=shared_device_vector[2],device_vector[3]=shared_device_vector[3];
}
/* todo:ntt加速的几个版本
1.全部放在global memory中，对于每一层设计不同的kernel代码，每层并行执行n/2个线程
(下面的放在shared memory中)
2.数据产生了分块，可以用几个不同的线程，把分块的数据都放到shared memory中(创建一个_shared数组，取值依据block的位置，然后基于这个来进行计算)，最后再进行一起的操作回global memory中（把shared memory拷贝回去）
3.当n很小时，可以全部放入shared memory中，每个线程处理2个数字，然后再进行类似于1的并行
(下面放到寄存器中)
4.依据正确的间隔load，相应数量的系数，然后每个线程在自己的ntt数组中进行计算。问题在于，如何使用一个kernel来进行实现呢？ （层数可以作为最外层的循环嘛？）
 */

#define NMAX 256
/* caculate g^m % mod*/
int power(int g, int m, int mod) {
    int res = 1;
    while (m) {
        if (m % 2) {
            res = (res * g)%mod; // 这里应该是乘法而不是加法
        }
        g = (g * g)%mod;
        m >>= 1;
    }
    return res;
}
void native_FFT_trick(int *a,int n,int W2n,int q,int *zetas,int k) {
    //debug:only do the butterfly
    for (int group_size=n; group_size > 1; group_size>>=1) {
        for (int group_start = 0; group_start < n; group_start += group_size) {
            int zeta = zetas[++k];// debug:这里应该是从1开始而非0
            for (int start = group_start; start < group_start + group_size / 2; start++) {
                // 蝴蝶操作
                int t = a[start];
                int u = (a[start + group_size / 2] * zeta)%q;
                a[start] = (t + u)%q;
                a[start + group_size / 2] = (t - u + q) % q;
            }
        }
    }
}
/* based on gcd,caculate the inverse of element a*/
int inv_with_gcd(int a, int b) {
    int u = a, v = b;// a<b,u<v
    int x1 = 1, x2 = 0;
    while (u!=1) {
        int q = v / u;
        int r = v - q*u;
        v = u, u = r;
        int x = x2 - q * x1;
        x2 = x1, x1 = x;// debug:这里一定要有一个中间值
    }
    return (x1 + b) % b;
}
/* based on binary,caculate the inverse of element a*/
int inv_with_binary(int a, int b) {
    int res = 1;
    int u = a, v = b;
    int x1 = 1, x2 = 0;
    while (u != 1 && v != 1) {// debug:这里是与条件句
        while (u % 2 == 0) {
            u >>= 1;
            if (x1 % 2) {
                x1 = (x1 + b) / 2;
            }
            else x1 >>= 1;
        }
        while (v % 2 == 0) {
            v >> 1;
            if (x2 % 2) {
                x2 = (x2 + b) / 2;
            }
            else x2 >>= 1;
        }
        if (u >= v) {
            u = u - v;
            x1 = x1 - x2;
        }
        else {
            v = v - u;
            x2 = x2 - x1;
        }
    }
    if (u == 1) return x1 % b;
    else return x2 % b;
}
void native_inv_FFT_trick(int* a, int n, int W2n, int q) {
    int k = n;
    int bits = 0;
    int cn = n;
    while (cn) {
        bits++;
        cn >>= 1;
    }
    bits--;//这里要减去一个
    int zetas[NMAX] = { 0 };//注意这里实际使用到的只有n个
    int bitrev[NMAX] = { 0 };
    for (int i = 1; i < n; i++) {
        bitrev[i] = (bitrev[i >> 1] >> 1) | ((i & 1) << (bits - 1));
    }
    for (int i = 1; i < n; i++) {
        zetas[i] = power(W2n, bitrev[i], q);
    }

    //比特反转系数
    for (int i = 0; i < n; i++) {
        if (i < bitrev[i]) {
            //手动实现一个swap
            int value = a[i];
            a[i] = a[bitrev[i]];
            a[bitrev[i]] = value;
        }
    }
    
    for (int group_size = 2; group_size <= n; group_size <<= 1) {
        for (int group_start = 0; group_start < n; group_start += group_size) {
            int zeta = (-zetas[--k] + q) % q;
            for (int start = group_start; start < group_start + group_size / 2; start++) {
                //蝴蝶操作
                int t = a[start]; 
                a[start] = (t + a[start + group_size / 2]) % q;
                a[start + group_size / 2] = (((t - a[start + group_size / 2] + q) % q) * zeta) % q;
            }
        }
    }
    
    //最后来延迟处理a[i]/2^n
    //首先计算逆元
    //int f = inv_with_gcd(n, q);
    int f = inv_with_binary(n, q);
    for (int i = 0; i < n; i++) {
        a[i] = (a[i] * f) % q;
    }

    for (int i = 0; i < n; i++) {
        printf("%d ", a[i]);
    }
    printf("\n");
}

/* 主要用于初始化系数的值 */
// TODO:可以写一个随机化生成系数的函数
void initial_vector(int * coe,int n){
    // 简单起见，先给定值
    coe[0]=1,coe[1]=2,coe[2]=7,coe[3]=11;
}
/* pre cac twiddles used in FFT-trick tree*/
void initial_twiddles(int *twi,int n){
    // to simply,here give the value directly
    //TODO:write the program here ti cac the twiddles with different q and n
    twi[0]=0,twi[1]=4,twi[2]=2,twi[3]=8;
    /*
    int k = 0;
    int bits = 0;
    int cn = n;
    while (cn) {
        bits++;
        cn >>= 1;
    }
    bits--;// 这里要减去一个
    int zetas[NMAX] = { 0 };// 注意这里实际使用到的只有n个
    int bitrev[NMAX] = { 0 };
    for (int i = 1; i < n; i++) {
         bitrev[i] = (bitrev[i >> 1] >> 1) | ((i & 1) << (bits - 1));
    }
    for (int i = 1; i < n; i++) {
        zetas[i] = power(W2n, bitrev[i],q);
    }
    */
}


int main()
{
    int n=4;// 代表的是数组的大小
    /* 分配cpu的内存 */
    int *host_vector;
    host_vector=(int *)malloc(sizeof(int)*n);

    int *host_twiddles;
    host_twiddles=(int *)malloc(sizeof(int)*n);

    int *device_vector;
    int *device_twiddles;
    cudaMalloc(&device_vector,sizeof(int)*n);
    cudaMalloc(&device_twiddles,sizeof(int)*n);

    /* 在cpu上进行计算并测速 */
#ifdef cpu_exp
    initial_vector(host_vector,n);
    initial_twiddles(host_twiddles,n);
    double cpu_time=0.0;
    auto start = std::chrono::high_resolution_clock::now();
    native_FFT_trick(host_vector, 4, 2, 17,host_twiddles,0);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed=end-start;
    cpu_time=elapsed.count();
    cout<< " cpu consuming time is " << cpu_time <<endl;
    //verify the cpu results
    for(int i=0;i<n;i++){
        cout<<host_vector[i]<<" ";
    }
    cout<<endl;
#endif

    /* gpu version1 
    Every layer has different kenel code.Every thread process two values and do butterfly operation,then copy the data to global memory*/
#ifdef gpu_exp1
    initial_vector(host_vector,n);
    initial_twiddles(host_twiddles,n);
    //cudaMalloc 
    //copy the vector to the device global memory
    cudaMemcpy(device_vector,host_vector,sizeof(int)*n,cudaMemcpyHostToDevice);
    cudaMemcpy(device_twiddles,host_twiddles,sizeof(int)*n,cudaMemcpyHostToDevice);

    double gpu_time=0.0;
    auto gpu_start = std::chrono::high_resolution_clock::now();
    gpu_v1<<<1,2>>>(device_vector,device_twiddles);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_elapsed=gpu_end-gpu_start;
    gpu_time=gpu_elapsed.count();
    cout<< " gpu1 consuming time is " << gpu_time <<endl;
    //verify the gpu results
    cudaMemcpy(host_vector,device_vector,sizeof(int)*n,cudaMemcpyDeviceToHost);
    for(int i=0;i<n;i++){
        cout<<host_vector[i]<<" ";
    }
    cout<<endl;
#endif

#ifdef gpu_exp2
    initial_vector(host_vector,n);
    initial_twiddles(host_twiddles,n);
    //cudaMalloc 
    //copy the vector to the device global memory
    cudaMemcpy(device_vector,host_vector,sizeof(int)*n,cudaMemcpyHostToDevice);
    cudaMemcpy(device_twiddles,host_twiddles,sizeof(int)*n,cudaMemcpyHostToDevice);

    double gpu2_time=0.0;
    auto gpu2_start = std::chrono::high_resolution_clock::now();
    gpu_v2<<<1,2>>>(device_vector,device_twiddles);
    auto gpu2_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu2_elapsed=gpu2_end-gpu2_start;
    gpu2_time=gpu2_elapsed.count();
    cout<< " gpu2 consuming time is " << gpu2_time <<endl;
    //verify the gpu results
    cudaMemcpy(host_vector,device_vector,sizeof(int)*n,cudaMemcpyDeviceToHost);
    for(int i=0;i<n;i++){
        cout<<host_vector[i]<<" ";
    }
    cout<<endl;
#endif

#ifdef gpu_exp3
    initial_vector(host_vector,n);
    initial_twiddles(host_twiddles,n);
    //cudaMalloc 
    //copy the vector to the device global memory
    cudaMemcpy(device_vector,host_vector,sizeof(int)*n,cudaMemcpyHostToDevice);
    cudaMemcpy(device_twiddles,host_twiddles,sizeof(int)*n,cudaMemcpyHostToDevice);

    double gpu3_time=0.0;
    auto gpu3_start = std::chrono::high_resolution_clock::now();
    gpu_v3<<<1,2>>>(device_vector,device_twiddles);
    auto gpu3_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu3_elapsed=gpu3_end-gpu3_start;
    gpu3_time=gpu3_elapsed.count();
    cout<< " gpu3 consuming time is " << gpu3_time <<endl;
    //verify the gpu results
    cudaMemcpy(host_vector,device_vector,sizeof(int)*n,cudaMemcpyDeviceToHost);
    for(int i=0;i<n;i++){
        cout<<host_vector[i]<<" ";
    }
    cout<<endl;
#endif

    //conclusion:by comparing gpu_exp1 and gpu_exp2,we can see that there is a huge improvement

    return 0;
}