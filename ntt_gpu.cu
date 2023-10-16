#include <iostream>
#include <chrono>
#include <ctime>
#include <cstdlib>
 
#define r 17492915097719143606//模数的原根
#define p 0xFFFFFFFF00000001//通常情况下的模数
 
void rand_vector(uint64_t *vec,size_t n,uint64_t maxn){
    srand(time(0));
    for(size_t i=0;i<n;i++)
        vec[i]=rand()%maxn;
}
 
/* base % mod */
inline __host__ __device__ uint64_t modulo(uint64_t base){
    uint64_t result=base%p;
    return result;
}
 
inline __host__ __device__ uint64_t mod_exp(uint64_t x,uint64_t y){
    uint64_t res=1;
    while(y){
        if(y&1) res=modulo(res*x);
        x=modulo(x*x);
        y>>=1;
    }
    return res;
}
 
inline __device__ uint64_t exp(uint64_t x,uint64_t y){
    uint64_t res=1;
    while(y){
        if(y&1) res=res*x;
        x=x*x;
        y>>=1;
    }
    return res;
}
 
void ntt_cpu(uint64_t *vec,size_t *rev,size_t n,int bits){
    rev[0]=0;
    for(size_t i=0;i<n;i++)
        rev[i]=(rev[i>>1]>>1)|((i&1)<<(bits-1));
    for(size_t i=0;i<n;i++)
        if(i<rev[i]) std::swap(vec[i],vec[rev[i]]);
    for(size_t i=1;i<n;i<<=1){
        uint64_t wn=mod_exp(r,(p-1)/(2*i));
        for(size_t j=0,d=(i<<1);j<n;j+=d){
            uint64_t w=1;
            for(size_t k=0;k<i;k++){
                uint64_t factor1=vec[j+k];
                uint64_t factor2=modulo(w*vec[j+k+i]);
                vec[j+k]=modulo(factor1+factor2);
                vec[j+k+i]=modulo(factor1-factor2);
                w=modulo(w*wn);
            }
        }
    }
}
 
__global__ void bit_reverse_gpu(uint64_t *d_vec,size_t n,int bits){
    size_t tid=threadIdx.x+blockIdx.x*blockDim.x;
    if(tid>=n) return;
    uint64_t val=d_vec[tid];
    size_t old_id=tid,new_id=0;
    for(int i=0;i<bits;i++){
        int b=old_id&1;
        new_id=(new_id<<1)|b;
        old_id>>=1;
    }
    if(tid<new_id){
        uint64_t temp=d_vec[new_id];
        d_vec[tid]=temp;
        d_vec[new_id]=val;
    }
}
 
__global__ void twiddle_factor_gpu(uint64_t *d_twiddles,size_t n,int bits){
    size_t tid=threadIdx.x+blockIdx.x*blockDim.x;
    size_t total=(n>>1)*bits;
    if(tid>=total) return;
    size_t size=(n>>1);
    size_t num=tid/size;// num-th iteration
    size_t res=tid%size;// num-th iteration, res-th postion
    size_t len=exp(2,num);// chunck size
    d_twiddles[tid]=mod_exp(r,(p-1)/(2*len)*(res%len));
}
 
/* n/2 threads per iteration */
__global__ void ntt_kernel(uint64_t *d_vec,uint64_t *d_twiddles,size_t n,size_t iter,size_t chunck_size){
    size_t tid=blockIdx.x*blockDim.x+threadIdx.x;
    size_t half=n>>1;
    if(tid>=half) return;
    size_t vec_idx=(tid/chunck_size)*(2*chunck_size)+(tid%chunck_size); // index of vector
    size_t twi_idx=iter*half+tid; // index of twiddles
    uint64_t factor1=d_vec[vec_idx]; 
    uint64_t factor2=modulo(d_twiddles[twi_idx]*d_vec[vec_idx+chunck_size]);
    d_vec[vec_idx]=modulo(factor1+factor2);
    d_vec[vec_idx+chunck_size]=modulo(factor1-factor2);
}
 
void ntt_gpu(uint64_t *d_vec,uint64_t *d_twiddles,size_t n,int bits){
    size_t block_size=128;
    size_t grid_size=(n+block_size-1)/block_size;
    bit_reverse_gpu<<<grid_size,block_size>>>(d_vec,n,bits);// bit_reverse: one thread for one number
    size_t total=(n>>1)*bits;
    grid_size=(total+block_size-1)/block_size;
    twiddle_factor_gpu<<<grid_size,block_size>>>(d_twiddles,n,bits);// preprocessing twiddle factor, bits * n/2
    size_t iter=0;
    for(size_t i=1;i<n;i<<=1){
        grid_size=((n>>1)+block_size-1)/block_size;
        ntt_kernel<<<grid_size,block_size>>>(d_vec,d_twiddles,n,iter,i);// ntt
        iter++;
    }
    cudaDeviceSynchronize();
}
 
bool check(uint64_t *cpu_result,uint64_t *gpu_result,size_t n){
    for(size_t i=0;i<n;i++){
        if(cpu_result[i]!=gpu_result[i])
            return false;
    }
    return true;
}
 
int main(int argc,char *argv[]){
    int bits=12;
    if(argc==2) bits=atoi(argv[1]);
    else if(argc>2){
        std::cerr<<"arguments error"<<std::endl;
        exit(-1);
    }
 
    double cpu_time=0.0;
    double gpu_time=0.0;
    size_t n=(size_t)1<<bits;
 
    /* allocate cpu memory */
    uint64_t *host_vector;
    size_t *host_rev;
    host_vector=(uint64_t*)malloc(sizeof(uint64_t)*n);
    host_rev=(size_t*)malloc(sizeof(size_t)*n);
 
    /* allocate gpu memory */
    uint64_t *device_vector;
    uint64_t *device_twiddles;
    // size_t *device_rev;
    cudaMalloc(&device_vector,sizeof(uint64_t)*n);
    cudaMalloc(&device_twiddles,sizeof(uint64_t)*(n>>1)*bits);// bits iterations, n/2 twiddles per iteration
    
    /* init vector */
    rand_vector(host_vector,n,n);
    /* host to device */
    cudaMemcpy(device_vector,host_vector,sizeof(uint64_t)*n,cudaMemcpyHostToDevice);
 
    /* ntt on cpu */
    auto start = std::chrono::high_resolution_clock::now();
    ntt_cpu(host_vector,host_rev,n,bits);//======== cpu ntt
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed=end-start;
    cpu_time=elapsed.count();
 
    /* ntt on gpu */
    start = std::chrono::high_resolution_clock::now();
    ntt_gpu(device_vector,device_twiddles,n,bits);//======== gpu ntt
    end = std::chrono::high_resolution_clock::now();
    elapsed=end-start;
    gpu_time=elapsed.count();
    /* copy result from gpu to cpu */
    uint64_t *gpu_result;
    gpu_result=(uint64_t*)malloc(sizeof(uint64_t)*n);
    cudaMemcpy(gpu_result,device_vector,sizeof(uint64_t)*n,cudaMemcpyDeviceToHost);
 
    std::cout<<"==== vector length: 2^"<<bits<<" ===="<<std::endl;
    std::cout<<"CPU time: "<<cpu_time<<" ms"<<std::endl;
    std::cout<<"GPU time: "<<gpu_time<<" ms"<<std::endl;
    // if(check(host_vector,gpu_result,n))  printf("all correct\n");
    // else printf("error\n"); // 检查CPU和GPU的输出结果是否相同
 
    free(host_vector);
    free(host_rev);
    free(gpu_result);
    cudaFree(device_vector);
    cudaFree(device_twiddles);
    
    return 0;
}