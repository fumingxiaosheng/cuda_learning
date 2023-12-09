//graph原理的讲解:https://zhuanlan.zhihu.com/p/593246168
/*
一个操作在图中形成一个节点。操作之间的依赖关系就是边。这些依赖关系限制了操作的执行顺序。

一个操作可以在它所依赖的节点完成后的任何时间被调度。调度是留给CUDA系统的。
*/

#include <cuda_runtime.h>

// CUDA 向量加法内核
__global__ void vectorAddKernel(const float* A, const float* B, float* C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

// 创建 CUDA 图并向其中添加节点
void createGraph(cudaGraph_t* pGraph, cudaStream_t stream, const float* d_A, const float* d_B, float* d_C, int numElements) {
    cudaGraph_t graph;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    // 添加 CUDA 节点：向量加法
    dim3 blockDim(256);
    dim3 gridDim((numElements + blockDim.x - 1) / blockDim.x);
    vectorAddKernel<<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_C, numElements);

    cudaStreamEndCapture(stream, &graph);
    *pGraph = graph;
}

int main() {
    const int numElements = 1000;
    size_t size = numElements * sizeof(float);

    // 分配和初始化主机上的数据
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    for (int i = 0; i < numElements; ++i) {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    // 在设备上分配内存
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // 将数据从主机复制到设备
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 创建 CUDA 流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 创建 CUDA 图并添加节点
    cudaGraph_t graph;
    createGraph(&graph, stream, d_A, d_B, d_C, numElements);

    // 实例化 CUDA 图执行对象
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

    // 启动 CUDA 图执行对象
    cudaGraphLaunch(graphExec, stream);

    // 等待 CUDA 流上的操作完成
    cudaStreamSynchronize(stream);

    // 将结果从设备复制回主机
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 打印结果
    for (int i = 0; i < numElements; ++i) {
        printf("%f + %f = %f\n", h_A[i], h_B[i], h_C[i])