//from:https://zhuanlan.zhihu.com/p/51402722
//cudaMemcpyAsync:https://blog.csdn.net/Small_Munich/article/details/103494881
//cuda stream概念原理详解:https://zhuanlan.zhihu.com/p/460278403
/*
1.cudaMemcpy函数的原型如下所示:

​cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )

Copies data between host and device.
Parameters
dst
- Destination memory address // 目的存储指针
src
- Source memory address // 数据源指针
count
- Size in bytes to copy // 数据大小
kind
- Type of transfer // 设备拷贝数据

2.cudaMemcpyAsync的函数原型如下所示:

__host__​__device__​cudaError_t cudaMemcpyAsync ( void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0 )

Copies data between host and device.
Parameters
dst
- Destination memory address // 目的指针
src
- Source memory address // 数据源指针
count
- Size in bytes to copy // 拷贝大小
kind
- Type of transfer // 设备迁移
stream
- Stream identifier // 默认streams还是指定streams


*/
/*host端使用cuda stream调用核函数的方法
kernel<<<Dg, Db, Ns, S>>>(param list);
Dg：int型或者dim3类型(x,y,z)，用于定义一个Grid中Block是如何组织的，如果是int型，则表示一维组织结构
Db：int型或者dim3类型(x,y,z)，用于定义一个Block中Thread是如何组织的，如果是int型，则表示一维组织结构
Ns：size_t类型，可缺省，默认为0；用于设置每个block除了静态分配的共享内存外，最多能动态分配的共享内存大小，单位为byte。0表示不需要动态分配。
S：cudaStream_t类型，可缺省，默认为0。表示该核函数位于哪个流

创建流的方式
cudaStream_t stream, stream1;
	cudaStreamCreate(&stream);
*/

#include <vector>
#include <random>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>    
#include <math.h>   


//#define kernel_async_test
//#define kernel_stream_test
#define native_cuda_stream_use

#ifdef DEBUG
#define CUDA_CALL(F)  if( (F) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__); exit(-1);}
#define CUDA_CHECK()  if( (cudaPeekAtLastError()) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__-1); exit(-1);}
#else
#define CUDA_CALL(F) (F)
#define CUDA_CHECK()
#endif


#ifdef cuda_stream_test
void PrintDeviceInfo();
void GenerateBgra8K(uint8_t* buffer, int dataSize);
void convertPixelFormatCpu(uint8_t* inputBgra, uint8_t* outputYuv, int numPixels);
__global__ void convertPixelFormat(uint8_t* inputBgra, uint8_t* outputYuv, int numPixels);

int main()
{
  PrintDeviceInfo();

  uint8_t* bgraBuffer;
  uint8_t* yuvBuffer;
  uint8_t* deviceBgraBuffer;
  uint8_t* deviceYuvBuffer;

  const int dataSizeBgra = 7680 * 4320 * 4;
  const int dataSizeYuv = 7680 * 4320 * 3;
  CUDA_CALL(cudaMallocHost(&bgraBuffer, dataSizeBgra));
  CUDA_CALL(cudaMallocHost(&yuvBuffer, dataSizeYuv));
  CUDA_CALL(cudaMalloc(&deviceBgraBuffer, dataSizeBgra));
  CUDA_CALL(cudaMalloc(&deviceYuvBuffer, dataSizeYuv));

  std::vector<uint8_t> yuvCpuBuffer(dataSizeYuv);

  cudaEvent_t start, stop;
  float elapsedTime;
  float elapsedTimeTotal;
  float dataRate;
  CUDA_CALL(cudaEventCreate(&start));
  CUDA_CALL(cudaEventCreate(&stop));

  std::cout << " " << std::endl;
  std::cout << "Generating 7680 x 4320 BRGA8888 image, data size: " << dataSizeBgra << std::endl;
  GenerateBgra8K(bgraBuffer, dataSizeBgra);

  std::cout << " " << std::endl;
  std::cout << "Computing results using CPU." << std::endl;
  std::cout << " " << std::endl;
  CUDA_CALL(cudaEventRecord(start, 0));
  convertPixelFormatCpu(bgraBuffer, yuvCpuBuffer.data(), 7680*4320);
  CUDA_CALL(cudaEventRecord(stop, 0));
  CUDA_CALL(cudaEventSynchronize(stop));
  CUDA_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));
  std::cout << "    Whole process took " << elapsedTime << "ms." << std::endl;

  std::cout << " " << std::endl;
  std::cout << "Computing results using GPU, default stream." << std::endl;
  std::cout << " " << std::endl;

  std::cout << "    Move data to GPU." << std::endl;
  CUDA_CALL(cudaEventRecord(start, 0));
  CUDA_CALL(cudaMemcpy(deviceBgraBuffer, bgraBuffer, dataSizeBgra, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaEventRecord(stop, 0));
  CUDA_CALL(cudaEventSynchronize(stop));
  CUDA_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));
  dataRate = dataSizeBgra/(elapsedTime/1000.0)/1.0e9;
  elapsedTimeTotal = elapsedTime;
  std::cout << "        Data transfer took " << elapsedTime << "ms." << std::endl;
  std::cout << "        Performance is " << dataRate << "GB/s." << std::endl;

  std::cout << "    Convert 8-bit BGRA to 8-bit YUV." << std::endl;
  CUDA_CALL(cudaEventRecord(start, 0));
  convertPixelFormat<<<32400, 1024>>>(deviceBgraBuffer, deviceYuvBuffer, 7680*4320);
  CUDA_CHECK();
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaEventRecord(stop, 0));
  CUDA_CALL(cudaEventSynchronize(stop));
  CUDA_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));
  dataRate = dataSizeBgra/(elapsedTime/1000.0)/1.0e9;
  elapsedTimeTotal += elapsedTime;
  std::cout << "        Processing of 8K image took " << elapsedTime << "ms." << std::endl;
  std::cout << "        Performance is " << dataRate << "GB/s." << std::endl;

  std::cout << "    Move data to CPU." << std::endl;
  CUDA_CALL(cudaEventRecord(start, 0));
  CUDA_CALL(cudaMemcpy(yuvBuffer, deviceYuvBuffer, dataSizeYuv, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaEventRecord(stop, 0));
  CUDA_CALL(cudaEventSynchronize(stop));
  CUDA_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));
  dataRate = dataSizeYuv/(elapsedTime/1000.0)/1.0e9;
  elapsedTimeTotal += elapsedTime;
  std::cout << "        Data transfer took " << elapsedTime << "ms." << std::endl;
  std::cout << "        Performance is " << dataRate << "GB/s." << std::endl;

  std::cout << "    Whole process took " << elapsedTimeTotal << "ms." <<std::endl;

  std::cout << "    Compare CPU and GPU results ..." << std::endl;
  bool foundMistake = false;
  for(int i=0; i<dataSizeYuv; i++){
    if(yuvCpuBuffer[i]!=yuvBuffer[i]){
      foundMistake = true;
      break;
    }
  }

  if(foundMistake){
    std::cout << "        Results are NOT the same." << std::endl;
  } else {
    std::cout << "        Results are the same." << std::endl;
  }

  const int nStreams = 16;

  std::cout << " " << std::endl;
  std::cout << "Computing results using GPU, using "<< nStreams <<" streams." << std::endl;
  std::cout << " " << std::endl;

  //这里创建了nStreams个cudastream
  cudaStream_t streams[nStreams];
  std::cout << "    Creating " << nStreams << " CUDA streams." << std::endl;
  for (int i = 0; i < nStreams; i++) {
    CUDA_CALL(cudaStreamCreate(&streams[i])); //创建函数为cudaStreamCreate
  }

  int brgaOffset = 0;
  int yuvOffset = 0;
  const int brgaChunkSize = dataSizeBgra / nStreams;
  const int yuvChunkSize = dataSizeYuv / nStreams;

  CUDA_CALL(cudaEventRecord(start, 0));
  for(int i=0; i<nStreams; i++)
  {
    std::cout << "        Launching stream " << i << "." << std::endl;
    brgaOffset = brgaChunkSize*i;
    yuvOffset = yuvChunkSize*i;
    //TODO:1.这里是否会造成自己没拷贝完结果就开始调用kernel函数呢 2.kernel的执行和host是否是同步的呢，还是异步的
    CUDA_CALL(cudaMemcpyAsync(  deviceBgraBuffer+brgaOffset,
                                bgraBuffer+brgaOffset,
                                brgaChunkSize,
                                cudaMemcpyHostToDevice,
                                streams[i] ));//cudaMencpyAsync所执行的操作同cudaMemcpy,但是cudaMencpy是同步的拷贝，即没有拷贝结束的时候程序是会被阻塞的，而这里的cudaMencpyAsync则是异步的拷贝,不会产生相应的阻塞操作,参考链接https://blog.csdn.net/Small_Munich/article/details/103494881

    convertPixelFormat<<<4096, 1024, 0, streams[i]>>>(deviceBgraBuffer+brgaOffset, deviceYuvBuffer+yuvOffset, brgaChunkSize/4);

    CUDA_CALL(cudaMemcpyAsync(  yuvBuffer+yuvOffset,
                                deviceYuvBuffer+yuvOffset,
                                yuvChunkSize,
                                cudaMemcpyDeviceToHost,
                                streams[i] ));
  }

  CUDA_CHECK();
  CUDA_CALL(cudaDeviceSynchronize());

  CUDA_CALL(cudaEventRecord(stop, 0));
  CUDA_CALL(cudaEventSynchronize(stop));
  CUDA_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));
  std::cout << "    Whole process took " << elapsedTime << "ms." << std::endl;

  std::cout << "    Compare CPU and GPU results ..." << std::endl;
  for(int i=0; i<dataSizeYuv; i++){
    if(yuvCpuBuffer[i]!=yuvBuffer[i]){
      foundMistake = true;
      break;
    }
  }

  if(foundMistake){
    std::cout << "        Results are NOT the same." << std::endl;
  } else {
    std::cout << "        Results are the same." << std::endl;
  }

  CUDA_CALL(cudaFreeHost(bgraBuffer));
  CUDA_CALL(cudaFreeHost(yuvBuffer));
  CUDA_CALL(cudaFree(deviceBgraBuffer));
  CUDA_CALL(cudaFree(deviceYuvBuffer));

  return 0;
}

void PrintDeviceInfo(){
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  std::cout << "Number of device(s): " << deviceCount << std::endl;
  if (deviceCount == 0) {
      std::cout << "There is no device supporting CUDA" << std::endl;
      return;
  }

  cudaDeviceProp info;
  for(int i=0; i<deviceCount; i++){
    cudaGetDeviceProperties(&info, i);
    std::cout << "Device " << i << std::endl;
    std::cout << "    Name:                    " << std::string(info.name) << std::endl;
    std::cout << "    Glocbal memory:          " << info.totalGlobalMem/1024.0/1024.0 << " MB"<< std::endl;
    std::cout << "    Shared memory per block: " << info.sharedMemPerBlock/1024.0 << " KB"<< std::endl;
    std::cout << "    Warp size:               " << info.warpSize<< std::endl;
    std::cout << "    Max thread per block:    " << info.maxThreadsPerBlock<< std::endl;
    std::cout << "    Thread dimension limits: " << info.maxThreadsDim[0]<< " x "
                                                 << info.maxThreadsDim[1]<< " x "
                                                 << info.maxThreadsDim[2]<< std::endl;
    std::cout << "    Max grid size:           " << info.maxGridSize[0]<< " x "
                                                 << info.maxGridSize[1]<< " x "
                                                 << info.maxGridSize[2]<< std::endl;
    std::cout << "    Compute capability:      " << info.major << "." << info.minor << std::endl;
  }
}

void GenerateBgra8K(uint8_t* buffer, int dataSize){

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> sampler(0, 255);

  for(int i=0; i<dataSize/4; i++){
    buffer[i*4] = sampler(gen);
    buffer[i*4+1] = sampler(gen);
    buffer[i*4+2] = sampler(gen);
    buffer[i*4+3] = 255;
  }
}

void convertPixelFormatCpu(uint8_t* inputBgra, uint8_t* outputYuv, int numPixels){
  short3 yuv16;
  char3 yuv8;
  for(int idx=0; idx<numPixels; idx++){
    yuv16.x = 66*inputBgra[idx*4+2] + 129*inputBgra[idx*4+1] + 25*inputBgra[idx*4];
    yuv16.y = -38*inputBgra[idx*4+2] + -74*inputBgra[idx*4+1] + 112*inputBgra[idx*4];
    yuv16.z = 112*inputBgra[idx*4+2] + -94*inputBgra[idx*4+1] + -18*inputBgra[idx*4];

    yuv8.x = (yuv16.x>>8)+16;
    yuv8.y = (yuv16.y>>8)+128;
    yuv8.z = (yuv16.z>>8)+128;

    *(reinterpret_cast<char3*>(&outputYuv[idx*3])) = yuv8;
  }
}

__global__ void convertPixelFormat(uint8_t* inputBgra, uint8_t* outputYuv, int numPixels){
  int stride = gridDim.x * blockDim.x;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  short3 yuv16;
  char3 yuv8;

  while(idx<=numPixels){
    if(idx<numPixels){
      yuv16.x = 66*inputBgra[idx*4+2] + 129*inputBgra[idx*4+1] + 25*inputBgra[idx*4];
      yuv16.y = -38*inputBgra[idx*4+2] + -74*inputBgra[idx*4+1] + 112*inputBgra[idx*4];
      yuv16.z = 112*inputBgra[idx*4+2] + -94*inputBgra[idx*4+1] + -18*inputBgra[idx*4];

      yuv8.x = (yuv16.x>>8)+16;
      yuv8.y = (yuv16.y>>8)+128;
      yuv8.z = (yuv16.z>>8)+128;

      *(reinterpret_cast<char3*>(&outputYuv[idx*3])) = yuv8;
    }
    idx += stride;
  }
}
#endif

#ifdef kernel_async_test
/*
测试结果为:

before sleep kernel
after sleep kernel 1
after sleep kernel 2
in sleep kernel 1
wake up 1
in sleep kernel 2
wake up 2
in sleep kernel 3
wake up 3
after sleep kernel 3

因此，可以知道kernel的执行和host代码是异步的;同时，使用cudaDeviceSynchronize可以实现cpu代码和gpu代码的同步
*/

__global__ void sleep_kernel(int i){
    
    //Sleep(10000);
    printf("in sleep kernel %d\n",i);
    int a=1<<31;
    while(a){
        a--;
        
    }
    printf("wake up %d\n",i);
}
int main(){
    printf("before sleep kernel\n");
    sleep_kernel<<<1,1>>>(1);
    printf("after sleep kernel 1\n");
    sleep_kernel<<<1,1>>>(2);
    printf("after sleep kernel 2\n");
    sleep_kernel<<<1,1>>>(3);
    cudaDeviceSynchronize();
    printf("after sleep kernel 3\n");
}

#endif


//#define only_one
#ifdef native_cuda_stream_use
/*
测试结果1:（没有加上同步stream和device的时间）
[2 streams] time consuming is:81.7727
[1 stream] time consuming is:63.8293
[no cuda stream] time consuming is:65.616

测试结果2:（加上同步的操作之后）
[2 streams] time consuming is:81.4873
[1 stream] time consuming is:64.392
[no cuda stream] time consuming is:66.7978

测试结果3:删掉2 streams的将值从device拷贝到cpu的代码
[2 streams] time consuming is:55.6956
[1 stream] time consuming is:64.8624
[no cuda stream] time consuming is:67.5877
*/
#define N (1024*1024)    
#define FULL_DATA_SIZE N*20    

//kernel主要实现的是数组a和数组b的均值求取
__global__ void kernel(int* a, int *b, int*c)
{
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
 
	if (threadID < N)
	{
		c[threadID] = (a[threadID] + b[threadID]) / 2;
	}
    if(threadID==1024){
        printf("thread done\n");
    }
}
 
int main()
{
	//获取设备属性  
	cudaDeviceProp prop;
	int deviceID;
	cudaGetDevice(&deviceID);
	cudaGetDeviceProperties(&prop, deviceID);
 
	//检查设备是否支持重叠功能  
	if (!prop.deviceOverlap)
	{
		printf("No device will handle overlaps. so no speed up from stream.\n");
		return 0;
	}
 
	//启动计时器  
	cudaEvent_t start, stop;
	float elapsedTime;
	
 
	//创建两个CUDA流  
	cudaStream_t stream, stream1;
	cudaStreamCreate(&stream);
	cudaStreamCreate(&stream1);
 
	int *host_a, *host_b, *host_c;
	int *dev_a, *dev_b, *dev_c;
	int *dev_a1, *dev_b1, *dev_c1;
 
	//在GPU上分配内存  
	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * sizeof(int));
 
	cudaMalloc((void**)&dev_a1, N * sizeof(int));
	cudaMalloc((void**)&dev_b1, N * sizeof(int));
	cudaMalloc((void**)&dev_c1, N * sizeof(int));
 
	//在CPU上分配页锁定内存  
	cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);

	//主机上的内存赋值 //初始化数组a和数组b的值 
	for (int i = 0; i < FULL_DATA_SIZE; i++)
	{
		host_a[i] = i;
		host_b[i] = FULL_DATA_SIZE - i;
	}

    cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	for (int i = 0; i < FULL_DATA_SIZE; i += 2 * N) //FULL_DATA_SIZE=20*N,每次使用两个流共处理2*N
	{
        //为stream1分配[i,i+N-1]的数值
		cudaMemcpyAsync(dev_a, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(dev_b, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);

        //为stream2分配[i+N,i+2*N]的数值
		cudaMemcpyAsync(dev_a1, host_a + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(dev_b1, host_b + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1);

        //TODO:没有拷贝完的话，这里会继续进行吗？
		kernel << <N / 1024, 1024, 0, stream >> > (dev_a, dev_b, dev_c);
		kernel << <N / 1024, 1024, 0, stream1 >> > (dev_a, dev_b, dev_c1);

        printf("FULL_DATA_SIZE done %d\n",i);
        //把计算得到的结果拷贝回去//把这个删掉试试
        /*
		cudaMemcpyAsync(host_c + i, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost, stream);
		cudaMemcpyAsync(host_c + i + N, dev_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream1);
        */
    }
 
	// 等待Stream流执行完成
	cudaStreamSynchronize(stream);
	cudaStreamSynchronize(stream1);
 
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
 
	std::cout << "[2 streams] time consuming is:" << elapsedTime << std::endl;


    //假如使用1个stream
    cudaEventRecord(start, 0);
    
    for (int i = 0; i < FULL_DATA_SIZE; i += N) { //每N个进行处理
        cudaMemcpyAsync(dev_a, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(dev_b, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);
        kernel <<<N / 1024, 1024,0,stream>>>(dev_a, dev_b, dev_c);
        printf("put 1 stream done %d\n",i);
    }

    cudaStreamSynchronize(stream);
    cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
 
	std::cout << "[1 stream] time consuming is:" << elapsedTime << std::endl;

    //不使用cuda stream技术
    cudaEventRecord(start, 0);
    
    for (int i = 0; i < FULL_DATA_SIZE; i += N) { //每N个进行处理
        cudaMemcpy(dev_a, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice);//cudaMemcpy操作是一个同步的过程，会使得cpu端发生阻塞
		cudaMemcpy(dev_b, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice);
        kernel<<<N/1024, 1024>>>(dev_a, dev_b, dev_c);
        printf("put kernel done %d\n",i);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
 
	std::cout <<"[no cuda stream] time consuming is:" << elapsedTime << std::endl;

    cudaEventRecord(start, 0);
    cudaMemcpyAsync(dev_a, host_a , N * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout <<"[cudaMemcpyAsync] time consuming is:" << elapsedTime << std::endl;
    
    cudaEventRecord(start, 0);
    cudaMemcpy(dev_a, host_a , N * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout <<"[cudaMemcpy] time consuming is:" << elapsedTime << std::endl;
	//输出前10个结果  
    /*
	for (int i = 0; i < 10; i++)
	{
		std::cout << host_c[i] << std::endl;
	}
    */
 
	//getchar();
 
	// free stream and mem    
	cudaFreeHost(host_a);
	cudaFreeHost(host_b);
	cudaFreeHost(host_c);
 
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
 
	cudaFree(dev_a1);
	cudaFree(dev_b1);
	cudaFree(dev_c1);
 
	cudaStreamDestroy(stream);
	cudaStreamDestroy(stream1);
	return 0;
}
#endif

#ifdef kernel_stream_test
/*
参考链接:https://zhuanlan.zhihu.com/p/460278403
问题:用来测试default stream中是把cpu线程核gpu线程都放到一个stream中了吗？
测试结果:
before sleep kernel
after sleep kernel 1
after sleep kernel 2
after sleep kernel 3
in sleep kernel 1
in sleep kernel 2
in sleep kernel 3
wake up 2
wake up 1
wake up 3

和kernel_async_test中的测试结果相比，可以得到如下的结论
1.cpu线程并没有被假如到default stream中，即cpu和各个gpu线程之间仍让是异步的
2.default stream规定了多个kernel之间的执行必须是同步的，即一个接着一个的，这可以从kernel_async_test中的"in sleep kernel"和"wake up"总是同步出现得出，而在kernel_stream_test中，三个kernel之间的执行是并行的。
3.为了解决cpu和kernel之间的异步问题，可以使用cudaDeviceSynchronize();当取消掉对于该语句的注释时，after sleep kernel3会最后被输出
4.为了解决cpu和stream[i]之间的异步问题，可以使用cudaStreamSynchronize(stream[i])来使得cpu必须等待stream[i]执行完所有的操作后才能接触阻塞
*/
__global__ void sleep_kernel(int i){
    
    //Sleep(10000);
    printf("in sleep kernel %d\n",i);
    int a=1<<31;
    while(a){
        a--;
        
    }
    printf("wake up %d\n",i);
}
int main(){
    //首先创建三个stream
    cudaStream_t streams[3];
    for(int i=0;i<3;i++){
        cudaStreamCreate(&streams[i]);
    }


    printf("before sleep kernel\n");
    sleep_kernel<<<1,1,0,streams[0]>>>(1);
    printf("after sleep kernel 1\n");
    //cudaStreamSynchronize(streams[0]);
    sleep_kernel<<<1,1,0,streams[1]>>>(2);
    printf("after sleep kernel 2\n");
    //cudaStreamSynchronize(streams[1]);
    sleep_kernel<<<1,1,0,streams[2]>>>(3);
    //cudaDeviceSynchronize();
    printf("after sleep kernel 3\n");
}
#endif