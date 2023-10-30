#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "../params.h"
#include "../randombytes.h"
#include "../util.cuh"

#define NTESTS 10000
#define QINV 58728449// q^(-1) mod 2^32
//#define DILITHIUM_N 256 //TODO:猜测这里是256嘛？
//#define DILITHIUM_K 这个常量代表什么

// GMEM //device表明是设备端的变量
__device__ static const int32_t gpu_zetas[DILITHIUM_N] = {
        0, 25847, -2608894, -518909, 237124, -777960, -876248, 466468,
        1826347, 2353451, -359251, -2091905, 3119733, -2884855, 3111497, 2680103,
        2725464, 1024112, -1079900, 3585928, -549488, -1119584, 2619752, -2108549,
        -2118186, -3859737, -1399561, -3277672, 1757237, -19422, 4010497, 280005,
        2706023, 95776, 3077325, 3530437, -1661693, -3592148, -2537516, 3915439,
        -3861115, -3043716, 3574422, -2867647, 3539968, -300467, 2348700, -539299,
        -1699267, -1643818, 3505694, -3821735, 3507263, -2140649, -1600420, 3699596,
        811944, 531354, 954230, 3881043, 3900724, -2556880, 2071892, -2797779,
        -3930395, -1528703, -3677745, -3041255, -1452451, 3475950, 2176455, -1585221,
        -1257611, 1939314, -4083598, -1000202, -3190144, -3157330, -3632928, 126922,
        3412210, -983419, 2147896, 2715295, -2967645, -3693493, -411027, -2477047,
        -671102, -1228525, -22981, -1308169, -381987, 1349076, 1852771, -1430430,
        -3343383, 264944, 508951, 3097992, 44288, -1100098, 904516, 3958618,
        -3724342, -8578, 1653064, -3249728, 2389356, -210977, 759969, -1316856,
        189548, -3553272, 3159746, -1851402, -2409325, -177440, 1315589, 1341330,
        1285669, -1584928, -812732, -1439742, -3019102, -3881060, -3628969, 3839961,
        2091667, 3407706, 2316500, 3817976, -3342478, 2244091, -2446433, -3562462,
        266997, 2434439, -1235728, 3513181, -3520352, -3759364, -1197226, -3193378,
        900702, 1859098, 909542, 819034, 495491, -1613174, -43260, -522500,
        -655327, -3122442, 2031748, 3207046, -3556995, -525098, -768622, -3595838,
        342297, 286988, -2437823, 4108315, 3437287, -3342277, 1735879, 203044,
        2842341, 2691481, -2590150, 1265009, 4055324, 1247620, 2486353, 1595974,
        -3767016, 1250494, 2635921, -3548272, -2994039, 1869119, 1903435, -1050970,
        -1333058, 1237275, -3318210, -1430225, -451100, 1312455, 3306115, -1962642,
        -1279661, 1917081, -2546312, -1374803, 1500165, 777191, 2235880, 3406031,
        -542412, -2831860, -1671176, -1846953, -2584293, -3724270, 594136, -3776993,
        -2013608, 2432395, 2454455, -164721, 1957272, 3369112, 185531, -1207385,
        -3183426, 162844, 1616392, 3014001, 810149, 1652634, -3694233, -1799107,
        -3038916, 3523897, 3866901, 269760, 2213111, -975884, 1717735, 472078,
        -426683, 1723600, -1803090, 1910376, -1667432, -1104333, -260646, -3833893,
        -2939036, -2235985, -420899, -2286327, 183443, -976891, 1612842, -3545687,
        -554416, 3919660, -48306, -1362209, 3937738, 1400424, -846154, 1976782};

// ptx .s32:signed 32b r:32b int | reduce usage of registers
//实现蒙哥玛丽乘法，调用了ptx指令，ptx指令不是最底层的指令，需要去验证
//__forceinline代表的是内联指令，使得代码量大但是会使得调用时间变少
//具体的指令可以自己去查询
//device关键词代表该函数是在设备上执行且仅从设备上调用
__device__ __forceinline__ int32_t gpu_montgomery_multiply(int32_t x, int32_t y) {
    int32_t t;

    asm(
            "{\n\t"
            " .reg .s32 a_hi, a_lo;\n\t" //.s代表的是signed hi是指high，lo代表的是low//声明了两个type为s32，状态空间为.reg的变量a_hi和a_lo//变量声明需要同时声明状态空间和数据类型
            " mul.hi.s32 a_hi, %1, %2;\n\t" //%1和%2代表x和y两个变量
            " mul.lo.s32 a_lo, %1, %2;\n\t"
            " mul.lo.s32 %0, a_lo, %4;\n\t"
            " mul.hi.s32 %0, %0, %3;\n\t"
            " sub.s32 %0, a_hi, %0;\n\t"
            "}"
            : "=r"(t) //r代表32bit的指数,对应参数%0
            : "r"(x), "r"(y), "r"(DILITHIUM_Q), "r"(QINV)); //分别对应参数%1 到 %4

    //    int64_t a = (int64_t) x * y;
    //    t = (int64_t) (int32_t) a * QINV;
    //    t = (a - (int64_t) t * DILITHIUM_Q) >> 32;

    return t;
}

// CT
//代表做CT蝴蝶变换的函数
__device__ __inline__ static void ntt_butt(int32_t &a, int32_t &b, const int32_t zeta) {
    int32_t t = gpu_montgomery_multiply(zeta, b);
    b = a - t;
    a = a + t;
}

// basic bank conflict
//32线程版本 s_ntt代表的是线程块内共享的内存
//regs[8]代表256/32=8，即每个线程处理的8个系数
__device__ void ntt_inner(int32_t regs[8], int32_t *s_ntt) {
    //关于每层zetas应该使用哪一个zetas可以通过自己进行推导
    // level 1
    //首先进行第一层的蝴蝶变换，第一层都是取的同样的w
    ntt_butt(regs[0], regs[4], gpu_zetas[1]);
    ntt_butt(regs[1], regs[5], gpu_zetas[1]);
    ntt_butt(regs[2], regs[6], gpu_zetas[1]);
    ntt_butt(regs[3], regs[7], gpu_zetas[1]);

    // level 2
    //第二层分为两组,前128使用w[2],后128使用w[3]
    ntt_butt(regs[0], regs[2], gpu_zetas[2]);
    ntt_butt(regs[1], regs[3], gpu_zetas[2]);
    ntt_butt(regs[4], regs[6], gpu_zetas[3]);
    ntt_butt(regs[5], regs[7], gpu_zetas[3]);
    // level 3
    //第三层分为四组,前64使用w[4],前中64使用w[5],后中64使用w[6],后64使用w[7]
    ntt_butt(regs[0], regs[1], gpu_zetas[4]);
    ntt_butt(regs[2], regs[3], gpu_zetas[5]);
    ntt_butt(regs[4], regs[5], gpu_zetas[6]);
    ntt_butt(regs[6], regs[7], gpu_zetas[7]);
    // SMEM exchange
#pragma unroll
    //这里是把计算的寄存器的值写入到共享内存中
    for (size_t i = 0; i < 8; i++)
        s_ntt[i * 32 + threadIdx.x] = regs[i];
    //Executing __syncwarp() guarantees memory ordering among threads participating in the barrier. Thus, threads within a warp that wish to communicate via memory can store to memory, execute __syncwarp(), and then safely read values stored by other threads in the warp.
    __syncwarp();//TODO:同步机制，为什么不能同步地完成呢？->猜测:可能是由于内存的操作本质上和指令的操作是不一样的
#pragma unroll
    for (size_t i = 0; i < 8; i++)
        regs[i] = s_ntt[(threadIdx.x / 4) * 32 + (threadIdx.x & 3) + i * 4];
    //现在是有8个组,一个组需要4个线程,使用threadIdx.x/4来进行分组
    // level 4
    //每个组使用相同的w,第一个组使用w[8],后续依次9、10、11...15。由于线程选定的系数一定位于同一个组内，因此，在level 4上，每个线程使用的w是相同的，由8 + threadIdx.x / 4来表示
    ntt_butt(regs[0], regs[4], gpu_zetas[8 + threadIdx.x / 4]);
    ntt_butt(regs[1], regs[5], gpu_zetas[8 + threadIdx.x / 4]);
    ntt_butt(regs[2], regs[6], gpu_zetas[8 + threadIdx.x / 4]);
    ntt_butt(regs[3], regs[7], gpu_zetas[8 + threadIdx.x / 4]);
    // level 5
    //16个组,第一个组使用w[16]。一个线程所涉及的系数，会被分成两个组。上半部分从16开始,下半部分从17开始。这16个组类似于是把每个八个组分半开，因此使用thread.x/4来表示属于哪一个组,然后再乘上2代表相应的间隔
    ntt_butt(regs[0], regs[2], gpu_zetas[16 + (threadIdx.x / 4) * 2]);
    ntt_butt(regs[1], regs[3], gpu_zetas[16 + (threadIdx.x / 4) * 2]);
    ntt_butt(regs[4], regs[6], gpu_zetas[17 + (threadIdx.x / 4) * 2]);
    ntt_butt(regs[5], regs[7], gpu_zetas[17 + (threadIdx.x / 4) * 2]);
    // level 6
    //32个组,和16个组的思路是一样的，一个线程涉及的8个系数，需要使用4个不同的旋转因子。32个组，本质上是将8个组中的每一个组划分成4份，因此分别从32、33、34、35开始，然后使用thread.x/4来定位属于哪一个组，然后乘上4代表相应的间隔
    ntt_butt(regs[0], regs[1], gpu_zetas[32 + (threadIdx.x / 4) * 4]);
    ntt_butt(regs[2], regs[3], gpu_zetas[33 + (threadIdx.x / 4) * 4]);
    ntt_butt(regs[4], regs[5], gpu_zetas[34 + (threadIdx.x / 4) * 4]);
    ntt_butt(regs[6], regs[7], gpu_zetas[35 + (threadIdx.x / 4) * 4]);
    // SMEM exchange
#pragma unroll
    for (size_t i = 0; i < 8; i++)
        s_ntt[(threadIdx.x / 4) * 32 + (threadIdx.x & 3) + i * 4] = regs[i];
    __syncwarp();
#pragma unroll
    //每个线程取连续的8个系数，此时是4个为一组
    for (size_t i = 0; i < 8; i++)
        regs[i] = s_ntt[threadIdx.x * 8 + i];
    // level 7
    //64个组，每四个元素为一组，共用同一个旋转因子。因此，在该层，每一个线程需要两个不同的旋转因子。在最上面，是从w[64]和2[65]开始的,每个线程占用两个连续的旋转因子
    ntt_butt(regs[0], regs[2], gpu_zetas[64 + threadIdx.x * 2]);
    ntt_butt(regs[1], regs[3], gpu_zetas[64 + threadIdx.x * 2]);
    ntt_butt(regs[4], regs[6], gpu_zetas[65 + threadIdx.x * 2]);
    ntt_butt(regs[5], regs[7], gpu_zetas[65 + threadIdx.x * 2]);
    // level 8
    //128个组，每个线程占用4个连续的旋转因子
    ntt_butt(regs[0], regs[1], gpu_zetas[128 + threadIdx.x * 4]);
    ntt_butt(regs[2], regs[3], gpu_zetas[129 + threadIdx.x * 4]);
    ntt_butt(regs[4], regs[5], gpu_zetas[130 + threadIdx.x * 4]);
    ntt_butt(regs[6], regs[7], gpu_zetas[131 + threadIdx.x * 4]);
}

// solve bank conflict
__device__ void ntt_inner_1(int32_t regs[8], int32_t s_ntt[DILITHIUM_N + 32]) {
    // level 1
    ntt_butt(regs[0], regs[4], gpu_zetas[1]);
    ntt_butt(regs[1], regs[5], gpu_zetas[1]);
    ntt_butt(regs[2], regs[6], gpu_zetas[1]);
    ntt_butt(regs[3], regs[7], gpu_zetas[1]);
    // level 2
    ntt_butt(regs[0], regs[2], gpu_zetas[2]);
    ntt_butt(regs[1], regs[3], gpu_zetas[2]);
    ntt_butt(regs[4], regs[6], gpu_zetas[3]);
    ntt_butt(regs[5], regs[7], gpu_zetas[3]);
    // level 3
    ntt_butt(regs[0], regs[1], gpu_zetas[4]);
    ntt_butt(regs[2], regs[3], gpu_zetas[5]);
    ntt_butt(regs[4], regs[5], gpu_zetas[6]);
    ntt_butt(regs[6], regs[7], gpu_zetas[7]);
    // SMEM exchange
#pragma unroll
    for (size_t i = 0; i < 8; i++)
        s_ntt[i * 36 + threadIdx.x] = regs[i];
    __syncwarp();
#pragma unroll
    for (size_t i = 0; i < 8; i++)
        regs[i] = s_ntt[(threadIdx.x / 4) * 36 + (threadIdx.x & 3) + i * 4];
    // level 4
    ntt_butt(regs[0], regs[4], gpu_zetas[8 + threadIdx.x / 4]);
    ntt_butt(regs[1], regs[5], gpu_zetas[8 + threadIdx.x / 4]);
    ntt_butt(regs[2], regs[6], gpu_zetas[8 + threadIdx.x / 4]);
    ntt_butt(regs[3], regs[7], gpu_zetas[8 + threadIdx.x / 4]);
    // level 5
    ntt_butt(regs[0], regs[2], gpu_zetas[16 + (threadIdx.x / 4) * 2]);
    ntt_butt(regs[1], regs[3], gpu_zetas[16 + (threadIdx.x / 4) * 2]);
    ntt_butt(regs[4], regs[6], gpu_zetas[17 + (threadIdx.x / 4) * 2]);
    ntt_butt(regs[5], regs[7], gpu_zetas[17 + (threadIdx.x / 4) * 2]);
    // level 6
    ntt_butt(regs[0], regs[1], gpu_zetas[32 + (threadIdx.x / 4) * 4]);
    ntt_butt(regs[2], regs[3], gpu_zetas[33 + (threadIdx.x / 4) * 4]);
    ntt_butt(regs[4], regs[5], gpu_zetas[34 + (threadIdx.x / 4) * 4]);
    ntt_butt(regs[6], regs[7], gpu_zetas[35 + (threadIdx.x / 4) * 4]);
    // SMEM exchange
#pragma unroll
    for (size_t i = 0; i < 8; i++)
        s_ntt[(threadIdx.x / 4) * 36 + ((threadIdx.x & 3) + i * 4) / 8 + (threadIdx.x & 3) + i * 4] = regs[i];
    __syncwarp();
#pragma unroll
    for (size_t i = 0; i < 8; i++)
        regs[i] = s_ntt[threadIdx.x * 9 + i];
    // level 7
    ntt_butt(regs[0], regs[2], gpu_zetas[64 + threadIdx.x * 2]);
    ntt_butt(regs[1], regs[3], gpu_zetas[64 + threadIdx.x * 2]);
    ntt_butt(regs[4], regs[6], gpu_zetas[65 + threadIdx.x * 2]);
    ntt_butt(regs[5], regs[7], gpu_zetas[65 + threadIdx.x * 2]);
    // level 8
    ntt_butt(regs[0], regs[1], gpu_zetas[128 + threadIdx.x * 4]);
    ntt_butt(regs[2], regs[3], gpu_zetas[129 + threadIdx.x * 4]);
    ntt_butt(regs[4], regs[5], gpu_zetas[130 + threadIdx.x * 4]);
    ntt_butt(regs[6], regs[7], gpu_zetas[131 + threadIdx.x * 4]);
}

// ignore interface
__device__ __inline__ static int32_t montgomery_multiply_c(int32_t x, const int32_t &y) {
    int32_t a_hi = __mulhi(x, y);//hi
    int32_t a_lo = x * y;//lo
    int32_t t = a_lo * QINV;//lo
    t = __mulhi(t, DILITHIUM_Q);//hi
    t = a_hi - t;
    return t;
}

//这里把所有的函数调用给展开了
__device__ void ntt_inner_unroll(int32_t regs[8], int32_t *s_poly) {
    size_t butt_idx;
    int32_t t;
    int32_t zeta;

    // level 1 128 58728449
    //zetas[1]*b
    t = regs[4] * 25847 * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(regs[4], 25847) - t;
    regs[4]=regs[0]-t;
    regs[0]=regs[0]+t;

    t = regs[5] * 25847 * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(regs[5], 25847) - t;
    regs[5]=regs[1]-t;
    regs[1]=regs[1]+t;

    t = regs[6] * 25847 * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(regs[6], 25847) - t;
    regs[6]=regs[2]-t;
    regs[2]=regs[2]+t;


    t = regs[7] * 25847 * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(regs[7], 25847) - t;
    regs[7]=regs[3]-t;
    regs[3]=regs[3]+t;

    //level2 64
    t = regs[2] * -2608894 * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(regs[2], -2608894) - t;
    regs[2]=regs[0]-t;
    regs[0]=regs[0]+t;

    t = regs[3] * -2608894 * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(regs[3], -2608894) - t;
    regs[3]=regs[1]-t;
    regs[1]=regs[1]+t;

    t = regs[6] * -518909 * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(regs[6], -518909) - t;
    regs[6]=regs[4]-t;
    regs[4]=regs[4]+t;

    t = regs[7] * -518909 * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(regs[7], -518909) - t;
    regs[7]=regs[5]-t;
    regs[5]=regs[5]+t;

    //    level3 32
    butt_idx = (threadIdx.x >> 3) + threadIdx.x;
    t = regs[1] * 237124 * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(regs[1], 237124) - t;
    s_poly[butt_idx]=regs[0]+t;
    s_poly[36 + butt_idx]=regs[0]-t;

    t = regs[3] * -777960 * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(regs[3], -777960) - t;
    s_poly[72  + butt_idx]=regs[2]+t;
    s_poly[108 + butt_idx]=regs[2]-t;

    t = regs[5] * -876248 * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(regs[5], -876248) - t;
    s_poly[144 + butt_idx]=regs[4]+t;
    s_poly[180 + butt_idx]=regs[4]-t;

    t = regs[7] * 466468 * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(regs[7], 466468) - t;
    s_poly[216 + butt_idx]=regs[6]+t;
    s_poly[252 + butt_idx]=regs[6]-t;

    //level4 16   (i * 4)/8 + i * 4
    butt_idx = (threadIdx.x >> 2) * 36 +(threadIdx.x & 3);
    regs[0]=s_poly[butt_idx];
    regs[4]=s_poly[18+butt_idx];
    zeta = gpu_zetas[8 + (threadIdx.x >> 2)];
    t = regs[4] * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(regs[4], zeta) - t;
    regs[4]=regs[0]-t;
    regs[0]=regs[0]+t;

    regs[1]=s_poly[4  + butt_idx];
    regs[5]=s_poly[22 + butt_idx];
    t = regs[5] * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(regs[5], zeta) - t;
    regs[5]=regs[1]-t;
    regs[1]=regs[1]+t;

    regs[2]=s_poly[9  + butt_idx];
    regs[6]=s_poly[27 + butt_idx];
    t = regs[6] * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(regs[6], zeta) - t;
    regs[6]=regs[2]-t;
    regs[2]=regs[2]+t;

    regs[3]=s_poly[13  + butt_idx];
    regs[7]=s_poly[31 + butt_idx];
    t = regs[7] * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(regs[7], zeta) - t;
    regs[7]=regs[3]-t;
    regs[3]=regs[3]+t;

    //level5 8
    zeta = gpu_zetas[16 + ((threadIdx.x >> 2) << 1)];
    t = regs[2] * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(regs[2], zeta) - t;
    regs[2]=regs[0]-t;
    regs[0]=regs[0]+t;

    t = regs[3] * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(regs[3], zeta) - t;
    regs[3]=regs[1]-t;
    regs[1]=regs[1]+t;

    zeta = gpu_zetas[17 + ((threadIdx.x >> 2) << 1)];
    t = regs[6] * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(regs[6], zeta) - t;
    regs[6]=regs[4]-t;
    regs[4]=regs[4]+t;

    t = regs[7] * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(regs[7], zeta) - t;
    regs[7]=regs[5]-t;
    regs[5]=regs[5]+t;

    //level6 4
    zeta = gpu_zetas[32 + ((threadIdx.x >> 2) << 2)];
    t = regs[1] * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(regs[1], zeta) - t;
    s_poly[butt_idx]=regs[0]+t;
    s_poly[4 + butt_idx]=regs[0]-t;

    zeta = gpu_zetas[33 + ((threadIdx.x >> 2) << 2)];
    t = regs[3] * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(regs[3], zeta) - t;
    s_poly[9 + butt_idx]=regs[2]+t;
    s_poly[13+ butt_idx]=regs[2]-t;

    zeta = gpu_zetas[34 + ((threadIdx.x >> 2) << 2)];
    t = regs[5] * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(regs[5], zeta) - t;
    s_poly[18+butt_idx]=regs[4]+t;
    s_poly[22 + butt_idx]=regs[4]-t;

    zeta = gpu_zetas[35 + ((threadIdx.x >> 2) << 2)];
    t = regs[7] * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(regs[7], zeta) - t;
    s_poly[27 + butt_idx]=regs[6]+t;
    s_poly[31 + butt_idx]=regs[6]-t;

    //level7 2  i
    butt_idx = threadIdx.x * 9;
    regs[0]=s_poly[butt_idx];
    regs[2]=s_poly[2 + butt_idx];
    zeta = gpu_zetas[64 + (threadIdx.x << 1)];
    t = regs[2] * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(regs[2], zeta) - t;
    regs[2]=regs[0]-t;
    regs[0]=regs[0]+t;

    regs[1]=s_poly[1 + butt_idx];
    regs[3]=s_poly[3 + butt_idx];
    t = regs[3] * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(regs[3], zeta) - t;
    regs[3]=regs[1]-t;
    regs[1]=regs[1]+t;

    regs[4]=s_poly[4 + butt_idx];
    regs[6]=s_poly[6 + butt_idx];
    zeta = gpu_zetas[65 + (threadIdx.x << 1)];
    t = regs[6] * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(regs[6], zeta) - t;
    regs[6]=regs[4]-t;
    regs[4]=regs[4]+t;

    regs[5]=s_poly[5 + butt_idx];
    regs[7]=s_poly[7 + butt_idx];
    t = regs[7] * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(regs[7], zeta) - t;
    regs[7]=regs[5]-t;
    regs[5]=regs[5]+t;

    //level8 1
    zeta = gpu_zetas[128 + (threadIdx.x << 2)];
    t = regs[1] * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(regs[1], zeta) - t;
    regs[1]=regs[0]-t;
    regs[0]=regs[0]+t;

    zeta = gpu_zetas[129 + (threadIdx.x << 2)];
    t = regs[3] * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(regs[3], zeta) - t;
    regs[3]=regs[2]-t;
    regs[2]=regs[2]+t;

    zeta = gpu_zetas[130 + (threadIdx.x << 2)];
    t = regs[5] * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(regs[5], zeta) - t;
    regs[5]=regs[4]-t;
    regs[4]=regs[4]+t;

    zeta = gpu_zetas[131 + (threadIdx.x << 2)];
    t = regs[7] * zeta * QINV;
    t = __mulhi(t, DILITHIUM_Q);
    t = __mulhi(regs[7], zeta) - t;
    regs[7]=regs[6]-t;
    regs[6]=regs[6]+t;
}

__device__ void invntt_inner_unroll(int32_t regs[8], int32_t *s_poly) {
#define F 41978
#define FMULZETA 3975713
    size_t butt_idx,w_idx;
    int32_t t,zeta;

    // level 8
    t = regs[0];
    regs[0] = t + regs[1];
    regs[1] = montgomery_multiply_c(t - regs[1], -gpu_zetas[255 - (threadIdx.x << 2)]);

    t = regs[2];
    regs[2] = t + regs[3];
    regs[3] = montgomery_multiply_c(t - regs[3], -gpu_zetas[254 - (threadIdx.x << 2)]);

    t = regs[4];
    regs[4] = t + regs[5];
    regs[5] = montgomery_multiply_c(t - regs[5], -gpu_zetas[253 - (threadIdx.x << 2)]);

    t = regs[6];
    regs[6] = t + regs[7];
    regs[7] = montgomery_multiply_c(t - regs[7], -gpu_zetas[252 - (threadIdx.x << 2)]);

    // level 7 i
    butt_idx = threadIdx.x * 9;
    zeta = -gpu_zetas[127 - (threadIdx.x << 1)];
    t = regs[0];
    s_poly[butt_idx] = t + regs[2];
    s_poly[2 + butt_idx] = montgomery_multiply_c(t - regs[2], zeta);

    t = regs[1];
    s_poly[1 + butt_idx] = t + regs[3];
    s_poly[3 + butt_idx] = montgomery_multiply_c(t - regs[3], zeta);

    zeta = -gpu_zetas[126 - (threadIdx.x << 1)];
    t = regs[4];
    s_poly[4 + butt_idx] = t + regs[6];
    s_poly[6 + butt_idx] = montgomery_multiply_c(t - regs[6], zeta);

    t = regs[5];
    s_poly[5 + butt_idx] = t + regs[7];
    s_poly[7 + butt_idx] = montgomery_multiply_c(t - regs[7], zeta);

    // level 6
    butt_idx = (threadIdx.x >> 2) * 36 +(threadIdx.x & 3);
    w_idx = (threadIdx.x >> 2) << 2;
    t = s_poly[butt_idx];
    regs[1] = s_poly[4 + butt_idx];
    regs[0] = t + regs[1];
    regs[1] = montgomery_multiply_c(t - regs[1], -gpu_zetas[63 - w_idx]);

    t = s_poly[9 + butt_idx];
    regs[3] = s_poly[13+ butt_idx];
    regs[2] = t + regs[3];
    regs[3] = montgomery_multiply_c(t - regs[3], -gpu_zetas[62 - w_idx]);

    t = s_poly[18 + butt_idx];
    regs[5] = s_poly[22 + butt_idx];
    regs[4] = t + regs[5];
    regs[5] = montgomery_multiply_c(t - regs[5], -gpu_zetas[61 - w_idx]);

    t = s_poly[27 + butt_idx];
    regs[7] = s_poly[31 + butt_idx];
    regs[6] = t + regs[7];
    regs[7] = montgomery_multiply_c(t - regs[7], -gpu_zetas[60 - w_idx]);

    // level 5
    w_idx = (threadIdx.x >> 2) << 1;
    zeta = -gpu_zetas[31 - w_idx];
    t = regs[0];
    regs[0] = t + regs[2];
    regs[2] = montgomery_multiply_c(t - regs[2], zeta);

    t = regs[1];
    regs[1] = t + regs[3];
    regs[3] = montgomery_multiply_c(t - regs[3], zeta);

    zeta = -gpu_zetas[30 - w_idx];
    t = regs[4];
    regs[4] = t + regs[6];
    regs[6] = montgomery_multiply_c(t - regs[6], zeta);

    t = regs[5];
    regs[5] = t + regs[7];
    regs[7] = montgomery_multiply_c(t - regs[7], zeta);

    // level 4
    zeta = -gpu_zetas[15 - (threadIdx.x >> 2)];
    t = regs[0];
    s_poly[butt_idx] = t + regs[4];
    s_poly[18+butt_idx] = montgomery_multiply_c(t - regs[4], zeta);

    t = regs[1];
    s_poly[4  + butt_idx] = t + regs[5];
    s_poly[22 + butt_idx] = montgomery_multiply_c(t - regs[5], zeta);

    t = regs[2];
    s_poly[9  + butt_idx] = t + regs[6];
    s_poly[27 + butt_idx] = montgomery_multiply_c(t - regs[6], zeta);

    t = regs[3];
    s_poly[13  + butt_idx] = t + regs[7];
    s_poly[31 + butt_idx] = montgomery_multiply_c(t - regs[7], zeta);

    // level 3
    butt_idx = (threadIdx.x >> 3) + threadIdx.x;
    t = s_poly[butt_idx];
    regs[1] = s_poly[36 + butt_idx];
    regs[0] = t + regs[1];
    regs[1] = montgomery_multiply_c(t - regs[1], -466468);

    t = s_poly[72  + butt_idx];
    regs[3] = s_poly[108 + butt_idx];
    regs[2] = t + regs[3];
    regs[3] = montgomery_multiply_c(t - regs[3], 876248);

    t = s_poly[144 + butt_idx];
    regs[5] = s_poly[180 + butt_idx];
    regs[4] = t + regs[5];
    regs[5] = montgomery_multiply_c(t - regs[5], 777960);

    t = s_poly[216 + butt_idx];
    regs[7] = s_poly[252 + butt_idx];
    regs[6] = t + regs[7];
    regs[7] = montgomery_multiply_c(t - regs[7], -237124);

    // level 2
    t = regs[0];
    regs[0] = t + regs[2];
    regs[2] = montgomery_multiply_c(t - regs[2], 518909);

    t = regs[1];
    regs[1] = t + regs[3];
    regs[3] = montgomery_multiply_c(t - regs[3], 518909);

    t = regs[4];
    regs[4] = t + regs[6];
    regs[6] = montgomery_multiply_c(t - regs[6], 2608894);

    t = regs[5];
    regs[5] = t + regs[7];
    regs[7] = montgomery_multiply_c(t - regs[7], 2608894);

    // level 1
    t = regs[0];
    regs[0] = montgomery_multiply_c(t + regs[4],F);
    regs[4] = montgomery_multiply_c(t - regs[4], FMULZETA);

    t = regs[1];
    regs[1] = montgomery_multiply_c(t + regs[5],F);
    regs[5] = montgomery_multiply_c(t - regs[5], FMULZETA);

    t = regs[2];
    regs[2] = montgomery_multiply_c(t + regs[6],F);
    regs[6] = montgomery_multiply_c(t - regs[6], FMULZETA);

    t = regs[3];
    regs[3] = montgomery_multiply_c(t + regs[7],F);
    regs[7] = montgomery_multiply_c(t - regs[7], FMULZETA);
}

#define K DILITHIUM_K
// 1 polyvec/block
/*
    功能:g_polyvec是一个二维的矩阵,由NIEST行组成,每行含有DILITHIUM_K个多项式,每个多项式含有DILITHIUM_N个系数。
        共启动了NIEST*32个线程,因为每一个线程块需要对一行中的DILITHIUM_K个多项式进行NTT操作。对于每一个多项式的
        NTT操作由32线程协同完成,主要定义在了ntt_inner函数中
    参数:第一个参数代表的是待测试的数组,每一行由K个多项式组成,共有NIEST行;
        第二个参数代表实际申请的是二维的空间中每一行的字节数
    说明:32个线程为一个线程块,共有NIESTS个线程块
*/

__global__ void k0_ntt(int32_t *g_polyvec, size_t g_polyvec_pitch) {//bank conflict
    __shared__ int32_t s_poly[DILITHIUM_N];//32个线程即一个线程块中所有线程所共享的内存，含256个元素
    int32_t regs[8]; //每一个线程所私有的寄存器内存
    //每个线程块对应的行为line=g_polyvec + blockIdx.x * g_polyvec_pitch / sizeof(int32_t)
    //line+=k*DILITHIUM_N可以来索引该行中的第k个多项式
    for (int k = 0; k < K; ++k) { //遍历每一个多项式
        int32_t *g_poly = g_polyvec + blockIdx.x * g_polyvec_pitch / sizeof(int32_t) + k * DILITHIUM_N;//索引的作用，每个块对应的具体的256个系数
        //下面开始对多项式g_poly进行NTT操作

        for (size_t i = 0; i < 8; ++i) //每个线程以32为间隔取出自己的系数
            regs[i] = g_poly[i * 32 + threadIdx.x];//将全局内存拷贝到寄存器中
        ntt_inner(regs, s_poly);
        for (size_t i = 0; i < 8; ++i)
            g_poly[threadIdx.x * 8 + i] = regs[i]; //把自己计算的结果写回到全局内存中
    }
}

__global__ void k0_ntt_noBC(int32_t *g_polyvec, size_t g_polyvec_pitch) {//no bank conflict
    __shared__ int32_t s_poly[DILITHIUM_N + 32];
    int32_t regs[8];
    for (int k = 0; k < K; ++k) {
        int32_t *g_poly = g_polyvec + blockIdx.x * g_polyvec_pitch / sizeof(int32_t) + k * DILITHIUM_N;
        for (size_t i = 0; i < 8; ++i)
            regs[i] = g_poly[i * 32 + threadIdx.x];
        ntt_inner_1(regs, s_poly);
        for (size_t i = 0; i < 8; ++i)
            g_poly[threadIdx.x * 8 + i] = regs[i];
    }
}

__global__ void k0_ntt_unroll(int32_t *g_polyvec, size_t g_polyvec_pitch) {//unroll
    __shared__ int32_t s_poly[DILITHIUM_N + 32];
    int32_t regs[8];
    for (int k = 0; k < K; ++k) {
        int32_t *g_poly = g_polyvec + blockIdx.x * g_polyvec_pitch / sizeof(int32_t) + k * DILITHIUM_N;
        for (size_t i = 0; i < 8; ++i)
            regs[i] = g_poly[i * 32 + threadIdx.x];
        ntt_inner_unroll(regs, s_poly);
        for (size_t i = 0; i < 8; ++i)
            g_poly[threadIdx.x * 8 + i] = regs[i];
    }
}

__global__ void k0_unpack(int32_t *g_polyvec, size_t g_polyvec_pitch,
                          const uint8_t *g_polyvec_packed, size_t g_polyvec_packed_pitch) {
    for (int k = 0; k < DILITHIUM_K; k++) {
        auto *g_poly_packed = g_polyvec_packed + blockIdx.x * g_polyvec_packed_pitch + k * POLYT0_PACKEDBYTES;
        int32_t *g_poly = g_polyvec + blockIdx.x * g_polyvec_pitch / sizeof(int32_t) + k * DILITHIUM_N;

        for (size_t i = 0; i < 8; i++) {
            uint32_t t = (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 + ((threadIdx.x & 7) / 2) * 3 + 0]) |
                         (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 + ((threadIdx.x & 7) / 2) * 3 + 1] << 8) |
                         (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 + ((threadIdx.x & 7) / 2) * 3 + 2] << 16) |
                         (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 + ((threadIdx.x & 7) / 2) * 3 + 3] << 24);
            t >>= (threadIdx.x & 7) * 13 - ((threadIdx.x & 7) / 2) * 3 * 8;
            t &= 0x1FFF;
            g_poly[i * 32 + threadIdx.x] = (1 << (DILITHIUM_D - 1)) - (int32_t) t;
        }
    }
}

// kernel fusing
__global__ void k1_unpack_fuse_ntt(int32_t *g_polyvec, size_t g_polyvec_pitch,
                                   const uint8_t *g_polyvec_packed, size_t g_polyvec_packed_pitch) {
    __shared__ int32_t s_ntt[DILITHIUM_N];
    int32_t regs[8];

    // unpack
    for (int k = 0; k < DILITHIUM_K; ++k) {
        auto *g_poly_packed = g_polyvec_packed + blockIdx.x * g_polyvec_packed_pitch + k * POLYT0_PACKEDBYTES;
        int32_t *g_poly = g_polyvec + blockIdx.x * g_polyvec_pitch / sizeof(int32_t) + k * DILITHIUM_N;
        for (size_t i = 0; i < 8; i++) {
            uint32_t t = (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 + ((threadIdx.x & 7) / 2) * 3 + 0]) |
                         (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 + ((threadIdx.x & 7) / 2) * 3 + 1] << 8) |
                         (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 + ((threadIdx.x & 7) / 2) * 3 + 2] << 16) |
                         (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 + ((threadIdx.x & 7) / 2) * 3 + 3] << 24);
            t >>= (threadIdx.x & 7) * 13 - ((threadIdx.x & 7) / 2) * 3 * 8;
            t &= 0x1FFF;
            g_poly[i * 32 + threadIdx.x] = (1 << (DILITHIUM_D - 1)) - (int32_t) t;
        }
    }

    // ntt
    for (int k = 0; k < DILITHIUM_K; ++k) {
        int32_t *g_poly = g_polyvec + blockIdx.x * g_polyvec_pitch / sizeof(int32_t) + k * DILITHIUM_N;
        for (size_t i = 0; i < 8; ++i)
            regs[i] = g_poly[i * 32 + threadIdx.x];
        ntt_inner(regs, s_ntt);
        for (size_t i = 0; i < 8; i++)
            g_poly[threadIdx.x * 8 + i] = regs[i];
    }
}

// merge two loops into one and use registers to store intermediate poly
__global__ void k2(int32_t *g_polyvec, size_t g_polyvec_pitch,
                   const uint8_t *g_polyvec_packed, size_t g_polyvec_packed_pitch) {
    __shared__ int32_t s_ntt[DILITHIUM_N];
    int32_t regs[8];

    for (int k = 0; k < DILITHIUM_K; ++k) {
        // unpack
        auto *g_poly_packed = g_polyvec_packed + blockIdx.x * g_polyvec_packed_pitch + k * POLYT0_PACKEDBYTES;
        for (size_t i = 0; i < 8; i++) {
            uint32_t t = (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 + ((threadIdx.x & 7) / 2) * 3 + 0]) |
                         (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 + ((threadIdx.x & 7) / 2) * 3 + 1] << 8) |
                         (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 + ((threadIdx.x & 7) / 2) * 3 + 2] << 16) |
                         (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 + ((threadIdx.x & 7) / 2) * 3 + 3] << 24);
            t >>= (threadIdx.x & 7) * 13 - ((threadIdx.x & 7) / 2) * 3 * 8;
            t &= 0x1FFF;
            regs[i] = (1 << (DILITHIUM_D - 1)) - (int32_t) t;
        }
        // ntt
        int32_t *g_poly = g_polyvec + blockIdx.x * g_polyvec_pitch / sizeof(int32_t) + k * DILITHIUM_N;
        ntt_inner(regs, s_ntt);
        for (size_t i = 0; i < 8; i++)
            g_poly[threadIdx.x * 8 + i] = regs[i];
    }
}

// avoid smem bank conflict in ntt
__global__ void k3(int32_t *g_polyvec, size_t g_polyvec_pitch,
                   const uint8_t *g_polyvec_packed, size_t g_polyvec_packed_pitch) {
    __shared__ int32_t s_ntt[DILITHIUM_N + 32];
    int32_t regs[8];

    for (int k = 0; k < DILITHIUM_K; ++k) {
        // unpack
        auto *g_poly_packed = g_polyvec_packed + blockIdx.x * g_polyvec_packed_pitch + k * POLYT0_PACKEDBYTES;
        for (size_t i = 0; i < 8; i++) {
            uint32_t t = (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 + ((threadIdx.x & 7) / 2) * 3 + 0]) |
                         (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 + ((threadIdx.x & 7) / 2) * 3 + 1] << 8) |
                         (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 + ((threadIdx.x & 7) / 2) * 3 + 2] << 16) |
                         (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 + ((threadIdx.x & 7) / 2) * 3 + 3] << 24);
            t >>= (threadIdx.x & 7) * 13 - ((threadIdx.x & 7) / 2) * 3 * 8;
            t &= 0x1FFF;
            regs[i] = (1 << (DILITHIUM_D - 1)) - (int32_t) t;
        }
        // ntt
        int32_t *g_poly = g_polyvec + blockIdx.x * g_polyvec_pitch / sizeof(int32_t) + k * DILITHIUM_N;
        ntt_inner_1(regs, s_ntt);
        for (size_t i = 0; i < 8; i++)
            g_poly[threadIdx.x * 8 + i] = regs[i];
    }
}

// avoid smem bank conflict in ntt + unroll
__global__ void k4(int32_t *g_polyvec, size_t g_polyvec_pitch,
                   const uint8_t *g_polyvec_packed, size_t g_polyvec_packed_pitch) {
    __shared__ int32_t s_ntt[DILITHIUM_N + 32];
    int32_t regs[8];

    for (int k = 0; k < DILITHIUM_K; ++k) {
        // unpack
        auto *g_poly_packed = g_polyvec_packed + blockIdx.x * g_polyvec_packed_pitch + k * POLYT0_PACKEDBYTES;
        for (size_t i = 0; i < 8; i++) {
            uint32_t t = (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 + ((threadIdx.x & 7) / 2) * 3 + 0]) |
                         (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 + ((threadIdx.x & 7) / 2) * 3 + 1] << 8) |
                         (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 + ((threadIdx.x & 7) / 2) * 3 + 2] << 16) |
                         (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 + ((threadIdx.x & 7) / 2) * 3 + 3] << 24);
            t >>= (threadIdx.x & 7) * 13 - ((threadIdx.x & 7) / 2) * 3 * 8;
            t &= 0x1FFF;
            regs[i] = (1 << (DILITHIUM_D - 1)) - (int32_t) t;
        }
        // ntt
        int32_t *g_poly = g_polyvec + blockIdx.x * g_polyvec_pitch / sizeof(int32_t) + k * DILITHIUM_N;
        ntt_inner_unroll(regs, s_ntt);
        for (size_t i = 0; i < 8; i++)
            g_poly[threadIdx.x * 8 + i] = regs[i];
    }
}

__global__ void test_ntt_correctness() {
    __shared__ int32_t s_ntt[DILITHIUM_N + 32];
    int32_t regs[8];

    for (size_t i = 0; i < 8; ++i)
        regs[i] = 32 * i + threadIdx.x;

    ntt_inner(regs, s_ntt);

    printf("%d ", regs[0]);
    if (threadIdx.x == 0) printf("\n");

    for (size_t i = 0; i < 8; ++i)
        regs[i] = 32 * i + threadIdx.x;

    ntt_inner_1(regs, s_ntt);

    printf("%d ", regs[0]);
    if (threadIdx.x == 0) printf("\n");

}

int main(void) {
    uint8_t *d_polyveck_packed;
    int32_t *d_polyveck;
    size_t d_polyveck_packed_pitch;
    size_t d_polyveck_pitch;

    cudaMallocPitch(&d_polyveck_packed, &d_polyveck_packed_pitch, DILITHIUM_K * POLYT0_PACKEDBYTES, NTESTS);//NTESTS代表的是并发度,也就是对应的行数，
    cudaMallocPitch(&d_polyveck, &d_polyveck_pitch, DILITHIUM_K * DILITHIUM_N * sizeof(int32_t), NTESTS); //TODO:这里的两个宏的具体含义是什么？->这里的K代表着一个多项式向量中的元素（指多项式）个数，N代表的是一个多项式含有多少个系数

    print_timer_banner();

    CUDATimer timer_k4("k4_no_BC_unroll");
    CUDATimer timer_k3("k3_no_BC");
    CUDATimer timer_k2("k2_fuse_loop");
    CUDATimer timer_k1("k1_unpack_fuse_ntt");
    CUDATimer timer_k0("k0_baseline");

//    CUDATimer timer_Unroll("ntt_Unroll");
//    CUDATimer timer_no_BC("ntt_no_BC");
//    CUDATimer timer_BC("ntt_BC");

    for (size_t i = 0; i < 1000; ++i) { //执行1000次来获取相应的平均或者最小的实践
//        
          //timer_BC.start();
//        k0_ntt<<<NTESTS, 32>>>(d_polyveck, d_polyveck_pitch); //分配NIESTS个线程块，每个线程块含有32个线程//可能是每个线程对应一个向量，是因为可能存在相应的访问冲突所以才会这个样子嘛？还是说不一定是32个线程为一组 ？
//        cudaDeviceSynchronize();
//        timer_BC.stop();
//
//        timer_no_BC.start();
//        k0_ntt_noBC<<<NTESTS, 32>>>(d_polyveck, d_polyveck_pitch);
//        cudaDeviceSynchronize();
//        timer_no_BC.stop();
//
//        timer_Unroll.start();
//        k0_ntt_unroll<<<NTESTS, 32>>>(d_polyveck, d_polyveck_pitch);
//        cudaDeviceSynchronize();
//        timer_Unroll.stop();

        timer_k0.start();
        k0_unpack<<<NTESTS, 32>>>(d_polyveck, d_polyveck_pitch, d_polyveck_packed, d_polyveck_packed_pitch);
        k0_ntt<<<NTESTS, 32>>>(d_polyveck, d_polyveck_pitch);
        cudaDeviceSynchronize();
        timer_k0.stop();

        timer_k1.start();
        k1_unpack_fuse_ntt<<<NTESTS, 32>>>(
                d_polyveck, d_polyveck_pitch,
                d_polyveck_packed, d_polyveck_packed_pitch);
        cudaDeviceSynchronize();
        timer_k1.stop();

        timer_k2.start();
        k2<<<NTESTS, 32>>>(
                d_polyveck, d_polyveck_pitch,
                d_polyveck_packed, d_polyveck_packed_pitch);
        cudaDeviceSynchronize();
        timer_k2.stop();

        timer_k3.start();
        k3<<<NTESTS, 32>>>(
                d_polyveck, d_polyveck_pitch,
                d_polyveck_packed, d_polyveck_packed_pitch);
        cudaDeviceSynchronize();
        timer_k3.stop();

        timer_k4.start();
        k4<<<NTESTS, 32>>>(
                d_polyveck, d_polyveck_pitch,
                d_polyveck_packed, d_polyveck_packed_pitch);
        cudaDeviceSynchronize();
        timer_k4.stop();
    }

    cudaFree(d_polyveck_packed);
    cudaFree(d_polyveck);

    CHECK_LAST_CUDA_ERROR();

    return 0;
}
