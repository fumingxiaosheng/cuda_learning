//AES基于电子密码本模式，将数据以128bit进行分组
//1.密钥的提取
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include<iostream>
#include <stdio.h>
#include <chrono>
#include <cstdlib>
#include<ctime>//debug:three header must be included
using namespace std;
#define uint_test

//uint4:16 byte; uint32_t:8 byte; uint8_t: 1byte;
#define u8 uint8_t
#define  u32 uint32_t
#define KEY_T uint4[11]

#define EXTRACT_D0( ((x)>>24uL)) //提取高8bit
#define EXTRACT_D1( ( (x) >> 16uL ) & 0xFFuL) //提取中高8bit
#define EXTRACT_D2( ( (x) >> 8uL) & 0xFFuL) //提取中低8bit
#define EXTRACT_D3( (x) & 0xFFuL) //提取低8bit

#define MIX_COL(a) ( ( (a)& 0x80uL) ? ((( (a)<<1uL) &0xFFuL) ^ 0xBu) : ((a)<<1uL) )
#define XOR_5(a,b,c,d,e) ( ((a)^(b)) ^ ((c)^(d)) ^ (e))

u8 * s_box_ptr ;//= s_box;//s盒代换 TODO:这个S盒是什么样子的？

uint4 * cipher_block(unsigned char * cipher){
    uint4 *cipher_blocks = reinterpret_cast<uint4 *>(cipher);
    return cipher_blocks;
}
void key_expansion(){

}

__host__  void AES_encrypt__4_4_reg(
    uint4 * const cipher_block,//128bit的密文数据
    KEY_T * const cipher_key,//KEY_T是由10个int4的变量组成，每个uint4会保留一个循环密钥
    const u32 round_nums //当前的轮数
    ){
    
    //step1:处理密文，将128bit转化为4*4的矩阵形式
    //每个wi是32bit
    u32 w0=cipher_block->w;
    u32 w1=cipher_block->x;
    u32 w2=cipher_block->y;
    u32 w3=cipher_block->z;

    u8 a0=EXTRACT_D0(w0);
    u8 a1=EXTRACT_D0(w1);
    u8 a2=EXTRACT_D0(w2);
    u8 a3=EXTRACT_D0(w3);

    u8 a4=EXTRACT_D1(w0);
    u8 a5=EXTRACT_D1(w1);
    u8 a6=EXTRACT_D1(w2);
    u8 a7=EXTRACT_D1(w3);

    u8 a8=EXTRACT_D2(w0);
    u8 a9=EXTRACT_D2(w1);
    u8 a10=EXTRACT_D2(w2);
    u8 a11=EXTRACT_D2(w3);

    u8 a12=EXTRACT_D3(w0);
    u8 a13=EXTRACT_D3(w1);
    u8 a14=EXTRACT_D3(w2);
    u8 a15=EXTRACT_D3(w3);

    //step2:从内存中拿出密钥,并进行最初的轮密钥异或操作，此处是第0轮
    u32 round_num=0;
    w0=(*cipher_key)[round_num].w;
    w1=(*cipher_key)[round_num].x;
    w2=(*cipher_key)[round_num].y;
    w3=(*cipher_key)[round_num].z;

    a0 ^=EXTRACT_D0(w0);
    a1 ^=EXTRACT_D0(w1);
    a2 ^=EXTRACT_D0(w2);
    a3 ^=EXTRACT_D0(w3);

    a4 ^= EXTRACT_D1(w0);
    a5 ^= EXTRACT_D1(w1);
    a6 ^= EXTRACT_D1(w2);
    a7 ^= EXTRACT_D1(w3);

    a8 ^= EXTRACT_D2(w0);
    a9 ^= EXTRACT_D2(w1);
    a10 ^= EXTRACT_D2(w2);
    a11 ^= EXTRACT_D2(w3);

    a12 ^= EXTRACT_D3(w0);
    a13 ^= EXTRACT_D3(w1);
    a14 ^= EXTRACT_D3(w2);
    a15 ^= EXTRACT_D3(w3);
    round_num ++;

    //step2:开始进行迭代
    while(round_num <= round_nums){
        //step2.1:取该轮对应的密钥
        {
        w0=(*cipher_key)[round_num].w;
        w1=(*cipher_key)[round_num].x;
        w2=(*cipher_key)[round_num].y;
        w3=(*cipher_key)[round_num].z;
        }

        //step2.2:字节代换
        {
        a0=s_box_ptr[a0];
        a1=s_box_ptr[a1];
        a2=s_box_ptr[a2];
        a3=s_box_ptr[a3];

        a4=s_box_ptr[a4];
        a5=s_box_ptr[a5];
        a6=s_box_ptr[a6];
        a7=s_box_ptr[a7];
        
        a8=s_box_ptr[a8];
        a9=s_box_ptr[a9];
        a10=s_box_ptr[a10];
        a11=s_box_ptr[a11];

        a12=s_box_ptr[a12];
        a13=s_box_ptr[a13];
        a14=s_box_ptr[a14];
        a15=s_box_ptr[a15];
        }

        u8 tmp0,tmp1,tmp2,tmp3;
        //step2.3:行移位变换
        {
        //a0,a4,a8,a12保持不变
        //a1,a5,a9,a13-> a5,a9,a13,a1
        tmp0=a1;
        a1=a5;
        a5=a9;
        a9=a13;
        a13=tmp0;
        //a2,a6,a10,a14-> a10,a14,a2,a6
        tmp0=a2;
        tmp1=a6;
        a2=a10;
        a6=a14;
        a10=tmp0;
        a14=tmp1;
        //a3,a7,a11,a15->a15,a3,a7,a11
        tmp0=a3;
        tmp1=a7;
        tmp2=a11;
        a3=a15;
        a7=tmp0;
        a11=tmp1;
        a15=tmp2;
        }

        //step2.4:在非最后一轮的情况下需要进行列混合变换(具体的实现参照P220的列混合讲解)
        if(round_num!=10){
            u8 b0 =MIX_COL(a0);
            u8 b1 =MIX_COL(a1);
            u8 b2 =MIX_COL(a2);
            u8 b3 =MIX_COL(a3);

            u8 b4 =MIX_COL(a4);
            u8 b5 =MIX_COL(a5);
            u8 b6 =MIX_COL(a6);
            u8 b7 =MIX_COL(a7);

            u8 b8 =MIX_COL(a8);
            u8 b9 =MIX_COL(a9);
            u8 b10 =MIX_COL(a10);
            u8 b11 =MIX_COL(a11);

            u8 b12 =MIX_COL(a12);
            u8 b13 =MIX_COL(a13);
            u8 b14 =MIX_COL(a14);
            u8 b15 =MIX_COL(a15);

            tmp0=XOR_5(b0,a3,a2,b1,a1);
            tmp1=XOR_5(b1,a0,a3,b2,a2);
            tmp2=XOR_5(b2,a1,a0,b3,a3);
            tmp3=XOR_5(b3,a2,a1,b0,a0);
            a0=tmp0,a1=tmp1,a2=tmp2,a3=tmp3;

            tmp0=XOR_5(b4,a7,a6,b5,a5);
            tmp1=XOR_5(b5,a4,a7,b6,a6);
            tmp2=XOR_5(b6,a5,a4,b7,a7);
            tmp3=XOR_5(b7,a6,a5,b4,a4);
            a4=tmp0,a5=tmp1,a6=tmp2,a7=tmp3;

            tmp0=XOR_5(b8,a11,a10,b9,a9);
            tmp1=XOR_5(b9,a8,a11,b10,a10);
            tmp2=XOR_5(b10,a9,a8,b11,a11);
            tmp3=XOR_5(b11,a10,a9,b8,a8);
            a8=tmp0,a9=tmp1,a10=tmp2,a11=tmp3;

            tmp0=XOR_5(b8,a11,a10,b9,a9);
            tmp1=XOR_5(b9,a8,a11,b10,a10);
            tmp2=XOR_5(b10,a9,a8,b11,a11);
            tmp3=XOR_5(b11,a10,a9,b8,a8);
            a8=tmp0,a9=tmp1,a10=tmp2,a11=tmp3;




        }
}
int main(){

#ifdef uint_test //debug:不能这样来用,uint4是cuda的内置数据类型
    printf("%d %d %d\n",sizeof(uint4),sizeof(uint32_t),sizeof(uint8_t));
    uint4 a = {1,2,3,4};
    u32 w0=(&a)->w;
    u32 w1=(&a)->x;
    u32 w2=(&a)->y;
    u32 w3=(&a)->z;
#endif

}