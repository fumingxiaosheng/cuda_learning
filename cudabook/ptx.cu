//https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#state-spaces 
//.param代表的是kernel函数的参数，或者是本地的参数

/*
ptx基础数据类型表
Table 8 Fundamental Type Specifiers
Basic Type  Fundamental Type Specifiers
Signed integer  .s8, .s16, .s32, .s64
Unsigned integer    .u8, .u16, .u32, .u64
Floating-point  .f16, .f16x2, .f32, .f64
Bits (untyped)  .b8, .b16, .b32, .b64, .b128
Predicate   .pred

ptx中的vector
ptx支持长度为2或者4的向量，分别使用.v2和.v4进行表示，例如 .v4 .s8代表长度为4的有8bit符号整数的向量。ptx要求，整个向量的大小不能超过128bit
*/
/*
ptx的状态空间表
Name    Description
.reg    Registers, fast.
.sreg   Special registers. Read-only; pre-defined; platform-specific.
.const  Shared, read-only memory.
.global Global memory, shared by all threads.
.local  Local memory, private to each thread.
.param  Kernel parameters, defined per-grid; or Function or local parameters, defined per-thread.
.shared Addressable memory, defined per CTA, accessible to all threads in the cluster throughout the lifetime of the CTA that defines it.
.tex    Global texture memory (deprecated).
*/
/*

每个指令里的操作数都要声明其类型，而且类型必须符合指令的模板，并没有自动的类型转换。
cvt指令一般完成的是数据类型的转化
ld指令从内存空间中将数值加载到寄存器中
st指令从寄存器中将数值存储到内存空间中
mov指令在寄存器间进行数值的传递

目的操作数往往是一个寄存器，用于存储相应的计算结果

1. 使用地址作为操作数时，使用[x]来进行寻址，其中x代表的是具体的地址值，x可以是var、reg、immAddr等，使用数组时，可以是用var[immAddr]来对数组中的元素进行索引

.shared .u16 x; 
.reg .u16 r0; 
.global .v4 .f32 V; 
.reg .v4 .f32 W; 
.const .s32 tbl[256];
.reg .b32 p; .reg .s32 q; 

ld.shared.u16 r0,[x]; 
ld.global.v4.f32 W, [V]; 
ld.const.s32 q, [tbl+12]; 
mov.u32 p, tbl;

2. 使用数组作为操作数时，可以使用具体计算的地址或者使用[]来索引数组中的元素
*/

/*
cuda内联ptx: https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html
asm("template-string" : "constraint"(output) : "constraint"(input));
其中template-string中包括多个以分号分隔的ptx指令,一条语句的结尾往往使用"\n\t",input和output中若包含多个参数，则可以使用逗号进行分割。在template-string中，可以使用%n来代表相应的输入和输出，%n对应了第n个参数

asm("add.s32 %0, %1, %2;" : "=r"(i) : "r"(j), "r"(k)); 这句话就等同于"add.s32 i, j, k;" ，其中r代表的是32bit的整数寄存器。
"=r"代表的是要写入的寄存器，"+r"代表的是要读取和写入的寄存器
asm("add.s32 %0, %0, %1;" : "+r"(i) : "r" (j));
此外，+也可以代表结果是条件的写入，如下所示
__device__ int cond (int x)
{
  int y = 0;
  asm("{\n\t"
      " .reg .pred %p;\n\t"
      " setp.eq.s32 %p, %1, 34;\n\t" // x == 34?
      " @%p mov.s32 %0, 1;\n\t"      // set y to 1 if true
      "}"                            // conceptually y = (x==34)?1:y
      : "+r"(y) : "r" (x));
  return y;
}

在没有输入的时候，可以省略掉最后一个冒号，在没有输出的时候，不能省略任何一个冒号

关于输入和输出，有如下的限制
"h" = .u16 reg
"r" = .u32 reg
"l" = .u64 reg
"f" = .f32 reg
"d" = .f64 reg
n代表的是一个立即数操作数，例如asm("add.u32 %0, %0, %1;" : "=r"(x) : "n"(42));

*/

/*
ptx中各种操作的详细介绍
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions
*/
.entry foo ( .param .b32 N, .param .align 8 .b8 buffer[64]) 
{ 
    .reg .u32 %n; 
    .reg .f64 %d; 
    ld.param.u32 %n, [N]; 
    ld.param.f64 %d, [buffer]; 
    ...