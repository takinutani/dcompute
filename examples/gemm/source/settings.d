extern (C):

// =================================================================================================
// Project: 
// Exploring the performance of general matrix-multiplication on an NVIDIA Tesla K40m GPU.
//
// File information:
// Institution.... SURFsara <www.surfsara.nl>
// Author......... Cedric Nugteren <cedric.nugteren@surfsara.nl>
// Changed at..... 2014-11-07
// License........ MIT license
// Tab-size....... 4 spaces
// Line length.... 100 characters
//
// =================================================================================================

// Select a kernel
enum KERNEL = 8;

// Constants for kernels 1 -- 5
enum TS = 32; // The square-root of the 2D tile-size (== work-group dims)

// Constants for kernels 3, 5
enum WPT = 8; // The amount of work-per-thread, i.e. the thread-coarsening factor
enum RTS = TS / WPT; // The reduced tile-size in one dimension

// Constants for kernels 4, 7 -- 10
enum WIDTH = 4; // The vector-width (in number of floats)

// Constants for kernel 5
enum TSDK = 16; // The tile-size in dimension K (for kernel 5 only)
enum LPT = (TSDK * WPT) / TS; // The amount of loads-per-thread (assume TSN==TSM)

// Constants for kernels 6 -- 10
enum TSM = 128; // The tile-size in dimension M
enum TSN = 128; // The tile-size in dimension N
enum TSK = 16; // The tile-size in dimension K
enum WPTM = 8; // The amount of work-per-thread in dimension M
enum WPTN = 8; // The amount of work-per-thread in dimension N
enum RTSM = TSM / WPTM; // The reduced tile-size in dimension M (== number of threads)
enum RTSN = TSN / WPTN; // The reduced tile-size in dimension N (== number of threads)
enum LPTA = (TSK * WPTM * WPTN) / TSN; // The amount of loads-per-thread for A
enum LPTB = (TSK * WPTM * WPTN) / TSM; // The amount of loads-per-thread for B

// Constraints on settings for kernels 6 -- 10
// Note: TSM/WPTM has to be integer
// Note: TSN/WPTN has to be integer
// Note: TSM/WIDTH has to be integer
// Note: TSN/WIDTH has to be integer
// Note: (TSK*WPTM*WPTN)/(TSN*WIDTH) has to be integer
// Note: (TSK*WPTM*WPTN)/(TSM*WIDTH) has to be integer

// Constants for kernel 11 (mimicing clBlas)
enum THREADSX = 8;
enum THREADSY = 8;
enum RX = 8;
enum RY = 4;
enum RK = RY;

// Constants for the supporting transpose kernel
enum TRANSPOSEX = 16;
enum TRANSPOSEY = 16;

// Constants for the supporting padding kernels
enum PADDINGX = 16;
enum PADDINGY = 16;

// Macros for host and kernel code
extern (D) auto MIN(T0, T1)(auto ref T0 a, auto ref T1 b)
{
    return (a > b) ? b : a;
}

extern (D) auto MAX(T0, T1)(auto ref T0 a, auto ref T1 b)
{
    return (a > b) ? a : b;
}

extern (D) auto CEIL_DIV(T0, T1)(auto ref T0 x, auto ref T1 y)
{
    return (x + y - 1) / y;
}

extern (D) auto MOD2(T0, T1)(auto ref T0 x, auto ref T1 y)
{
    return x % y;
}

extern (D) auto DIV2(T0, T1)(auto ref T0 x, auto ref T1 y)
{
    return x / y;
}

// =================================================================================================
