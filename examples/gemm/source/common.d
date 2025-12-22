import core.stdc.config;

extern (C):

// =================================================================================================
// Project: 
// Exploring the performance of general matrix-multiplication on an NVIDIA Tesla K40m GPU.
//
// File information:
// Institution.... SURFsara <www.surfsara.nl>
// Author......... Cedric Nugteren <cedric.nugteren@surfsara.nl>
// Changed at..... 2014-11-17
// License........ MIT license
// Tab-size....... 4 spaces
// Line length.... 100 characters
//
// =================================================================================================

// Common C includes

// #include <sys/time.h>
// #include "sys/time.h"

// =================================================================================================

// Repeat all kernels multiple times to get an average timing result
enum NUM_RUNS = 4;

// Squared matrices are tested within a certain range (e.g. 1024x1024, 2048x2048, 4096x4096)
enum MINSIZE = 1024;
enum MAXSIZE = 4 * 1024;

// Set the alpha and beta values for the cuBLAS and clBlas libraries. Note that the myGEMM kernels
// for simplicity only support alpha values of 1 and beta values of 0.
enum ALPHA = 1.0f;
enum BETA = 0.0f;

// Define the current GPU's parameters
enum GPU_NAME = "Tesla K40m";
enum GPU_CLOCK = 0.745; // Core clock in GHz
enum GPU_CORES = 2880; // Total number of CUDA cores
enum GPU_MOD = 2; // Fused multiply-add

// OpenCL settings
enum MAX_NUM_DEVICES = 16;
enum MAX_DEVICE_NAME = 1024;
enum CURRENT_DEVICE = 0;

// =================================================================================================

// Timer structure
struct profile_t
{
    double t; // Time
    long kf; // KFlops
}

// Number of timers
enum NUM_TIMERS = 10;

// Global variable holding the timing results
extern __gshared profile_t[NUM_TIMERS] timers;

// =================================================================================================

// Forward declarations of BLAS functions
void libcublas (float* A, float* B, float* C, int K, int M, int N, int timerID);
void libclblas (float* A, float* B, float* C, int K, int M, int N, int timerID);
void mycublas (float* A, float* B, float* C, int K, int M, int N, int timerID);
void myclblas (float* A, float* B, float* C, int K, int M, int N, int timerID);

// Forward declarations of the timer functions
double timer ();
double wtime (profile_t timer);
double gflops (profile_t timer);

// Other forward declarations
char* readKernelFile (const(char)* filename, c_long* _size);

// =================================================================================================
