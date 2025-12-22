module clgemm;
import std.experimental.allocator : theAllocator;
import std.file : read;
import dcompute.driver.ocl;
import settings;
import common;
import std.math.traits : isNaN;
import std.stdio;

import kernels;

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

// // Common include
// #include "common.h"

// // Include OpenCL 
// #include <CL/cl.h>

// // Include kernel constants
// #include "settings.h"

// // Forward declaration of the OpenCL error checking function
// void checkError(cl_int error, int line);

// =================================================================================================

// // Set the locations of the OpenCL kernel files
// #define CL_INCLUDE_FILE "src/settings.h"
// #define CL_KERNEL_FILE "src/kernels.cl"

// // Determine the location where to output the PTX code
// #define CL_PTX_FILE "bin/myGEMM.cl.ptx"

// // Define OpenCL compiler options, such as "-cl-nv-maxrregcount=127"
// #define COMPILER_OPTIONS ""

// =================================================================================================
enum CL_PLATFORM_INDEX = 2;
enum CURRENT_DEVICE = 0;

enum COMPILER_OPTIONS = "";

// Matrix-multiplication using a custom OpenCL SGEMM kernel. This function also copies the input
// matrices to the GPU, runs SGEMM, and copies the output matrix back to the CPU.
// void myclblas(float* A, float* B, float* C,
//               int K, int M, int N,
//               int timerID) {
void myclblas(float[] A, float[] B, float[] C,
              int K, int M, int N,
              int timerID) {

    // // In case of myGEMM10, compute matrix sizes K, M, N as rounded-up to form complete tiles
    // #if KERNEL == 10
    //     int K_XL = CEIL_DIV(K, TSK) * TSK;
    //     int M_XL = CEIL_DIV(M, TSM) * TSM;
    //     int N_XL = CEIL_DIV(N, TSN) * TSN;
    // #else
        int K_XL = K;
        int M_XL = M;
        int N_XL = N;
    // #endif

    // // Define OpenCL variables
    // cl_int err;
    // cl_platform_id platform = 0;
    // cl_device_id device = 0;
    // cl_device_id devices[MAX_NUM_DEVICES];
    // cl_uint numDevices = 0;
    cl_context_properties[3] props = [CL_CONTEXT_PLATFORM, 0, 0];
    // cl_context context = 0;
    // // cl_command_queue queue = 0;
    // cl_event event = NULL;
    // // cl_program program = null;
    // char deviceName[MAX_DEVICE_NAME];

    // // Configure the OpenCL environment
    // err = clGetPlatformIDs(1, &platform, NULL);
    // err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    // err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
    // device = devices[CURRENT_DEVICE];
    // props[1] = (cl_context_properties)platform;
    // context = clCreateContext(props, 1, &device, NULL, NULL, &err);
    // queue = clCreateCommandQueue(context, device, 0, &err);
    // err = clGetDeviceInfo(device, CL_DEVICE_NAME, MAX_DEVICE_NAME, deviceName, NULL);
    // checkErrors();
    // //writef("## %d devices, running on %d: '%s'\n", numDevices, CURRENT_DEVICE, deviceName);

    auto platforms = Platform.getPlatforms(theAllocator);
    auto platform = platforms[CL_PLATFORM_INDEX];
    DerelictCL.reload(CLVersion.CL21);
    props[1] = cast(cl_context_properties)platform.raw;

    auto devices  = platform.getDevices(theAllocator);
    auto device = devices[CURRENT_DEVICE];
    // writefln("## %d devices, running on %d: '%s'", devices.length, CURRENT_DEVICE, device.name);

    auto plist    = propertyList!(Context.Properties)(Context.Properties.platform, platform.raw);
    writeln(plist);
    // auto ctx      = Context(devices[0 ..1], null /*FIXME: plist[]*/);
    auto ctx      = Context(devices[0 ..1], plist);
    // auto ctx      = Context(devices[0 ..1], props);
    scope(exit) ctx.release();

    auto queue    = ctx.createQueue(device, Queue.Properties.outOfOrderExecution);
    scope(exit) queue.release();

    // // Read the kernel file from disk
    // long sizeHeader, sizeSource;
    // char* header = readKernelFile(CL_INCLUDE_FILE, &sizeHeader);
    // char* source = readKernelFile(CL_KERNEL_FILE, &sizeSource);
    // long size = 2 + sizeHeader + sizeSource;
    // char* code = (char*)malloc(size*sizeof(char));
    // for (int c=0; c<size; c++) { code[c] = '\0'; }
    // strcat(code, header);
    // strcat(code, source);
    // const char* constCode = code;
    // free(header);
    // free(source);

    // // Compile the kernel file
    // program = clCreateProgramWithSource(context, 1, &constCode, NULL, &err);
    // checkErrors();
    // err = clBuildProgram(program, 0, NULL, COMPILER_OPTIONS, NULL, NULL);

    Program.globalProgram = ctx.createProgram(cast(ubyte[]) read("./kernels_ocl200_64.spv"));
    try
    {
        Program.globalProgram.build(devices, COMPILER_OPTIONS);
    }
    catch(DComputeDriverException e)
    {
        auto b = Build(Program.globalProgram, device);
        writeln(b.log);
    }
    scope(exit) Program.globalProgram.release();

    cl_program program = Program.globalProgram.raw;

    // Check for compilation errors
    version (Debug) {
        size_t logSize;
        clGetProgramBuildInfo(cast(cl_program)program, device.raw, CL_PROGRAM_BUILD_LOG, 0, null, &logSize);
        checkErrors();
        char[] messages = new char[logSize + 1];
        clGetProgramBuildInfo(cast(cl_program)program, device.raw, CL_PROGRAM_BUILD_LOG, logSize, messages.ptr, null);
        checkErrors();
        messages[$-1] = '\0';
        if (logSize > 10) { writefln("## Compiler message: %s", messages); }
    }

    // // Retrieve the PTX code from the OpenCL compiler and output it to disk
    // size_t binSize;
    // err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binSize, NULL);
    // checkErrors();
    // unsigned char *bin = (unsigned char *)malloc(binSize);
    // err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char *), &bin, NULL);
    // checkErrors();
    // FILE* file = fopen(CL_PTX_FILE, "wb");
    // fwrite(bin, sizeof(char), binSize, file);
    // fclose(file);
    // free(bin);

    // Prepare OpenCL memory objects
    // cl_mem bufA    = clCreateBuffer(context, CL_MEM_READ_ONLY,  M*K*sizeof(*A), NULL, &err);
    // cl_mem bufB    = clCreateBuffer(context, CL_MEM_READ_ONLY,  K*N*sizeof(*B), NULL, &err);
    // cl_mem bufB_TR = clCreateBuffer(context, CL_MEM_READ_ONLY,  N*K*sizeof(*B), NULL, &err);
    // cl_mem bufC    = clCreateBuffer(context, CL_MEM_READ_WRITE, M*N*sizeof(*C), NULL, &err);
    Buffer!(float) bufA    = ctx.createBuffer(A, Memory.Flags.useHostPointer | Memory.Flags.readOnly);
    scope(exit) bufA.release();
    Buffer!(float) bufB    = ctx.createBuffer(B, Memory.Flags.useHostPointer | Memory.Flags.readOnly);
    scope(exit) bufB.release();
    Buffer!(float) bufB_TR = ctx.createBuffer(B, Memory.Flags.useHostPointer | Memory.Flags.readOnly);
    scope(exit) bufB_TR.release();
    Buffer!(float) bufC    = ctx.createBuffer(C, Memory.Flags.useHostPointer | Memory.Flags.readWrite);
    scope(exit) bufC.release();
    checkErrors();

    // Copy matrices to the GPU (also C to erase the results of the previous run)
    // err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, M*K*sizeof(*A), A, 0, NULL, NULL);
    // err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, K*N*sizeof(*B), B, 0, NULL, NULL);
    // err = clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0, M*N*sizeof(*C), C, 0, NULL, NULL);
    queue.write!(float)(bufA, A);
    queue.write!(float)(bufB, B);
    queue.write!(float)(bufC, C);
    checkErrors();

    // // Create extra objects for rounded-up sizes (only needed in case of myGEMM10)
    // cl_mem bufA_XL    = clCreateBuffer(context, CL_MEM_READ_ONLY,  M_XL*K_XL*sizeof(*A), NULL, &err);
    // cl_mem bufB_TR_XL = clCreateBuffer(context, CL_MEM_READ_ONLY,  N_XL*K_XL*sizeof(*B), NULL, &err);
    // cl_mem bufC_XL    = clCreateBuffer(context, CL_MEM_READ_WRITE, M_XL*N_XL*sizeof(*C), NULL, &err);
    // checkErrors();

    // // Configure the myGEMM kernel
    // char kernelname[100];
    // swritef(kernelname, "myGEMM%d", KERNEL);
    // cl_kernel kernel1 = clCreateKernel(program, kernelname, &err);
    // checkErrors();

    // // Set the arguments of the myGEMM kernel
    // #if KERNEL == 10
    //     err = clSetKernelArg(kernel1, 0, sizeof(int), (void*)&M_XL);
    //     err = clSetKernelArg(kernel1, 1, sizeof(int), (void*)&N_XL);
    //     err = clSetKernelArg(kernel1, 2, sizeof(int), (void*)&K_XL);
    //     err = clSetKernelArg(kernel1, 3, sizeof(cl_mem), (void*)&bufA_XL);
    //     err = clSetKernelArg(kernel1, 4, sizeof(cl_mem), (void*)&bufB_TR_XL);
    //     err = clSetKernelArg(kernel1, 5, sizeof(cl_mem), (void*)&bufC_XL);
    // #else
    //     err = clSetKernelArg(kernel1, 0, sizeof(int), (void*)&M);
    //     err = clSetKernelArg(kernel1, 1, sizeof(int), (void*)&N);
    //     err = clSetKernelArg(kernel1, 2, sizeof(int), (void*)&K);
    //     err = clSetKernelArg(kernel1, 3, sizeof(cl_mem), (void*)&bufA);
    //     #if KERNEL == 5 || KERNEL == 6 || KERNEL == 7 || KERNEL == 8 || KERNEL == 9
    //         err = clSetKernelArg(kernel1, 4, sizeof(cl_mem), (void*)&bufB_TR);
    //     #else
    //         err = clSetKernelArg(kernel1, 4, sizeof(cl_mem), (void*)&bufB);
    //     #endif
    //     err = clSetKernelArg(kernel1, 5, sizeof(cl_mem), (void*)&bufC);
    // #endif
    // checkErrors();

    // // Configure the supporting transpose kernel and set its arguments (only for certain myGEMMs)
    // #if KERNEL == 5 || KERNEL == 6 || KERNEL == 7 || KERNEL == 8 || KERNEL == 9 || KERNEL == 10
    //     cl_kernel kernel2 = clCreateKernel(program, "transpose", &err);
    //     checkErrors();
    //     err = clSetKernelArg(kernel2, 0, sizeof(int), (void*)&K);
    //     err = clSetKernelArg(kernel2, 1, sizeof(int), (void*)&N);
    //     err = clSetKernelArg(kernel2, 2, sizeof(cl_mem), (void*)&bufB);
    //     err = clSetKernelArg(kernel2, 3, sizeof(cl_mem), (void*)&bufB_TR);
    //     checkErrors();
    //     const size_t tLocal[2] = { TRANSPOSEX, TRANSPOSEY };
    //     const size_t tGlobal[2] = { (size_t)K, (size_t)N };
    // #endif

    // // Configure the supporting padding kernels and set their arguments (only for myGEMM10)
    // #if KERNEL == 10
    //     cl_kernel kernel3a = clCreateKernel(program, "paddingAddZeroes", &err);
    //     checkErrors();
    //     err = clSetKernelArg(kernel3a, 0, sizeof(int), (void*)&M);
    //     err = clSetKernelArg(kernel3a, 1, sizeof(int), (void*)&K);
    //     err = clSetKernelArg(kernel3a, 2, sizeof(cl_mem), (void*)&bufA);
    //     err = clSetKernelArg(kernel3a, 3, sizeof(int), (void*)&M_XL);
    //     err = clSetKernelArg(kernel3a, 4, sizeof(int), (void*)&K_XL);
    //     err = clSetKernelArg(kernel3a, 5, sizeof(cl_mem), (void*)&bufA_XL);
    //     checkErrors();
    //     cl_kernel kernel3b = clCreateKernel(program, "paddingAddZeroes", &err);
    //     checkErrors();
    //     err = clSetKernelArg(kernel3b, 0, sizeof(int), (void*)&N);
    //     err = clSetKernelArg(kernel3b, 1, sizeof(int), (void*)&K);
    //     err = clSetKernelArg(kernel3b, 2, sizeof(cl_mem), (void*)&bufB_TR);
    //     err = clSetKernelArg(kernel3b, 3, sizeof(int), (void*)&N_XL);
    //     err = clSetKernelArg(kernel3b, 4, sizeof(int), (void*)&K_XL);
    //     err = clSetKernelArg(kernel3b, 5, sizeof(cl_mem), (void*)&bufB_TR_XL);
    //     checkErrors();
    //     cl_kernel kernel3c = clCreateKernel(program, "paddingRemoveZeroes", &err);
    //     checkErrors();
    //     err = clSetKernelArg(kernel3c, 0, sizeof(int), (void*)&M_XL);
    //     err = clSetKernelArg(kernel3c, 1, sizeof(int), (void*)&N_XL);
    //     err = clSetKernelArg(kernel3c, 2, sizeof(cl_mem), (void*)&bufC_XL);
    //     err = clSetKernelArg(kernel3c, 3, sizeof(int), (void*)&M);
    //     err = clSetKernelArg(kernel3c, 4, sizeof(int), (void*)&N);
    //     err = clSetKernelArg(kernel3c, 5, sizeof(cl_mem), (void*)&bufC);
    //     checkErrors();
    //     const size_t pLocal[2] = { PADDINGX, PADDINGY };
    //     const size_t pAGlobal[2] = { (size_t)M_XL, (size_t)K_XL };
    //     const size_t pBGlobal[2] = { (size_t)N_XL, (size_t)K_XL };
    //     const size_t pCGlobal[2] = { (size_t)M, (size_t)N };
    // #endif

    // Configure the thread/work-group dimensions of the myGEMM kernel
    const size_t[] local = [ TS, TS ];
    const size_t[] global = [ M, N ];
    // #if KERNEL == 1 || KERNEL == 2
    //     const size_t local[2] = { TS, TS };
    //     const size_t global[2] = { (size_t)M, (size_t)N };
    // #elif KERNEL == 3 || KERNEL == 5
    //     const size_t local[2] = { TS, TS/WPT };
    //     const size_t global[2] = { (size_t)M, (size_t)(N/WPT) };
    // #elif KERNEL == 4
    //     const size_t local[2] = { TS/WIDTH, TS };
    //     const size_t global[2] = { (size_t)(M/WIDTH), (size_t)N };
    // #elif KERNEL == 6 || KERNEL == 7 || KERNEL == 8 || KERNEL == 9
    //     const size_t local[2] = { TSM/WPTM, TSN/WPTN };
    //     const size_t global[2] = { (size_t)(M/WPTM), (size_t)(N/WPTN) };
    // #elif KERNEL == 10
    //     const size_t local[2] = { TSM/WPTM, TSN/WPTN };
    //     const size_t global[2] = { (size_t)(M_XL/WPTM), (size_t)(N_XL/WPTN) };
    // #elif KERNEL == 11
    //     const size_t local[2] = { THREADSX, THREADSY };
    //     const size_t global[2] = { (size_t)(M/RX), (size_t)(N/RY) };
    // #endif

    // // Start the timed loop
    // double startTime = timer();
    for (int r=0; r<NUM_RUNS; r++) {

        // // Run the transpose kernel first
        // #if KERNEL == 5 || KERNEL == 6 || KERNEL == 7 || KERNEL == 8 || KERNEL == 9 || KERNEL == 10
        //     err = clEnqueueNDRangeKernel(queue, kernel2, 2, NULL, tGlobal, tLocal, 0, NULL, &event);
        // #endif

        // // Make the inputs extra large with padded zeros
        // #if KERNEL == 10
        //     err = clEnqueueNDRangeKernel(queue, kernel3a, 2, NULL, pAGlobal, pLocal, 0, NULL, &event);
        //     err = clEnqueueNDRangeKernel(queue, kernel3b, 2, NULL, pBGlobal, pLocal, 0, NULL, &event);
        // #endif

        // Run the myGEMM kernel
        // err = clEnqueueNDRangeKernel(queue, kernel1, 2, NULL, global, local, 0, NULL, &event);
        Event e = queue.enqueue!(myGEMM1)(global, null, local)(M, N, K, bufA, bufB, bufC);

        // // Remove padded zeroes from the larger output
        // #if KERNEL == 10
        //     err = clEnqueueNDRangeKernel(queue, kernel3c, 2, NULL, pCGlobal, pLocal, 0, NULL, &event);
        // #endif

        // Wait for calculations to be finished
        checkErrors();

        e.wait();
        // err = clWaitForEvents(1, &event);
    }

    // // End the timed loop
    // timers[timerID].t += (timer() - startTime) / (double)NUM_RUNS;
    // timers[timerID].kf += ((long)K * (long)M * (long)N * 2) / 1000;

    // Copy the output matrix C back to the CPU memory
    // err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, M*N*sizeof(*C), C, 0, NULL, NULL);
    if (isNaN(C[0])) {
        queue.read!(float)(bufC, C);
    }
    checkErrors();

    // // Free the memory objects
    // clReleaseMemObject(bufA);
    // clReleaseMemObject(bufB);
    // clReleaseMemObject(bufB_TR);
    // clReleaseMemObject(bufC);
    // clReleaseMemObject(bufA_XL);
    // clReleaseMemObject(bufB_TR_XL);
    // clReleaseMemObject(bufC_XL);

    // Clean-up OpenCL 
    // clReleaseCommandQueue(queue);
    // clReleaseContext(context);
    // clReleaseProgram(program);
    // clReleaseKernel(kernel1);
    // #if KERNEL == 5 || KERNEL == 6 || KERNEL == 7 || KERNEL == 8 || KERNEL == 9 || KERNEL == 10
    //     clReleaseKernel(kernel2);
    // #endif
    // #if KERNEL == 10
    //     clReleaseKernel(kernel3a);
    //     clReleaseKernel(kernel3b);
    //     clReleaseKernel(kernel3c);
    // #endif
}

