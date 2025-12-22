@compute(CompileFor.deviceOnly)
module kernels;
pragma(LDC_no_moduleinfo);

import ldc.dcompute;
import dcompute.std.index;

// First naive implementation
@kernel void myGEMM1(const int M, const int N, const int K,
    const GlobalPointer!(float) A,
    const GlobalPointer!(float) B,
    GlobalPointer!(float) C) {

    // Thread identifiers
    typeof(GlobalIndex.x) globalRow = GlobalIndex.x; // Row ID of C (0..M)
    typeof(GlobalIndex.y) globalCol = GlobalIndex.y; // Col ID of C (0..N)

    // Compute a single element (loop over K)
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += A[k * M + globalRow] * B[globalCol * K + k];
    }

    // Store the result
    C[globalCol * M + globalRow] = acc;
}
