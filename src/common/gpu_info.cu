#include <cuda_runtime.h>
#include <stdio.h>

#include "gpu_info.h"

void gpu_info::print() {
    int device;
    cudaGetDevice(&device);
    printf("Using device: %d\n", device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("1. GPU card's name: %s\n", prop.name);
    printf("2. GPU computation capabilities: %d.%d\n", prop.major, prop.minor);
    printf(
        "3. Maximum number of block dimensions: (%d, %d, %d)\n",
        prop.maxThreadsDim[0],
        prop.maxThreadsDim[1],
        prop.maxThreadsDim[2]
    );
    printf(
        "4. Maximum number of grid dimensions: (%d, %d, %d)\n",
        prop.maxGridSize[0],
        prop.maxGridSize[1],
        prop.maxGridSize[2]
    );
    printf(
        "5. Maximum size of GPU memory: %.2f GB\n",
        (double)prop.totalGlobalMem / (1 << 30)
    );
    printf(
        "6. Amount of constant memory: %.2f KB\n",
        (double)prop.totalConstMem / 1024.0
    );
    printf(
        "   Amount of shared memory per block: %.2f KB\n",
        (double)prop.sharedMemPerBlock / 1024.0
    );
    printf("7. Warp size: %d\n", prop.warpSize);
}