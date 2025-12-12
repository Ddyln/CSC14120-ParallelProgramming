cmake_minimum_required(VERSION 3.18)
project(ParallelProgrammingFinal LANGUAGES C CXX CUDA)

# Enable optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# CUDA specific settings
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find all source files from subdirectories
file(GLOB COMMON_SOURCES src/common/*.cu src/common/*.cpp)
file(GLOB CPU_SOURCES src/cpu/*.cpp)
file(GLOB GPU_SOURCES src/gpu/*.cu)
file(GLOB GPU2_SOURCES src/gpu2/*.cu)
set(MAIN_SOURCE src/main.cu)
set(SOURCES ${COMMON_SOURCES} ${CPU_SOURCES} ${GPU_SOURCES} ${GPU2_SOURCES} ${MAIN_SOURCE})

# Generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set CUDA architectures (Pascal=61, Turing=75, Ampere=86)
set(CMAKE_CUDA_ARCHITECTURES 75 86)

# Create executable
add_executable(extract_features ${SOURCES})

target_include_directories(
    extract_features 
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Enable separable compilation for CUDA
set_target_properties(extract_features PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
)