project(ParallemProgrammingFinal LANGUAGES C CXX CUDA)
cmake_minimum_required(VERSION 3.18)

# Thêm thư mục include/
include_directories(${CMAKE_SOURCE_DIR}/include)

# Lấy tất cả file .cu và .cpp trong src/
file(GLOB SOURCES src/*.cu src/*.cpp)

# Tạo executable
set(CMAKE_CUDA_ARCHITECTURES 61 75)
add_executable(extract_features ${SOURCES})
