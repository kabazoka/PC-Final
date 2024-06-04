cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(RGBFL LANGUAGES CUDA CXX)

set(CMAKE_BUILD_TYPE Release)

set(GMP_INCLUDE_DIR "/usr/include")
set(GMP_LIBRARIES "/usr/lib/x86_64-linux-gnu/libgmp.so")

# Specify the path to CGAL
set(CGAL_DIR "third_party/CGAL-5.6.1")

# Specify the path to CUDA (if necessary)
# set(CUDA_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7")
# set(CUDA_HOST_COMPILER "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.37.32822/bin/Hostx64/x64/cl.exe")
# set(CUDA_ARCH "86") 

# Find required packages
find_package(CGAL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)
# # Link cuBLAS
# find_package(CUBLAS REQUIRED)

# Include directories for GMP
include_directories(${GMP_INCLUDE_DIR})
# Include directories for CGAL
include_directories(${CGAL_INCLUDE_DIRS})
# Include directories for other dependencies
include_directories(${CMAKE_SOURCE_DIR}/third_party/eigen-3.4.0)
include_directories(/usr/include)  # Adjust this path to the correct GMP include directory if necessary

# Enable CUDA and C++ languages
enable_language(CUDA)
enable_language(CXX)

# Set CUDA flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --std=c++14)
set(CMAKE_CUDA_COMPILER_FORCED TRUE)

# If OpenMP is found, set the compiler flags
if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

cuda_add_library(cuda_interpolation src/cuda_interpolation.cu)

# Add the executable
cuda_add_executable(rgbfl src/main.cpp)
target_link_libraries(rgbfl ${GMP_LIBRARIES} ${CGAL_LIBRARIES} ${OpenMP_LIBS} ${CUDA_LIBRARIES} ${CUBLAS_LIBRARIES} cuda_interpolation)