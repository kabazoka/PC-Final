cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(RGBFL LANGUAGES CUDA)

# Set the policy to use the updated CUDA support
cmake_policy(SET CMP0146 NEW)

# Specify the path to CGAL
set(CGAL_DIR "C:/cgal-5.6.1")

# Specify the path to CUDA
set(CUDA_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7")
set(CUDA_HOST_COMPILER "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.37.32822/bin/Hostx64/x64/cl.exe")
set(CUDA_ARCH "86") 

# include_directories(${OpenCV_INCLUDE_DIRS})
find_package(CGAL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)

enable_language(CUDA)

# Set CUDA flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --std=c++11)
set(CMAKE_CUDA_COMPILER_FORCED TRUE)

# if(OpenMP_FOUND)
#     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# endif()

# include_directories(${CGAL_INCLUDE_DIRS})
# include_directories(${CMAKE_SOURCE_DIR}/third_party/eigen-3.4.0)


# # Add the executable
# # add_executable(rgbfl src/single_threaded.cpp)
# add_executable(rgbfl src/openmp.cpp)
# target_link_libraries(rgbfl CGAL::CGAL ${Eigen3_LIBRARIES} ${OpenMP_LIBS})

# Add the executable
cuda_add_executable(rgbfl src/cuda.cu)
target_link_libraries(rgbfl ${CGAL_LIBRARIES} ${OpenMP_LIBS} ${CUDA_LIBRARIES})