cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(RGBFL VERSION 1.0)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# # Find the OpenCV package
# find_package(OpenCV REQUIRED)

# # Find the Eigen package
# find_package(Eigen3 3.3 REQUIRED NO_MODULE) # Requires Eigen 3.3 or later; adjust version as needed

# # Include directories
# include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories(${CMAKE_SOURCE_DIR}/third_party/eigen-3.4.0)
# include_directories(include)

# Specify the path to CGAL
set(CGAL_DIR "C:/cgal-5.6.1")

find_package(CGAL REQUIRED)

# Add the executable
add_executable(rgbfl src/main.cpp)
target_link_libraries(rgbfl CGAL::CGAL)