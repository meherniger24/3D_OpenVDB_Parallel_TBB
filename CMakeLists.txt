cmake_minimum_required(VERSION 3.16)

# Project name and languages used
project(3D_vdb LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)

# Specify the paths to the necessary include directories
include_directories(
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    /home/meher/openvdb/build/include/
)

# Specify the path to the library directory
link_directories(/home/meher/openvdb/build/lib)
link_directories(${TBB_LIBRARY_DIRS})
link_libraries(tbb)  


find_package(TBB REQUIRED)
find_package(OpenMP REQUIRED)
include_directories(${TBB_INCLUDE_DIRS})


# Define the executable target
add_executable(3D_vdb main.cpp)

# Link the executable with the necessary libraries
target_link_libraries(3D_vdb PUBLIC openvdb ${TBB_LIBRARIES} OpenMP::OpenMP_CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# For MS Visual Studio specific options
if(MSVC)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
    set(LIBRARY_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
    set(LIBRARY_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-D_SCL_SECURE_NO_WARNINGS)
endif()
