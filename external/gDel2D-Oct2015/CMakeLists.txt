project(gflip2d)
cmake_minimum_required(VERSION 2.8)
find_package(CUDA REQUIRED)

# Visualization (OpenGL / GLEW / GLUT) is disabled for this build.
# Original lines kept for reference:
# find_package(OpenGL REQUIRED)
# find_package(GLEW REQUIRED)
# find_package(GLUT REQUIRED)

set(CMAKE_BUILD_TYPE Release)
set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -gencode arch=compute_89,code=sm_89)
#add_definitions(-DREAL_TYPE_FP32)
include_directories(
    src
    )

# Core gDel2D library (no visualization / no main)
cuda_add_library(gdel2d_lib
      src/DelaunayChecker.cpp
      src/HashTable.cpp
      src/RandGen.cpp
      src/InputCreator.cpp
      src/gDel2D/CPU/PredWrapper.cpp
      src/gDel2D/CPU/predicates.cpp
      src/gDel2D/GPU/GpuDelaunay.cu
      src/gDel2D/GPU/ThrustWrapper.cu
      src/gDel2D/GPU/KerPredicates.cu
      src/gDel2D/GPU/KerDivision.cu
      src/gDel2D/GPU/SmallCounters.cu
)

# Optional standalone executable using the library (kept for reference)
cuda_add_executable(
    ${PROJECT_NAME}
      src/Main.cpp
)

target_link_libraries(${PROJECT_NAME}
    gdel2d_lib
)

# Keep CMake output non-verbose to avoid noisy build logs.
set(CMAKE_VERBOSE_MAKEFILE "OFF")

# When visualization is enabled, link against OpenGL / GLEW / GLUT:
# target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLUT_LIBRARIES})
