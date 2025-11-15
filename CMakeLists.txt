cmake_minimum_required(VERSION 3.18)

project(CUDA2_MD LANGUAGES C CXX CUDA)

# ---- language setup ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---- core packages ----
find_package(MPI REQUIRED COMPONENTS C CXX)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(ZLIB REQUIRED)

# MPI check
if(DEFINED ENV{OMPI_CUDA_PREFIX})
    set(OMPI_CUDA_INCLUDE "$ENV{OMPI_CUDA_PREFIX}/include")
    message(STATUS "OMPI_CUDA_INCLUDE = ${OMPI_CUDA_INCLUDE}")
    include_directories(SYSTEM "${OMPI_CUDA_INCLUDE}")
else()
    message(WARNING "OMPI_CUDA_PREFIX is not set; CUDA-aware MPI headers will not be added explicitly.")
endif()

# ---- NumPy / matplotlib checks ----
execute_process(
    COMMAND "${Python3_EXECUTABLE}" "-c" "import numpy, sys; sys.stdout.write(numpy.get_include())"
    OUTPUT_VARIABLE PYTHON_NUMPY_INCLUDE_DIR
    RESULT_VARIABLE NUMPY_STATUS
)
if(NOT NUMPY_STATUS EQUAL 0 OR PYTHON_NUMPY_INCLUDE_DIR STREQUAL "")
    message(FATAL_ERROR "NumPy not found for Python: ${Python3_EXECUTABLE}. Please ensure python3-numpy is installed.")
endif()

execute_process(
    COMMAND "${Python3_EXECUTABLE}" "-c" "import matplotlib"
    RESULT_VARIABLE MATPLOTLIB_IMPORT_RESULT
)
if(NOT MATPLOTLIB_IMPORT_RESULT EQUAL 0)
    message(FATAL_ERROR "Python3 matplotlib module not found for interpreter: ${Python3_EXECUTABLE}")
endif()

# ---- project directories ----
set(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(INCLUDE_DIR "${PROJECT_ROOT}/include")
set(CUDA_DIR    "${PROJECT_ROOT}/src")
set(SRC_DIR     "${PROJECT_ROOT}/src")
set(TESTS_DIR   "${PROJECT_ROOT}/tests")
set(MATPLOTLIB_CPP_DIR "${PROJECT_ROOT}/external/matplotlib-cpp")

# ---- sources ----
file(GLOB_RECURSE PROJECT_HEADERS "${INCLUDE_DIR}/*.hpp")
file(GLOB_RECURSE PROJECT_SOURCES "${SRC_DIR}/*.cpp")
file(GLOB_RECURSE PROJECT_CUDA    "${CUDA_DIR}/*.cu")

add_library(md_particle_lib
    ${PROJECT_SOURCES}
    ${PROJECT_CUDA}
)

set_source_files_properties(
    ${PROJECT_CUDA}
    PROPERTIES
        LANGUAGE CUDA
)

# ---- robust MPI include detection ----
set(MPI_ALL_INCLUDES "")

# 1) From imported target MPI::MPI_CXX (most reliable)
if(TARGET MPI::MPI_CXX)
    get_target_property(_mpi_cxx_inc MPI::MPI_CXX INTERFACE_INCLUDE_DIRECTORIES)
    if(_mpi_cxx_inc)
        list(APPEND MPI_ALL_INCLUDES ${_mpi_cxx_inc})
    endif()
endif()

# 2) From classic variables (if CMake set them)
if(MPI_CXX_INCLUDE_DIRS)
    list(APPEND MPI_ALL_INCLUDES ${MPI_CXX_INCLUDE_DIRS})
endif()
if(MPI_C_INCLUDE_PATH)
    list(APPEND MPI_ALL_INCLUDES ${MPI_C_INCLUDE_PATH})
endif()
if(MPI_INCLUDE_PATH)
    list(APPEND MPI_ALL_INCLUDES ${MPI_INCLUDE_PATH})
endif()

list(REMOVE_DUPLICATES MPI_ALL_INCLUDES)

message(STATUS "MPI_ALL_INCLUDES = ${MPI_ALL_INCLUDES}")

# If still empty, you *can* hard-code as a last resort on this machine:
# list(APPEND MPI_ALL_INCLUDES "/usr/lib/x86_64-linux-gnu/openmpi/include"
#                              "/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi")

# Apply globally so nvcc also sees them
if(MPI_ALL_INCLUDES)
    include_directories(SYSTEM ${MPI_ALL_INCLUDES})
endif()

# ---- includes / links for md_particle_lib ----
target_include_directories(md_particle_lib
    PUBLIC
        "${PROJECT_ROOT}"
        "${INCLUDE_DIR}"
        "${MATPLOTLIB_CPP_DIR}"
        ${Python3_INCLUDE_DIRS}
        ${PYTHON_NUMPY_INCLUDE_DIR}
        ${OMPI_CUDA_INCLUDE}
)

target_link_libraries(md_particle_lib
    PUBLIC
        fmt::fmt
        Python3::Python
        nlohmann_json::nlohmann_json
        MPI::MPI_CXX
        ZLIB::ZLIB
)

# ---- tests ----
file(GLOB_RECURSE TEST_SOURCES "${TESTS_DIR}/*/*.cpp")

foreach(test_src IN LISTS TEST_SOURCES)
    get_filename_component(test_name "${test_src}" NAME_WE)
    get_filename_component(test_dir "${test_src}" DIRECTORY)
    get_filename_component(test_dir_name "${test_dir}" NAME)

    set(exe_name "${test_dir_name}_${test_name}")

    add_executable("${exe_name}" "${test_src}")

    target_link_libraries("${exe_name}"
        PRIVATE
            md_particle_lib
            MPI::MPI_CXX
    )

    target_include_directories("${exe_name}"
        PRIVATE
            "${PROJECT_ROOT}"
            "${INCLUDE_DIR}"
            "${MATPLOTLIB_CPP_DIR}"
            ${Python3_INCLUDE_DIRS}
            ${PYTHON_NUMPY_INCLUDE_DIR}
            ${MPI_ALL_INCLUDES}
    )
endforeach()
