cmake_minimum_required(VERSION 3.18)
project(CUDA2_MD LANGUAGES C CXX CUDA)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_compile_definitions(RAFT_HIDE_DEPRECATION_WARNINGS)

# ---- Language Standards ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Some site toolchains leave CMake's feature tables empty, which breaks imported
# targets that request cxx_std_Xx. Seed a minimal set so target_compile_features
# works even when compiler detection is incomplete.
if(NOT CMAKE_CXX_KNOWN_FEATURES)
    set(CMAKE_CXX_KNOWN_FEATURES cxx_std_98 cxx_std_11 cxx_std_14 cxx_std_17 cxx_std_20 cxx_std_23)
    set(CMAKE_CXX_COMPILE_FEATURES ${CMAKE_CXX_KNOWN_FEATURES})
endif()
if(NOT CMAKE_C_KNOWN_FEATURES)
    set(CMAKE_C_KNOWN_FEATURES c_std_90 c_std_99 c_std_11 c_std_17 c_std_23)
    set(CMAKE_C_COMPILE_FEATURES ${CMAKE_C_KNOWN_FEATURES})
endif()
# Disable linker dependency files (ld.bfd on this system does not support
# --dependency-file), otherwise linking fails with "unrecognized option".
set(CMAKE_LINK_DEPENDS_USE_LINKER FALSE)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared CACHE STRING "Use shared CUDA runtime" FORCE)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89" CACHE STRING
        "CUDA architectures to compile for" FORCE)
endif()

# Ensure the local vcpkg manifest install tree stays on CMAKE_PREFIX_PATH even
# when users override it on the command line (needed for fmt, delaunator-cpp, etc.).
if(NOT VCPKG_TARGET_TRIPLET)
    set(VCPKG_TARGET_TRIPLET "x64-linux")
endif()
set(_LOCAL_VCPKG_PREFIX "${CMAKE_SOURCE_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}")
if(EXISTS "${_LOCAL_VCPKG_PREFIX}")
    list(PREPEND CMAKE_PREFIX_PATH "${_LOCAL_VCPKG_PREFIX}")
endif()

# ---- Core Packages ----
find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(raft REQUIRED)

# ---- Header-only Libs ----
find_path(
    DELAUNATOR_CPP_INCLUDE_DIRS
    "delaunator-header-only.hpp"
    HINTS
        "${_LOCAL_VCPKG_PREFIX}/include"
        "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include"
        "${CMAKE_SOURCE_DIR}/build/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include"
    PATH_SUFFIXES
        include
)
if(NOT DELAUNATOR_CPP_INCLUDE_DIRS)
    message(FATAL_ERROR
        "Could not find delaunator-header-only.hpp. "
        "Checked local vcpkg prefixes including '${_LOCAL_VCPKG_PREFIX}/include' "
        "and '${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include'.")
endif()

# ---- MPI / CUDA Helper ----
if(DEFINED ENV{OMPI_CUDA_PREFIX})
    set(OMPI_CUDA_INCLUDE "$ENV{OMPI_CUDA_PREFIX}/include")
    message(STATUS "OMPI_CUDA_INCLUDE = ${OMPI_CUDA_INCLUDE}")
    include_directories(SYSTEM "${OMPI_CUDA_INCLUDE}")
else()
    message(WARNING "OMPI_CUDA_PREFIX is not set; CUDA-aware MPI headers will not be added explicitly.")
endif()

# --------------------------------------------------------
# FIX 2: Robust MPI Include Detection (Fixes 'mpi.h not found' in CUDA)
# --------------------------------------------------------
set(MPI_ALL_INCLUDES "")

# Method A: Try Standard CMake Target properties
if(TARGET MPI::MPI_CXX)
    get_target_property(_mpi_inc MPI::MPI_CXX INTERFACE_INCLUDE_DIRECTORIES)
    if(_mpi_inc)
        list(APPEND MPI_ALL_INCLUDES ${_mpi_inc})
    endif()
endif()

# Method B: Ask the compiler wrapper directly (Crucial for OpenMPI modules)
if(NOT MPI_ALL_INCLUDES)
    message(STATUS "CMake failed to detect MPI includes automatically. Querying mpicxx...")
    execute_process(
        COMMAND bash -c "mpicxx --showme:incdirs"
        OUTPUT_VARIABLE MPI_RAW_INC_OUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(MPI_RAW_INC_OUT)
        string(REPLACE " " ";" MPI_RAW_INC_LIST "${MPI_RAW_INC_OUT}")
        list(APPEND MPI_ALL_INCLUDES ${MPI_RAW_INC_LIST})
        message(STATUS "Detected MPI Includes via mpicxx: ${MPI_ALL_INCLUDES}")
    endif()
endif()

# MPI::MPI_CXX normally supplies these includes. The explicit list also makes
# them visible while NVCC compiles CUDA translation units.
include_directories(SYSTEM ${MPI_ALL_INCLUDES})

# ---- Python Checks ----
execute_process(
    COMMAND "${Python3_EXECUTABLE}" "-c" "import numpy, sys; sys.stdout.write(numpy.get_include())"
    OUTPUT_VARIABLE PYTHON_NUMPY_INCLUDE_DIR
    RESULT_VARIABLE NUMPY_STATUS
)
if(NOT NUMPY_STATUS EQUAL 0 OR PYTHON_NUMPY_INCLUDE_DIR STREQUAL "")
    message(FATAL_ERROR "NumPy not found for Python: ${Python3_EXECUTABLE}")
endif()


# ---- Project Directories ----
set(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(INCLUDE_DIR "${PROJECT_ROOT}/include")
set(CUDA_DIR    "${PROJECT_ROOT}/src")
set(SRC_DIR     "${PROJECT_ROOT}/src")
set(TESTS_DIR   "${PROJECT_ROOT}/tests")
set(RUNS_DIR   "${PROJECT_ROOT}/run")

# ---- External libraries ----
# Build the GPU 2D Delaunay triangulation package (gDel2D)
add_subdirectory(external/gDel2D-Oct2015)

# ---- Sources ----
file(GLOB_RECURSE PROJECT_HEADERS "${INCLUDE_DIR}/*.hpp")
file(GLOB_RECURSE PROJECT_SOURCES "${SRC_DIR}/*.cpp")
file(GLOB_RECURSE PROJECT_CUDA    "${CUDA_DIR}/*.cu")

# ---- Main Library ----
add_library(md_particle_lib
    ${PROJECT_SOURCES}
    ${PROJECT_CUDA}
)

set_source_files_properties(${PROJECT_CUDA} PROPERTIES LANGUAGE CUDA)

target_include_directories(md_particle_lib
    PUBLIC
        "${PROJECT_ROOT}"
        "${INCLUDE_DIR}"
        ${Python3_INCLUDE_DIRS}
        ${PYTHON_NUMPY_INCLUDE_DIR}
        ${OMPI_CUDA_INCLUDE}
        ${DELAUNATOR_CPP_INCLUDE_DIRS}
        ${MPI_ALL_INCLUDES}  # <--- This is the fix for CUDA
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(md_particle_lib
    PUBLIC
        fmt::fmt-header-only
        Python3::Python
        nlohmann_json::nlohmann_json
        gdel2d_lib
        MPI::MPI_CXX
        ZLIB::ZLIB
        raft::raft
        CUDA::cusparse
        CUDA::cublas
        CUDA::cusolver
        CUDA::cudart
)
set_target_properties(md_particle_lib PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

# ---- Tests ----
file(GLOB_RECURSE TEST_SOURCES
    "${TESTS_DIR}/*/*.cpp"
    "${TESTS_DIR}/*/*.cu"
)

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
            ${Python3_INCLUDE_DIRS}
            ${PYTHON_NUMPY_INCLUDE_DIR}
            ${MPI_ALL_INCLUDES}
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
endforeach()

# Avoid assembler incompatibility with -gdwarf-4 on this target
if (TARGET run_test_run_test)
    target_compile_options(run_test_run_test PRIVATE -g0)
endif()

# formal runs
file(GLOB_RECURSE RUN_SOURCES "${RUNS_DIR}/*.cpp")

foreach(run_src IN LISTS RUN_SOURCES)
    get_filename_component(run_name "${run_src}" NAME_WE)
    get_filename_component(run_dir "${run_src}" DIRECTORY)
    get_filename_component(run_dir_name "${run_dir}" NAME)

    set(exe_name "${run_dir_name}_${run_name}")

    add_executable("${exe_name}" "${run_src}")

    target_link_libraries("${exe_name}"
        PRIVATE
            md_particle_lib
            MPI::MPI_CXX
    )
    target_include_directories("${exe_name}"
        PRIVATE
            "${PROJECT_ROOT}"
            "${INCLUDE_DIR}"
            ${Python3_INCLUDE_DIRS}
            ${PYTHON_NUMPY_INCLUDE_DIR}
            ${MPI_ALL_INCLUDES}
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
endforeach()
