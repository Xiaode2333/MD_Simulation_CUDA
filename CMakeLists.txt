cmake_minimum_required(VERSION 3.18)

project(CUDA2_MD LANGUAGES C CXX CUDA)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Hide noisy RAFT deprecation #pragma messages that come from transitive includes.
# We rely on the installed RAFT API and just want clean build output.
add_compile_definitions(RAFT_HIDE_DEPRECATION_WARNINGS)

# --------------------------------------------------------
# Debug info flag compatibility
# --------------------------------------------------------
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-gdwarf-4" HAS_GDWARF4)
if (HAS_GDWARF4)
    add_compile_options("-gdwarf-4")
else()
    message(WARNING "Compiler/assembler does not support -gdwarf-4; disabling debug info with -g0")
    add_compile_options("-g0")
endif()
# CUDA host compilation can still emit unsupported DWARF flags; force no debug
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-g0>)

# ---- Language Standards ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
# Disable linker dependency files (ld.bfd on this system does not support
# --dependency-file), otherwise linking fails with "unrecognized option".
set(CMAKE_LINK_DEPENDS_USE_LINKER FALSE)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared CACHE STRING "Use shared CUDA runtime" FORCE)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
set(GCC_LIB_PATH "/apps/software/2024a/software/GCCcore/13.3.0/lib64")
set(CMAKE_CUDA_ARCHITECTURES 89)
link_directories(/apps/software/2024a/software/CUDA/12.6.0/lib64)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/apps/software/2024a/software/CUDA/12.6.0/lib64")


add_link_options("-L${GCC_LIB_PATH}" "-Wl,-rpath,${GCC_LIB_PATH}")
# Keep the CUDA 12.6 toolchain libs ahead of any conda CUDA shims at runtime.
set(CUDA_TOOLCHAIN_LIB "/apps/software/2024a/software/CUDA/12.6.0/lib64")
add_link_options("-Wl,-rpath,${CUDA_TOOLCHAIN_LIB}")
list(APPEND CMAKE_BUILD_RPATH "${CUDA_TOOLCHAIN_LIB}")
list(APPEND CMAKE_INSTALL_RPATH "${CUDA_TOOLCHAIN_LIB}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# ---- Core Packages ----
find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(ZLIB REQUIRED)
set(raft_DIR "/home/bh692/.conda/envs/py3/lib/cmake/raft")
find_package(raft REQUIRED)

# ---- Header-only Libs ----
find_path(DELAUNATOR_CPP_INCLUDE_DIRS "delaunator-header-only.hpp")

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

# Method C: Hardcoded Fallback (Based on your cluster logs)
if(NOT MPI_ALL_INCLUDES)
    message(WARNING "Using hardcoded MPI path as fallback.")
    list(APPEND MPI_ALL_INCLUDES "/apps/software/2024a/software/OpenMPI/5.0.3-GCC-13.3.0-CUDA-12.6.0/include")
endif()

# Apply globally so headers are found
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
target_link_directories(md_particle_lib PUBLIC /apps/software/2024a/software/CUDA/12.6.0/lib64)

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
    target_link_directories("${exe_name}" PRIVATE /apps/software/2024a/software/CUDA/12.6.0/lib64)

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
    target_link_directories("${exe_name}" PRIVATE /apps/software/2024a/software/CUDA/12.6.0/lib64)

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
