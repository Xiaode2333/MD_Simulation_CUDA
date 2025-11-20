cmake_minimum_required(VERSION 3.18)

project(CUDA2_MD LANGUAGES C CXX CUDA)

# --------------------------------------------------------
# FIX 1: Force older debug info format (Fixes 'as' error on cluster)
# --------------------------------------------------------
add_compile_options("-gdwarf-4")

# ---- Language Standards ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(GCC_LIB_PATH "/apps/software/2024a/software/GCCcore/13.3.0/lib64")
set(CMAKE_CUDA_ARCHITECTURES 89)


add_link_options("-L${GCC_LIB_PATH}" "-Wl,-rpath,${GCC_LIB_PATH}")

# ---- Core Packages ----
find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(ZLIB REQUIRED)

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
        fmt::fmt
        Python3::Python
        nlohmann_json::nlohmann_json
        MPI::MPI_CXX
        ZLIB::ZLIB
)

# ---- Tests ----
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
            ${Python3_INCLUDE_DIRS}
            ${PYTHON_NUMPY_INCLUDE_DIR}
            ${MPI_ALL_INCLUDES}
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
endforeach()

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
