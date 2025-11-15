# cmake -B build -S .  -DCMAKE_TOOLCHAIN_FILE=/home/bryson/vcpkg/scripts/buildsystems/vcpkg.cmake  -DVCPKG_TARGET_TRIPLET=x64-linux     -DPython3_EXECUTABLE=/usr/bin/python3

cmake_minimum_required(VERSION 3.18)

project(CUDA2_MD LANGUAGES C CXX CUDA)

# ---- language setup ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---- CUDA toolkit (for cuda_runtime.h etc.) ----
# find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(ZLIB REQUIRED)


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

# ---- nlohmann-json via vcpkg ----


# ---- project directories ----
set(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(INCLUDE_DIR "${PROJECT_ROOT}/include")
set(CUDA_DIR    "${PROJECT_ROOT}/src")
set(SRC_DIR     "${PROJECT_ROOT}/src")
set(TESTS_DIR   "${PROJECT_ROOT}/tests")
set(MATPLOTLIB_CPP_DIR "${PROJECT_ROOT}/external/matplotlib-cpp")

# ---- core sources ----
file(GLOB_RECURSE PROJECT_HEADERS "${INCLUDE_DIR}/*.hpp")
file(GLOB_RECURSE PROJECT_SOURCES "${SRC_DIR}/*.cpp")
file(GLOB_RECURSE PROJECT_CUDA    "${CUDA_DIR}/*.cu")

add_library(md_particle_lib
    # ${PROJECT_HEADERS}
    ${PROJECT_SOURCES}
    ${PROJECT_CUDA}
)

set_source_files_properties(
    # ${PROJECT_HEADERS}
    # ${PROJECT_SOURCES}
    ${PROJECT_CUDA}
    PROPERTIES
        LANGUAGE CUDA
)

target_include_directories(md_particle_lib
    PUBLIC
        "${PROJECT_ROOT}"
        "${INCLUDE_DIR}"
        "${MATPLOTLIB_CPP_DIR}"
        ${Python3_INCLUDE_DIRS}
        ${PYTHON_NUMPY_INCLUDE_DIR}
        # ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(md_particle_lib
    PUBLIC
        fmt::fmt
        Python3::Python
        # Python3::NumPy
        nlohmann_json::nlohmann_json
        # CUDA::cudart
        MPI::MPI_CXX
        ZLIB::ZLIB
)
# target_compile_definitions(md_particle_lib
#     PUBLIC
#         FMT_HEADER_ONLY
# )

# ---- test executables: tests/*/*.cpp ----
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
    )

    target_include_directories("${exe_name}"
        PRIVATE
            "${PROJECT_ROOT}"
            "${INCLUDE_DIR}"
            "${MATPLOTLIB_CPP_DIR}"
            ${Python3_INCLUDE_DIRS}
            ${Python3_NumPy_INCLUDE_DIRS}
            ${CUDAToolkit_INCLUDE_DIRS}
    )
endforeach()

# End of Add
