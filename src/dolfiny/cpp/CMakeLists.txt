cmake_minimum_required(VERSION 3.28)
project(dolfiny_cpp LANGUAGES CXX)

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

# Locate nanobind CMake dir by querying Python, since pip may install it outside system paths
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE NB_DIR
    RESULT_VARIABLE NB_RESULT
)

if(NOT NB_RESULT EQUAL 0)
    message(
        FATAL_ERROR
        "Failed to find nanobind. Install it with: pip install nanobind"
    )
endif()

list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

find_package(DOLFINX REQUIRED)

# Running error analysis header-only library, provides running_error.h.
include(FetchContent)
FetchContent_Declare(
    rea
    GIT_REPOSITORY https://github.com/fenics-dolfiny/rea.git
    GIT_TAG main
)
FetchContent_MakeAvailable(rea)

nanobind_add_module(_cpp STABLE_ABI NB_STATIC bindings.cpp)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import sys; print(sys.executable)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PY_EXE_CHECK
)
message(STATUS "CMake is using Python: ${PY_EXE_CHECK}")

# Query Python for wrapper header dirs — pip may install these outside system include paths
execute_process(
    COMMAND
        "${Python_EXECUTABLE}" -c
        "import dolfinx, mpi4py, petsc4py; print(';'.join([dolfinx.get_include(), mpi4py.get_include(), petsc4py.get_include()]))"
    OUTPUT_VARIABLE DOLFINX_PY_INCLUDE_DIRS
    RESULT_VARIABLE DOLFINX_PY_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT DOLFINX_PY_RESULT EQUAL 0)
    message(FATAL_ERROR "Could not find dolfinx Python wrapper headers.")
endif()

target_include_directories(_cpp PRIVATE ${DOLFINX_PY_INCLUDE_DIRS})
target_compile_features(_cpp PRIVATE cxx_std_23)
target_link_libraries(_cpp PRIVATE dolfinx::dolfinx rea)
target_include_directories(_cpp PRIVATE ${DOLFINX_INCLUDE_DIRS})

install(TARGETS _cpp LIBRARY DESTINATION dolfiny/cpp)
# Installed next to the _cpp extension module so that dolfiny.cpp.get_include()
# finds it both in regular and in editable installs.
install(
    FILES "${rea_SOURCE_DIR}/include/running_error.h"
    DESTINATION dolfiny/cpp
)
