cmake_minimum_required(VERSION 3.15)
project(banded_solvers CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import jax.ffi; print(jax.ffi.include_dir())"
    OUTPUT_VARIABLE JAX_FFI_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _jax_result
)
if(NOT _jax_result EQUAL 0)
    message(FATAL_ERROR "Could not determine JAX FFI include dir. "
            "Make sure jax is installed in the active Python environment.")
endif()
message(STATUS "JAX FFI include dir: ${JAX_FFI_INCLUDE_DIR}")

foreach(target penta_ffi tri_ffi)
    add_library(${target} SHARED ${target}.cpp)
    target_include_directories(${target} PRIVATE ${JAX_FFI_INCLUDE_DIR})
    target_compile_options(${target} PRIVATE -O2 -fPIC)
    set_target_properties(${target} PROPERTIES
        PREFIX ""
        SUFFIX ".so"
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    if(APPLE)
        target_link_options(${target} PRIVATE -undefined dynamic_lookup)
    endif()
endforeach()
