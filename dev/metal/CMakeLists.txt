cmake_minimum_required(VERSION 3.16)
cmake_policy(VERSION 3.16)

project(layernorm_forward CXX)

# Use Clang-17
find_program(CLANGPP_17 clang++-17)
find_program(CLANG_17 clang-17)

if (NOT CLANGPP_17)
    message(FATAL_ERROR "clang++-17 not found")
endif()

if (NOT CLANG_17)
    message(FATAL_ERROR "clang-17 not found")
endif()

set(CMAKE_CXX_COMPILER "${CLANGPP_17}")
set(CMAKE_C_COMPILER "${CLANG_17}")

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libc++")

# Check for required environment variables
if("$ENV{TT_METAL_HOME}" STREQUAL "")
    message(FATAL_ERROR "TT_METAL_HOME is not set")
endif()
if("$ENV{ARCH_NAME}" STREQUAL "")
    message(FATAL_ERROR "ARCH_NAME is not set")
endif()

# Print debugging information
message(STATUS "TT_METAL_HOME: $ENV{TT_METAL_HOME}")
message(STATUS "ARCH_NAME: $ENV{ARCH_NAME}")

# Find Boost
# find_package(Boost REQUIRED COMPONENTS thread filesystem system regex)

# Find NUMA library
find_library(NUMA_LIBRARY NAMES numa)
if (NOT NUMA_LIBRARY)
    message(FATAL_ERROR "NUMA library not found")
endif()

set(REFLECT_INCLUDE_DIR "$ENV{TT_METAL_HOME}/.cpmcache/reflect/e75434c4c5f669e4a74e4d84e0a30d7249c1e66f")

# Manually configure the stdlib and metal_common_libs interface libraries
add_library(stdlib INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    find_library(LIBC++ c++)
    find_library(LIBC++ABI c++abi)
    if(NOT LIBC++ OR NOT LIBC++ABI)
        message(FATAL_ERROR "libc++ or libc++abi not found. Make sure you have libc++ and libc++abi installed and in your PATH")
    endif()
    target_link_libraries(stdlib INTERFACE ${LIBC++} ${LIBC++ABI})
    target_compile_options(stdlib INTERFACE -stdlib=libc++)
else()
    target_link_libraries(stdlib INTERFACE stdc++)
    target_compile_options(stdlib INTERFACE -fsized-deallocation)
endif()

add_library(metal_common_libs INTERFACE)
target_link_libraries(metal_common_libs INTERFACE
    dl z pthread atomic hwloc numa stdlib  # system libraries and stdlib interface
)

# Define a function to add targets with common properties
function(add_common_target target_name)
    add_executable(${target_name} ${target_name}.cpp)
    
    set(ARCH_NAME $ENV{ARCH_NAME})
    set(ARCH_DEFINES "${ARCH_NAME}_defines")
    set(UMD_HOME "$ENV{TT_METAL_HOME}/tt_metal/third_party/umd")
    
    set(INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}
        $ENV{TT_METAL_HOME}
        $ENV{TT_METAL_HOME}/tt_metal
        $ENV{TT_METAL_HOME}/tt_metal/detail
        $ENV{TT_METAL_HOME}/tt_metal/common
        $ENV{TT_METAL_HOME}/tt_metal/third_party/umd
        $ENV{TT_METAL_HOME}/tt_metal/third_party/fmt
        $ENV{TT_METAL_HOME}/tt_metal/hw/inc/wormhole
        $ENV{TT_METAL_HOME}/tt_metal/hw/inc/wormhole/wormhole_b0_defines
        $ENV{TT_METAL_HOME}/tt_metal/hw/inc/
        $ENV{TT_METAL_HOME}/tt_metal/hw/inc/common
        $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/include
        $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/include/wormhole_b0
        $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/src/firmware/riscv/wormhole
        $ENV{TT_METAL_HOME}/tt_metal/hostdevcommon
        ${UMD_HOME}/device/${ARCH_NAME}
        ${UMD_HOME}/src/firmware/riscv/${ARCH_NAME}
        $ENV{TT_METAL_HOME}/tt_metal/third_party/reflect
        ${REFLECT_INCLUDE_DIR}
    )

    target_include_directories(${target_name} PRIVATE ${INCLUDE_DIRS})
    
    # Print include directories for debugging
    message(STATUS "Include directories for ${target_name}:")
    foreach(dir ${INCLUDE_DIRS})
        message(STATUS "  ${dir}")
    endforeach()

    target_compile_definitions(${target_name} PRIVATE
        FMT_HEADER_ONLY
        TT_METAL_VERSIM_DISABLED
    )


    target_compile_options(${target_name} PRIVATE
        -mavx2 -fPIC -DFMT_HEADER_ONLY -fvisibility-inlines-hidden -fno-lto -stdlib=libc++
        -Werror -Wdelete-non-virtual-dtor -Wreturn-type -Wswitch -Wuninitialized -Wno-unused-parameter
        -Wno-deprecated-this-capture  # Add this line to disable the 'this' capture warning
    )

    target_link_directories(${target_name} PRIVATE
        $ENV{TT_METAL_HOME}/build_Release/lib
    )
    target_link_libraries(${target_name} PRIVATE
        tt_metal 
        #yaml-cpp
        dl z pthread atomic hwloc ${NUMA_LIBRARY}
        stdlib
    )
endfunction()

# Add multiple targets using the common function
add_common_target(layernorm_forward)
add_common_target(layernorm_backward)
add_common_target(dram_loopback)
