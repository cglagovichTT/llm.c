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

# Define a function to add targets with common properties
function(add_common_target target_name)
    add_executable(${target_name} ${target_name}.cpp)
    
    set(ARCH_NAME $ENV{ARCH_NAME})
    set(ARCH_DEFINES "${ARCH_NAME}_defines")
    
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
        -mavx2 -fPIC -fvisibility-inlines-hidden -fno-lto -stdlib=libc++
    )
    target_link_directories(${target_name} PRIVATE
        $ENV{TT_METAL_HOME}/build/lib
    )
    target_link_libraries(${target_name} PRIVATE
        tt_metal 
        yaml-cpp
        dl z boost_thread boost_filesystem boost_system boost_regex pthread atomic hwloc numa
        c++ c++abi
    )
endfunction()

# Add multiple targets using the common function
add_common_target(layernorm_forward)
add_common_target(dram_loopback)