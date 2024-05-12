cmake_minimum_required(VERSION 3.16)
project(layernorm_forward CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if("$ENV{TT_METAL_HOME}" STREQUAL "")
    message(FATAL_ERROR "TT_METAL_HOME is not set")
endif()
if("$ENV{ARCH_NAME}" STREQUAL "")
    message(FATAL_ERROR "ARCH_NAME is not set")
endif()

message($ENV{TT_METAL_HOME}/tt_metal/third_party/src/firmware/riscv/wormhole)

# Define a function to add targets with common properties
function(add_common_target target_name)
    add_executable(${target_name} ${target_name}.cpp)
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        $ENV{TT_METAL_HOME}
        $ENV{TT_METAL_HOME}/tt_metal
        $ENV{TT_METAL_HOME}/tt_metal/detail
        $ENV{TT_METAL_HOME}/tt_metal/third_party/umd
        $ENV{TT_METAL_HOME}/tt_metal/third_party/fmt
        $ENV{TT_METAL_HOME}/tt_metal/hw/inc/wormhole
        $ENV{TT_METAL_HOME}/tt_metal/hw/inc/wormhole/wormhole_b0_defines
        $ENV{TT_METAL_HOME}/tt_metal/hw/inc/
        $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/src/firmware/riscv/wormhole
        $ENV{TT_METAL_HOME}/tt_eager
    )
    add_definitions(-DFMT_HEADER_ONLY)
    target_link_directories(${target_name} PRIVATE
        $ENV{TT_METAL_HOME}/build/lib
    )
    target_link_libraries(${target_name} PRIVATE
        tt_metal 
        #tt_eager
        yaml-cpp
        dl z boost_thread boost_filesystem boost_system boost_regex pthread atomic hwloc
    )
endfunction()

# Add multiple targets using the common function
add_common_target(layernorm_forward)
add_common_target(dram_loopback)