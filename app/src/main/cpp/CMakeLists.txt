cmake_minimum_required(VERSION 3.22.1)
project("tensor_affinity")

add_library(
    tensor_affinity
    SHARED
    tensor_affinity.cpp
)

# WHY: Android 15-17 requires 16 KB (16384-byte) ELF segment alignment for all 64-bit shared libraries.
target_link_options(
    tensor_affinity
    PRIVATE
    "-Wl,-z,max-page-size=16384"
    "-Wl,-z,common-page-size=16384"
)

find_library(log-lib log)

target_link_libraries(
    tensor_affinity
    ${log-lib}
)
