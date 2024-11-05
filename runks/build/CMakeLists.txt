cmake_minimum_required(VERSION 3.15)
project(lagaus)

set(CMAKE_CXX_STANDARD 17)

find_package(xsimd CONFIG REQUIRED)

add_executable(laGAUS_JORDAN.cpp)
target_link_libraries(lagaus xsimd::xsimd)
