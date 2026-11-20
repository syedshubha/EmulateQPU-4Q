cmake_minimum_required(VERSION 3.20)
project(emulateqpu_fullwave_overlap LANGUAGES CXX)

set(PALACE_SOURCE "" CACHE PATH "Source matching the Palace binary and MFEM build")
if(NOT EXISTS "${PALACE_SOURCE}/palace/utils/constants.hpp")
  message(FATAL_ERROR "Set PALACE_SOURCE to the Palace source tree used for the saved fields")
endif()

find_package(MFEM CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
add_executable(emulateqpu_fullwave_overlap
  src/crossmesh_overlap.cpp)
target_compile_features(emulateqpu_fullwave_overlap PRIVATE cxx_std_17)
target_include_directories(emulateqpu_fullwave_overlap PRIVATE "${PALACE_SOURCE}/palace")
target_link_libraries(emulateqpu_fullwave_overlap PRIVATE mfem nlohmann_json::nlohmann_json)
