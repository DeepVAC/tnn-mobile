cmake_minimum_required(VERSION 3.1.0)

if(${CMAKE_VERSION} VERSION_LESS 3.11)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else()
  cmake_policy(VERSION 3.11)
endif()

project(DeepvacTNNHelper)

set(CMAKE_BUILD_TYPE "RELEASE")

file(GLOB TH_SRC src/*.cc)
file(GLOB TH_HEADERS include/*.h)

add_library(DeepvacTNNHelper STATIC ${TH_SRC})

target_include_directories(DeepvacTNNHelper PUBLIC 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/../..>
)
