cmake_minimum_required(VERSION 3.1.0)

if(${CMAKE_VERSION} VERSION_LESS 3.11)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else()
  cmake_policy(VERSION 3.11)
endif()

project(DeepvacClothesSeg)

set(CMAKE_BUILD_TYPE "RELEASE")

file(GLOB CS_SRC src/*.cc)
file(GLOB CS_HEADERS include/*.h)

add_subdirectory(${PROJECT_SOURCE_DIR}/../tnn_helper DeepvacTNNHelper)
add_library(DeepvacClothesSeg STATIC ${CS_SRC})

target_include_directories(DeepvacClothesSeg PUBLIC 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/../..>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/../tnn_helper/include>
)

target_link_libraries(
    DeepvacClothesSeg
    DeepvacTNNHelper
)