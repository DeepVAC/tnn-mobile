cmake_minimum_required(VERSION 3.1.0)

if(${CMAKE_VERSION} VERSION_LESS 3.11)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else()
  cmake_policy(VERSION 3.11)
endif()

project(DeepvacKannaRotate)

set(CMAKE_BUILD_TYPE "RELEASE")

file(GLOB KR_SRC kannarotate-android-lib/src/*.c)
file(GLOB KR_HEADERS kannarotate-android-lib/include/*.h kannarotate-android-lib/src/*.h *.h)

add_library(DeepvacKannaRotate STATIC ${KR_SRC})

target_include_directories(DeepvacKannaRotate PUBLIC 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/kannarotate-android-lib/include>
)
