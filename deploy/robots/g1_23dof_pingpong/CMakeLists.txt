cmake_minimum_required(VERSION 3.12)
project(g1_23dof_pingpong)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_RPATH "/usr/local/lib")
set(CMAKE_INSTALL_RPATH "/usr/local/lib")

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

message(STATUS "Building g1_23dof_pingpong with ROS2 Humble topic support.")

include_directories(
  /usr/include/eigen3
  /usr/local/include/ddscxx # dds
  /usr/local/include/iceoryx/v2.0.2 # dds
  include
  ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/include
  ${PROJECT_SOURCE_DIR}/../../include/
  ${rclcpp_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
  ${nav_msgs_INCLUDE_DIRS}
)

link_libraries(
  /usr/local/lib/libunitree_sdk2.a
  /usr/local/lib/libddsc.so.0
  /usr/local/lib/libddscxx.so.0
  rt pthread z # dds
  libboost_program_options.a libyaml-cpp.a fmt
  cnpy
  ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so.1.22.0
)

file(GLOB_RECURSE ADD_SRC_LIST
  ${PROJECT_SOURCE_DIR}/src/*.cpp
)

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST})
ament_target_dependencies(${PROJECT_NAME}_lib rclcpp geometry_msgs nav_msgs)
link_libraries(${PROJECT_NAME}_lib)

add_executable(g1_pingpong_ctrl main.cpp)
target_link_libraries(g1_pingpong_ctrl ${PROJECT_NAME}_lib)
target_link_options(g1_pingpong_ctrl PRIVATE "-Wl,--disable-new-dtags" "-Wl,-rpath,/usr/local/lib")
