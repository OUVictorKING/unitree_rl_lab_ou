cmake_minimum_required(VERSION 3.12)
project(g1_23dof_pingpong)

set(CMAKE_CXX_STANDARD 17)

# unitree_sdk2 is built but not installed to /usr/local on this machine.
# Point at its local install tree (headers + libs) instead.
set(UNITREE_SDK2_DIR "/home/victorking/projects/HumanoidProject/unitree_sdk2/install"
    CACHE PATH "unitree_sdk2 install prefix")

set(CMAKE_BUILD_RPATH "${UNITREE_SDK2_DIR}/lib")
set(CMAKE_INSTALL_RPATH "${UNITREE_SDK2_DIR}/lib")

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

message(STATUS "Building g1_23dof_pingpong with ROS2 Humble topic support.")

include_directories(
  /usr/include/eigen3
  ${UNITREE_SDK2_DIR}/include # unitree headers (unitree/dds_wrapper/...)
  ${UNITREE_SDK2_DIR}/include/ddscxx # dds
  include
  ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/include
  ${PROJECT_SOURCE_DIR}/../../thirdparty/cnpy # cnpy.h (npy reader)
  ${PROJECT_SOURCE_DIR}/../../include/
  ${rclcpp_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
  ${nav_msgs_INCLUDE_DIRS}
)

link_libraries(
  ${UNITREE_SDK2_DIR}/lib/libunitree_sdk2.a
  ${UNITREE_SDK2_DIR}/lib/libddsc.so.0
  ${UNITREE_SDK2_DIR}/lib/libddscxx.so.0
  rt pthread z # dds
  libboost_program_options.a libyaml-cpp.a fmt
  ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so.1.22.0
)

file(GLOB_RECURSE ADD_SRC_LIST
  ${PROJECT_SOURCE_DIR}/src/*.cpp
)
# cnpy is vendored as source (only cnpy.cpp, depends on zlib which is linked above)
list(APPEND ADD_SRC_LIST ${PROJECT_SOURCE_DIR}/../../thirdparty/cnpy/cnpy.cpp)

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST})
ament_target_dependencies(${PROJECT_NAME}_lib rclcpp geometry_msgs nav_msgs)
link_libraries(${PROJECT_NAME}_lib)

add_executable(g1_pingpong_ctrl main.cpp)
target_link_libraries(g1_pingpong_ctrl ${PROJECT_NAME}_lib)
target_link_options(g1_pingpong_ctrl PRIVATE "-Wl,--disable-new-dtags" "-Wl,-rpath,${UNITREE_SDK2_DIR}/lib")
