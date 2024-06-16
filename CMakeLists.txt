################################################################################
# Set minimum required version of cmake, project name and compile options
################################################################################
cmake_minimum_required(VERSION 3.0.2)
project(nicol_rh8d VERSION 0.0.1)

add_compile_options(-std=c++11)

################################################################################
# Find catkin packages and libraries for catkin and system dependencies
################################################################################
find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    sensor_msgs
    cmake_modules
    rospy
    cv_bridge
    image_transport
    camera_info_manager
    realsense2_camera
)

################################################################################
# Setup for python modules and scripts
################################################################################

################################################################################
# Declare ROS messages, services and actions
################################################################################

################################################################################
## Declare ROS dynamic reconfigure parameters
################################################################################

################################################################################
# Declare catkin specific configuration to be passed to dependent projects
################################################################################
catkin_package(
  CATKIN_DEPENDS  roscpp
                  std_msgs
                  sensor_msgs
                  cmake_modules
                  rospy
                  cv_bridge
                  image_transport
                  camera_info_manager
                  realsense2_camera
)

################################################################################
# Build
################################################################################
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# add_library(${PROJECT_NAME} SHARED src/video_stream.cpp)
# target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
# add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencfg)

################################################################################
# Install
################################################################################
# install(TARGETS ${PROJECT_NAME}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

# install(PROGRAMS
#   scripts/set_defaults_SEE_CAM
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}  
# )

install(DIRECTORY launch scripts
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

# install(FILES nodelet_plugins.xml
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

################################################################################
# Test
################################################################################
