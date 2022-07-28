cmake_minimum_required(VERSION 3.8)
project(ipm_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(vision_msgs REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PlaneStamped.msg"
  "msg/Point2DStamped.msg"
  "srv/ProjectPoint.srv"
  "srv/ProjectPointCloud2.srv"
  DEPENDENCIES geometry_msgs sensor_msgs shape_msgs std_msgs vision_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
