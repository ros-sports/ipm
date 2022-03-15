# Copyright (c) 2022 Hamburg Bit-Bots
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
import tf2_ros as tf2
from rclpy.node import Node
from rclpy.duration import Duration
from ipm_interfaces.srv import ProjectPoint, ProjectPointCloud2
from ipm_interfaces.msg import PlaneStamped
from sensor_msgs.msg import CameraInfo
from sensor_msgs_py.point_cloud2 import read_points_numpy, create_cloud_xyz32
from std_msgs.msg import Header
from ipm_library.ipm import IPM


class IPMService(Node):
    def __init__(self) -> None:
        super().__init__("ipm_service")
        self.tf_buffer = tf2.Buffer(Duration(seconds=5))   # TODO param
        self.tf_listener = tf2.TransformListener(self.tf_buffer, self)
        self.ipm = IPM(self.tf_buffer)
        self.camera_info_sub = self.create_subscription(CameraInfo, 'camera_info', self.ipm.set_camera_info)
        self.point_srv = self.create_service(ProjectPoint, 'project_point', self.point_projection_callback)
        self.point_cloud_srv = self.create_service(ProjectPointCloud2, 'project_pointcloud2', self.point_cloud_projection_callback)

    def point_projection_callback(self, request, response):
        # Map optional marking from "" to None
        if request.output_frame == "":
            output_frame = None
        else:
            output_frame = request.output_frame
        # Project the given point
        response.point = self.ipm.project_point(
            request.plane,
            request.point,
            output_frame).point
        return response

    def point_cloud_projection_callback(self, request, response):
        # Map optional marking from "" to None
        if request.output_frame == "":
            output_frame = self.ipm.get_camera_info().header.frame_id
        else:
            output_frame = request.output_frame
        # Project the given points
        projected_points = self.ipm.project_points(
            request.plane,
            read_points_numpy(request.points),
            output_frame)
        # Convert them into a PointCloud2
        response.points = create_cloud_xyz32(
            Header(
                stamp=request.plane.stamp,
                frame_id=output_frame),
            projected_points)
        return response



def main(args=None):
    rclpy.init(args=args)

    minimal_service = IPMService()

    rclpy.spin(minimal_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
