# Copyright (c) 2022 Kenji Brameld
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

from ipm_interfaces.msg import PlaneStamped
from ipm_interfaces.srv import MapPoint, MapPointCloud2
from ipm_library.ipm import IPM
from ipm_service.ipm import IPMService
import numpy as np
import rclpy
from sensor_msgs.msg import CameraInfo
from sensor_msgs_py.point_cloud2 import create_cloud, PointField, read_points_numpy
from shape_msgs.msg import Plane
from std_msgs.msg import Header
from tf2_ros import Buffer
from vision_msgs.msg import Point2D


def test_topics_and_services():

    rclpy.init()
    ipm_service_node = IPMService()

    # Check subscriptions
    dict_topics = dict(ipm_service_node.get_topic_names_and_types())

    assert '/camera_info' in dict_topics
    assert 'sensor_msgs/msg/CameraInfo' in dict_topics['/camera_info']

    # Check services
    dict_services = dict(ipm_service_node.get_service_names_and_types())

    assert '/map_point' in dict_services
    assert 'ipm_interfaces/srv/MapPoint' in dict_services['/map_point']

    assert '/map_pointcloud2' in dict_services
    assert 'ipm_interfaces/srv/MapPointCloud2' in dict_services['/map_pointcloud2']

    rclpy.shutdown()


def test_map_point_no_camera_info():

    rclpy.init()
    ipm_service_node = IPMService()
    test_node = rclpy.node.Node('test')

    client = test_node.create_client(MapPoint, 'map_point')
    future = client.call_async(MapPoint.Request())
    rclpy.spin_once(ipm_service_node, timeout_sec=0.1)

    rclpy.spin_once(test_node, timeout_sec=0.1)

    assert future.result() is not None
    assert future.result().result == MapPoint.Response.RESULT_NO_CAMERA_INFO

    rclpy.shutdown()


def test_map_point_invalid_plane():

    rclpy.init()
    ipm_service_node = IPMService()
    test_node = rclpy.node.Node('test')

    camera_info_pub = test_node.create_publisher(CameraInfo, 'camera_info', 10)
    camera_info = CameraInfo()
    camera_info_pub.publish(camera_info)
    rclpy.spin_once(ipm_service_node, timeout_sec=0.1)

    client = test_node.create_client(MapPoint, 'map_point')
    # Request with the default plane a=b=c=0 should be an invalid plane
    future = client.call_async(MapPoint.Request())
    rclpy.spin_once(ipm_service_node, timeout_sec=0.1)

    rclpy.spin_once(test_node, timeout_sec=0.1)

    assert future.result() is not None
    assert future.result().result == MapPoint.Response.RESULT_INVALID_PLANE

    rclpy.shutdown()


def test_map_point_no_intersection_error():

    rclpy.init()
    ipm_service_node = IPMService()
    test_node = rclpy.node.Node('test')

    camera_info_pub = test_node.create_publisher(CameraInfo, 'camera_info', 10)
    camera_info = CameraInfo(
        width=2048,
        height=1536,
        binning_x=4,
        binning_y=4,
        k=[1338.64532, 0., 1026.12387, 0., 1337.89746, 748.42213, 0., 0., 1.])
    camera_info_pub.publish(camera_info)
    rclpy.spin_once(ipm_service_node, timeout_sec=0.1)

    client = test_node.create_client(MapPoint, 'map_point')
    req = MapPoint.Request(plane=PlaneStamped(plane=Plane(coef=[0, 0, 1, 1])))
    future = client.call_async(req)
    rclpy.spin_once(ipm_service_node, timeout_sec=0.1)

    rclpy.spin_once(test_node, timeout_sec=0.1)

    assert future.result() is not None
    assert future.result().result == MapPoint.Response.RESULT_NO_INTERSECTION

    rclpy.shutdown()


def test_map_point():

    rclpy.init()
    ipm_service_node = IPMService()
    test_node = rclpy.node.Node('test')

    camera_info_pub = test_node.create_publisher(CameraInfo, 'camera_info', 10)
    camera_info = CameraInfo(
        header=Header(frame_id='camera_optical_frame'),
        width=2048,
        height=1536,
        binning_x=4,
        binning_y=4,
        k=[1338.64532, 0., 1026.12387, 0., 1337.89746, 748.42213, 0., 0., 1.])
    camera_info_pub.publish(camera_info)
    rclpy.spin_once(ipm_service_node, timeout_sec=0.1)

    point = Point2D(x=100.0, y=100.0)

    # XY-plane at z = 1.0
    # Create Plane in the same frame as our camera with 1m distance facing the camera
    plane = PlaneStamped()
    plane.header.frame_id = 'camera_optical_frame'
    plane.plane.coef[2] = 1.0  # Normal in z direction
    plane.plane.coef[3] = -1.0  # 1 meter distance

    client = test_node.create_client(MapPoint, 'map_point')
    req = MapPoint.Request(point=point, plane=plane)
    future = client.call_async(req)
    rclpy.spin_once(ipm_service_node, timeout_sec=0.1)

    rclpy.spin_once(test_node, timeout_sec=0.1)

    assert future.result() is not None
    assert future.result().result == MapPoint.Response.RESULT_SUCCESS

    ipm = IPM(Buffer(), camera_info)
    expected_point = ipm.map_point(plane, point)
    assert future.result().point == expected_point

    rclpy.shutdown()


def test_map_point_cloud_no_camera_info():

    rclpy.init()
    ipm_service_node = IPMService()
    test_node = rclpy.node.Node('test')

    client = test_node.create_client(MapPointCloud2, 'map_pointcloud2')
    future = client.call_async(MapPointCloud2.Request())
    rclpy.spin_once(ipm_service_node, timeout_sec=0.1)

    rclpy.spin_once(test_node, timeout_sec=0.1)

    assert future.result() is not None
    assert future.result().result == MapPointCloud2.Response.RESULT_NO_CAMERA_INFO

    rclpy.shutdown()


def test_map_point_cloud_invalid_plane():

    rclpy.init()
    ipm_service_node = IPMService()
    test_node = rclpy.node.Node('test')

    camera_info_pub = test_node.create_publisher(CameraInfo, 'camera_info', 10)
    camera_info = CameraInfo()
    camera_info_pub.publish(camera_info)
    rclpy.spin_once(ipm_service_node, timeout_sec=0.1)

    point_cloud = create_cloud(
        header=Header(),
        fields=[
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1)],
        points=[])

    client = test_node.create_client(MapPointCloud2, 'map_pointcloud2')
    # Request with the default plane a=b=c=0 should be an invalid plane
    future = client.call_async(MapPointCloud2.Request(points=point_cloud))
    rclpy.spin_once(ipm_service_node, timeout_sec=0.1)

    rclpy.spin_once(test_node, timeout_sec=0.1)

    assert future.result() is not None
    assert future.result().result == MapPointCloud2.Response.RESULT_INVALID_PLANE

    rclpy.shutdown()


def test_map_point_cloud():
    rclpy.init()
    ipm_service_node = IPMService()
    test_node = rclpy.node.Node('test')

    camera_info_pub = test_node.create_publisher(CameraInfo, 'camera_info', 10)
    camera_info = CameraInfo(
        header=Header(frame_id='camera_optical_frame'),
        width=2048,
        height=1536,
        binning_x=4,
        binning_y=4,
        k=[1338.64532, 0., 1026.12387, 0., 1337.89746, 748.42213, 0., 0., 1.])
    camera_info_pub.publish(camera_info)
    rclpy.spin_once(ipm_service_node, timeout_sec=0.1)

    # Create input point cloud
    points = np.arange(100).reshape(-1, 2)
    point_cloud = create_cloud(
        header=Header(frame_id='camera_optical_frame'),
        fields=[
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1)],
        points=points)

    # XY-plane at z = 1.0
    # Create Plane in the same frame as our camera with 1m distance facing the camera
    plane = PlaneStamped()
    plane.header.frame_id = 'camera_optical_frame'
    plane.plane.coef[2] = 1.0  # Normal in z direction
    plane.plane.coef[3] = -1.0  # 1 meter distance

    client = test_node.create_client(MapPointCloud2, 'map_pointcloud2')
    req = MapPointCloud2.Request(points=point_cloud, plane=plane)
    future = client.call_async(req)
    rclpy.spin_once(ipm_service_node, timeout_sec=0.1)

    rclpy.spin_once(test_node, timeout_sec=0.1)

    assert future.result() is not None
    assert future.result().result == MapPointCloud2.Response.RESULT_SUCCESS

    ipm = IPM(Buffer(), camera_info)
    expected_points = ipm.map_points(plane, points)

    np.testing.assert_allclose(
        read_points_numpy(future.result().points),
        expected_points,
        rtol=1e-06)

    rclpy.shutdown()
