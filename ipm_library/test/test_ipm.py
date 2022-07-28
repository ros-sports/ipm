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

from geometry_msgs.msg import TransformStamped
from ipm_library.exceptions import NoIntersectionError
from ipm_library.ipm import IPM
from ipm_interfaces.msg import PlaneStamped, Point2DStamped
import numpy as np
import pytest
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header
import tf2_ros as tf2
from vision_msgs.msg import Point2D


def test_ipm_camera_info():
    """Test if the camera info is handled correctly."""
    # We need to create a dummy tf buffer
    tf_buffer = tf2.Buffer()
    # Dummy camera info
    cam = CameraInfo()
    # Create an IPM
    ipm1 = IPM(tf_buffer, cam)
    assert ipm1.camera_info_received(), 'Failed to set camera info in constructor'
    # Create another IPM without the CameraInfo
    ipm2 = IPM(tf_buffer)
    assert not ipm2.camera_info_received(), 'Missing camera info not recognized'
    # Set camera info
    ipm2.set_camera_info(cam)
    assert ipm1.camera_info_received(), 'Failed to set camera info'
    # Set another camera info
    ipm2.set_camera_info(CameraInfo(header=Header(frame_id='test')))
    assert ipm2._camera_info != cam, 'Camera info not updated'


def test_ipm_project_point_no_transform():
    """Project Point2DStamped without doing any tf transforms."""
    # We need to create a dummy tf buffer
    tf_buffer = tf2.Buffer()
    # Dummy camera info
    cam = CameraInfo(
        header=Header(
            frame_id='camera_optical_frame',
        ),
        width=2048,
        height=1536,
        binning_x=4,
        binning_y=4,
        k=[1338.64532, 0., 1026.12387, 0., 1337.89746, 748.42213, 0., 0., 1.])
    # Create an IPM
    ipm = IPM(tf_buffer, cam)
    # Create Plane in the same frame as our camera with 1m distance facing the camera
    plane = PlaneStamped()
    plane.header.frame_id = 'camera_optical_frame'
    plane.plane.coef[2] = 1.0  # Normal in z direction
    plane.plane.coef[3] = -1.0  # 1 meter distance
    # Create Point2DStamped with the center pixel of the camera
    point_original_x = 100.0  # in pixels
    point_original_y = 200.0  # in pixels
    point_original = np.array([[point_original_x], [point_original_y]])
    point_original_msg = Point2DStamped(
        header=cam.header, point=Point2D(x=point_original_x, y=point_original_y))
    # Project points
    point_projected_msg = ipm.project_point(plane, point_original_msg)
    # Check header
    assert point_projected_msg.header == cam.header, 'Point header do not match'

    # Perform projection back into 2D image using projection matrix K to ensure that
    # it's the same as the original point
    point_projected_vec = np.array([[point_projected_msg.point.x],
                                    [point_projected_msg.point.y],
                                    [point_projected_msg.point.z]], dtype=np.float64)
    projection_matrix = np.reshape(cam.k, (3, 3))
    point_reprojected_2d_vec = np.matmul(projection_matrix, point_projected_vec)
    point_reprojected_2d = point_reprojected_2d_vec[0:2]
    # Projection doesn't consider the binning, so we need to correct for that
    point_reprojected_2d[0] = point_reprojected_2d[0] / cam.binning_x
    point_reprojected_2d[1] = point_reprojected_2d[1] / cam.binning_y
    assert np.allclose(point_original, point_reprojected_2d, rtol=0.0001), \
        'Projected point differs too much'


def test_ipm_project_points_no_transform():
    """Project points from NumPy array without doing any tf transforms."""
    # We need to create a dummy tf buffer
    tf_buffer = tf2.Buffer()
    # Dummy camera info
    cam = CameraInfo(
        header=Header(
            frame_id='camera_optical_frame',
        ),
        width=2048,
        height=1536,
        binning_x=4,
        binning_y=4,
        k=[1338.64532, 0., 1026.12387, 0., 1337.89746, 748.42213, 0., 0., 1.])
    # Create an IPM
    ipm = IPM(tf_buffer, cam)
    # Create Plane in the same frame as our camera with 1m distance facing the camera
    plane = PlaneStamped()
    plane.header.frame_id = 'camera_optical_frame'
    plane.plane.coef[2] = 1.0  # Normal in z direction
    plane.plane.coef[3] = -1.0  # 1 meter distance
    # Create two Points on the center pixel of the camera
    points = np.array([
        # Center
        [float(cam.width // cam.binning_x // 2),
         float(cam.height // cam.binning_y // 2), 0],
        # Diagonal Corners
        [float(cam.width // cam.binning_x),
         float(cam.height // cam.binning_y), 0],
        [0, 0, 0]
    ])
    # Project points
    projected_points = ipm.project_points(plane, points, cam.header)
    # Make goal points array, x and y are not exactly 0 because of the camera calibration as
    # well as an uneven amount of pixels
    goal_point_array = np.array([
        [-0.0015865,  0.014633, 1],
        [0.7633658,  0.588668, 1],
        [-0.7665390, -0.559401, 1]
    ])
    assert np.allclose(goal_point_array, projected_points, rtol=0.0001), \
        'Projected point differs too much'


def test_ipm_project_point_no_transform_no_intersection():
    """Impossible projection of Point2DStamped without doing any tf transforms."""
    # We need to create a dummy tf buffer
    tf_buffer = tf2.Buffer()
    # Dummy camera info
    cam = CameraInfo(
        header=Header(
            frame_id='camera_optical_frame',
        ),
        width=2048,
        height=1536,
        binning_x=4,
        binning_y=4,
        k=[1338.64532, 0., 1026.12387, 0., 1337.89746, 748.42213, 0., 0., 1.])
    # Create an IPM
    ipm = IPM(tf_buffer, cam)
    # Create Plane in the same frame as our camera but 1m behind it
    plane = PlaneStamped()
    plane.header.frame_id = 'camera_optical_frame'
    plane.plane.coef[2] = 1.0  # Normal in z direction
    plane.plane.coef[3] = 1.0  # 1 meter distance
    # Create Point2DStamped with the center pixel of the camera
    point = Point2DStamped()
    point.header = cam.header
    point.point.x = float(cam.width // cam.binning_x // 2)
    point.point.y = float(cam.height // cam.binning_y // 2)
    # Test if a NoIntersectionError is raised
    with pytest.raises(NoIntersectionError):
        # Project points
        ipm.project_point(plane, point)


def test_ipm_project_points_no_transform_no_intersection():
    """Impossible projection of points from NumPy array without doing any tf transforms."""
    # We need to create a dummy tf buffer
    tf_buffer = tf2.Buffer()
    # Dummy camera info
    cam = CameraInfo(
        header=Header(
            frame_id='camera_optical_frame',
        ),
        width=2048,
        height=1536,
        binning_x=4,
        binning_y=4,
        k=[1338.64532, 0., 1026.12387, 0., 1337.89746, 748.42213, 0., 0., 1.])
    # Create an IPM
    ipm = IPM(tf_buffer, cam)
    # Create Plane in the same frame as our camera with 1m distance facing the camera
    plane = PlaneStamped()
    plane.header.frame_id = 'camera_optical_frame'
    plane.plane.coef[2] = 1.0  # Normal in z direction
    plane.plane.coef[3] = 1.0  # 1 meter distance
    # Create two Points on the center pixel of the camera
    points = np.array([
        # Corner
        [0, 0, 0]
    ])
    # Project points
    projected_points = ipm.project_points(plane, points, cam.header)
    # Make goal points array, x and y are not exactly 0 because of the camera calibration as
    # well as an uneven amount of pixels
    goal_point_array = np.array([
        [np.nan,  np.nan, np.nan]
    ])
    np.testing.assert_equal(
        projected_points,
        goal_point_array,
        err_msg='Not all axes are none even tho the plane is invisible')


def test_ipm_project_point():
    """Project Point2DStamped without doing any tf transforms."""
    # We need to create a dummy tf buffer
    tf_buffer = tf2.Buffer()
    transform = TransformStamped()
    transform.header.frame_id = 'camera_optical_frame'
    transform.child_frame_id = 'base_footprint'
    transform.transform.rotation.w = 1.0
    transform.transform.translation.z = 1.0
    tf_buffer.set_transform_static(transform, '')
    # Dummy camera info
    cam = CameraInfo(
        header=Header(
            frame_id='camera_optical_frame',
        ),
        width=2048,
        height=1536,
        binning_x=4,
        binning_y=4,
        k=[1338.64532, 0., 1026.12387, 0., 1337.89746, 748.42213, 0., 0., 1.])
    # Create an IPM
    ipm = IPM(tf_buffer, cam)
    # Create Plane in the same frame as our camera with 1m distance facing the camera
    plane = PlaneStamped()
    plane.header.frame_id = 'base_footprint'
    plane.plane.coef[2] = 1.0  # Normal in z direction
    # Create Point2DStamped with the center pixel of the camera
    point = Point2DStamped()
    point.header = cam.header
    point.point.x = float(cam.width // cam.binning_x // 2)
    point.point.y = float(cam.height // cam.binning_y // 2)
    # Project points
    projected_point = ipm.project_point(
        plane, point, output_frame=plane.header.frame_id)
    # Check header
    assert projected_point.header.frame_id == plane.header.frame_id, 'Point header do not match'
    # Make goal point array, x and y are not exactly 0 because of the camera calibration as
    # well as an uneven amount of pixels
    goal_point_array = np.array([-0.0015865, 0.014633, 0])
    # Convert projected point to array
    projected_point_array = np.array([
        projected_point.point.x,
        projected_point.point.y,
        projected_point.point.z,
    ])
    assert np.allclose(goal_point_array, projected_point_array, rtol=0.0001), \
        'Projected point differs too much'


def test_ipm_project_points():
    """Project project of points from NumPy array."""
    # We need to create a dummy tf buffer
    tf_buffer = tf2.Buffer()
    transform = TransformStamped()
    transform.header.frame_id = 'camera_optical_frame'
    transform.child_frame_id = 'base_footprint'
    transform.transform.rotation.w = 1.0
    transform.transform.translation.z = 1.0
    tf_buffer.set_transform_static(transform, '')
    # Dummy camera info
    cam = CameraInfo(
        header=Header(
            frame_id='camera_optical_frame',
        ),
        width=2048,
        height=1536,
        binning_x=4,
        binning_y=4,
        k=[1338.64532, 0., 1026.12387, 0., 1337.89746, 748.42213, 0., 0., 1.])
    # Create an IPM
    ipm = IPM(tf_buffer, cam)
    # Create Plane in the same frame as our camera with 1m distance facing the camera
    plane = PlaneStamped()
    plane.header.frame_id = 'base_footprint'
    plane.plane.coef[2] = 1.0  # Normal in z direction
    # Create two Points on the center pixel of the camera
    points = np.array([
        # Center
        [float(cam.width // cam.binning_x // 2),
         float(cam.height // cam.binning_y // 2), 0],
        # Diagonal Corners
        [float(cam.width // cam.binning_x),
         float(cam.height // cam.binning_y), 0],
        [0, 0, 0]
    ])
    # Project points
    projected_points = ipm.project_points(
        plane,
        points=points,
        points_header=cam.header,
        output_frame=plane.header.frame_id)
    # Make goal points array, x and y are not exactly 0 because of the camera calibration as
    # well as an uneven amount of pixels
    goal_point_array = np.array([
        [-0.0015865,  0.014633, 0],
        [0.7633658,  0.588668, 0],
        [-0.7665390, -0.559401, 0]
    ])
    assert np.allclose(goal_point_array, projected_points, rtol=0.0001), \
        'Projected point differs too much'
