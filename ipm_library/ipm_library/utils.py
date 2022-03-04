import math
import tf2_ros
import transforms3d
import numpy as np
from rclpy.duration import Duration
from shape_msgs.msg import Plane
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import TransformStamped
from tf2_geometry_msgs import PointStamped
from numba import jit
from typing import Tuple


def transform_to_normal_plane(plane: Plane) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a plane msg from the form `ax + by + cz + d = 0` to a normal vector
    and a base point.

    :param plane: The input plane
    :returns: A tuple with the normal vector and the base_point
    """
    # ax + by + cz + d = 0 where a, b, c are the normal vector
    a, b, c, d = plane.coef
    normal = np.linalg.norm(np.array([a, b, c]))
    base_point = normal * d
    return normal, base_point

def transform_plane_to_frame(
        plane: Tuple[np.ndarray, np.ndarray],
        input_frame: str,
        output_frame: str,
        stamp,
        buffer: tf2_ros.Buffer,
        timeout: Duration = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """
    returns a plane which an object is believed to be on as a tuple of a point on this plane and a normal
    #TODO docs
    """

    field_normal = PointStamped()
    field_normal.header.frame_id = input_frame
    field_normal.header.stamp = stamp
    field_normal.point.x = plane[0][0] + plane[1][0]
    field_normal.point.y = plane[0][1] + plane[1][1]
    field_normal.point.z = plane[0][2] + plane[1][2]
    field_normal = buffer.transform(field_normal, output_frame, timeout=timeout)

    field_point = PointStamped()
    field_point.header.frame_id = input_frame
    field_point.header.stamp = stamp
    field_point.point.x = plane[1][0]
    field_point.point.y = plane[1][1]
    field_point.point.z = plane[1][2]
    field_point = buffer.transform(field_point, output_frame, timeout=timeout)

    field_normal = np.array([field_normal.point.x, field_normal.point.y, field_normal.point.z])
    field_point = np.array([field_point.point.x, field_point.point.y, field_point.point.z])

    # field normal is a vector! so it stats at field point and goes up in z direction
    field_normal = field_point - field_normal
    return field_normal, field_point

def get_field_intersection_for_pixels(
        camera_info: CameraInfo,
        points: np.ndarray,
        plane_normal: np.ndarray,
        plane_base_point: np.ndarray,
        scale: float = 1.0) -> np.ndarray:
    """
    Projects an numpy array of points to the correspoding places on the field plane (in the camera frame).
    :param points: A nx3 array with n being the number of points
    :param plane_normal: The normal vektor of the projection plane
    :param plane_base_point: The base point of the projection plane
    :param scale: A scaling factor used if e.g. a mask with a lower resolution is transformed
    """
    camera_projection_matrix = camera_info.k

    # Calculate binning and scale
    binning_x = max(camera_info.binning_x, 1) / scale
    binning_y = max(camera_info.binning_y, 1) / scale

    # Create rays
    points[:, 0] = (points[:, 0] - (camera_projection_matrix[2] / binning_x)) / (camera_projection_matrix[0] / binning_x)
    points[:, 1] = (points[:, 1] - (camera_projection_matrix[5] / binning_y)) / (camera_projection_matrix[4] / binning_y)
    points[:, 2] = 1

    # Calculate ray -> plane intersections
    intersections = line_plane_intersections(plane_normal, plane_base_point, points)

    return intersections


def line_plane_intersections(
        plane_normal: np.ndarray,
        plane_base_point: np.ndarray,
        ray_directions: np.ndarray) -> np.ndarray:
    """
    Calculates the intersections of rays with a plane described by a normal and a point

    :param plane_normal: The normal vektor of the projection plane
    :param plane_base_point: The base point of the projection plane
    :param ray_directions: A nx3 array with n being the number of rays
    """
    n_dot_u = np.tensordot(plane_normal, ray_directions, axes=([0],[1]))
    relative_ray_distance = -plane_normal.dot(-plane_base_point) / n_dot_u

    # we are casting a ray, intersections need to be in front of the camera
    relative_ray_distance[relative_ray_distance <= 0] = np.nan

    ray_directions[:,0] = np.multiply(relative_ray_distance, ray_directions[:,0])
    ray_directions[:,1] = np.multiply(relative_ray_distance, ray_directions[:,1])
    ray_directions[:,2] = np.multiply(relative_ray_distance, ray_directions[:,2])

    return ray_directions

@jit
def transform_points(point_cloud: np.ndarray, transform: TransformStamped) -> np.ndarray:
    """
    Transforms a bulk of points from an numpy array using a provided `TransformStamped`.

    :param point_cloud: nx3 Array of points where n is the number of points
    :param transform: TF2 transform used for the transformation
    :returns: Array with the same shape as the input array, but with the transformation applied
    """
    # Build affine transformation
    T = [
        transform.transform.translation.x,
        transform.transform.translation.y,
        transform.transform.translation.z
    ]
    R = transforms3d.quaternions.quat2mat(
        [
            transform.transform.rotation.w,
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z
        ])
    affine = transforms3d.affines.compose(T, R, np.ones(3))

    # Apply transformation to all points
    for i in range(len(point_cloud)):
        p = transforms3d.affines.compose(point_cloud[i], np.eye(3), np.ones(3))
        point_cloud[i] = transforms3d.affines.decompose(np.matmul(p, affine))[0]

    return point_cloud