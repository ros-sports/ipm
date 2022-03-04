#!/usr/bin/env python3
import tf2_ros
import numpy as np
from sensor_msgs.msg import CameraInfo
from shape_msgs.msg import Plane
from std_msgs.msg import Header
from tf2_geometry_msgs import PointStamped
from typing import Tuple
from ipm_library import utils

class IPM:
    _camera_info = None
    _tf_buffer = None

    def __init__(self, tf_buffer: tf2_ros.Buffer, camera_info: CameraInfo = None) -> None:
        self._tf_buffer = tf_buffer # Needs a listener that is init in the node context, so we need a reference
        self.set_camera_info(camera_info)

    def set_camera_info(self, camera_info: CameraInfo) -> None:
        """
        Set a new `CameraInfo` message

        :param camera_info: The updated camera info message.
        """
        self._camera_info = camera_info

    def camera_info_recived(self) -> bool:
        """
        Returns if `CameraInfo` message has been recived

        :returns: If the message was recived
        """
        return self._camera_info is not None

    def project_point(
            self,
            plane: Tuple[Plane, str],
            point: PointStamped,
            output_frame: str = None) -> PointStamped:
        """
        Projects a `PointStamped` onto a given plane using the latest CameraInfo intrinsics.

        :param plane: Plane in which the projection should happen
        :param point: Point that should be projected
        :param output_frame: TF2 frame in which the output should be provided
        :returns: The point projected onto the given plane in the output frame
        """
        # Convert point to numpy and utilize numpy projection function
        np_point = self.project_points(
            plane,
            np.array([[point.point.x, point.point.y, point.point.z]]),
            point.header,
            output_frame = None)[0]

        # Check if we have any nan values, aka if we have a valid intersection
        if np.isnan(np_point).any():
            raise Exception() # TODO custom exception

        # Create output point
        intersection_stamped = PointStamped()
        intersection_stamped.point.x = np_point[0]
        intersection_stamped.point.y = np_point[1]
        intersection_stamped.point.z = np_point[2]
        intersection_stamped.header.stamp = point.header.stamp
        intersection_stamped.header.frame_id = self._camera_info.header.frame_id

        # Transform output point if output frame if needed
        if output_frame is not None:
            intersection_stamped = self._tf_buffer.transform(intersection_stamped, output_frame)

        return intersection_stamped

    def project_points(
            self,
            plane: Tuple[Plane, str],
            points: np.ndarray,
            points_header: Header,
            output_frame: str = None) -> np.ndarray:
        """
        Projects a `PointStamped` onto a given plane using the latest CameraInfo intrinsics.

        :param plane: Plane in which the projection should happen
        :param points: Points that should be projected in the form of
            a nx3 numpy array where n is the number of points
        :param points_header: Header for the numpy message containing the frame and time stamp
        :param output_frame: TF2 frame in which the output should be provided
        :returns: The points projected onto the given plane in the output frame
        """
        assert points_header.stamp == plane.header.stamp, \
            "Plane and Point need to have the same time stamp"
        assert self.camera_info_recived(), "No camera info set"
        assert self._camera_info.header.frame_id == points_header.frame_id, \
            "Points needs to be in frame described in the camera info message"

        # Convert plane to normal format
        plane = utils.transform_to_normal_plane(plane)

        # View plane from camera frame
        plane_normal, plane_base_point = utils.transform_plane_to_frame(
            plane=plane,
            input_frame=plane.header.frame_id,
            output_frame=self._camera_info.header.frame_id,
            stamp=points_header.stamp,
            buffer=self._tf_buffer)

        # Get intersection points with plane
        np_points = utils.get_field_intersection_for_pixels(
            self._camera_info,
            points,
            plane_normal,
            plane_base_point)[0]

        # Transform output point if output frame if needed
        if output_frame is not None:
            output_transformation = self._tf_buffer.lookup_transform(
                    output_frame,
                    self._camera_info.header.frame_id,
                    points_header.stamp)
            np_points = utils.transform_points(np_points, output_transformation)

        return np_points
