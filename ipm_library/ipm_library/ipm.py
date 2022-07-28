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

from typing import Optional

from ipm_library import utils
from ipm_library.exceptions import NoIntersectionError
from ipm_interfaces.msg import PlaneStamped, Point2DStamped
import numpy as np
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header
from tf2_geometry_msgs import PointStamped
import tf2_ros


class IPM:
    _camera_info: Optional[CameraInfo] = None

    def __init__(
            self,
            tf_buffer: tf2_ros.Buffer,
            camera_info: Optional[CameraInfo] = None) -> None:
        """
        Create a new inverse perspective mapper instance.

        :param tf_buffer: This module needs access to a sufficiently large tf2 buffer
        :param camera_info: `CameraInfo` Message containing the
            camera intrinsics, camera frame, ...
            The camera info can be updated later on using the setter or
            provided directly if it is unlikly to change
        """
        # TF needs a listener that is init in the node context, so we need a reference
        self._tf_buffer = tf_buffer
        self.set_camera_info(camera_info)

    def set_camera_info(self, camera_info: CameraInfo) -> None:
        """
        Set a new `CameraInfo` message.

        :param camera_info: The updated camera info message.
        """
        self._camera_info = camera_info

    def camera_info_received(self) -> bool:
        """
        Return if `CameraInfo` message has been received.

        :returns: If the message was received
        """
        return self._camera_info is not None

    def project_point(
            self,
            plane: PlaneStamped,
            point: Point2DStamped,
            output_frame: Optional[str] = None) -> PointStamped:
        """
        Projects a `Point2DStamped` onto a given plane using the latest CameraInfo intrinsics.

        :param plane: Plane in which the projection should happen
        :param point: Point that should be projected
        :param output_frame: TF2 frame in which the output should be provided
        :returns: The point projected onto the given plane in the output frame
        """
        # Convert point to numpy and utilize numpy projection function
        np_point = self.project_points(
            plane,
            np.array([[point.point.x, point.point.y]]),
            point.header,
            output_frame=None)[0]

        # Check if we have any nan values, aka if we have a valid intersection
        if np.isnan(np_point).any():
            raise NoIntersectionError

        # Create output point
        intersection_stamped = PointStamped()
        intersection_stamped.point.x = np_point[0]
        intersection_stamped.point.y = np_point[1]
        intersection_stamped.point.z = np_point[2]
        intersection_stamped.header.stamp = point.header.stamp
        intersection_stamped.header.frame_id = self._camera_info.header.frame_id

        # Transform output point if output frame if needed
        if output_frame is not None:
            intersection_stamped = self._tf_buffer.transform(
                intersection_stamped, output_frame)

        return intersection_stamped

    def project_points(
            self,
            plane_msg: PlaneStamped,
            points: np.ndarray,
            points_header: Header,
            output_frame: Optional[str] = None) -> np.ndarray:
        """
        Projects image points onto a given plane using the latest CameraInfo intrinsics.

        :param plane_msg: Plane in which the projection should happen
        :param points: Points that should be projected in the form of
            a nx2 numpy array where n is the number of points
        :param points_header: Header for the numpy message containing the frame and time stamp
        :param output_frame: TF2 frame in which the output should be provided
        :returns: The points projected onto the given plane in the output frame
        """
        assert points_header.stamp == plane_msg.header.stamp, \
            'Plane and Point need to have the same time stamp'
        assert self.camera_info_received(), 'No camera info set'
        assert self._camera_info.header.frame_id == points_header.frame_id, \
            'Points need to be in frame described in the camera info message'

        # Convert plane from general form to point normal form
        plane = utils.plane_general_to_point_normal(plane_msg.plane)

        # View plane from camera frame
        plane_base_point, plane_normal = utils.transform_plane_to_frame(
            plane=plane,
            input_frame=plane_msg.header.frame_id,
            output_frame=self._camera_info.header.frame_id,
            stamp=points_header.stamp,
            buffer=self._tf_buffer)

        # Convert points to float if they aren't allready
        if points.dtype.char not in np.typecodes['AllFloat']:
            points = points.astype(np.float32)

        # Get intersection points with plane
        np_points = utils.get_field_intersection_for_pixels(
            self._camera_info,
            points,
            plane_normal,
            plane_base_point)

        # Transform output point if output frame if needed
        if output_frame is not None:
            output_transformation = self._tf_buffer.lookup_transform(
                output_frame,
                self._camera_info.header.frame_id,
                points_header.stamp)
            np_points = utils.transform_points(
                np_points, output_transformation.transform)

        return np_points
