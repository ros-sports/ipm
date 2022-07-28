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

from typing import List, Optional, Union

from geometry_msgs.msg import TransformStamped
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from soccer_ipm.soccer_ipm import SoccerIPM
from soccer_vision_2d_msgs.msg import Ball, BallArray, GoalpostArray, RobotArray
import soccer_vision_3d_msgs.msg as sv3dm
from soccer_vision_attribute_msgs.msg import Confidence
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage

# Dummy CameraInfo Message
camera_info = CameraInfo(
        header=Header(
            frame_id='camera_optical_frame',
        ),
        width=2048,
        height=1536,
        binning_x=4,
        binning_y=4,
        k=[1338.64532, 0., 1024.0, 0., 1337.89746, 768.0, 0., 0., 1.])


def standard_ipm_test_case(
        input_msg_type: type,
        input_topic: str,
        input_msg: Union[BallArray, GoalpostArray, RobotArray],
        output_msg_type: type,
        output_topic: str) -> Union[
            sv3dm.BallArray, sv3dm.GoalpostArray, sv3dm.RobotArray]:
    # Init ros
    rclpy.init()
    # Create IPM node
    node = SoccerIPM()
    # Create test node which comunicates with the IPM node
    test_node = Node('test_handler')
    # Create publishers to send data to the IPM node
    ball_pub = test_node.create_publisher(
        input_msg_type, input_topic, 10)
    camera_info_pub = test_node.create_publisher(
        CameraInfo, 'camera_info', 10)
    tf_pub = test_node.create_publisher(
        TFMessage, 'tf', 10)

    # Create a shared reference to the recived message in the local scope
    received_msg: List[Optional[output_msg_type]] = [None]

    # Create a callback with sets this reference
    def callback(msg):
        received_msg[0] = msg

    # Subscribe to IPM results
    test_node.create_subscription(
        output_msg_type, output_topic, callback, 10)

    # Create header message for the current time stamp in the camera frame
    header = Header(
        stamp=node.get_clock().now().to_msg(),
        frame_id="camera_optical_frame")

    # Create a dummy transform from the camera to the base_footprint frame
    tf = TransformStamped(
        header=header,
        child_frame_id="base_footprint",
    )
    tf.transform.translation.z = 1.0
    tf.transform.rotation.x = 0.0
    tf.transform.rotation.w = 1.0

    # Publish the dummy transform
    tf_pub.publish(TFMessage(
        transforms=[tf]
    ))
    # Spin the ipm to process the new data
    rclpy.spin_once(node, timeout_sec=0.1)

    # Send camera info message to the IPM
    camera_info.header.stamp = header.stamp
    camera_info_pub.publish(camera_info)
    # Spin the IPM to process the new data
    rclpy.spin_once(node, timeout_sec=0.1)

    # Send image space detection
    input_msg.header = header
    ball_pub.publish(input_msg)
    # Spin the IPM to process the new data
    rclpy.spin_once(node, timeout_sec=0.1)

    # Spin the test__node to recive the results from the IPM
    rclpy.spin_once(test_node, timeout_sec=0.1)

    # Assert that we recived a message
    assert received_msg[0] is not None

    # Clean shutdown of the nodes
    rclpy.shutdown()
    node.destroy_node()
    test_node.destroy_node()

    return received_msg[0], input_msg


def test_ipm_empty_ball():
    standard_ipm_test_case(
        BallArray,
        'balls_in_image',
        BallArray(),
        sv3dm.BallArray,
        'balls_relative')


def test_ipm_ball():
    # Create ball detection
    ball = Ball()
    ball.center.x = camera_info.width / camera_info.binning_x // 2
    ball.center.y = camera_info.height / camera_info.binning_y // 2
    ball.confidence = Confidence(confidence=0.42)
    ball_detection = BallArray(balls=[ball])

    out, inp = standard_ipm_test_case(
        BallArray,
        'balls_in_image',
        ball_detection,
        sv3dm.BallArray,
        'balls_relative')

    # Assert that we recived the correct message
    assert len(out.balls) == 1, 'Got too many balls'
    assert out.header.stamp == inp.header.stamp, 'Time stamp got changed by the ipm'
    assert out.header.frame_id == 'base_footprint', \
        'Output frame is not "base_footprint"'
    ball_relative: sv3dm.Ball = out.balls[0]
    np.testing.assert_allclose(
        ball_relative.confidence.confidence,
        ball.confidence.confidence)
    np.testing.assert_allclose(
        [ball_relative.center.x, ball_relative.center.y, ball_relative.center.z],
        [0.0, 0.0, 0.0])
