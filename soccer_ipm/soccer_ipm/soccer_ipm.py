import cv2
from cv_bridge import CvBridge
from ipm_library.exceptions import NoIntersectionError
from ipm_library.ipm import IPM
from ipm_msgs.msg import PlaneStamped, Point2DStamped
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from soccer_vision_2d_msgs.msg import (BallArray, FieldBoundary, GoalpostArray,
                                       RobotArray)
import soccer_vision_3d_msgs.msg as sv3dm
from std_msgs.msg import Header
import tf2_ros as tf2
from vision_msgs.msg import BoundingBox2D, Point2D


class SoccerIPM(Node):
    def __init__(self) -> None:
        super().__init__('soccer_ipm')
        # We need to create a dummy tf buffer
        self.tf_buffer = tf2.Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = tf2.TransformListener(self.tf_buffer, self)

        # Create an IPM instance
        self.ipm = IPM(self.tf_buffer)

        # Create CvBride
        self._cv_bridge = CvBridge()

        # Declare params
        self.declare_parameter('ball.ball_radius', 0.0)
        self.declare_parameter('goalposts.bar_height', 0.0)
        self.declare_parameter('base_footprint_frame', "base_footprint")
        self.declare_parameter('obstacles.footpoint_out_of_image_threshold', 0.0)
        self.declare_parameter('goalposts.footpoint_out_of_image_threshold', 0.0)
        self.declare_parameter('camera_info.camera_info_topic', "camera_info")
        self.declare_parameter('ball.ball_topic', "balls_in_image")
        self.declare_parameter('goalposts.goalposts_topic', "goalposts_in_image")
        self.declare_parameter('obstacles.obstacles_topic', "obstacles_in_image")
        self.declare_parameter('field_boundary.field_boundary_topic', "field_boundary_in_image")
        self.declare_parameter('masks.line_mask.topic', "line_masks_in_image")
        self.declare_parameter('masks.line_mask.scale', 0.0)

        # Parameters
        self._ball_height = self.get_parameter("ball.ball_radius").value
        self._bar_height = self.get_parameter("goalposts.bar_height").value
        self._base_footprint_frame = self.get_parameter("base_footprint_frame").value
        self._obstacle_footpoint_out_of_image_threshold = \
            self.get_parameter("obstacles.footpoint_out_of_image_threshold").value
        self._goalpost_footpoint_out_of_image_threshold = \
            self.get_parameter("goalposts.footpoint_out_of_image_threshold").value
        camera_info_topic = self.get_parameter("camera_info.camera_info_topic").value
        balls_in_image_topic = self.get_parameter("ball.ball_topic").value
        goalposts_in_image_topic = self.get_parameter("goalposts.goalposts_topic").value
        obstacles_in_image_topic = self.get_parameter("obstacles.obstacles_topic").value
        field_boundary_in_image_topic = \
            self.get_parameter("field_boundary.field_boundary_topic").value
        line_mask_in_image_topic = self.get_parameter("masks.line_mask.topic").value
        line_mask_scaling = self.get_parameter("masks.line_mask.scale").value

        # Subscribe to camera info
        self.create_subscription(CameraInfo, camera_info_topic, self.ipm.set_camera_info, 1)

        self.balls_pub = self.create_publisher(
            sv3dm.BallArray, "balls_relative", 1)
        self.line_mask__pub = self.create_publisher(
            PointCloud2, "line_mask_relative_pc", 1)
        self.goalposts_pub = self.create_publisher(
            sv3dm.GoalpostArray, "goal_posts_relative", 1)
        self.robots__pub = self.create_publisher(
            sv3dm.RobotArray, "robots_relative", 1)
        self.field_boundary_pub = self.create_publisher(
            sv3dm.FieldBoundary, "field_boundary_relative", 1)

        # Subscribe to image space data topics
        self.create_subscription(
            BallArray, balls_in_image_topic, self.callback_ball, 1)
        self.create_subscription(
            GoalpostArray, goalposts_in_image_topic, self.callback_goalposts, 1)
        self.create_subscription(
            RobotArray, obstacles_in_image_topic, self.callback_robots, 1)
        self.create_subscription(
            FieldBoundary, field_boundary_in_image_topic, self.callback_field_boundary, 1)
        self.create_subscription(
            Image,
            line_mask_in_image_topic,
            lambda msg: self.callback_masks(
                msg,
                self.line_mask__pub,
                scale=line_mask_scaling), 1)

    def get_field(self, time, heigh_offset=0):
        plane = PlaneStamped()
        plane.header.frame_id = self._base_footprint_frame
        plane.header.stamp = time
        plane.plane.coef[2] = 1.0  # Normal in z direction
        plane.plane.coef[3] = -heigh_offset  # Distance above the ground
        return plane

    def callback_ball(self, msg: BallArray):
        field = self.get_field(msg.header.stamp, self._ball_height)

        balls_relative = sv3dm.BallArray()
        balls_relative.header.stamp = msg.header.stamp
        balls_relative.header.frame_id = self._base_footprint_frame

        for ball in msg.balls:
            ball_point = Point2DStamped(
                header=msg.header,
                point=ball.center)

            try:
                transformed_ball = self.ipm.project_point(
                    field,
                    ball_point,
                    output_frame=self._base_footprint_frame)

                ball_relative = sv3dm.Ball()
                ball_relative.center = transformed_ball.point
                ball_relative.confidence = ball.confidence
                balls_relative.balls.append(ball_relative)
            except NoIntersectionError:
                self.get_logger().warn(
                    "Got a ball at ({},{}) I could not transform.".format(
                        ball.center.x,
                        ball.center.y),
                    throttle_duration_sec=5)

        self.balls_pub.publish(balls_relative)

    def callback_goalposts(self, msg: GoalpostArray):
        field = self.get_field(msg.header.stamp)

        # Create new message
        goalposts_relative_msg = sv3dm.GoalpostArray()
        goalposts_relative_msg.header.stamp = msg.header.stamp
        goalposts_relative_msg.header.frame_id = self._base_footprint_frame

        # Transform goal posts
        for goal_post_in_image in msg.posts:
            # Check if post is not going out of the image at the bottom
            if not self._object_at_bottom_of_image(
                    self._bb_footpoint(goal_post_in_image.bb).y,
                    self._goalpost_footpoint_out_of_image_threshold):
                # Create footpoint
                footpoint = Point2DStamped(
                    header=msg.header,
                    point=self._bb_footpoint(goal_post_in_image.bb)
                )
                # Project point from image onto field plane
                try:
                    relative_foot_point = self.ipm.project_point(
                        field,
                        footpoint,
                        output_frame=self._base_footprint_frame)

                    post_relative = sv3dm.Goalpost()
                    post_relative.attributes = goal_post_in_image.attributes
                    post_relative.bb.center.position = relative_foot_point.point
                    post_relative.bb.size.x = 0.1  # TODO better size estimation
                    post_relative.bb.size.y = 0.1  # TODO better size estimation
                    post_relative.bb.size.z = 1.5  # TODO better size estimation
                    post_relative.confidence = goal_post_in_image.confidence
                    goalposts_relative_msg.posts.append(post_relative)
                except NoIntersectionError:
                    self.get_logger().warn(
                        "Got a post with foot point ({},{}) I could not transform.".format(
                            footpoint.point.x,
                            footpoint.point.y),
                        throttle_duration_sec=5)

        self.goalposts_pub.publish(goalposts_relative_msg)

    def callback_robots(self, msg: RobotArray):
        field = self.get_field(msg.header.stamp, 0.0)

        robots = sv3dm.RobotArray()
        robots.header.stamp = msg.header.stamp
        robots.header.frame_id = self._base_footprint_frame

        for robot in msg.robots:

            # Check if post is not going out of the image at the bottom
            if not self._object_at_bottom_of_image(
                    self._bb_footpoint(robot.bb).y,
                    self._goalpost_footpoint_out_of_image_threshold):
                # Create footpoint
                footpoint = Point2DStamped(
                    header=msg.header,
                    point=self._bb_footpoint(robot.bb)
                )
                # Project point from image onto field plane
                try:
                    relative_foot_point = self.ipm.project_point(
                        field,
                        footpoint,
                        output_frame=self._base_footprint_frame)

                    transformed_robot = sv3dm.Robot()
                    transformed_robot.attributes = robot.attributes
                    transformed_robot.confidence = robot.confidence
                    transformed_robot.bb.center.position = relative_foot_point.point
                    transformed_robot.bb.size.x = 0.3   # TODO better size estimation
                    transformed_robot.bb.size.y = 0.3   # TODO better size estimation
                    transformed_robot.bb.size.z = 0.5   # TODO better size estimation
                    robots.robots.append(transformed_robot)
                except NoIntersectionError:
                    self.get_logger().warn(
                        "Got a robot with foot point ({},{}) I could not transform.".format(
                            footpoint.point.x,
                            footpoint.point.y),
                        throttle_duration_sec=5)

        self.robots__pub.publish(robots)

    def callback_field_boundary(self, msg: FieldBoundary):
        field = self.get_field(msg.header.stamp, 0.0)

        field_boundary = sv3dm.FieldBoundary()
        field_boundary.header.stamp = msg.header.stamp
        field_boundary.header.frame_id = self._base_footprint_frame
        field_boundary.confidence = field_boundary.confidence

        for p in msg.points:
            image_point = Point2DStamped(
                header=msg.header,
                point=p)
            # Project point from image onto field plane
            try:
                relative_foot_point = self.ipm.project_point(
                    field,
                    image_point,
                    output_frame=self._base_footprint_frame)

                field_boundary.points.append(relative_foot_point.point)
            except NoIntersectionError:
                self.get_logger().warn(
                    "Got a field boundary point ({},{}) I could not transform.".format(
                        image_point.point.x,
                        image_point.point.y),
                    throttle_duration_sec=5)

        self.field_boundary_pub.publish(field_boundary)

    def callback_masks(self, msg: Image, publisher, encoding='8UC1', scale: float = 1.0):
        """
        Projects a mask from the input image as a pointcloud on the field plane.

        :param msg: Mask msg type
        :param publisher: Publisher which should be used to publish the projected point cloud
        :param encoding: Encoding of the input mask. For the exact format see the cv_bride docs.
        :param scale: Downsampling which is applied to the mask before the projection.
        """
        # Get field plane
        field = self.get_field(msg.header.stamp, 0.0)
        if field is None:
            return

        # Convert subsampled image
        image = cv2.resize(
            self._cv_bridge.imgmsg_to_cv2(msg, encoding),
            (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        # Get indices for all non 0 pixels (the pixels which should be displayed in the pointcloud)
        point_idx_tuple = np.where(image != 0)

        # Restructure index tuple to a array
        point_idx_array = np.empty((point_idx_tuple[0].shape[0], 3))
        point_idx_array[:, 0] = point_idx_tuple[1] / scale
        point_idx_array[:, 1] = point_idx_tuple[0] / scale

        # Project points
        points_on_plane = self.ipm.project_points(
                    field,
                    point_idx_array,
                    msg.header,
                    output_frame=self._base_footprint_frame)

        # Make a pointcloud2 out of them
        pc = create_cloud_xyz32(
            Header(
                stamp=msg.header.stamp,
                frame_id=self._base_footprint_frame
            ),
            points_on_plane)

        # Publish point cloud
        publisher.publish(pc)

    def _object_at_bottom_of_image(self, position, thresh):
        """
        Check if the objects y position is at the bottom of the image.

        :param position: Y-position of the object
        :param thresh: Threshold defining the region at the bottom of the image which is
            counted as 'the bottom' as a fraction of the image height
        """
        image_height = self.ipm.get_camera_info().height
        image_height /= max(self.ipm.get_camera_info().binning_y, 1)

        scaled_thresh = thresh * image_height
        return position > scaled_thresh

    def _bb_footpoint(self, bounding_box: BoundingBox2D) -> Point2D:  # TODO rotated bb
        return Point2D(
            x=float(bounding_box.center.position.x),
            y=float(bounding_box.center.position.y + bounding_box.size_y // 2),
        )


def main(args=None):
    rclpy.init(args=args)
    node = SoccerIPM()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    ex.spin()
    node.destroy_node()
    rclpy.shutdown()
