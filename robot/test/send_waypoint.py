#!/usr/bin/env python3
"""
Debug script to send a single (X, Y) waypoint to the PathFollower.
Usage: python3 send_waypoint.py <X> <Y>
"""

import sys
import time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('debug_waypoint_sender')

    # Ensure the user provided X and Y arguments
    if len(sys.argv) != 3:
        node.get_logger().error("Usage: python3 send_waypoint.py <X> <Y>")
        node.get_logger().error("Example: python3 send_waypoint.py 1.5 -0.8")
        sys.exit(1)

    try:
        target_x = float(sys.argv[1])
        target_y = float(sys.argv[2])
    except ValueError:
        node.get_logger().error("X and Y coordinates must be numbers.")
        sys.exit(1)

    # Publisher matches the topic in PathFollower
    pub = node.create_publisher(Path, '/exploration_path', 10)

    # Wait a brief moment for the ROS 2 network to discover the publisher
    time.sleep(0.5)

    # Construct the Path message
    msg = Path()
    msg.header.frame_id = 'map'  # Change to 'odom' if your SLAM uses the odom frame
    msg.header.stamp = node.get_clock().now().to_msg()

    # Create the single waypoint
    pose = PoseStamped()
    pose.pose.position.x = target_x
    pose.pose.position.y = target_y
    pose.pose.position.z = 0.0
    
    # Orientation doesn't matter since PathFollower calculates heading based on dx/dy
    pose.pose.orientation.w = 1.0 

    msg.poses.append(pose)

    node.get_logger().info(f"Publishing single debug waypoint: ({target_x}, {target_y})")
    pub.publish(msg)

    # Spin briefly to ensure the message goes out, then exit
    rclpy.spin_once(node, timeout_sec=0.1)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()