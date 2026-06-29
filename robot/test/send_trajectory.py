#!/usr/bin/env python3
"""
Debug script to send a multi-waypoint trajectory to the PathFollower.
Usage: 
  python3 send_trajectory.py                  <-- Pings the robot for its location
  python3 send_trajectory.py <X1> <Y1> ...    <-- Sends a trajectory
"""

import sys
import time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('debug_trajectory_sender')

    path_points = []

    # If no arguments are provided, send an empty path as a "Ping"
    if len(sys.argv) == 1:
        node.get_logger().info("No coordinates provided. Sending a location ping to the robot...")
        node.get_logger().info("👉 Check your main PathFollower terminal to see the coordinates!")
    # Ensure the user provided pairs of X and Y arguments
    elif len(sys.argv) % 2 == 0:
        node.get_logger().error("Usage: python3 send_trajectory.py <X1> <Y1> <X2> <Y2> ...")
        node.get_logger().error("Example: python3 send_trajectory.py 1.0 0.0 1.0 1.0")
        sys.exit(1)
    else:
        # Parse arguments into a list of (x, y) tuples
        try:
            raw_coords = [float(arg) for arg in sys.argv[1:]]
            path_points = [(raw_coords[i], raw_coords[i+1]) for i in range(0, len(raw_coords), 2)]
        except ValueError:
            node.get_logger().error("All coordinates must be numbers.")
            sys.exit(1)

    # Publisher matches the topic in PathFollower
    path_pub = node.create_publisher(Path, '/exploration_path', 10)

    # Wait a brief moment for the ROS 2 network to discover the publisher
    time.sleep(0.5)

    path_msg = Path()
    path_msg.header.frame_id = 'map'
    path_msg.header.stamp = node.get_clock().now().to_msg()

    for x, y in path_points:
        p = PoseStamped()
        p.header = path_msg.header
        p.pose.position = Point(x=x, y=y, z=0.0)
        p.pose.orientation.w = 1.0
        path_msg.poses.append(p)

    if path_points:
        node.get_logger().info(f"Publishing trajectory with {len(path_points)} waypoints...")
        
    path_pub.publish(path_msg)
    rclpy.spin_once(node, timeout_sec=0.1)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()