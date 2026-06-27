#!/usr/bin/env python3
"""
Simulated testing environment for PathFollower.
"""
import math
import time
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from geometry_msgs.msg import Twist
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your actual PathFollower class
# (Ensure your follower script is named path_follower.py)
from control.path_follower import PathFollower

class MockSLAM(Node):
    def __init__(self):
        super().__init__('mock_slam')
        # Start at the origin facing forward
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.last_time = time.time()
        
        # Listen to the follower's velocity commands to simulate movement
        self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)

    @property
    def pose(self):
        # This property matches exactly what your PathFollower expects
        return (self.x, self.y, self.theta)

    def cmd_callback(self, msg):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        
        # Local velocities commanded by PathFollower
        vx = msg.linear.x
        vy = msg.linear.y
        w = msg.angular.z
        
        # Convert local omni-directional velocity to global map movement
        gx = vx * math.cos(self.theta) - vy * math.sin(self.theta)
        gy = vx * math.sin(self.theta) + vy * math.cos(self.theta)
        
        # Integrate position (update the fake robot's location)
        self.x += gx * dt
        self.y += gy * dt
        self.theta += w * dt
        
        # Keep heading mapped correctly
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

class MockExplorer:
    def notify_path_done(self, reason):
        print(f"\n✅ [MOCK EXPLORER] Path done notification received! Reason: '{reason}'\n")

def main(args=None):
    rclpy.init(args=args)
    
    mock_slam = MockSLAM()
    mock_explorer = MockExplorer()
    
    # Instantiate your node with the mocked dependencies.
    # Lidar and preprocessor are None since we just want to test path following, not avoidance.
    follower = PathFollower(slam=mock_slam, lidar=None, preprocessor=None)
    follower.explorer = mock_explorer
    
    # We need an executor to run both the MockSLAM node and the PathFollower node simultaneously
    executor = SingleThreadedExecutor()
    executor.add_node(mock_slam)
    executor.add_node(follower)
    
    print("🚀 Simulated PathFollower is running! Robot is at (0.0, 0.0).")
    print("Open another terminal and use send_waypoint.py to give it a goal.")
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        print("\nShutting down simulation...")
    finally:
        executor.shutdown()
        follower.destroy_node()
        mock_slam.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()