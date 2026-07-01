#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
class StraightCmdTest(Node):
    def __init__(self):
        super().__init__("straight_cmd_test")
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.speed = 0.08       # m/s, safe slow speed
        self.duration = 8.0     # seconds
        self.rate_hz = 20.0
        self.get_logger().info(
            f"Straight test: publishing /cmd_vel x={self.speed} m/s for {self.duration} s"
        )
    def run_test(self):
        msg = Twist()
        msg.linear.x = self.speed
        msg.linear.y = 0.0
        msg.angular.z = 0.0
        dt = 1.0 / self.rate_hz
        start = time.time()
        while rclpy.ok() and (time.time() - start) < self.duration:
            self.pub.publish(msg)
            time.sleep(dt)
        stop = Twist()
        for _ in range(20):
            self.pub.publish(stop)
            time.sleep(dt)
        self.get_logger().info("Straight test finished. Robot stopped.")
def main():
    rclpy.init()
    node = StraightCmdTest()
    try:
        node.run_test()
    finally:
        node.destroy_node()
        rclpy.shutdown()
if __name__ == "__main__":
    main()
