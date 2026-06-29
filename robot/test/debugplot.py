
#!/usr/bin/env python3
"""
Debug plot — Live trajectory plot for /odom and /slam_pose.

Run this on the laptop terminal.

It subscribes to:
    /odom
    /slam_pose

It does not publish /cmd_vel.
It does not run SLAM.
It only plots the data coming from the robot.
"""

import threading

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry

import matplotlib.pyplot as plt


class DebugPlotNode(Node):
    """
    Records /odom and /slam_pose to plot them live.
    """

    def __init__(self):
        super().__init__("debug_plot_node")

        self.odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_cb,
            10
        )

        self.slam_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            "/slam_pose",
            self.slam_cb,
            10
        )

        self.odom_x, self.odom_y = [], []
        self.slam_x, self.slam_y = [], []

        self.lock = threading.Lock()

        self.get_logger().info(
            "Debug plot started: listening to /odom and /slam_pose."
        )

    def odom_cb(self, msg: Odometry):
        with self.lock:
            self.odom_x.append(msg.pose.pose.position.x)
            self.odom_y.append(msg.pose.pose.position.y)

    def slam_cb(self, msg: PoseWithCovarianceStamped):
        with self.lock:
            self.slam_x.append(msg.pose.pose.position.x)
            self.slam_y.append(msg.pose.pose.position.y)

    def get_data(self):
        with self.lock:
            return (
                list(self.odom_x),
                list(self.odom_y),
                list(self.slam_x),
                list(self.slam_y),
            )


class LivePlot:
    def __init__(self, node: DebugPlotNode):
        self.node = node
        self.setup_plot()

    def setup_plot(self):
        plt.ion()

        self.fig, self.ax = plt.subplots(figsize=(8, 6))

        self.odom_line, = self.ax.plot(
            [],
            [],
            label="Odometry (Raw)",
            linestyle="--",
            color="blue",
            alpha=0.7
        )

        self.slam_line, = self.ax.plot(
            [],
            [],
            label="EKF SLAM (Corrected)",
            color="red",
            linewidth=2
        )

        self.ax.set_title("Live Trajectory: Odometry vs. SLAM")
        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Y (meters)")
        self.ax.legend()
        self.ax.grid(True)

        # Force equal aspect ratio so 1 m in X looks the same as 1 m in Y
        self.ax.set_aspect("equal", adjustable="datalim")

    def update_plot(self):
        odom_x, odom_y, slam_x, slam_y = self.node.get_data()

        if not odom_x and not slam_x:
            return

        self.odom_line.set_data(odom_x, odom_y)
        self.slam_line.set_data(slam_x, slam_y)

        self.ax.relim()
        self.ax.autoscale_view()

        self.fig.canvas.draw_idle()

    def run(self):
        try:
            while rclpy.ok() and plt.fignum_exists(self.fig.number):
                self.update_plot()
                plt.pause(0.1)

        except KeyboardInterrupt:
            pass


def main():
    rclpy.init()

    node = DebugPlotNode()

    ros_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True
    )
    ros_thread.start()

    plot = LivePlot(node)

    try:
        plot.run()

    finally:
        node.destroy_node()
        rclpy.shutdown()
        ros_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()

