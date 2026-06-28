#!/usr/bin/env python3
"""
Main entry point — B07 Escape Room Robot.
"""
import json
import os
import sys
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mapping.rrt_exploration  import RRTExplorer
#from control.path_follower    import PathFollower
from sensors.lidar    import LidarSensor
from sensors.camera   import CameraSensor
from sensors.odometry import OdometrySensor
from perception.scan_preprocessor import ScanPreprocessor
from perception.aruco_detector     import ArucoDetector
from perception.motion_model       import OmniMotionModel
from control.aruco_monitor     import ArucoMonitor
from state_estimation.ekf_slam import EKFLidarSLAM


def load_calibration(path: str, camera_id: str = "11"):
    with open(path) as f:
        data = json.load(f)
    cam = data[camera_id]
    return (
        np.array(cam["camera_matrix"], dtype=np.float64),
        np.array(cam["dist_coeffs"],   dtype=np.float64),
    )


def main() -> None:
    rclpy.init()

    # ── Sensors ───────────────────────────────────────────────────────
    lidar  = LidarSensor(topic_name="/scan",    min_range=0.10, max_range=8.0)
    camera = CameraSensor(topic_name="/image_raw")
    odom   = OdometrySensor(topic_name="/odom", qos_profile=30)

    # ── Shared helpers (plain objects, not nodes) ──────────────────────
    preprocessor = ScanPreprocessor(
        min_range=0.10,
        max_range=8.0,
        apply_smoothing=True,
        smoothing_kernel_size=5,
    )
    motion_model = OmniMotionModel()

    # ── Nodes ─────────────────────────────────────────────────────────
    calib_path = os.path.join(
    os.path.dirname(__file__),
    "config",
    "camera_calibration.json"
    )

    camera_matrix, distortion_coeffs = load_calibration(
        calib_path
    )

    aruco = ArucoDetector(
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs,
        marker_size=0.05,
    )
    slam = EKFLidarSLAM(
        lidar=lidar,
        odom=odom,
        scan_preprocessor=preprocessor,
        motion_model=motion_model,
    )

    # ── Explorer ──────────────────────────────────────────────────────
    # New RRTExplorer is a self-contained node (like the two working files):
    # it reads the map off a topic, the pose off TF (map -> base_footprint),
    # and /scan, then publishes /cmd_vel. It no longer takes slam/lidar/
    # preprocessor objects. Only the SLAM map topic name needs to match what
    # EKFLidarSLAM actually publishes.
    rrt = RRTExplorer(map_topic="/slam_map")

    aruco_monitor = ArucoMonitor(
        camera=camera,
        aruco=aruco,          # not aruco_detector
        slam=slam,
        explorer=rrt
    )

    # stop motors at startup (publisher is named cmd_pub on the new node)
    rrt.cmd_pub.publish(Twist())

    # ── Executor ──────────────────────────────────────────────────────
    executor = MultiThreadedExecutor()
    for node in [
        lidar,
        camera,
        odom,
        aruco_monitor,
        slam,
        rrt
    ]:
        executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        aruco_monitor.print_summary()
        slam.print_debug_summary()
        rrt.cmd_pub.publish(Twist())   # stop motors

        for node in [
            rrt,
            slam,
            aruco_monitor,
            odom,
            camera,
            lidar
        ]:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()