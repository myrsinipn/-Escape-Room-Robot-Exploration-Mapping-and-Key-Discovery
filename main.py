#!/usr/bin/env python3
import json
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sensors.lidar import LidarSensor
from sensors.camera import CameraSensor
from perception.scan_preprocessor import ScanPreprocessor
from perception.aruco_detector import ArucoDetector
from control.safe_lidar_motion import SafeLidarMotion
from control.aruco_monitor import ArucoMonitor


def load_calibration(path: str, camera_id: str = "11"):
    with open(path) as f:
        data = json.load(f)
    cam = data[camera_id]
    camera_matrix     = np.array(cam["camera_matrix"], dtype=np.float64)
    distortion_coeffs = np.array(cam["dist_coeffs"],   dtype=np.float64)
    return camera_matrix, distortion_coeffs


def main():
    rclpy.init()

    # ── Sensors ───────────────────────────────────────────────────
    lidar  = LidarSensor(topic_name='/scan',       min_range=0.10, max_range=8.0)
    camera = CameraSensor(topic_name='/image_raw')

    # ── Perception ────────────────────────────────────────────────
    preprocessor = ScanPreprocessor(
        min_range=0.10,
        max_range=8.0,
        apply_smoothing=True,
        smoothing_kernel_size=5,
    )

    calib_path = os.path.join(
        os.path.dirname(__file__), 'config', 'camera_calibration.json'
    )
    camera_matrix, distortion_coeffs = load_calibration(calib_path)

    aruco = ArucoDetector(
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs,
        marker_size=0.05,  # measure your markers
    )

    # ── Control ───────────────────────────────────────────────────
    motion  = SafeLidarMotion(lidar, preprocessor)
    monitor = ArucoMonitor(camera, aruco)

    # ── Executor ──────────────────────────────────────────────────
    executor = MultiThreadedExecutor()
    executor.add_node(lidar)
    executor.add_node(camera)
    executor.add_node(motion)
    executor.add_node(monitor)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Print ArUco summary before shutting down
        monitor.print_summary()

        from geometry_msgs.msg import Twist
        stop = Twist()
        motion.cmd_pub.publish(stop)
        for node in [motion, monitor, camera, lidar]:
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()