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
from control.path_follower    import PathFollower
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

    path_follower = PathFollower(
        slam,
        lidar=lidar,
        preprocessor=preprocessor
    )

    rrt = RRTExplorer(
        slam=slam,
        step_size=0.15,                 # Sliced from 0.35 -> 0.15m (3 cells) so branches can navigate tight corridors
        max_iterations=1200,            # Raised from 600 -> 1200 to give the shorter steps plenty of growth attempts
        frontier_cluster_radius=0.6,
        robot_radius_cells=1,           
        sampling_padding_cells=45,      
        slam_map_topic="/slam_map",     
        min_known_cells=500,             
        min_plan_interval=2.0,          
    )

    aruco_monitor = ArucoMonitor(
        camera=camera,
        aruco=aruco,          # not aruco_detector
        slam=slam,
        explorer=rrt
    )

    # ── Wire explorer ↔ follower ─────────────────────

    rrt.path_follower = path_follower
    path_follower.explorer = rrt

    # ── Executor ──────────────────────────────────────────────────────
    executor = MultiThreadedExecutor()
    for node in [
        lidar,
        camera,
        odom,
        aruco_monitor,
        path_follower,
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
        path_follower._cmd_pub.publish(Twist())   # stop motors  ← underscore

        for node in [
            rrt,
            path_follower,
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