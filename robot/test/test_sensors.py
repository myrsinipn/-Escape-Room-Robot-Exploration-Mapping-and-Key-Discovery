import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cv2
import rclpy
from sensors.lidar import LidarSensor
from sensors.camera import CameraSensor
from sensors.odometry import OdometrySensor

def main():

    rclpy.init()

    lidar = LidarSensor()
    camera = CameraSensor()
    odometry = OdometrySensor()

    print("\n=== SENSOR TEST STARTED ===")
    print("Press ESC to exit.\n")

    try:

        while rclpy.ok():

            # Process ROS callbacks
            rclpy.spin_once(
                lidar,
                timeout_sec=0.01,
            )

            rclpy.spin_once(
                camera,
                timeout_sec=0.01,
            )

            rclpy.spin_once(
                odometry,
                timeout_sec=0.01,
            )

            # ---------------------------------
            # LIDAR
            # ---------------------------------

            scan = lidar.get_scan()

            if scan is not None:

                ranges = scan["ranges"]

                print(
                    f"[LIDAR] "
                    f"Beams={len(ranges)} "
                    f"Min={ranges.min():.2f}m "
                    f"Max={ranges.max():.2f}m"
                )

            # ---------------------------------
            # CAMERA
            # ---------------------------------

            frame_data = camera.get_frame()

            if frame_data is not None:

                frame = frame_data["frame"]

                cv2.imshow(
                    "Camera Test",
                    frame,
                )

            # ---------------------------------
            # ODOMETRY
            # ---------------------------------

            pose_data = odometry.get_pose()

            if pose_data is not None:

                pose = pose_data["pose"]

                print(
                    f"[ODOM] "
                    f"x={pose[0]:.2f} "
                    f"y={pose[1]:.2f} "
                    f"theta={pose[2]:.2f}"
                )

            # ---------------------------------
            # EXIT
            # ---------------------------------

            key = cv2.waitKey(1)

            if key == 27:  # ESC
                break

    except KeyboardInterrupt:

        pass

    finally:

        cv2.destroyAllWindows()

        lidar.destroy_node()
        camera.destroy_node()
        odometry.destroy_node()

        rclpy.shutdown()

        print("\n=== SENSOR TEST FINISHED ===")


if __name__ == "__main__":
    main()