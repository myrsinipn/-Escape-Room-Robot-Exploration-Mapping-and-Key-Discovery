from typing import List, Dict, Any, Optional

import cv2
import numpy as np


class ArucoDetector:
    """
    ArUco marker detector.

    Responsibilities:
    - Detect ArUco markers
    - Estimate marker poses
    - Return clean semantic detections


    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        distortion_coeffs: np.ndarray,
        marker_size: float,
        dictionary_name: int = cv2.aruco.DICT_4X4_50,
    ) -> None:
        """
        Parameters
        ----------
        camera_matrix : np.ndarray
            Camera intrinsic matrix.

        distortion_coeffs : np.ndarray
            Camera distortion coefficients.

        marker_size : float
            Marker size in meters.

        dictionary_name : int
            OpenCV ArUco dictionary.
        """

        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs

        self.marker_size = marker_size

        # aruco dictionary
        self.dictionary = cv2.aruco.getPredefinedDictionary(
            dictionary_name
        )

        # detector parameters
        self.detector_params = (
            cv2.aruco.DetectorParameters()
        )

        # detector object
        self.detector = cv2.aruco.ArucoDetector(
            self.dictionary,
            self.detector_params,
        )

    def detect(
        self,
        frame: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Detect ArUco markers in image.

        Returns:
        [
            {
                "marker_id": int,
                "corners": np.ndarray,
                "rvec": np.ndarray,
                "tvec": np.ndarray,
                "T_cm": np.ndarray,
            }
        ]
        """

        detections = []

        # grayscale image
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        # detect markers
        corners, ids, _ = self.detector.detectMarkers(
            gray
        )

        if ids is None:
            return detections

        # estimate pose
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            self.marker_size,
            self.camera_matrix,
            self.distortion_coeffs,
        )

        for i, marker_id in enumerate(ids.flatten()):

            rvec = rvecs[i].flatten()
            tvec = tvecs[i].flatten()

            T_cm = self.create_transformation_matrix(
                rvec,
                tvec,
            )

            detections.append(
                {
                    "marker_id": int(marker_id),
                    "corners": corners[i],
                    "rvec": rvec,
                    "tvec": tvec,
                    "T_cm": T_cm,
                }
            )

        return detections

    def create_transformation_matrix(
        self,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> np.ndarray:
        """
        Creates homogeneous transformation matrix.

        T_cm:
        transform from marker frame -> camera frame
        """

        R_cm, _ = cv2.Rodrigues(rvec)

        T_cm = np.eye(4)

        T_cm[:3, :3] = R_cm
        T_cm[:3, 3] = tvec

        return T_cm

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Draw markers and axes on frame.
        """

        output = frame.copy()

        for detection in detections:

            corners = detection["corners"]
            rvec = detection["rvec"]
            tvec = detection["tvec"]
            marker_id = detection["marker_id"]

            # draw marker borders
            cv2.aruco.drawDetectedMarkers(
                output,
                [corners],
                np.array([[marker_id]]),
            )

            # draw coordinate axes
            cv2.drawFrameAxes(
                output,
                self.camera_matrix,
                self.distortion_coeffs,
                rvec,
                tvec,
                self.marker_size * 0.5,
            )

        return output