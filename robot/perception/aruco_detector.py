from typing import List, Dict, Any, Optional

import cv2
import numpy as np


class ArucoDetector:
    """Detects ArUco markers in a camera frame and estimates their 3-D poses.

    For each detected marker the detector returns:
      - marker_id  : integer ID encoded in the marker pattern
      - corners    : 2-D pixel corners of the marker in the image
      - rvec       : rotation vector (Rodrigues) from camera to marker frame
      - tvec       : translation vector (metres) from camera to marker centre
      - T_cm       : 4×4 homogeneous transform from marker frame to camera frame
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
            3×3 camera intrinsic matrix (from calibration).
        distortion_coeffs : np.ndarray
            Distortion coefficients (from calibration).
        marker_size : float
            Physical side length of the marker in metres.
        dictionary_name : int
            OpenCV ArUco dictionary constant (default: DICT_4X4_50).
        """
        self.camera_matrix    = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        self.marker_size      = marker_size

        self.dictionary       = cv2.aruco.getPredefinedDictionary(dictionary_name)
        self.detector_params  = cv2.aruco.DetectorParameters()
        self.detector         = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect all ArUco markers in ``frame`` and estimate their poses.

        Returns a list of detection dicts (one per marker); empty if none found.
        """
        detections = []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is None:
            return detections

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            self.marker_size,
            self.camera_matrix,
            self.distortion_coeffs,
        )

        for i, marker_id in enumerate(ids.flatten()):
            rvec = rvecs[i].flatten()
            tvec = tvecs[i].flatten()
            detections.append(
                {
                    "marker_id": int(marker_id),
                    "corners":   corners[i],
                    "rvec":      rvec,
                    "tvec":      tvec,
                    "T_cm":      self.create_transformation_matrix(rvec, tvec),
                }
            )

        return detections

    def create_transformation_matrix(
        self,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> np.ndarray:
        """Build the 4×4 homogeneous transform T_cm (marker frame → camera frame)."""
        R_cm, _ = cv2.Rodrigues(rvec)
        T_cm = np.eye(4)
        T_cm[:3, :3] = R_cm
        T_cm[:3, 3]  = tvec
        return T_cm

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Overlay marker borders and coordinate axes on a copy of ``frame``."""
        output = frame.copy()
        for detection in detections:
            corners   = detection["corners"]
            rvec      = detection["rvec"]
            tvec      = detection["tvec"]
            marker_id = detection["marker_id"]

            cv2.aruco.drawDetectedMarkers(output, [corners], np.array([[marker_id]]))
            cv2.drawFrameAxes(
                output,
                self.camera_matrix,
                self.distortion_coeffs,
                rvec,
                tvec,
                self.marker_size * 0.5,
            )

        return output