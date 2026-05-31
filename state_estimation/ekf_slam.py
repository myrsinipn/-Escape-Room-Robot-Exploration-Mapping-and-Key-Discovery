from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import numpy.typing as npt
import warnings

from helpers import (
    DEFAULT_MAX_MARKERS,
    LM_DIM,
    ROBOT_DIM,
    lm_slice,
    pose2d_to_Twb,
    wrap_to_pi,
)

ArrayF = npt.NDArray[np.float64]
ArrayB = npt.NDArray[np.bool_]

MotionModelCallback = Callable[
    [ArrayF, float, float, float, float, bool],
    Tuple[ArrayF, ArrayF]
]

# (marker_id, T_cm)
Detection = Tuple[int, ArrayF]


class EKFSLAM:
    """
    EKF-SLAM for omnidirectional robot + planar landmarks.

    Robot state:
        [x, y, theta]

    Landmark state:
        [mx, my]

    Full state:
        [robot_state, landmark_0, landmark_1, ...]

    Measurements:
        z = [range, bearing]
    """

    MIN_RANGE = 1e-6

    def __init__(
        self,
        max_markers: int = DEFAULT_MAX_MARKERS,
        T_bc: Optional[npt.ArrayLike] = None,
        mu: Optional[npt.ArrayLike] = None,
        Sigma: Optional[npt.ArrayLike] = None,
        marker_seen: Optional[npt.ArrayLike] = None,
        R: Optional[npt.ArrayLike] = None,
        Q: Optional[npt.ArrayLike] = None,
        motion_model_callback: Optional[MotionModelCallback] = None,
    ) -> None:

        self.max_markers = int(max_markers)

        self.state_dim = (
            ROBOT_DIM
            + self.max_markers * LM_DIM
        )

        # camera pose wrt robot base
        self.T_bc = (
            np.eye(4)
            if T_bc is None
            else np.array(T_bc, dtype=float).reshape(4, 4)
        )

        # state mean
        self._mu = (
            np.zeros(self.state_dim, dtype=float)
            if mu is None
            else np.array(mu, dtype=float).reshape(self.state_dim)
        )

        self._mu[2] = wrap_to_pi(self._mu[2])

        # covariance
        if Sigma is None:

            Sigma = np.eye(
                self.state_dim,
                dtype=float,
            ) * 1e-3

            # unknown landmarks
            Sigma[ROBOT_DIM:, ROBOT_DIM:] = (
                np.eye(
                    self.state_dim - ROBOT_DIM,
                    dtype=float,
                ) * 1e6
            )

        self._Sigma = np.array(
            Sigma,
            dtype=float,
        ).reshape(self.state_dim, self.state_dim)

        # landmark observation flags
        self._marker_seen = (
            np.zeros(self.max_markers, dtype=bool)
            if marker_seen is None
            else np.array(
                marker_seen,
                dtype=bool,
            ).reshape(self.max_markers)
        )

        # process noise
        self._R = (
            np.diag([
                0.03**2,
                0.03**2,
                np.deg2rad(2.0)**2,
            ])
            if R is None
            else np.array(R, dtype=float).reshape(3, 3)
        )

        # measurement noise
        self._Q = (
            np.diag([
                0.05**2,
                np.deg2rad(3.0)**2,
            ])
            if Q is None
            else np.array(Q, dtype=float).reshape(2, 2)
        )

        if motion_model_callback is None:

            raise ValueError(
                "motion_model_callback is required."
            )

        self._motion_model_callback = motion_model_callback

    @property
    def mu(self) -> ArrayF:
        return self._mu.copy()

    @property
    def Sigma(self) -> ArrayF:
        return self._Sigma.copy()

    @property
    def marker_seen(self) -> ArrayB:
        return self._marker_seen.copy()

    def landmark_slice(
        self,
        marker_index: int,
    ) -> slice:

        return lm_slice(
            marker_index,
            robot_dim=ROBOT_DIM,
        )

    @staticmethod
    def measurement_body_from_T_cm(
        T_bc: npt.ArrayLike,
        T_cm_meas: npt.ArrayLike,
    ) -> ArrayF:
        """
        Converts camera transform -> body range/bearing.
        """

        pb = (
            T_bc
            @ T_cm_meas
            @ np.array([0.0, 0.0, 0.0, 1.0])
        )

        px = float(pb[0])
        py = float(pb[1])

        r = np.hypot(px, py)

        phi = np.arctan2(py, px)

        return np.array([r, phi])

    def predicted_range_bearing(
        self,
        mu: npt.ArrayLike,
        marker_index: int,
    ) -> ArrayF:
        """
        Predicted measurement h(mu).
        """

        mu = np.array(mu).reshape(self.state_dim)

        x_r = mu[0]
        y_r = mu[1]
        theta_r = mu[2]

        sl = self.landmark_slice(marker_index)

        mx = mu[sl.start]
        my = mu[sl.start + 1]

        dx = mx - x_r
        dy = my - y_r

        r = np.hypot(dx, dy)

        phi = np.arctan2(dy, dx) - theta_r

        phi = wrap_to_pi(phi)

        return np.array([r, phi])

    def measurement_residual(
        self,
        mu: npt.ArrayLike,
        marker_index: int,
        T_cm_meas: npt.ArrayLike,
    ) -> ArrayF:
        """
        Innovation:
            y = z - h(mu)
        """

        z = self.measurement_body_from_T_cm(
            self.T_bc,
            T_cm_meas,
        )

        h = self.predicted_range_bearing(
            mu,
            marker_index,
        )

        y = z - h

        y[1] = wrap_to_pi(y[1])

        return y

    def analytic_measurement_jacobian(
        self,
        mu: npt.ArrayLike,
        marker_index: int,
    ) -> ArrayF:
        """
        Measurement Jacobian H.
        """

        mu = np.array(mu).reshape(self.state_dim)

        x_r = mu[0]
        y_r = mu[1]

        sl = self.landmark_slice(marker_index)

        mx = mu[sl.start]
        my = mu[sl.start + 1]

        dx = mx - x_r
        dy = my - y_r

        q = dx**2 + dy**2

        q = max(q, self.MIN_RANGE)

        sqrt_q = np.sqrt(q)

        H = np.zeros((2, self.state_dim))

        # range
        H[0, 0] = -dx / sqrt_q
        H[0, 1] = -dy / sqrt_q

        H[0, sl.start] = dx / sqrt_q
        H[0, sl.start + 1] = dy / sqrt_q

        # bearing
        H[1, 0] = dy / q
        H[1, 1] = -dx / q
        H[1, 2] = -1.0

        H[1, sl.start] = -dy / q
        H[1, sl.start + 1] = dx / q

        return H

    def initialize_landmark(
        self,
        mu: npt.ArrayLike,
        Sigma: npt.ArrayLike,
        marker_index: int,
        T_cm_meas: npt.ArrayLike,
    ):
        """
        Initialize landmark in world frame.
        """

        T_wb = pose2d_to_Twb(
            mu[0],
            mu[1],
            mu[2],
        )

        T_wc = T_wb @ self.T_bc

        T_wm = T_wc @ T_cm_meas

        mx = T_wm[0, 3]
        my = T_wm[1, 3]

        mu_landmark = np.array([mx, my])

        # simple initialization covariance
        Sigma_landmark = np.diag([
            0.05**2,
            0.05**2,
        ])

        return mu_landmark, Sigma_landmark

    def prediction_step(
        self,
        u_body: np.ndarray,
        dt: float,
    ) -> None:
        """
        EKF prediction.
        """

        vx, vy, omega = np.array(
            u_body,
            dtype=float,
        ).reshape(3)

        mu_robot_pred, F_r = (
            self._motion_model_callback(
                mu=self._mu[:3],
                vx=vx,
                vy=vy,
                omega=omega,
                dt=dt,
                velocities_in_body_frame=True,
            )
        )

        self._mu[:3] = mu_robot_pred

        # full Jacobian
        F = np.eye(self.state_dim)

        F[:3, :3] = F_r

        # process noise
        R_full = np.zeros(
            (self.state_dim, self.state_dim)
        )

        R_full[:3, :3] = self._R

        self._Sigma = (
            F
            @ self._Sigma
            @ F.T
            + R_full
        )

    def correction_step(
        self,
        detections: Iterable[Detection],
    ) -> None:
        """
        EKF correction using ArUco detections.
        """

        I = np.eye(self.state_dim)

        mu_bar = self._mu.copy()
        Sigma_bar = self._Sigma.copy()

        for marker_id, T_cm_meas in detections:

            # invalid marker id
            if (
                marker_id < 0
                or marker_id >= self.max_markers
            ):

                warnings.warn(
                    f"Ignoring invalid marker_id={marker_id}",
                    RuntimeWarning,
                )

                continue

            # first observation
            if not self._marker_seen[marker_id]:

                mu_landmark, Sigma_landmark = (
                    self.initialize_landmark(
                        mu_bar,
                        Sigma_bar,
                        marker_id,
                        T_cm_meas,
                    )
                )

                sl = self.landmark_slice(marker_id)

                mu_bar[sl] = mu_landmark

                Sigma_bar[sl, sl] = Sigma_landmark

                self._marker_seen[marker_id] = True

                continue

            # innovation
            y = self.measurement_residual(
                mu_bar,
                marker_id,
                T_cm_meas,
            )

            # Jacobian
            H = self.analytic_measurement_jacobian(
                mu_bar,
                marker_id,
            )

            # innovation covariance
            S = (
                H
                @ Sigma_bar
                @ H.T
                + self._Q
            )

            # Kalman gain
            K = (
                Sigma_bar
                @ H.T
                @ np.linalg.solve(
                    S,
                    np.eye(2),
                )
            )

            # state update
            mu_bar = mu_bar + K @ y

            mu_bar[2] = wrap_to_pi(mu_bar[2])

            # covariance update
            Sigma_bar = (
                I - K @ H
            ) @ Sigma_bar

        self._mu = mu_bar
        self._Sigma = Sigma_bar

    def step(
        self,
        u_body: npt.ArrayLike,
        detections: Iterable[Detection],
        dt: float,
    ) -> None:
        """
        Full EKF-SLAM step.
        """

        self.prediction_step(
            u_body,
            dt,
        )

        self.correction_step(
            detections,
        )