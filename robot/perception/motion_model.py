import numpy as np

from helpers import wrap_to_pi


class OmniMotionModel:
    """
    Omnidirectional robot motion model.

    State:
        mu = [x, y, theta]

    Control input:
        u = [vx, vy, omega]

    Supports:
    - body-frame velocities
    - world-frame velocities

    Returns:
    - predicted state
    - motion Jacobian
    """

    def __init__(self) -> None:

        pass

    def predict(
        self,
        mu: np.ndarray,
        vx: float,
        vy: float,
        omega: float,
        dt: float,
        velocities_in_body_frame: bool = True,
    ):
        """
        EKF-compatible motion model.

        Parameters
        ----------
        mu : np.ndarray
            Current robot state [x, y, theta]

        vx : float
            Linear x velocity

        vy : float
            Linear y velocity

        omega : float
            Angular velocity

        dt : float
            Time step

        velocities_in_body_frame : bool
            If True:
                velocities are expressed in robot frame

            If False:
                velocities are expressed in world frame

        Returns
        -------
        mu_pred : np.ndarray
            Predicted state

        F : np.ndarray
            Motion Jacobian
        """

        x, y, theta = mu

        # =========================
        # BODY FRAME VELOCITIES
        # =========================

        if velocities_in_body_frame:

            c = np.cos(theta)
            s = np.sin(theta)

            # transform body velocity -> world frame
            vx_world = c * vx - s * vy
            vy_world = s * vx + c * vy

            x_new = x + vx_world * dt
            y_new = y + vy_world * dt

            theta_new = theta + omega * dt

            theta_new = wrap_to_pi(theta_new)

            mu_pred = np.array(
                [
                    x_new,
                    y_new,
                    theta_new,
                ],
                dtype=float,
            )

            # Jacobian wrt state
            F = np.array(
                [
                    [
                        1.0,
                        0.0,
                        (-s * vx - c * vy) * dt,
                    ],
                    [
                        0.0,
                        1.0,
                        (c * vx - s * vy) * dt,
                    ],
                    [
                        0.0,
                        0.0,
                        1.0,
                    ],
                ],
                dtype=float,
            )

        # =========================
        # WORLD FRAME VELOCITIES
        # =========================

        else:

            x_new = x + vx * dt
            y_new = y + vy * dt

            theta_new = theta + omega * dt

            theta_new = wrap_to_pi(theta_new)

            mu_pred = np.array(
                [
                    x_new,
                    y_new,
                    theta_new,
                ],
                dtype=float,
            )

            # linear model
            F = np.eye(3)

        return mu_pred, F