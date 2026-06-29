#!/usr/bin/env python3
"""
Live debug visualizer for RRT exploration.

Shows:
  - Blue dot + arrow  : robot current position and heading (from /slam_pose)
  - Blue trail        : actual trajectory the robot has driven
  - Orange line/dots  : planned RRT path that should be followed (/exploration_path)
  - Red star          : current goal (last waypoint in planned path)

Run:
    ROS_DOMAIN_ID=7 python3 debug_viz.py

Requires:  rclpy, matplotlib, numpy
"""

import math
import sys
import threading

import matplotlib
matplotlib.use("TkAgg")          # change to Qt5Agg / Agg if TkAgg is unavailable
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, PointStamped
from nav_msgs.msg import Path


# ── colours ────────────────────────────────────────────────────────────────
C_ROBOT    = "#1f77b4"   # blue       — robot dot + heading arrow
C_TRAJ     = "#aec7e8"   # light blue — actual trajectory
C_PATH     = "#ff7f0e"   # orange     — planned path
C_WP       = "#d62728"   # red        — waypoint dots on planned path
C_GOAL     = "#d62728"   # red        — goal star
C_START    = "#2ca02c"   # green      — starting point of planned path
C_CURWP    = "#00bcd4"   # cyan       — current target waypoint


class DebugViz(Node):
    """ROS2 node — collects pose + path data for the live plot."""

    MAX_TRAIL = 2000   # keep last N robot positions

    def __init__(self):
        super().__init__("debug_viz")

        self._lock = threading.Lock()

        # Robot pose
        self._rx: float | None = None
        self._ry: float | None = None
        self._rth: float | None = None

        # Trajectory history
        self._trail_x: list[float] = []
        self._trail_y: list[float] = []

        # Planned path
        self._path_x: list[float] = []
        self._path_y: list[float] = []

        # Current target waypoint
        self._cur_wp_x: float | None = None
        self._cur_wp_y: float | None = None

        self.create_subscription(
            PoseWithCovarianceStamped,
            "/slam_pose",
            self._pose_cb,
            10,
        )
        self.create_subscription(
            Path,
            "/exploration_path",
            self._path_cb,
            10,
        )
        self.create_subscription(
            PointStamped,
            "/exploration_current_wp",
            self._cur_wp_cb,
            10,
        )

        self.get_logger().info(
            "debug_viz ready — waiting for /slam_pose, /exploration_path, /exploration_current_wp"
        )

    # ── callbacks ──────────────────────────────────────────────────────────

    def _pose_cb(self, msg: PoseWithCovarianceStamped) -> None:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        theta = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        with self._lock:
            self._rx, self._ry, self._rth = x, y, theta
            self._trail_x.append(x)
            self._trail_y.append(y)
            if len(self._trail_x) > self.MAX_TRAIL:
                self._trail_x = self._trail_x[-self.MAX_TRAIL :]
                self._trail_y = self._trail_y[-self.MAX_TRAIL :]

    def _path_cb(self, msg: Path) -> None:
        xs = [p.pose.position.x for p in msg.poses]
        ys = [p.pose.position.y for p in msg.poses]
        with self._lock:
            self._path_x = xs
            self._path_y = ys

    def _cur_wp_cb(self, msg: PointStamped) -> None:
        with self._lock:
            self._cur_wp_x = msg.point.x
            self._cur_wp_y = msg.point.y

    # ── data accessor (called from main thread) ────────────────────────────

    def snapshot(self):
        with self._lock:
            return (
                list(self._trail_x),
                list(self._trail_y),
                self._rx,
                self._ry,
                self._rth,
                list(self._path_x),
                list(self._path_y),
                self._cur_wp_x,
                self._cur_wp_y,
            )


# ── matplotlib setup ───────────────────────────────────────────────────────

def build_figure():
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_title(
        "RRT Debug  |  Trail (blue) vs Planned path (orange)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("X  [m]")
    ax.set_ylabel("Y  [m]")

    # --- static legend proxies ---
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color=C_TRAJ,  lw=2,  label="Actual trajectory"),
        Line2D([0], [0], marker="o", color=C_ROBOT, lw=0, markersize=10,
               label="Robot position"),
        Line2D([0], [0], color=C_PATH,  lw=2,  label="Planned path"),
        Line2D([0], [0], marker="o", color=C_WP,   lw=0, markersize=7,
               label="Planned waypoints"),
        Line2D([0], [0], marker="P", color=C_CURWP, lw=0, markersize=13,
               label="Current target wp (cyan)"),
        Line2D([0], [0], marker="*", color=C_GOAL, lw=0, markersize=14,
               label="Goal"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", fontsize=9)

    return fig, ax


def main():
    rclpy.init()
    node = DebugViz()

    # ROS spin in a daemon thread so the plot stays responsive
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True
    )
    spin_thread.start()

    fig, ax = build_figure()

    # ── artist objects (updated every frame) ──────────────────────────────
    (trail_line,)   = ax.plot([], [], "-",  color=C_TRAJ,  lw=1.5, alpha=0.7)
    (robot_dot,)    = ax.plot([], [], "o",  color=C_ROBOT, ms=10,  zorder=5)
    quiv            = ax.quiver([], [], [], [], color=C_ROBOT, scale=5,
                                width=0.006, zorder=6)   # heading arrow
    (path_line,)    = ax.plot([], [], "-",  color=C_PATH,  lw=2)
    (wp_dots,)      = ax.plot([], [], "o",  color=C_WP,    ms=7,   zorder=4)
    (cur_wp_dot,)   = ax.plot([], [], "P",  color=C_CURWP, ms=16,  zorder=7,
                               markeredgecolor="white", markeredgewidth=1.5)
    (goal_star,)    = ax.plot([], [], "*",  color=C_GOAL,  ms=16,  zorder=5)
    (path_start,)   = ax.plot([], [], "s",  color=C_START, ms=8,   zorder=4)

    info_box = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes,
        va="top", fontsize=8, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                  edgecolor="gray", alpha=0.85),
    )

    # ── update function ────────────────────────────────────────────────────

    def update(_frame):
        trail_x, trail_y, rx, ry, rth, path_x, path_y, cwx, cwy = node.snapshot()

        # --- actual trajectory ---
        if len(trail_x) >= 2:
            trail_line.set_data(trail_x, trail_y)
        else:
            trail_line.set_data([], [])

        # --- robot dot + heading arrow ---
        if rx is not None:
            robot_dot.set_data([rx], [ry])
            arrow_len = 0.18
            dx = arrow_len * math.cos(rth)
            dy = arrow_len * math.sin(rth)
            quiv.set_offsets([[rx, ry]])
            quiv.set_UVC([dx], [dy])
        else:
            robot_dot.set_data([], [])

        # --- planned path ---
        if len(path_x) >= 2:
            path_line.set_data(path_x, path_y)
            wp_dots.set_data(path_x[1:-1], path_y[1:-1])  # intermediate waypoints
            goal_star.set_data([path_x[-1]], [path_y[-1]])
            path_start.set_data([path_x[0]], [path_y[0]])
        else:
            path_line.set_data([], [])
            wp_dots.set_data([], [])
            goal_star.set_data([], [])
            path_start.set_data([], [])

        # --- current target waypoint (cyan cross) ---
        if cwx is not None:
            cur_wp_dot.set_data([cwx], [cwy])
        else:
            cur_wp_dot.set_data([], [])

        # --- auto-zoom (keep robot + path visible with margin) ---
        all_x = trail_x + path_x + ([rx] if rx is not None else [])
        all_y = trail_y + path_y + ([ry] if ry is not None else [])
        if all_x and all_y:
            margin = 0.8
            cx = (min(all_x) + max(all_x)) / 2
            cy = (min(all_y) + max(all_y)) / 2
            half = max((max(all_x) - min(all_x)) / 2,
                       (max(all_y) - min(all_y)) / 2,
                       1.0) + margin
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)

        # --- info text ---
        lines = [
            f"Trail pts : {len(trail_x)}",
            f"Path wps  : {len(path_x)}",
        ]
        if rx is not None:
            lines.append(f"Robot     : ({rx:+.2f}, {ry:+.2f})")
            lines.append(f"Heading   : {math.degrees(rth):+.0f}°")
        if cwx is not None and rx is not None:
            d_wp = math.hypot(cwx - rx, cwy - ry)
            bearing = math.degrees(math.atan2(cwy - ry, cwx - rx))
            lines.append(f"Cur WP    : ({cwx:+.2f}, {cwy:+.2f})")
            lines.append(f"Dist WP   : {d_wp:.2f} m  bear={bearing:.0f}°")
        if path_x and rx is not None:
            dx = path_x[-1] - rx
            dy = path_y[-1] - ry
            lines.append(f"Dist goal : {math.hypot(dx, dy):.2f} m")
        info_box.set_text("\n".join(lines))

        return (trail_line, robot_dot, path_line, wp_dots,
                cur_wp_dot, goal_star, path_start, info_box)

    ani = animation.FuncAnimation(
        fig, update, interval=200, blit=False, cache_frame_data=False
    )

    plt.tight_layout()
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
