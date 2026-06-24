#!/usr/bin/env python3
import json
import time
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.ion()
fig, ax = plt.subplots(figsize=(10, 10))

trail_x = []
trail_y = []

while True:
    try:
        # ── Load RRT debug data ──────────────────────────────────────
        with open("/tmp/rrt_debug.json", "r") as f:
            data = json.load(f)

        ax.clear()

        # ── Draw tree ────────────────────────────────────────────────
        for edge in data.get("tree", []):
            ax.plot(
                [edge["x1"], edge["x2"]],
                [edge["y1"], edge["y2"]],
                color="lightgray",
                linewidth=0.7,
                zorder=1,
            )

        # ── Draw path ────────────────────────────────────────────────
        path = data.get("path", [])
        if len(path) > 0:
            px = [p[0] for p in path]
            py = [p[1] for p in path]
            ax.plot(
                px,
                py,
                color="gold",
                linewidth=3,
                zorder=3,
                label="Planned path",
            )
            # Mark each waypoint
            ax.scatter(px, py, color="orange", s=30, zorder=4)

        # ── Draw goal ────────────────────────────────────────────────
        goal = data.get("goal")
        if goal is not None:
            ax.scatter(
                goal["x"],
                goal["y"],
                s=200,
                color="red",
                marker="*",
                zorder=5,
                label=f"Goal ({goal['x']:.2f}, {goal['y']:.2f})",
            )

        # ── Draw target (optional) ───────────────────────────────────
        target = data.get("target")
        if target:
            ax.scatter(
                target["x"],
                target["y"],
                s=150,
                color="purple",
                marker="x",
                zorder=5,
                label="Target",
            )

        # ── Live robot pose (20 Hz from SLAM) ────────────────────────
        robot = None
        try:
            with open("/tmp/slam_pose.json", "r") as f:
                robot = json.load(f)
            trail_x.append(robot["x"])
            trail_y.append(robot["y"])
        except Exception:
            robot = data.get("robot")   # fallback to rrt_debug snapshot

        if robot is not None:
            rx = robot["x"]
            ry = robot["y"]
            th = robot["theta"]

            # Trail
            if len(trail_x) > 1:
                ax.plot(
                    trail_x,
                    trail_y,
                    "b--",
                    linewidth=1,
                    alpha=0.5,
                    zorder=2,
                    label="Robot trail",
                )

            # Robot dot
            ax.scatter(
                rx, ry,
                marker="o",
                s=120,
                color="blue",
                zorder=6,
                label=f"Robot ({rx:.2f}, {ry:.2f})",
            )

            # Heading arrow
            ax.arrow(
                rx, ry,
                0.3 * math.cos(th),
                0.3 * math.sin(th),
                head_width=0.07,
                head_length=0.05,
                fc="blue",
                ec="blue",
                zorder=6,
            )

            # Pose text
            ax.text(
                rx + 0.1,
                ry + 0.1,
                f"({rx:.2f}, {ry:.2f})\n{math.degrees(th):.0f}°",
                fontsize=7,
                color="blue",
                zorder=7,
            )

        # ── Stats overlay ────────────────────────────────────────────
        tree_size   = len(data.get("tree", []))
        path_len    = len(path)
        stats_text  = f"Tree edges: {tree_size}  |  Path pts: {path_len}"
        ax.set_title(stats_text, fontsize=10)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.axis("equal")
        ax.grid(True, linewidth=0.4, alpha=0.4)
        ax.legend(loc="upper left", fontsize=8)

        plt.tight_layout()
        plt.draw()
        plt.pause(0.05)

    except FileNotFoundError:
        # rrt_debug.json not written yet — just wait
        ax.clear()
        ax.set_title("Waiting for /tmp/rrt_debug.json ...")
        plt.draw()
        plt.pause(0.5)

    except Exception as e:
        print("ERROR:", e)
        time.sleep(0.5)