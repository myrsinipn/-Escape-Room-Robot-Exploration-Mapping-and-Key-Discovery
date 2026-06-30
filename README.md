# Escape Room Robot — Exploration, Mapping & Key Discovery

An autonomous robot system that explores an unknown escape-room environment, builds a live occupancy map, detects ArUco marker "keys", and navigates through locked "doors" as they are unlocked.

Built on an **ElephantRobotics myAGV Plus** omnidirectional robot running a hybrid **ROS 1 (Noetic) / ROS 2 (Galactic)** stack.

---

## How it works

The robot operates a continuous sense–plan–act loop:

1. **Map** — A log-odds occupancy grid is built in real time from YDLiDAR X2 scans, updated via Bresenham ray tracing.
2. **Localize** — An EKF-SLAM node fuses LiDAR corner features (extracted with Ramer–Douglas–Peucker segmentation) and wheel odometry to maintain a pose estimate.
3. **Explore** — An RRT planner selects frontier cells (boundaries between free and unknown space) and generates paths toward them. A reactive LiDAR controller executes those paths with real-time obstacle avoidance.
4. **Detect keys** — A camera node streams frames to an ArUco detector. Each confirmed marker is localized in the map frame and matched against the key/door configuration.
5. **Unlock doors** — Detecting a key marker removes the corresponding virtual wall from the RRT costmap, interrupts exploration, and drives the robot to the door center. Exploration resumes once the robot arrives.

---

## Repository layout

```
.
├── robot/                          # All code that runs on the robot
│   ├── main.py                     # Primary entry point (presentation mode)
│   ├── main2.py                    # Alternate entry point (full autonomous control)
│   ├── run_robot_all.sh            # tmux launcher: roscore → drivers → bridge → camera
│   ├── bridge.yaml                 # ros1_bridge topic configuration
│   │
│   ├── config/
│   │   ├── camera_calibration.json # Per-camera intrinsic parameters
│   │   └── key_doors.json          # ArUco key ↔ door mapping
│   │
│   ├── sensors/                    # Thin ROS 2 subscriber wrappers
│   │   ├── lidar.py                # YDLiDAR X2  →  /scan
│   │   ├── camera.py               # USB camera  →  /image_raw
│   │   └── odometry.py             # Wheel odometry  →  /odom
│   │
│   ├── perception/
│   │   ├── aruco_detector.py       # OpenCV ArUco detection + pose estimation
│   │   ├── door_localizer.py       # Projects marker detections into the map frame
│   │   ├── door_registry.py        # Tracks which marker pairs form each door
│   │   ├── motion_model.py         # Omnidirectional kinematic motion model
│   │   └── scan_preprocessor.py   # LiDAR scan filtering and smoothing
│   │
│   ├── state_estimation/
│   │   ├── ekf_slam.py             # EKF-SLAM (corner features + log-odds grid)
│   │   └── key_door_registry.py   # Runtime key/door unlock state
│   │
│   ├── mapping/
│   │   ├── rrt_exploration.py      # RRT frontier planner + waypoint follower
│   │   ├── rrt_exploration2.py     # Variant with costmap virtual walls
│   │   └── rrt_exploartion1.py     # Presentation-mode planner (path only, no motion)
│   │
│   ├── control/
│   │   ├── safe_lidar_motion.py    # Reactive LiDAR obstacle avoidance controller
│   │   └── aruco_monitor.py        # ArUco detection loop + key/door event publisher
│   │
│   ├── decision/
│   │   └── frontier_detector.py    # Identifies frontier cells from the occupancy grid
│   │
│   ├── slam_pose_to_tf.py          # Re-publishes SLAM pose as a TF transform
│   ├── slam_wheel_animator.py      # Publishes wheel joint states for RViz animation
│   ├── helpers.py                  # Shared math utilities
│   └── test/                       # Debug and visualisation scripts
│
└── laptop/
    └── myagv_plus_description/     # ROS 2 URDF package for RViz visualisation
        ├── urdf/                   # Robot URDF model
        ├── meshes/                 # 3-D meshes
        ├── launch/                 # Launch files
        └── rviz/                   # RViz configurations
```

---

## Hardware

| Component | Model |
|-----------|-------|
| Robot base | ElephantRobotics myAGV Plus (omnidirectional) |
| LiDAR | YDLiDAR X2 |
| Camera | v4l2_camera |
| On-board computer | Raspberry Pi (runs ROS 1 Noetic + ROS 2 Galactic) |

---

## Software requirements

| Requirement | Version |
|-------------|---------|
| ROS 1 | Noetic |
| ROS 2 | Galactic |
| ros1_bridge | Galactic |
| Python | 3.8+ |
| OpenCV | 4.x (with `aruco` contrib) |
| NumPy | any recent |
| SciPy | any recent |
| tmux | any |

---

## Configuration

### Key and door mapping — `robot/config/key_doors.json`

Each door is defined by a pair of ArUco marker IDs. Each key ArUco maps to a door ID.
Adjust these to match the physical deployment before running.

```json
{
  "door_marker_pairs": [
    {"door_id": 1, "markers": [5, 6]}
  ],
  "key_to_door": {"10": 1},
  "confirmation_frames": 3
}
```

- `door_marker_pairs` — two ArUco IDs that flank a physical door
- `key_to_door` — which key ArUco unlocks which door
- `confirmation_frames` — consecutive camera frames that must confirm a marker before it is accepted

At runtime, once both door markers are localized their connecting segment is added to the RRT costmap as a virtual wall. Detecting the matching key removes that wall, interrupts frontier exploration, and sends the robot to the already-mapped door center. Exploration resumes when the robot arrives.

### Camera calibration — `robot/config/camera_calibration.json`

Stores per-camera-device intrinsic matrix and distortion coefficients, keyed by camera ID (default `"11"`). Re-calibrate with a checkerboard if you change the camera.

### ROS bridge topics — `robot/bridge.yaml`

Lists every topic forwarded between ROS 1 and ROS 2 via `ros1_bridge`:

| Topic | Type |
|-------|------|
| `/odom` | `nav_msgs/Odometry` |
| `/scan` | `sensor_msgs/LaserScan` |
| `/cmd_vel` | `geometry_msgs/Twist` |
| `/imu` | `sensor_msgs/Imu` |
| `/tf`, `/tf_static` | `tf2_msgs/TFMessage` |

---

## Running on the robot

`run_robot_all.sh` orchestrates the full startup sequence inside a `tmux` session called `robot_ros`.
It waits for each service to become ready before starting the next.

```bash
cd /path/to/repo/robot
chmod +x run_robot_all.sh
./run_robot_all.sh
```

The script opens five tmux windows in order:

| Window | Service |
|--------|---------|
| `1_roscore` | ROS 1 core |
| `2_myagv_active` | myAGV odometry + YDLiDAR hardware driver |
| `3_ydlidar` | YDLiDAR ROS 1 driver (publishes `/scan`) |
| `4_bridge` | `ros1_bridge parameter_bridge` (ROS 1 ↔ ROS 2) |
| `5_camera` | `v4l2_camera_node` (publishes `/image_raw`) |

After all drivers are up, launch the robot brain in a new terminal:

```bash
source /opt/ros/galactic/setup.bash
export ROS_DOMAIN_ID=7
cd /path/to/repo/robot
python3 main.py        # presentation mode (SafeLidarMotion drives the robot)
# or
python3 main2.py       # full autonomous mode (RRTExplorer owns cmd_vel directly)
```

---

## Visualising on the laptop

Start these terminals in order — robot terminals first, then laptop terminals.

### Robot — Terminal 2 (TF bridge)

Run this after `run_robot_all.sh` and `main.py` are up:

```bash
source /opt/ros/galactic/setup.bash
export ROS_DOMAIN_ID=7
python3 ~/ros2_ws/src/slam_pose_to_tf.py
```

This re-publishes the SLAM pose as a `/tf` transform so RViz can place the robot model correctly.

---

### Laptop — Terminal 1 (robot_state_publisher)

Converts the URDF into live joint-state transforms:

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=7
xacro ~/ros2_ws/install/myagv_plus_description/share/myagv_plus_description/urdf/myagv_plus.urdf.xacro \
  > /tmp/myagv_plus.urdf
ros2 run robot_state_publisher robot_state_publisher \
  --ros-args -p robot_description:="$(cat /tmp/myagv_plus.urdf)"
```

### Laptop — Terminal 2 (RViz via Docker)

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=7

xhost +local:docker

docker run --rm -it \
  --network=host \
  --env DISPLAY=$DISPLAY \
  --volume /tmp/.X11-unix:/tmp/.X11-unix \
  --volume /tmp:/tmp \
  --volume $HOME/ros2_ws:/root/ros2_ws \
  -e ROS_DOMAIN_ID=7 \
  osrf/ros:galactic-desktop \
  bash -c "source /opt/ros/galactic/setup.bash && \
           source /root/ros2_ws/install/setup.bash && \
           rviz2"
```

Inside RViz: add a **RobotModel** display and set **Description File** to `/tmp/myagv_plus.urdf`.

### Laptop — Terminal 3 (wheel animator)

Publishes animated wheel joint states so the URDF model shows the wheels spinning:

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=7
export ROS_LOCALHOST_ONLY=0
python3 ~/ros2_ws/src/slam_wheel_animator.py
```

---

### Key RViz topics

| Topic | Type | Content |
|-------|------|---------|
| `/slam_map` | `OccupancyGrid` | Live log-odds occupancy grid |
| `/robot_pose` | `PoseWithCovarianceStamped` | SLAM pose estimate |
| `/exploration_path` | `Path` | Current RRT path |
| `/rrt_goal` | `Marker` | Active frontier goal |
| `/aruco/markers_viz` | `MarkerArray` | Detected doors/keys (orange spheres + labels) |

---

## Key algorithms

### EKF-SLAM (`state_estimation/ekf_slam.py`)
- Extracts corners from LiDAR scans using **Ramer–Douglas–Peucker** polyline simplification — distance-independent and robust to varying scan density.
- Each corner candidate is validated by checking arm length on both sides to reject laser noise spikes.
- Data association uses **Mahalanobis distance** (threshold 5.99); landmarks beyond 0.8 m from any known landmark are registered as new.
- Occupancy grid updates are paused while the robot spins in place to prevent wall-smearing artifacts.

### Log-odds occupancy mapping (`control/safe_lidar_motion.py`)
- Each LiDAR ray is traced with **Bresenham's line algorithm** over a 20 × 20 m grid at 5 cm resolution.
- Free cells accumulate `log_free = −0.35`; occupied endpoints accumulate `log_occ = +0.85`, clamped to `[log_min, log_max]`.
- Published as a standard `nav_msgs/OccupancyGrid`.

### RRT frontier exploration (`mapping/rrt_exploration2.py`)
- A **slow planner timer (3 s)** builds an RRT toward the nearest frontier cell using 30 % frontier-biased random sampling.
- A **fast control timer (20 Hz)** drives the robot through waypoints with a two-phase controller: pure pivot alignment (**TURN**) followed by forward + steering (**DRIVE**).
- Locked doors are injected as virtual wall segments in the costmap; detecting the matching key removes them and redirects the robot.
- Path collision checking uses an inflated robot radius (5 cells = 0.25 m clearance).

### ArUco key detection (`control/aruco_monitor.py`)
- Runs a detection loop against the latest camera frame using OpenCV's DICT_4X4_50 dictionary.
- A marker must appear in `confirmation_frames` **distinct** frames before it is accepted, preventing false positives from motion blur.
- Confirmed markers are localized in the map frame (accounting for a 0.13 m forward camera offset) and published as RViz `MarkerArray` with orange sphere + text label pairs.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Map smears during rotation | Spin-guard threshold not matched to actual angular velocity | Tune the angular velocity threshold in `ekf_slam.py` |
| ArUco markers never confirmed | `confirmation_frames` too high, or camera frame rate too low | Lower `confirmation_frames` in `key_doors.json` |
| Bridge topics not forwarding | `ROS_DOMAIN_ID` mismatch | Set the same domain ID (default `7`) in **all** terminals |
| Robot freezes near walls | LiDAR obstacle threshold too conservative | Adjust `safe_lidar_motion.py` stop-distance parameter |
