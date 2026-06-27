#!/bin/bash
#
# launch_rviz_robot.sh  —  Run this ON THE ROBOT (inside your SSH session).
#
# Run this BEFORE launch_rviz_laptop.sh on the laptop.
#
# Opens 3 tmux windows:
#   1. main.py            (SLAM: publishes /slam_pose, /slam_map, /slam_landmarks)
#   2. slam_pose_to_tf.py (converts /slam_pose -> TF map->base_footprint)
#   3. slam/tf debug      (optional: topic list + tf2_echo for checking)

SESSION="rviz_robot"
DOMAIN_ID=7
PROJECT_DIR="$HOME/B07/-Escape-Room-Robot-Exploration-Mapping-and-Key-Discovery"

# Clean start
tmux kill-session -t $SESSION 2>/dev/null

echo "Starting RViz robot-side pipeline on Domain: $DOMAIN_ID..."

# ---------------------------------------------------------
# STEP 1: MAIN.PY  (SLAM node)
# ---------------------------------------------------------
echo "[1/3] Launching main.py (SLAM)..."
tmux new-session -d -s $SESSION -n "1_main"
tmux send-keys -t $SESSION:1_main "cd $PROJECT_DIR" C-m
tmux send-keys -t $SESSION:1_main "source /opt/ros/galactic/setup.bash" C-m
tmux send-keys -t $SESSION:1_main "export ROS_DOMAIN_ID=$DOMAIN_ID" C-m
tmux send-keys -t $SESSION:1_main "python3 main.py" C-m

sleep 5
echo "      -> main.py launched!"

# ---------------------------------------------------------
# STEP 2: SLAM_POSE_TO_TF
# ---------------------------------------------------------
echo "[2/3] Launching slam_pose_to_tf.py..."
tmux new-window -t $SESSION -n "2_slam_pose_to_tf"
tmux send-keys -t $SESSION:2_slam_pose_to_tf "source /opt/ros/galactic/setup.bash" C-m
tmux send-keys -t $SESSION:2_slam_pose_to_tf "export ROS_DOMAIN_ID=$DOMAIN_ID" C-m
tmux send-keys -t $SESSION:2_slam_pose_to_tf "python3 ~/ros2_ws/src/slam_pose_to_tf.py" C-m

sleep 3
echo "      -> slam_pose_to_tf.py launched!"

# ---------------------------------------------------------
# STEP 3: SLAM / TF DEBUG  (optional — useful for verifying)
# ---------------------------------------------------------
echo "[3/3] Launching slam/tf debug terminal (optional)..."
tmux new-window -t $SESSION -n "3_slam_debug"
tmux send-keys -t $SESSION:3_slam_debug "source /opt/ros/galactic/setup.bash" C-m
tmux send-keys -t $SESSION:3_slam_debug "export ROS_DOMAIN_ID=$DOMAIN_ID" C-m
tmux send-keys -t $SESSION:3_slam_debug "echo '--- SLAM topics ---'" C-m
tmux send-keys -t $SESSION:3_slam_debug "ros2 topic list | grep slam" C-m
tmux send-keys -t $SESSION:3_slam_debug "echo '--- TF map->base_footprint ---'" C-m
tmux send-keys -t $SESSION:3_slam_debug "ros2 run tf2_ros tf2_echo map base_footprint" C-m

echo ""
echo "All robot-side terminals launched!"
echo ""
echo "Now run launch_rviz_laptop.sh on the laptop."
echo ""
echo "Ctrl+b then 1/2/3 to switch between tmux windows."

tmux attach-session -t $SESSION