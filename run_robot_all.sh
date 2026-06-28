#!/bin/bash

SESSION="robot_ros"
MYAGV_WS="/home/er/myagv_ros"
BRIDGE_YAML="/home/er/B07/-Escape-Room-Robot-Exploration-Mapping-and-Key-Discovery/bridge.yaml"
DOMAIN_ID=7

# We must source ROS 1 in the main script so it can check if the nodes are done booting
source /opt/ros/noetic/setup.bash

# Clean start
tmux kill-session -t $SESSION 2>/dev/null

echo "Starting ROS system strictly step-by-step on Domain: $DOMAIN_ID..."

# ---------------------------------------------------------
# STEP 1: ROSCORE
# ---------------------------------------------------------
echo "[1/5] Launching roscore..."
tmux new-session -d -s $SESSION -n "1_roscore"
tmux send-keys -t $SESSION:1_roscore "export ROS_DOMAIN_ID=$DOMAIN_ID" C-m
tmux send-keys -t $SESSION:1_roscore "source /opt/ros/noetic/setup.bash" C-m
tmux send-keys -t $SESSION:1_roscore "roscore" C-m

# The script pauses here and waits for roscore to appear
while ! rostopic list > /dev/null 2>&1; do 
    sleep 1 
done
echo "      -> roscore is ready!"

# ---------------------------------------------------------
# STEP 2: MYAGV ACTIVE (ODOMETRY)
# ---------------------------------------------------------
echo "[2/5] Launching AGV Base..."
tmux new-window -t $SESSION -n "2_myagv_active"
tmux send-keys -t $SESSION:2_myagv_active "export ROS_DOMAIN_ID=$DOMAIN_ID" C-m
tmux send-keys -t $SESSION:2_myagv_active "cd $MYAGV_WS/src/myagv_odometry/scripts" C-m
tmux send-keys -t $SESSION:2_myagv_active "./start_ydlidar.sh &" C-m
tmux send-keys -t $SESSION:2_myagv_active "sleep 2" C-m
tmux send-keys -t $SESSION:2_myagv_active "source $MYAGV_WS/devel/setup.bash" C-m
tmux send-keys -t $SESSION:2_myagv_active "roslaunch myagv_odometry myagv_active.launch" C-m

# The script pauses here and waits for the odometry topic to appear
while ! rostopic list | grep -q '/odom'; do 
    sleep 1 
done
echo "      -> AGV Base is ready!"

# ---------------------------------------------------------
# STEP 3: YDLIDAR
# ---------------------------------------------------------
echo "[3/5] Launching Lidar..."
tmux new-window -t $SESSION -n "3_ydlidar"
tmux send-keys -t $SESSION:3_ydlidar "export ROS_DOMAIN_ID=$DOMAIN_ID" C-m
tmux send-keys -t $SESSION:3_ydlidar "source $MYAGV_WS/devel/setup.bash" C-m
tmux send-keys -t $SESSION:3_ydlidar "roslaunch ydlidar_ros_driver X2.launch" C-m

# The script pauses here and waits for the Lidar to physically scan
while ! rostopic list | grep -q '/scan'; do 
    sleep 1 
done
echo "      -> Lidar is ready!"

# ---------------------------------------------------------
# STEP 4: ROS 1 - ROS 2 BRIDGE
# ---------------------------------------------------------
echo "[4/5] Launching Parameter Bridge..."
tmux new-window -t $SESSION -n "4_bridge"
tmux send-keys -t $SESSION:4_bridge "export ROS_DOMAIN_ID=$DOMAIN_ID" C-m
tmux send-keys -t $SESSION:4_bridge "source /opt/ros/noetic/setup.bash" C-m
tmux send-keys -t $SESSION:4_bridge "source $MYAGV_WS/devel/setup.bash" C-m
tmux send-keys -t $SESSION:4_bridge "rosparam load $BRIDGE_YAML" C-m
tmux send-keys -t $SESSION:4_bridge "source /opt/ros/galactic/setup.bash" C-m
tmux send-keys -t $SESSION:4_bridge "export ROS_DOMAIN_ID=$DOMAIN_ID" C-m
tmux send-keys -t $SESSION:4_bridge "ros2 run ros1_bridge parameter_bridge" C-m

# Give the bridge a mandatory 3 seconds to map all the topics together
sleep 3
echo "      -> Bridge initialized!"

# ---------------------------------------------------------
# STEP 5: CAMERA
# ---------------------------------------------------------
echo "[5/5] Launching Camera..."
tmux new-window -t $SESSION -n "5_camera"
tmux send-keys -t $SESSION:5_camera "export ROS_DOMAIN_ID=$DOMAIN_ID" C-m
tmux send-keys -t $SESSION:5_camera "source /opt/ros/galactic/setup.bash" C-m
tmux send-keys -t $SESSION:5_camera "ros2 run v4l2_camera v4l2_camera_node --ros-args -p video_device:=/dev/video0" C-m

echo "All systems launched sequentially!"

# Attach to tmux so you can monitor all panes
tmux attach-session -t $SESSION