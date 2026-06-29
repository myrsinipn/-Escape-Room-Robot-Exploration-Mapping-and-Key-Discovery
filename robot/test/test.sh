#!/bin/bash
# Run this on the robot to diagnose why the robot isn't moving.
# Usage: bash diagnose.sh

echo "=== Topic list ==="
ros2 topic list

echo ""
echo "=== /exploration_path publisher & subscriber count ==="
ros2 topic info /exploration_path

echo ""
echo "=== /cmd_vel publisher & subscriber count ==="
ros2 topic info /cmd_vel

echo ""
echo "=== Last 3 messages on /exploration_path (5s timeout) ==="
timeout 5 ros2 topic echo /exploration_path --once 2>/dev/null | head -20 || echo "(no message received)"

echo ""
echo "=== Last cmd_vel (5s timeout) ==="
timeout 5 ros2 topic echo /cmd_vel --once 2>/dev/null || echo "(no message received)"

echo ""
echo "=== Node list ==="
ros2 node list