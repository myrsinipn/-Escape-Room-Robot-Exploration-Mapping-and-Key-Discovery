# Escape Room Robot

## Key and door configuration

Door/key IDs are configured in `config/key_doors.json`. Each door is formed by
exactly two ArUco markers, and each key ArUco maps to a known door ID:

```json
{
  "door_marker_pairs": [
    {"door_id": 1, "markers": [5, 6]}
  ],
  "key_to_door": {"10": 1},
  "confirmation_frames": 3
}
```

The default therefore means ArUcos 5 and 6 form locked door 1, while ArUco 10
is its key. Change these IDs to match the physical deployment before running.

At runtime the camera must confirm an ID in distinct frames. Once both door
markers have been localized, their connecting segment is added to the RRT
costmap as a locked virtual wall. Detecting the matching key removes that wall,
interrupts frontier exploration, and sends the robot to the already-mapped door
center. Frontier exploration resumes when the robot reaches the door.
