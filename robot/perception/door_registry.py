#!/usr/bin/env python3
"""
DoorRegistry: maps pairs of ArUco markers to doors and tracks
their world-frame positions as sightings accumulate.

Data relationships maintained:
  (markerA, markerB) → door_id   (two markers define one doorway)
  key_id             → door_id   (delegated to KeyDoorRegistry)
"""

from typing import Dict, Tuple
import numpy as np

from state_estimation.key_door_registry import KeyDoorRegistry


class DoorRegistry:
    """Manages door geometry and key–door unlock state.

    Door positions are computed from pairs of ArUco marker sightings.
    Each call to register_marker_position() updates an incremental mean
    so the position stabilises as more frames are processed.
    """

    def __init__(
        self,
        door_marker_pairs: Dict[Tuple[int, int], int],
        key_to_door: Dict[int, int],
    ):
        """
        Parameters
        ----------
        door_marker_pairs : dict
            Maps (markerA_id, markerB_id) → door_id.
            Marker order within each pair does not matter.
        key_to_door : dict
            Maps key_id → door_id (forwarded to KeyDoorRegistry).
        """
        self.registry = KeyDoorRegistry(key_to_door)

        # Normalise marker pairs so (a, b) and (b, a) map to the same door.
        self.marker_pair_to_door = {}
        for pair, door_id in door_marker_pairs.items():
            a, b = sorted(pair)
            self.marker_pair_to_door[(a, b)] = door_id

        self.marker_world_positions  = {}   # marker_id → (wx, wy)
        self._marker_observations    = {}   # marker_id → observation count

        self.door_world_positions    = {}   # door_id → {left, right, center, _blocked}
        self.key_world_positions     = {}   # key_id  → (wx, wy)

    def register_marker_position(self, marker_id: int, wx: float, wy: float) -> None:
        """Update the running mean position of a marker from a new sighting.

        Camera-pose estimates jitter frame to frame.  Accumulating an
        incremental mean prevents a door from jumping around the map every
        time a new detection is processed.
        """
        count = self._marker_observations.get(marker_id, 0)
        old   = np.asarray(
            self.marker_world_positions.get(marker_id, (wx, wy)),
            dtype=float,
        )
        new   = (old * count + np.asarray((wx, wy), dtype=float)) / (count + 1)

        self._marker_observations[marker_id]   = count + 1
        self.marker_world_positions[marker_id] = tuple(new)

        # Rebuild any door positions that depend on this marker.
        self._try_build_doors()

    def _try_build_doors(self) -> None:
        """Recompute door geometry for any pair whose both markers are known."""
        for (a, b), door_id in self.marker_pair_to_door.items():
            if a not in self.marker_world_positions:
                continue
            if b not in self.marker_world_positions:
                continue

            p1 = np.array(self.marker_world_positions[a])
            p2 = np.array(self.marker_world_positions[b])

            center   = (p1 + p2) / 2
            previous = self.door_world_positions.get(door_id, {})
            self.door_world_positions[door_id] = {
                "left":     tuple(p1),
                "right":    tuple(p2),
                "center":   tuple(center),
                # Preserve the blocked flag so an unlock is not lost on recompute.
                "_blocked": previous.get("_blocked", False),
            }

    def register_key(self, key_id: int, position=None) -> bool:
        """Record that a key has been found.  Optionally store its world position.

        Returns True if this is the first time the key is seen.
        """
        if position is not None:
            self.key_world_positions[key_id] = tuple(position)
        return self.registry.register_key_detection(key_id)

    def is_door_unlocked(self, door_id: int) -> bool:
        """Return True if the key for this door has been collected."""
        return self.registry.is_door_unlocked(door_id)

    def get_door(self, door_id: int):
        """Return the geometry dict for a door, or None if not yet localised."""
        return self.door_world_positions.get(door_id, None)