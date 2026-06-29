#!/usr/bin/env python3

from typing import Dict, Tuple
import numpy as np

from state_estimation.key_door_registry import KeyDoorRegistry


class DoorRegistry:
    """
    Stores:

    (markerA, markerB) -> door_id
    key_id -> door_id

    and estimated world coordinates of doors.
    """

    def __init__(
        self,
        door_marker_pairs: Dict[Tuple[int,int], int],
        key_to_door: Dict[int,int]
    ):

        self.registry = KeyDoorRegistry(key_to_door)

        self.marker_pair_to_door = {}

        for pair, door_id in door_marker_pairs.items():

            a,b = sorted(pair)

            self.marker_pair_to_door[(a,b)] = door_id

        self.marker_world_positions = {}
        self._marker_observations = {}

        self.door_world_positions = {}
        self.key_world_positions = {}

    def register_marker_position(
        self,
        marker_id: int,
        wx: float,
        wy: float
    ):

        # Camera pose estimates jitter.  Keep an incremental mean so a door
        # does not jump around the map (or leave stale blocked-door overlays)
        # every time another frame is processed.
        count = self._marker_observations.get(marker_id, 0)
        old = np.asarray(
            self.marker_world_positions.get(marker_id, (wx, wy)),
            dtype=float,
        )
        new = (old * count + np.asarray((wx, wy), dtype=float)) / (count + 1)
        self._marker_observations[marker_id] = count + 1
        self.marker_world_positions[marker_id] = tuple(new)

        self._try_build_doors()

    def _try_build_doors(self):

        for (a,b), door_id in self.marker_pair_to_door.items():

            if a not in self.marker_world_positions:
                continue

            if b not in self.marker_world_positions:
                continue

            p1 = np.array(
                self.marker_world_positions[a]
            )

            p2 = np.array(
                self.marker_world_positions[b]
            )

            center = (p1+p2)/2

            previous = self.door_world_positions.get(door_id, {})
            self.door_world_positions[door_id] = {

                "left":tuple(p1),
                "right":tuple(p2),
                "center":tuple(center),
                "_blocked": previous.get("_blocked", False),
            }

    def register_key(
        self,
        key_id:int,
        position=None,
    ):
        if position is not None:
            self.key_world_positions[key_id] = tuple(position)
        return self.registry.register_key_detection(key_id)

    def is_door_unlocked(
        self,
        door_id:int
    ):
        return self.registry.is_door_unlocked(
            door_id
        )

    def get_door(self,door_id):

        return self.door_world_positions.get(
            door_id,
            None
        )
