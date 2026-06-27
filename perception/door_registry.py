#!/usr/bin/env python3

from typing import Dict, Tuple, Optional
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

        self.door_world_positions = {}

    def register_marker_position(
        self,
        marker_id: int,
        wx: float,
        wy: float
    ):

        self.marker_world_positions[marker_id] = (
            wx,
            wy
        )

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

            self.door_world_positions[door_id] = {

                "left":tuple(p1),
                "right":tuple(p2),
                "center":tuple(center)
            }

    def register_key(
        self,
        key_id:int
    ):
        self.registry.register_key_detection(
            key_id
        )

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