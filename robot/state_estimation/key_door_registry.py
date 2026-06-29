from typing import Dict, List, Optional, Set


class KeyDoorRegistry:
    """Tracks which keys have been found and which doors they unlock.

    A door is considered unlocked once its corresponding key has been
    registered via register_key_detection().
    """

    def __init__(self, key_to_door_map: Dict[int, int]) -> None:
        """
        Parameters
        ----------
        key_to_door_map : dict
            Maps key_id → door_id.  Example:
                {10: 3, 11: 7, 12: 1}
            means key 10 unlocks door 3, key 11 unlocks door 7, etc.
        """
        self.key_to_door_map = key_to_door_map

        # Build the reverse lookup so we can answer "which key opens door X?"
        self.door_to_key_map: Dict[int, int] = {
            door_id: key_id
            for key_id, door_id in key_to_door_map.items()
        }

        self.discovered_keys: Set[int] = set()

    def register_key_detection(self, key_id: int) -> bool:
        """Mark a key as discovered.

        Returns True if this is the first time the key is registered,
        False if it was already known or is not in the key-to-door map.
        """
        if key_id not in self.key_to_door_map:
            return False
        is_new = key_id not in self.discovered_keys
        self.discovered_keys.add(key_id)
        return is_new

    def has_key(self, key_id: int) -> bool:
        """Return True if the key has been discovered."""
        return key_id in self.discovered_keys

    def is_door_unlocked(self, door_id: int) -> bool:
        """Return True if the key required for this door has been found."""
        if door_id not in self.door_to_key_map:
            return False
        return self.door_to_key_map[door_id] in self.discovered_keys

    def get_required_key(self, door_id: int) -> Optional[int]:
        """Return the key_id that opens this door, or None if unknown."""
        return self.door_to_key_map.get(door_id, None)

    def get_unlocked_doors(self) -> List[int]:
        """Return the list of doors whose key has been collected."""
        return [
            door_id
            for door_id, key_id in self.door_to_key_map.items()
            if key_id in self.discovered_keys
        ]

    def get_locked_doors(self) -> List[int]:
        """Return the list of doors still waiting for their key."""
        return [
            door_id
            for door_id, key_id in self.door_to_key_map.items()
            if key_id not in self.discovered_keys
        ]

    def reset(self) -> None:
        """Clear all discovered keys (useful for testing or restart)."""
        self.discovered_keys.clear()

    def print_status(self) -> None:
        """Print a human-readable summary of all door lock states."""
        print("\n========== KEY-DOOR STATUS ==========")
        for door_id, key_id in self.door_to_key_map.items():
            status = "UNLOCKED" if key_id in self.discovered_keys else "LOCKED"
            print(f"Door {door_id} (Key {key_id}) -> {status}")
        print("=====================================\n")