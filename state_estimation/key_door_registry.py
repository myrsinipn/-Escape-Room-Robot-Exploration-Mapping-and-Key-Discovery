from typing import Dict, Optional, Set


class KeyDoorRegistry:
    """
    Registry that stores relationships between:
    - keys
    - doors

    Responsibilities:
    - Store key-door mappings
    - Track discovered keys
    - Check if a door is unlocked

    """

    def __init__(
        self,
        key_to_door_map: Dict[int, int],
    ) -> None:
        """
        Parameters
        ----------
        key_to_door_map : Dict[int, int]

            Example:
            {
                10: 3,
                11: 7,
                12: 1
            }

            Meaning:
            key 10 unlocks door 3
            key 11 unlocks door 7
            ...
        """

        self.key_to_door_map = key_to_door_map

        # reverse lookup
        self.door_to_key_map = {
            door_id: key_id
            for key_id, door_id
            in key_to_door_map.items()
        }

        # discovered keys
        self.discovered_keys: Set[int] = set()

    def register_key_detection(
        self,
        key_id: int,
    ) -> bool:
        """
        Mark a key as discovered.
        """

        if key_id not in self.key_to_door_map:
            return False

        is_new = key_id not in self.discovered_keys
        self.discovered_keys.add(key_id)
        return is_new

    def has_key(
        self,
        key_id: int,
    ) -> bool:
        """
        Returns True if key was discovered.
        """

        return key_id in self.discovered_keys

    def is_door_unlocked(
        self,
        door_id: int,
    ) -> bool:
        """
        Returns True if corresponding key was found.
        """

        if door_id not in self.door_to_key_map:

            return False

        required_key = self.door_to_key_map[door_id]

        return required_key in self.discovered_keys

    def get_required_key(
        self,
        door_id: int,
    ) -> Optional[int]:
        """
        Returns key required for a door.
        """

        return self.door_to_key_map.get(
            door_id,
            None,
        )

    def get_unlocked_doors(self):
        """
        Returns list of unlocked doors.
        """

        unlocked = []

        for door_id, key_id in self.door_to_key_map.items():

            if key_id in self.discovered_keys:

                unlocked.append(door_id)

        return unlocked

    def get_locked_doors(self):
        """
        Returns list of still locked doors.
        """

        locked = []

        for door_id, key_id in self.door_to_key_map.items():

            if key_id not in self.discovered_keys:

                locked.append(door_id)

        return locked

    def reset(self) -> None:
        """
        Clears discovered keys.
        """

        self.discovered_keys.clear()

    def print_status(self) -> None:
        """
        Prints current registry state.
        """

        print("\n========== KEY-DOOR STATUS ==========")

        for door_id, key_id in self.door_to_key_map.items():

            unlocked = key_id in self.discovered_keys

            status = (
                "UNLOCKED"
                if unlocked
                else "LOCKED"
            )

            print(
                f"Door {door_id} "
                f"(Key {key_id}) -> {status}"
            )

        print("=====================================\n")
