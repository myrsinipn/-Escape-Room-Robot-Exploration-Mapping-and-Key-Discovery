import numpy as np


class FrontierDetector:

    def detect(self, grid: np.ndarray):

        frontiers = []

        h, w = grid.shape

        for y in range(1, h - 1):
            for x in range(1, w - 1):

                cell = grid[y, x]

                # occupied
                if cell >= 50:
                    continue

                # unknown
                if cell == -1:
                    continue

                neighborhood = grid[y - 1:y + 2, x - 1:x + 2]

                if np.any(neighborhood == -1):
                    frontiers.append((x, y))

        return frontiers