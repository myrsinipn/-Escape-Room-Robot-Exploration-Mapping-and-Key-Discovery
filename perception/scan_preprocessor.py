import numpy as np


class ScanPreprocessor:
    """
    LiDAR scan preprocessing module.

    Responsibilities:
    - Remove invalid values
    - Clip ranges
    - Smooth scan
    - Downsample scan
    - Convert polar -> Cartesian

    """

    def __init__(
        self,
        min_range: float = 0.05,
        max_range: float = 8.0,
        apply_smoothing: bool = True,
        smoothing_kernel_size: int = 5,
        downsample_factor: int = 1,
    ) -> None:

        self.min_range = min_range
        self.max_range = max_range

        self.apply_smoothing = apply_smoothing
        self.smoothing_kernel_size = smoothing_kernel_size

        self.downsample_factor = downsample_factor

    def preprocess(
        self,
        scan: dict,
    ) -> dict:
    

        ranges = scan["ranges"].copy()
        angles = scan["angles"].copy()

        # remove invalid values
        ranges = self.remove_invalid_ranges(ranges)

        # clip ranges
        ranges = self.clip_ranges(ranges)

        # smoothing
        if self.apply_smoothing:

            ranges = self.smooth_scan(ranges)

        # downsampling
        if self.downsample_factor > 1:

            ranges = ranges[::self.downsample_factor]
            angles = angles[::self.downsample_factor]

        return {
            "ranges": ranges,
            "angles": angles,
            "timestamp": scan["timestamp"],
        }

    def remove_invalid_ranges(
        self,
        ranges: np.ndarray,
    ) -> np.ndarray:
        """
        Replace NaN and inf values.
        """

        cleaned = ranges.copy()

        invalid_mask = (
            np.isnan(cleaned)
            | np.isinf(cleaned)
        )

        cleaned[invalid_mask] = self.max_range

        return cleaned

    def clip_ranges(
        self,
        ranges: np.ndarray,
    ) -> np.ndarray:
        """
        Clip ranges to valid interval.
        """

        return np.clip(
            ranges,
            self.min_range,
            self.max_range,
        )

    def smooth_scan(
        self,
        ranges: np.ndarray,
    ) -> np.ndarray:
        """
        Moving average smoothing.
        """

        kernel_size = self.smoothing_kernel_size

        if kernel_size <= 1:
            return ranges

        kernel = np.ones(kernel_size) / kernel_size

        smoothed = np.convolve(
            ranges,
            kernel,
            mode="same",
        )

        return smoothed

    def polar_to_cartesian(
        self,
        ranges: np.ndarray,
        angles: np.ndarray,
    ) -> np.ndarray:
        """
        Converts polar LiDAR scan to Cartesian points.

        Returns:
            Nx2 array of [x, y]
        """

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)

        points = np.column_stack((x, y))

        return points