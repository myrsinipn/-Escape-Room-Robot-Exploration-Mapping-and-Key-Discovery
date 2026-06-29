import numpy as np


class ScanPreprocessor:
    """Cleans and conditions raw LiDAR scans before they reach SLAM or control.

    Pipeline (in order):
      1. Replace NaN / inf values with max_range.
      2. Clip all ranges to [0, max_range].
      3. Optionally smooth with a moving-average kernel.
      4. Optionally downsample by a fixed stride.
    """

    def __init__(
        self,
        min_range: float = 0.05,
        max_range: float = 8.0,
        apply_smoothing: bool = True,
        smoothing_kernel_size: int = 5,
        downsample_factor: int = 1,
    ) -> None:
        self.min_range            = min_range
        self.max_range            = max_range
        self.apply_smoothing      = apply_smoothing
        self.smoothing_kernel_size = smoothing_kernel_size
        self.downsample_factor    = downsample_factor

    def preprocess(self, scan: dict) -> dict:
        """Run the full preprocessing pipeline on a raw scan dict.

        Input dict must contain 'ranges' (np.ndarray), 'angles' (np.ndarray),
        and 'timestamp'.  Returns a new dict with the same keys.
        """
        ranges = scan["ranges"].copy()
        angles = scan["angles"].copy()

        ranges = self.remove_invalid_ranges(ranges)
        ranges = self.clip_ranges(ranges)

        if self.apply_smoothing:
            ranges = self.smooth_scan(ranges)

        if self.downsample_factor > 1:
            ranges = ranges[::self.downsample_factor]
            angles = angles[::self.downsample_factor]

        return {
            "ranges":    ranges,
            "angles":    angles,
            "timestamp": scan["timestamp"],
        }

    def remove_invalid_ranges(self, ranges: np.ndarray) -> np.ndarray:
        """Replace NaN and inf entries with max_range. Below-min values are left as-is."""
        cleaned = ranges.copy()
        cleaned[np.isnan(cleaned) | np.isinf(cleaned)] = self.max_range
        return cleaned

    def clip_ranges(self, ranges: np.ndarray) -> np.ndarray:
        """Cap readings at max_range; do not raise readings below min_range."""
        return np.minimum(ranges, self.max_range)

    def get_sector_min(
        self,
        processed_scan: dict,
        angle_min_deg: float,
        angle_max_deg: float,
    ) -> float:
        """Return the minimum valid range in an angular sector.

        Readings outside [min_range, max_range] are excluded before
        taking the minimum, which filters both too-close and saturated rays.
        Returns inf if no valid reading exists in the sector.
        """
        ranges     = processed_scan["ranges"]
        angles_deg = np.degrees(processed_scan["angles"])
        angles_deg = (angles_deg + 180) % 360 - 180   # normalise to [-180, 180]

        valid_mask = (
            (ranges     >= self.min_range)
            & (ranges   <= self.max_range)
            & (angles_deg >= angle_min_deg)
            & (angles_deg <= angle_max_deg)
        )

        sector_ranges = ranges[valid_mask]
        if len(sector_ranges) == 0:
            return float("inf")
        return float(np.min(sector_ranges))

    def smooth_scan(self, ranges: np.ndarray) -> np.ndarray:
        """Apply a uniform moving-average kernel to reduce per-beam noise."""
        k = self.smoothing_kernel_size
        if k <= 1:
            return ranges
        kernel   = np.ones(k) / k
        smoothed = np.convolve(ranges, kernel, mode="same")
        return smoothed

    def polar_to_cartesian(
        self,
        ranges: np.ndarray,
        angles: np.ndarray,
    ) -> np.ndarray:
        """Convert polar (range, angle) pairs to Cartesian (x, y) points.

        Returns an N×2 array where column 0 is x (forward) and column 1 is y.
        """
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        return np.column_stack((x, y))