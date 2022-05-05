import numpy as np
from PIL import Image


def decode_image(raw_img, w, h, format="bgr") -> Image.Image:
    if format == "bgr":
        img = Image.frombytes(mode="RGB", size=(w, h), data=raw_img)
        b, g, r = img.split()
        img = Image.merge("RGB", (r, g, b))
        return img
    elif format == "jpeg":
        img = Image.open(io.BytesIO(raw_img))
        img.load()  # is required, because Pillow uses lazy loading which causes problems later on
        return img
    else:
        raise NotImplementedError(f"image format {format} is not supported.")

class GazeSample:
    """
    Data structure for gaze data. It enforces
    (1) that the origin of gaze coordinates is at the upper left, and
    (2) the gaze coordinates are normalized.
    """

    # see also https://pillow.readthedocs.io/en/stable/handbook/concepts.html#coordinate-system
    ORIGIN_BOTTOM_LEFT = "bl"
    ORIGIN_TOP_LEFT = "tl"

    def __init__(self, gaze, reference_size, normalized=True, origin="bl"):
        self._max_width = reference_size[0]
        self._max_height = reference_size[1]
        self._gaze = self._normalize(gaze, normalized, origin)


    def _normalize(self, gaze, normalized, origin):
        x, y = tuple(gaze)
        if not normalized:
            x /= float(self.max_width)
            y /= float(self.max_height)

        if origin == self.ORIGIN_BOTTOM_LEFT:
            # convert to top-left coordinate
            y = 1. - y

        return x, y

    @property
    def x(self):
        return self.gaze[0]

    @property
    def y(self):
        return self.gaze[1]

    @property
    def gaze(self):
        return self._gaze

    @property
    def x_scaled(self):
        return self.x * self.max_width

    @property
    def y_scaled(self):
        return self.y * self.max_height

    @property
    def gaze_scaled(self):
        return self.x_scaled, self.y_scaled

    @property
    def max_width(self):
        return self._max_width

    @property
    def max_height(self):
        return self._max_height


class FixationEvent:

    def __init__(self, fixation_id: int, timestamp: float, duration: float, norm_pos: np.ndarray, video_resolution):
        self._is_complete = False
        self._fixation_id = fixation_id
        self._timestamps = []
        self._durations = []
        self._norm_pos = []
        self._video_resolution = video_resolution
        self.add_partial_fixation(fixation_id, timestamp, duration, norm_pos)

    def add_partial_fixation(self, fixation_id: int, timestamp: float, duration: float, norm_pos: np.ndarray):
        assert self._fixation_id == fixation_id
        self._timestamps.append(timestamp)
        self._norm_pos.append(norm_pos)
        self._durations.append(duration)

    def finalize(self) -> 'Fixation':
        self._is_complete = True
        return self

    @property
    def finalized(self) -> bool:
        return self._is_complete

    @property
    def num_samples(self):
        return len(self._timestamps)

    @property
    def index(self):
        return self._fixation_id

    @property
    def timestamp(self) -> float:
        return self._timestamps[0]

    @property
    def duration(self):
        if self._is_complete:
            return (self._timestamps[-1] - self._timestamps[0]) * 1000. + self._durations[-1]
        return None

    @property
    def fixation_point(self):
        norm_positions = np.vstack(self._norm_pos)
        fixation = GazeSample(
            gaze=norm_positions.mean(axis=0),
            normalized=True,
            reference_size=self._video_resolution,
            origin=GazeSample.ORIGIN_BOTTOM_LEFT
        )
        return fixation
