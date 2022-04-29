import time
from typing import Optional, List, Tuple
from PIL import Image
from multisensor_pipeline.dataframe import MSPDataFrame, Topic
from multisensor_pipeline.modules.base import BaseSource
from .pupil_remote import PupilRemote
from .util import FixationEvent, GazeSample


class PupilCaptureSource(BaseSource):

    def __init__(self, address: str = "127.0.0.1", port: int = 50020, streams: Optional[List[str]] = None):
        super().__init__()
        self._pupil_remote = PupilRemote(address=address, port=port, streams=streams)

        # prepare topics
        self._img_topic = Topic(name="scene_video", dtype=Image.Image)
        self._gaze_topic = Topic(name="gaze", dtype=GazeSample)
        self._fixation_topic = Topic(name="fixation", dtype=FixationEvent)

    @property
    def scene_video_topic(self) -> Topic:
        return self._img_topic

    @property
    def gaze_topic(self) -> Topic:
        return self._gaze_topic

    @property
    def fixation_topic(self) -> Topic:
        return self._fixation_topic

    @property
    def output_topics(self) -> Optional[List[Topic]]:
        return [self._img_topic, self._gaze_topic, self._fixation_topic]

    def on_start(self):
        self._pupil_remote.connect()

    def on_update(self) -> Optional[MSPDataFrame]:
        topic, payload = self._pupil_remote.get_next_event()

        # do nothing, if empty messages arrive
        if topic is None:
            return

        if topic == PupilRemote.Streams.SCENE_VIDEO:
            return MSPDataFrame(
                topic=self._img_topic,
                data=payload["image"],
                timestamp=payload["timestamp"]
            )

        if topic == PupilRemote.Streams.GAZE:
            return MSPDataFrame(
                topic=self._gaze_topic,
                data=payload["gaze"],
                timestamp=payload["timestamp"]
            )

        if topic == PupilRemote.Streams.FIXATIONS:
            assert isinstance(payload, FixationEvent)
            return MSPDataFrame(
                topic=self._fixation_topic,
                data=payload,
                duration=payload.duration,
                timestamp=payload.timestamp
            )

    def on_stop(self):
        self._pupil_remote.close()

