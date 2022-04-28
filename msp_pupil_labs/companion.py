import time
from queue import Queue
from threading import Thread
from typing import Optional, List, Tuple
from PIL import Image
from multisensor_pipeline.dataframe import MSPDataFrame, Topic
from multisensor_pipeline.modules.base import BaseSource
from pupil_labs.realtime_api.simple import discover_one_device


class PupilCompanionSource(BaseSource):

    def __init__(self):
        super().__init__()
        device = discover_one_device()
        if device is None:
            print("No device found.")
            raise SystemExit(-1)
        self._device = device
        self._queue = Queue()
        self._gaze_thread = Thread(target=self.get_gaze)
        self._video_thread = Thread(target=self.get_video)
        self._time_offset = 0

        # prepare topics
        self._img_topic = Topic(name="scene_video", dtype=Image.Image)
        self._gaze_topic = Topic(name="gaze", dtype=Tuple[int, int])

    def on_start(self):
        sample = self._device.receive_gaze_datum()
        current_time = time.perf_counter()
        self._time_offset = sample.timestamp_unix_seconds - current_time
        self._gaze_thread.start()
        self._video_thread.start()

    @property
    def scene_video_topic(self) -> Topic:
        return self._img_topic

    @property
    def gaze_topic(self) -> Topic:
        return self._gaze_topic

    @property
    def output_topics(self) -> Optional[List[Topic]]:
        return [self._img_topic, self._gaze_topic, ]

    def get_gaze(self):
        while self.active:
            self._queue.put((self.gaze_topic, self._device.receive_gaze_datum()))

    def get_video(self):
        while self.active:
            self._queue.put((self.scene_video_topic, self._device.receive_scene_video_frame()))

    def on_update(self) -> Optional[MSPDataFrame]:
        topic, data = self._queue.get()
        if topic.name == "gaze":
            return MSPDataFrame(
                topic=topic,
                data=(data[0], data[1]),
                timestamp=data.timestamp_unix_seconds - self._time_offset
            )
        else:
            return MSPDataFrame(
                topic=topic,
                data=Image.fromarray(data.bgr_pixels),
                timestamp=data.timestamp_unix_seconds - self._time_offset
            )

    def on_stop(self):
        self._gaze_thread.join()
        self._video_thread.join()
        self._device.streaming_stop()
        self._device.close()