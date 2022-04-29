from typing import Optional, List
import msgpack
import zmq
import time
from .util import decode_image, FixationEvent, GazeSample


class PupilRemote:
    class Streams:
        GAZE = "gaze."
        FIXATIONS = "fixations"
        SURFACES = "surfaces"
        SCENE_VIDEO = "frame.world"

        # lists plugin requirements for all available signal streams
        _plugin_requirements = {
            FIXATIONS: {"subject": "start_plugin", "name": "Fixation_Detector"},
            SURFACES: {"subject": "start_plugin", "name": "Surface_Tracker_Online"},
            SCENE_VIDEO: {'subject': 'frame_publishing.set_format', 'format': 'bgr'}
        }

        @staticmethod
        def requirement(stream):
            if stream in PupilRemote.Streams._plugin_requirements:
                return PupilRemote.Streams._plugin_requirements[stream]
            return None

    def __init__(self, address: str = "127.0.0.1", port: int = 50020, streams: Optional[List[str]] = None):
        self.ctx = zmq.Context()
        self.pupil_remote = zmq.Socket(self.ctx, zmq.REQ)
        self.pupil_remote_ip = address
        self.pupil_remote_port = port
        self.pupil_sub_port = 0
        self.pupil_pub_port = 0
        self.subscriber = None
        self.video_resolution = None
        self.last_fixation_id = None
        self._current_fixation = None

        self._streams = streams
        if streams is None:
            self._streams = [
                PupilRemote.Streams.GAZE,
                PupilRemote.Streams.FIXATIONS,
                PupilRemote.Streams.SCENE_VIDEO
            ]

    def connect(self):
        self.pupil_remote.connect(f"tcp://{self.pupil_remote_ip}:{self.pupil_remote_port}")
        self.pupil_remote.send_string('SUB_PORT')
        self.pupil_sub_port = self.pupil_remote.recv_string()
        self.pupil_remote.send_string('PUB_PORT')
        self.pupil_pub_port = self.pupil_remote.recv_string()

        self._start_required_plugins()
        self._start_subscription()
        self.sync_timestamp()

    def close(self):
        self.subscriber.close()
        self.pupil_remote.close()

    def _start_required_plugins(self):
        for stream in self._streams:
            requirement = self.Streams.requirement(stream)
            if requirement is not None:
                self._send_pupil_notification(requirement)

    def _send_pupil_notification(self, payload):
        topic = f"notify.{payload['subject']}"
        payload = msgpack.packb(payload, use_bin_type=True)
        self.pupil_remote.send_string(topic, flags=zmq.SNDMORE)
        self.pupil_remote.send(payload)
        return self.pupil_remote.recv_string()

    def _start_subscription(self):
        self.subscriber = self.ctx.socket(zmq.SUB)
        self.subscriber.connect(f"tcp://{self.pupil_remote_ip}:{self.pupil_sub_port}")
        for event in self._streams:
            self.subscriber.subscribe(event)

    def _recv_sub_event(self):
        try:
            # no block because of closing error
            topic = self.subscriber.recv_string(flags=zmq.NOBLOCK)
            payload = self.subscriber.recv_multipart(flags=zmq.NOBLOCK)
            message = msgpack.loads(payload[0])
            if len(payload) > 1:
                message["raw_img"] = payload[1:]  # e.g., the encoded image of the scene camera
        except Exception as e:
            if e.errno == 11 or 36:  # Resource is temporarily not available
                return None, None
            else:
                raise e
        return topic, message

    def _handle_partial_fixations(self, message) -> Optional[FixationEvent]:
        fixation_id = message["id"]
        timestamp = message["timestamp"]
        norm_pos = message["norm_pos"]
        duration = message["duration"]

        if self._current_fixation is None:
            self._current_fixation = FixationEvent(
                fixation_id=fixation_id,
                timestamp=timestamp,
                duration=duration,
                norm_pos=norm_pos,
                video_resolution=self.video_resolution
            )
            return None  # [self._current_fixation]
        elif self._current_fixation.index == fixation_id:
            self._current_fixation.add_partial_fixation(
                fixation_id=fixation_id,
                timestamp=timestamp,
                duration=duration,
                norm_pos=norm_pos
            )
            return None

        complete_fixation = self._current_fixation.finalize()
        self._current_fixation = FixationEvent(
            fixation_id=fixation_id,
            timestamp=timestamp,
            duration=duration,
            norm_pos=norm_pos,
            video_resolution=self.video_resolution
        )
        return complete_fixation  # , self._current_fixation

    def get_next_event(self):
        topic, message = self._recv_sub_event()

        if topic is not None:
            if topic == PupilRemote.Streams.SCENE_VIDEO:
                encoded_image = message["raw_img"][0]
                if encoded_image is None:
                    return
                image = decode_image(encoded_image, message["width"], message["height"], format=message["format"])
                self.video_resolution = image.size
                message["image"] = image
                self.correct_timestamp(message)
                return topic, message

            elif topic.startswith(PupilRemote.Streams.GAZE) and self.video_resolution is not None:
                topic = PupilRemote.Streams.GAZE
                # see https://docs.pupil-labs.com/developer/core/overview/#gaze-datum-format
                gaze_sample = GazeSample(
                    gaze=message["norm_pos"],
                    normalized=True,
                    reference_size=self.video_resolution,
                    origin=GazeSample.ORIGIN_BOTTOM_LEFT
                )
                message["gaze"] = gaze_sample
                self.correct_timestamp(message)
                return topic, message

            elif topic.startswith(PupilRemote.Streams.FIXATIONS) and self.video_resolution is not None:
                self.correct_timestamp(message)
                fixation = self._handle_partial_fixations(message)
                if fixation is not None:
                    topic = PupilRemote.Streams.FIXATIONS
                    return topic, fixation

        return None, None

    def correct_timestamp(self, message):
        message["timestamp"] = message["timestamp"] - self.pupil_time_offset

    def sync_timestamp(self):
        # set current Pupil time to timestamp
        local_clock = time.perf_counter
        offset = self.measure_clock_offset(clock_function=local_clock)
        # print(f"Clock offset (1 measurement): {offset} seconds")
        number_of_measurements = 10
        stable_offset_mean = self.measure_clock_offset_stable(
            clock_function=local_clock, nsamples=number_of_measurements
        )
        # print(
        #     f"Mean clock offset ({number_of_measurements} measurements): "
        #     f"{stable_offset_mean} seconds"
        # )

        # 5. Infer pupil clock time from "local" clock measurement
        local_time = local_clock()
        pupil_time_calculated_locally = local_time + stable_offset_mean
        # print(f"Local time: {local_time}")
        # print(f"Pupil time (calculated locally): {pupil_time_calculated_locally}")
        self.pupil_time_offset = stable_offset_mean

    def request_pupil_time(self):
        """Uses an existing Pupil Core software connection to request the remote time.
        Returns the current "pupil time" at the timepoint of reception.
        See https://docs.pupil-labs.com/core/terminology/#pupil-time for more information
        about "pupil time".
        """
        self.pupil_remote.send_string("t")
        pupil_time = self.pupil_remote.recv()
        return float(pupil_time)

    def measure_clock_offset(self, clock_function):
        """Calculates the offset between the Pupil Core software clock and a local clock.
        Requesting the remote pupil time takes time. This delay needs to be considered
        when calculating the clock offset. We measure the local time before (A) and
        after (B) the request and assume that the remote pupil time was measured at (A+B)/2,
        i.e. the midpoint between A and B.
        As a result, we have two measurements from two different clocks that were taken
        assumingly at the same point in time. The difference between them ("clock offset")
        allows us, given a new local clock measurement, to infer the corresponding time on
        the remote clock.
        """
        local_time_before = clock_function()
        pupil_time = self.request_pupil_time()
        local_time_after = clock_function()

        local_time = (local_time_before + local_time_after) / 2.0
        clock_offset = pupil_time - local_time
        return clock_offset

    def measure_clock_offset_stable(self, clock_function, nsamples=10):
        """Returns the mean clock offset after multiple measurements to reduce the effect
        of varying network delay.
        Since the network connection to Pupil Capture/Service is not necessarily stable,
        one has to assume that the delays to send and receive commands are not symmetrical
        and might vary. To reduce the possible clock-offset estimation error, this function
        repeats the measurement multiple times and returns the mean clock offset.
        The variance of these measurements is expected to be higher for remote connections
        (two different computers) than for local connections (script and Core software
        running on the same computer). You can easily extend this function to perform
        further statistical analysis on your clock-offset measurements to examine the
        accuracy of the time sync.
        """
        assert nsamples > 0, "Requires at least one sample"
        offsets = [self.measure_clock_offset(clock_function) for x in range(nsamples)]
        return sum(offsets) / len(offsets)  # mean offset
