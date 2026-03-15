"""Gymnasium wrapper that records separate video files for each camera view."""

from __future__ import annotations

import os
import subprocess

import gymnasium as gym
import numpy as np


class MultiCameraRecordVideo(gym.Wrapper):
    """Records one video per camera angle using the env's ``capture_multi_cameras()`` method.

    Output files follow the pattern::

        <video_folder>/rl-video-step-<N>-<cam_name>.mp4

    The wrapper does **not** call ``env.render()`` — it goes directly to the
    multi-camera capture method on the unwrapped Isaac Lab environment.
    """

    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        step_trigger: callable,
        video_length: int,
        fps: int = 20,
        disable_logger: bool = True,
    ):
        super().__init__(env)
        self.video_folder = video_folder
        self.step_trigger = step_trigger
        self.video_length = video_length
        self.fps = fps

        os.makedirs(video_folder, exist_ok=True)

        self._step_count = 0
        self._recording = False
        self._record_start_step = 0
        self._frames_recorded = 0
        self._video_writers: dict[str, subprocess.Popen] = {}

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if not self._recording and self.step_trigger(self._step_count):
            self._start_recording()

        if self._recording:
            self._capture_frame()
            if self._frames_recorded >= self.video_length:
                self._stop_recording()

        self._step_count += 1
        return obs, reward, terminated, truncated, info

    def close(self):
        if self._recording:
            self._stop_recording()
        super().close()

    # ── internal ──────────────────────────────────────────────────────

    def _start_recording(self):
        self._recording = True
        self._record_start_step = self._step_count
        self._frames_recorded = 0
        self._video_writers = {}

    def _capture_frame(self):
        if hasattr(self.unwrapped, "capture_multi_cameras"):
            frames = self.unwrapped.capture_multi_cameras()
        else:
            frame = self.render()
            if frame is None:
                raise RuntimeError("Video recording requested, but the environment did not return an rgb frame.")
            frames = {"main": np.asarray(frame)}
        for name, frame in frames.items():
            writer = self._video_writers.get(name)
            if writer is None:
                filename = f"rl-video-step-{self._record_start_step}-{name}.mp4"
                path = os.path.join(self.video_folder, filename)
                writer = _open_video_writer(frame, path, self.fps)
                self._video_writers[name] = writer
            if writer.stdin is not None:
                writer.stdin.write(frame.tobytes())
        self._frames_recorded += 1

    def _stop_recording(self):
        self._recording = False
        for cam_name, writer in self._video_writers.items():
            if writer.stdin is not None:
                writer.stdin.close()
            stderr = b""
            if writer.stderr is not None:
                stderr = writer.stderr.read()
            returncode = writer.wait()
            if returncode != 0:
                error = stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    f"ffmpeg failed while writing camera '{cam_name}' video: {error or 'unknown error'}"
                )
        self._video_writers = {}


def _open_video_writer(frame: np.ndarray, path: str, fps: int) -> subprocess.Popen:
    """Create an ffmpeg process that accepts raw RGB frames on stdin."""
    h, w, _ = frame.shape
    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast",
        path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
