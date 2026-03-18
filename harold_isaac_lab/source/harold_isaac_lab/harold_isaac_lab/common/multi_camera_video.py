"""Gymnasium wrapper that records separate video files for each camera view.

Includes HUD overlay (frame/step/reset count) and red-border reset flash
so video review agents can easily parse training progress.
"""

from __future__ import annotations

import os
import subprocess

import gymnasium as gym
import numpy as np

# HUD views that get the text overlay (skip top/front to keep them clean)
_HUD_VIEWS = {"side", "iso"}

# Reset flash: red border thickness in pixels
_RESET_BORDER_PX = 6
_RESET_COLOR = np.array([220, 40, 40], dtype=np.uint8)

# Number of warmup render calls before recording to avoid first-frame artifacts
_WARMUP_RENDERS = 2

# Axis widget: 2D projections of world X/Y/Z axes for each camera view.
# Isaac Sim world: +X = forward, +Y = left, +Z = up.
# Each entry maps axis label -> (dx, dy) in pixel space (right=+dx, down=+dy).
# Arrow length is scaled by _AXIS_LENGTH.
_AXIS_LENGTH = 30
_AXIS_PROJECTIONS: dict[str, dict[str, tuple[int, int]]] = {
    # Side camera looks from -Y: image right = +X (fwd), image up = +Z (up)
    "side":  {"+X fwd": (1, 0), "+Z up": (0, -1)},
    # Front camera looks from +X: image left = +Y (left), image up = +Z (up)
    "front": {"+Y left": (-1, 0), "+Z up": (0, -1)},
    # Top camera looks from +Z down: image right = +X (fwd), image up = -Y (right)
    "top":   {"+X fwd": (1, 0), "+Y left": (0, 1)},
    # Iso camera from (+X, -Y, +Z) — approximate projected directions
    "iso":   {"+X fwd": (-3, 1), "+Z up": (0, -4), "+Y left": (-3, -1)},
}


def _draw_text_pil(frame: np.ndarray, text: str) -> np.ndarray:
    """Burn white text on a semi-transparent dark strip into bottom-left of frame."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame  # PIL not available — skip overlay silently

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Measure text
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    padding = 4
    h = frame.shape[0]
    x, y = padding, h - th - padding * 3

    # Dark background strip
    draw.rectangle(
        [x - padding, y - padding, x + tw + padding, y + th + padding],
        fill=(0, 0, 0, 180),
    )
    draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
    return np.array(img)


def _draw_reset_border(frame: np.ndarray) -> np.ndarray:
    """Draw a red border and 'RESET' label on the frame."""
    f = frame.copy()
    b = _RESET_BORDER_PX
    # Top/bottom borders
    f[:b, :] = _RESET_COLOR
    f[-b:, :] = _RESET_COLOR
    # Left/right borders
    f[:, :b] = _RESET_COLOR
    f[:, -b:] = _RESET_COLOR

    # Burn "RESET" text into top-left corner
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.fromarray(f)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()
        draw.text((b + 4, b + 2), "RESET", fill=(255, 255, 255), font=font)
        f = np.array(img)
    except ImportError:
        pass  # PIL not available — border alone is still useful

    return f


# Axis colors: X=red, Y=green, Z=blue (standard RGB convention)
_AXIS_COLORS = {
    "+X fwd": (220, 60, 60),
    "+Y left": (60, 180, 60),
    "+Z up": (60, 100, 220),
}


def _draw_axes(frame: np.ndarray, cam_name: str) -> np.ndarray:
    """Draw a small axis indicator widget in the bottom-right corner of the frame."""
    projections = _AXIS_PROJECTIONS.get(cam_name)
    if projections is None:
        return frame

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 11)
    except (OSError, IOError):
        font = ImageFont.load_default()

    h, w = frame.shape[:2]
    # Origin of the axis widget: bottom-right corner with padding
    ox, oy = w - 70, h - 50

    # Semi-transparent background circle
    radius = 45
    draw.ellipse(
        [ox - radius, oy - radius, ox + radius, oy + radius],
        fill=(0, 0, 0, 120),
    )

    for label, (dx, dy) in projections.items():
        # Normalize direction and scale to _AXIS_LENGTH
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1e-6:
            continue
        ex = int(ox + dx / length * _AXIS_LENGTH)
        ey = int(oy + dy / length * _AXIS_LENGTH)
        color = _AXIS_COLORS.get(label, (200, 200, 200))

        # Draw arrow line (thick)
        draw.line([(ox, oy), (ex, ey)], fill=color, width=3)
        # Draw arrowhead dot
        draw.ellipse([ex - 3, ey - 3, ex + 3, ey + 3], fill=color)
        # Draw label at the tip
        tx = int(ox + dx / length * (_AXIS_LENGTH + 12))
        ty = int(oy + dy / length * (_AXIS_LENGTH + 12))
        draw.text((tx - 8, ty - 6), label, fill=color, font=font)

    return np.array(img)


class MultiCameraRecordVideo(gym.Wrapper):
    """Records one video per camera angle using the env's ``capture_multi_cameras()`` method.

    Output files follow the pattern::

        <video_folder>/rl-video-step-<N>-<cam_name>.mp4

    The wrapper does **not** call ``env.render()`` — it goes directly to the
    multi-camera capture method on the unwrapped Isaac Lab environment.

    Visual enhancements for review agents:
        - **HUD overlay** (side/iso views): frame number, training step, reset count
        - **Red border + "RESET" label**: flashed on frames where an episode reset is detected
        - **Warmup renders**: first few render calls are discarded to avoid pipeline artifacts
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

        # Reset detection state
        self._prev_episode_len: int | None = None
        self._reset_count = 0

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

    def _get_episode_length(self) -> int | None:
        """Read episode_length_buf[0] from the unwrapped env, if available."""
        env = self.unwrapped
        buf = getattr(env, "episode_length_buf", None)
        if buf is not None and len(buf) > 0:
            return int(buf[0].item()) if hasattr(buf[0], "item") else int(buf[0])
        return None

    def _detect_reset(self) -> bool:
        """Return True if env 0 was just reset (episode length dropped)."""
        cur = self._get_episode_length()
        if cur is None:
            return False
        was_reset = self._prev_episode_len is not None and cur < self._prev_episode_len
        self._prev_episode_len = cur
        if was_reset:
            self._reset_count += 1
        return was_reset

    def _start_recording(self):
        self._recording = True
        self._record_start_step = self._step_count
        self._frames_recorded = 0
        self._video_writers = {}
        self._reset_count = 0
        self._prev_episode_len = self._get_episode_length()

        # Warmup renders to avoid first-frame artifacts
        if hasattr(self.unwrapped, "capture_multi_cameras"):
            for _ in range(_WARMUP_RENDERS):
                self.unwrapped.capture_multi_cameras()

    def _capture_frame(self):
        if hasattr(self.unwrapped, "capture_multi_cameras"):
            frames = self.unwrapped.capture_multi_cameras()
        else:
            frame = self.render()
            if frame is None:
                raise RuntimeError("Video recording requested, but the environment did not return an rgb frame.")
            frames = {"main": np.asarray(frame)}

        # Detect reset for this frame
        is_reset = self._detect_reset()

        for name, frame in frames.items():
            # Apply reset flash border
            if is_reset:
                frame = _draw_reset_border(frame)

            # Apply HUD overlay on selected views
            if name in _HUD_VIEWS:
                hud_text = f"F:{self._frames_recorded}  S:{self._record_start_step}  R:{self._reset_count}"
                frame = _draw_text_pil(frame, hud_text)

            # Draw axis orientation widget
            frame = _draw_axes(frame, name)

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
