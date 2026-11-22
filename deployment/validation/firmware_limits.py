"""Map StreamingControl's encoder-angle clamps into manager policy joint coordinates."""

from pathlib import Path
import re

import numpy as np

from common.policy_config import JOINT_CATEGORIES, policy_wire_sign, resolve_deployment_joint_sign
from deployment.inference.stance import load_hw_default_pose, load_rl_default_pose


ROOT = Path(__file__).resolve().parents[2]
FIRMWARE = ROOT / "firmware/StreamingControl/HaroldStreamingControl/HaroldStreamingControl.ino"


def manager_joint_target_limits(firmware_path: Path = FIRMWARE) -> tuple[np.ndarray, np.ndarray]:
    """Return absolute lower and upper RL joint targets in radians, before wire/tick rounding.

    Uses the same stance and serial-target signs as ActionConverter. StreamingControl
    multiplies the serial target by JOINT_SIGN before clipping the physical angle.
    Its per-servo direction table is applied afterwards and does not affect the bounds.
    """
    source = re.sub(r"//[^\n]*|/\*.*?\*/", "", firmware_path.read_text(), flags=re.DOTALL)
    array = re.search(r"\bconst\s+float\s+JOINT_SIGN\s*\[\s*12\s*\]\s*=\s*\{([^}]*)\}\s*;", source)
    if array is None:
        raise ValueError(f"Missing JOINT_SIGN in {firmware_path}")
    sign_literals = array[1].strip().rstrip(",").split(",")
    firmware_sign = np.array([float(s.strip().rstrip("fF")) for s in sign_literals])
    config_dir = ROOT / "deployment/config"
    physical_joint_sign = resolve_deployment_joint_sign(hardware_path=config_dir / "hardware.yaml")
    wire_sign = np.asarray(policy_wire_sign(physical_joint_sign))
    for signs in (firmware_sign, wire_sign):
        if signs.shape != (12,) or not np.isin(signs, [-1, 1]).all():
            raise ValueError("Expected 12 joint signs of +1 or -1")

    limits = {}
    for joint, side, literal in re.findall(
        r"\bconst\s+float\s+(SHOULDER|THIGH|CALF)_(MIN|MAX)_DEG\s*=\s*([^;]+);", source
    ):
        limits[joint.lower(), side] = np.deg2rad(float(literal.strip().rstrip("fF")))

    try:
        physical = np.array([[limits[joint, side] for joint in JOINT_CATEGORIES] for side in ("MIN", "MAX")])
    except KeyError as missing:
        raise ValueError(f"Missing firmware clamp {missing} in {firmware_path}") from missing
    if not np.isfinite(physical).all() or not (physical[0] < physical[1]).all():
        raise ValueError("Invalid physical clamp interval")

    wire = physical / firmware_sign
    manager = load_rl_default_pose() + (wire - load_hw_default_pose()) / wire_sign
    lower, upper = np.sort(manager.astype(np.float32), axis=0)
    return lower, upper
