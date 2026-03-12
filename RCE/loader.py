"""
Streamlit-free Middlebury stereo loader for RCE notebook.
Lists scenes and loads left (im0) / right (im1) images only.
"""

import os
from pathlib import Path

import cv2
import numpy as np

# Default: project_root/data/middlebury (RCE/ is under project root)
_DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data" / "middlebury"


def get_scene_list(root_path=None):
    """
    List subdirectories that contain both im0.png and im1.png.
    Returns sorted list of scene names.
    """
    if root_path is None:
        root_path = _DEFAULT_ROOT
    root_path = Path(root_path)
    if not root_path.is_dir():
        return []
    scenes = []
    for entry in sorted(os.listdir(root_path)):
        scene_dir = root_path / entry
        if not scene_dir.is_dir():
            continue
        if (scene_dir / "im0.png").is_file() and (scene_dir / "im1.png").is_file():
            scenes.append(entry)
    return scenes


def load_stereo_pair(root_path, scene_name):
    """
    Load left (im0) and right (im1) images for a scene.
    root_path: path to data/middlebury (or equivalent).
    scene_name: e.g. 'artroom1'.
    Returns dict with 'left' and 'right' (BGR ndarrays).
    """
    root_path = Path(root_path)
    scene_dir = root_path / scene_name
    left_path = scene_dir / "im0.png"
    right_path = scene_dir / "im1.png"
    left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
    if left is None or right is None:
        raise FileNotFoundError(f"Missing im0.png or im1.png in {scene_dir}")
    return {"left": left, "right": right}
