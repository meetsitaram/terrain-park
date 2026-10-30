"""Shared paths: the X2 model comes from agibot-x2-record-and-replay (set X2_RECORDER to its checkout; default: a
sibling folder of this repo)."""
import os
from pathlib import Path

X2_RECORDER = Path(os.environ.get('X2_RECORDER', Path(__file__).resolve().parents[2] / 'agibot-x2-record-and-replay'))
X2_XML = X2_RECORDER / 'sim' / 'x2_ultra.xml'
KEY_N, KEY_P, KEY_SPACE = 78, 80, 32          # MuJoCo viewer (GLFW) key codes
