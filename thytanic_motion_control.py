"""
Connection to the Thymio (aka the Thytanic) and control it.

Classes:
- State: Define the three operational states of the Thymio.
- ObstacleDirection: Define the direction (left or right) of the detected obstacle.
- ControlThytanic: Control the Thymio.
"""

from enum import Enum
import numpy as np
from tdmclient import ClientAsync, aw

class State(Enum):
    """Define the three operational states of the Thymio."""

    GLOBAL_NAVIGATION = 0
    OBSTACLE_AVOIDANCE = 1
    STOP = 2

class ObstacleDirection(Enum):
    """"Define the direction (left or right) of the detected obstacle"""

    LEFT = 0
    RIGHT = 1


