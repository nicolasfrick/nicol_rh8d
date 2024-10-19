#!/usr/bin/env python3

import os
import sys
import rospy
from typing import Optional, Any, Tuple
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
import open_manipulator_msgs.msg
import open_manipulator_msgs.srv

class MoveRobot():

    def __init__(self) -> None:
        pass