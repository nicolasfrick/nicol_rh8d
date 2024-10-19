#!/usr/bin/env python3

import time
import rospy
import numpy as np
from collections import deque
from sensor_pkg.msg import AllSensors
from sensor_msgs.msg  import JointState
from open_manipulator_msgs.srv import SetJointPosition, SetJointPositionRequest, SetKinematicsPose, SetKinematicsPoseRequest

class MoveRobot():

    HEAD_NAME = "/NICOL/head"
    ROBOT_NAME = "/right/open_manipulator_p"
    JOINT_GOAL_SERVICE = '/goal_joint_space_path'
    TASK_GOAL_SERVICE = '/goal_task_space_path'
    SENSOR_TOPIC = ' /right/AllSensors'
    ACTUATOR_STATES = '/actuator_states'
    HEAD_JOINTS = ['joint_head_y', 'joint_head_z']
    HEAD_INIT_POS= [0.0, 0.0]

    def __init__(self) -> None:

        self.head_joint_goal_client = rospy.ServiceProxy(self.HEAD_NAME + self.JOINT_GOAL_SERVICE, SetJointPosition)
        self.head_joint_goal_client.wait_for_service(5)
        self.head_task_goal_client = rospy.ServiceProxy(self.HEAD_NAME + self.TASK_GOAL_SERVICE, SetKinematicsPose)
        self.head_task_goal_client.wait_for_service(5)
        rospy.wait_for_message(self.HEAD_NAME + self.ACTUATOR_STATES, JointState, 5)
        self.head_states_sub = rospy.Subscriber(self.HEAD_NAME + self.ACTUATOR_STATES, JointState, self.headJointStatesCB, queue_size=100)

        self.right_joint_goal_client = rospy.ServiceProxy(self.ROBOT_NAME + self.JOINT_GOAL_SERVICE, SetJointPosition)
        self.right_joint_goal_client.wait_for_service(5)
        rospy.wait_for_message(self.ROBOT_NAME + self.ACTUATOR_STATES, JointState, 5)
        self.right_states_sub = rospy.Subscriber(self.ROBOT_NAME + self.ACTUATOR_STATES, JointState, self.rightJointStatesCB)
        rospy.wait_for_message(self.SENSOR_TOPIC, AllSensors, 5)
        self.right_finger_sensor_sub = rospy.Subscriber(self.SENSOR_TOPIC, AllSensors, self.rightSensorsCB)

        self.states = deque(maxlen=1)

    def reachPositionBlocking(self, cmd: dict, t_path: float) -> bool:
        if not self.moveArmJointSpace(cmd, t_path):
            return False
        
        cnt = 0
        vals = list(cmd.values())
        goal = np.array(vals)
        crnt = np.ones(len(vals)) * np.inf
        while not np.isclose(goal, crnt):
            time.sleep(0.1)
            if len(self.states):
                crnt = self.states.pop()
            cnt += 1
            if cnt > 500:
                print("Position reach timeout, goal: ", list(goal), "current", list(crnt))
                return False

        return True

    def moveArmJointSpace(self, cmd: dict, t_path: float) -> bool:
        names = list(cmd.keys())
        pos = list(cmd.values())
        req = SetJointPositionRequest()
        req.joint_position.joint_name = names
        req.joint_position.position = pos
        req.path_time = t_path
        try:
            return self.right_joint_goal_client.call(req)
        except rospy.ServiceException as e:
            print(e)
            return False

    def moveHeadJointSpace(self, pitch: float, yaw: float, t_path: float) -> bool:
        req = SetJointPositionRequest()
        req.joint_position.joint_name = self.HEAD_JOINTS
        req.joint_position.position = [pitch, yaw]
        req.path_time = t_path
        try:
            return self.head_joint_goal_client.call(req)
        except rospy.ServiceException as e:
            print(e)
            return False

    def moveHeadTaskSpace(self, x: float, y: float, z: float, t_path: float) -> bool:
        req = SetKinematicsPoseRequest()
        req.kinematics_pose.pose.position.x = x
        req.kinematics_pose.pose.position.x = y
        req.kinematics_pose.pose.position.x = z
        req.path_time = t_path
        try:
            return self.head_task_goal_client.call(req)
        except rospy.ServiceException as e:
            print(e)
            return False

    def rightJointStatesCB(self, msg: JointState) -> None:
        self.states.append(np.array(msg.position))
            
    def headJointStatesCB(self, msg: JointState) -> None:
        pass

    def rightSensorsCB(self, msg: AllSensors) -> None:
        pass

