#!/usr/bin/env python3

import os
import sys
import time
import rospy
import datetime
import numpy as np
import pandas as pd
from collections import deque
from typing import Tuple, Union
from sensor_pkg.msg import AllSensors
from sensor_msgs.msg  import JointState
from open_manipulator_msgs.srv import SetJointPosition, SetJointPositionRequest, SetKinematicsPose, SetKinematicsPoseRequest
np.set_printoptions(threshold=sys.maxsize, suppress=True)

WAYPOINT_PTH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets/detection/keypoint')

class MoveRobot():

	M_PI = np.pi
	M_PI_2 = np.pi*0.5

	RH8D_VEL = 1.225 # rad/s # 0.13s/60°
	ACTUATOR_VEL = 0.10472 # rad/s
	ACTUATOR_ACCEL = 0.17453 # rad/s^2

	HEAD_NAME = "/NICOL/head"
	ROBOT_NAME = "/right/open_manipulator_p"
	JOINT_GOAL_SERVICE = '/goal_joint_space_path'
	TASK_GOAL_SERVICE = '/goal_task_space_path'
	SENSOR_TOPIC = '/right/AllSensors'
	ACTUATOR_STATES = '/actuator_states'

	HEAD_JOINTS = ['joint_head_y', 'joint_head_z']

	INTERLEAVED_JOINTS = [ "jointI1", "jointM1", "jointL1R1"]

	ROBOT_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8", "jointI1", "jointL1R1", "jointM1", "jointT0", "jointT1"]
	RH8D_JOINTS    = ["joint8", "jointI1", "jointI2", "jointI3", "jointL1", "jointL2", "jointL3", "jointM1", "jointM2", "jointM3", "jointR1", "jointR2", "jointR3", "jointT0", "jointT1", "jointT2", "jointT3"]
	ROBOT_JOINTS_INDEX = dict(zip(ROBOT_JOINTS, range(len(ROBOT_JOINTS))))

	HEAD_LOW_LIM = [-0.584, -1.57]
	HEAD_UP_LIM = [1.282, 1.57]
	HEAD_INIT = [-0.58, 0.0]
	HEAD_HOME = [0.0, 0.0]
	HEAD_START = [0.8, -1.0]

	ROBOT_LOW_LIM = [0.0, -1.5, -2.25, -2.9, -M_PI_2, -M_PI, -M_PI, -M_PI, -M_PI, -M_PI, -M_PI, -M_PI, -M_PI]
	ROBOT_UP_LIM = [2.5, 1.8, 1.5, 2.9, M_PI_2, M_PI, M_PI, M_PI, M_PI, M_PI, M_PI, M_PI, M_PI]
	ROBOT_INIT = [0.31, 0.2, -0.2, 1.5, 1.28, 0.0, 0.0, 0.0, -M_PI, -M_PI, -M_PI, -M_PI, -M_PI]
	ROBOT_HOME = [M_PI_2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -M_PI, -M_PI, -M_PI, -M_PI, -M_PI]
	ROBOT_EXP_START = [1.91, 0.471, -1.131, -M_PI_2, M_PI_2, -M_PI, 0.0, 0.0, -M_PI, -M_PI, -M_PI, -M_PI, -M_PI]

	JOINT5_IDX = ROBOT_JOINTS_INDEX['joint5']
	JOINT6_IDX = ROBOT_JOINTS_INDEX['joint6']
	JOINT5_MAX = 1.56
	JOINT5_MIN = -1.56
	JOINT6_MAX = 3
	JOINT6_MIN = -3
	RH8D_WRIST_MAX = 2.8
	RH8D_WRIST_MIN = -2.8
	RH8D_FINGER_MAX = 3
	RH8D_FINGER_MIN = -3

	def __init__(self, sensors: bool=False) -> None:

		self.states = deque(maxlen=1) # np.ndarray
		
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
		
		if sensors:
			rospy.wait_for_message(self.SENSOR_TOPIC, AllSensors, 5)
			self.right_finger_sensor_sub = rospy.Subscriber(self.SENSOR_TOPIC, AllSensors, self.rightSensorsCB)

	def reachInitBlocking(self, t_path: float) -> bool:
		cmd = dict(zip(self.ROBOT_JOINTS, self.ROBOT_INIT))
		return self.reachPositionBlocking(cmd, t_path)

	def reachHomeBlocking(self, t_path: float) -> bool:
		cmd = dict(zip(self.ROBOT_JOINTS, self.ROBOT_HOME))
		return self.reachPositionBlocking(cmd, t_path)

	def reachPositionBlocking(self, cmd: dict, t_path: float, t_settle: float=0.0) -> Tuple[bool, Union[dict, None]]:
		# send position
		if not self.moveArmJointSpace(cmd, t_path):
			if len(self.states):
				crnt = self.states.pop()
				return False, dict(zip(list(cmd.keys()), list(crnt)))
			else:
				return False, None
		
		vals = list(cmd.values())
		goal = np.array(vals)
		crnt = np.ones(len(vals)) * np.inf
		t_start = time.time()
		# wait for reach or abort
		while not all( np.isclose(np.round(goal, 2), np.round(crnt, 2)) ):
			time.sleep(0.1)
			
			# abort
			if (time.time() - t_start) > t_path:
				print("Position reach timeout, goal:\n", list(goal), "\ncurrent:\n", list(crnt))
				return False, dict(zip(self.ROBOT_JOINTS, list(crnt)))

		time.sleep(t_settle)
		return True, dict(zip(self.ROBOT_JOINTS, list(crnt)))

	def moveArmJointSpace(self, cmd: dict, t_path: float) -> bool:
		names = list(cmd.keys())
		pos = list(cmd.values())
		self.limRobotPositions(names, pos)
		
		req = SetJointPositionRequest()
		req.joint_position.joint_name = names
		req.joint_position.position = pos
		req.path_time = t_path
		try:
			return self.right_joint_goal_client.call(req)
		except rospy.ServiceException as e:
			print(e)
			return False

	def limRobotPositions(self, names: list, vals: list) -> bool:
		assert(len(names) == len(self.ROBOT_JOINTS))
		assert(len(vals) == len(self.ROBOT_JOINTS))
		for idx in range(len(vals)):
			assert(names[idx] == self.ROBOT_JOINTS[idx])
			vals[idx] = max(min(vals[idx], self.ROBOT_UP_LIM[idx]), self.ROBOT_LOW_LIM[idx])

	def moveHeadInit(self, t_path: float=2.0) -> bool:
		req = SetJointPositionRequest()
		req.joint_position.joint_name = self.HEAD_JOINTS
		req.joint_position.position = self.HEAD_INIT
		req.path_time = t_path
		try:
			return self.head_joint_goal_client.call(req)
		except rospy.ServiceException as e:
			print(e)
			return False

	def moveHeadHome(self, t_path: float=2.0) -> bool:
		req = SetJointPositionRequest()
		req.joint_position.joint_name = self.HEAD_JOINTS
		req.joint_position.position = self.HEAD_HOME
		req.path_time = t_path
		try:
			return self.head_joint_goal_client.call(req)
		except rospy.ServiceException as e:
			print(e)
			return False

	def moveHeadJointSpace(self, pitch: float, yaw: float, t_path: float) -> bool:
		joint_y = max(min(pitch, self.HEAD_UP_LIM[0]), self.HEAD_LOW_LIM[0])
		joint_z = max(min(yaw, self.HEAD_UP_LIM[1]), self.HEAD_LOW_LIM[1])
		req = SetJointPositionRequest()
		req.joint_position.joint_name = self.HEAD_JOINTS
		req.joint_position.position = [joint_y, joint_z]
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

	@classmethod
	def estimateMoveTime(self, current: list, goal: list) -> float:
		omp_distance =	np.abs(np.array(current[ : self.JOINT6_IDX]) - np.array(goal[ : self.JOINT6_IDX])) 
		rh8d_distance = np.abs(np.array(current[self.JOINT6_IDX :]) - np.array(goal[self.JOINT6_IDX :])) 
		t_move_omp = np.max(omp_distance / self.ACTUATOR_VEL) + (self.ACTUATOR_VEL / self.ACTUATOR_ACCEL)
		t_move_rh8d = np.max(rh8d_distance / self.RH8D_VEL)
		return max(t_move_omp, t_move_rh8d)

	@classmethod
	def generateWaypointsSequential(self, 
					   								robot_resolution_deg: float=25.0, 
					   								wrist_resolution_deg: float=30.0, 
													finger_resolution_deg: float=20.0, 
													interleaving_offset_deg: float=10.0,
													t_exp: float=2.0,
													) -> None:
		"""	   	  1. move home position
					2. move start position
					3.1 move joint5 in save range from up to low
							3.2 move joint6 in save range from up to low
								3.1.3 move wrist flexion
								3.1.4 move wrist abduction
								3.1.5 move thumb abduction
								3.1.6 move thumb flexion
								3.1.7 move fingers interleaved:  index -> middle -> little/ring
		"""

		assert(interleaving_offset_deg <= finger_resolution_deg/2)

		t_move_home_start = self.estimateMoveTime(self.ROBOT_HOME, self.ROBOT_EXP_START)
		cols = self.ROBOT_JOINTS.copy()
		cols.extend(["description", "t_travel"])
		home = self.ROBOT_HOME.copy()
		home.extend(["home", t_move_home_start])
		start = self.ROBOT_EXP_START.copy()
		start.extend(["start", t_move_home_start])

		wps_df = pd.DataFrame(columns=cols)
		wps_df = pd.concat([wps_df, pd.DataFrame([dict(zip(cols, home))])], ignore_index=True) # 1st waypoint is home
		wps_df = pd.concat([wps_df, pd.DataFrame([dict(zip(cols, start))])], ignore_index=True) # 2nd waypoint is experiment start
		interleaving_offset = np.deg2rad(interleaving_offset_deg)
		
		# joint5
		steps1 = int( (abs(self.JOINT5_MIN) + abs(self.JOINT5_MAX)) / np.deg2rad(robot_resolution_deg) )
		joint5_waypoints = np.linspace(self.JOINT5_MIN, self.JOINT5_MAX, steps1)

		# joint6
		steps2 = int( (abs(self.JOINT6_MIN) + abs(self.JOINT6_MAX)) / np.deg2rad(robot_resolution_deg) )
		joint6_waypoints = np.linspace(self.JOINT6_MIN, self.JOINT6_MAX, steps2)

		# joint7-8
		steps3 = int( (abs(self.RH8D_WRIST_MIN) + abs(self.RH8D_WRIST_MAX)) / np.deg2rad(wrist_resolution_deg) )
		wrist_waypoints = np.linspace(self.RH8D_WRIST_MIN + np.deg2rad(wrist_resolution_deg), self.RH8D_WRIST_MAX, steps3)

		# jointI1-T1
		steps4 = int( (abs(self.RH8D_FINGER_MIN) + abs(self.RH8D_FINGER_MAX)) / np.deg2rad(finger_resolution_deg) )
		finger_waypoints = np.linspace(self.RH8D_FINGER_MIN + np.deg2rad(finger_resolution_deg), self.RH8D_FINGER_MAX, steps4)

		for joint5_wp in joint5_waypoints:
			for joint6_wp in joint6_waypoints:
				print("new sequence")

				# wrist abduction
				for rh8d_wp in wrist_waypoints:
					print("wrist abduction", len(wps_df))
					waypoint = self.ROBOT_EXP_START[ : self.JOINT5_IDX] # joint 1,2,3,4 fixed
					wrist_only = [joint5_wp, joint6_wp, rh8d_wp] # joint7 moving
					waypoint.extend(wrist_only)
					waypoint.extend(self.ROBOT_EXP_START[self.ROBOT_JOINTS_INDEX['joint8'] : ]) # joint8 - jointT1 fixed
					last_wp = wps_df.loc[wps_df.last_valid_index()][:-2].values
					t_travel = self.estimateMoveTime(last_wp, waypoint)
					waypoint.extend(["wrist abduction", t_travel])
					waypoint = dict(zip(cols, waypoint))
					wps_df = pd.concat([wps_df, pd.DataFrame([waypoint])],  ignore_index=True)

				# wrist flexion
				for rh8d_wp in wrist_waypoints:
					print("wrist flexion", len(wps_df))
					waypoint = self.ROBOT_EXP_START[ : self.JOINT5_IDX] # joint 1,2,3,4 fixed
					wrist_only = [joint5_wp, joint6_wp, self.ROBOT_EXP_START[self.ROBOT_JOINTS_INDEX['joint7']], rh8d_wp] # joint8 moving
					waypoint.extend(wrist_only)
					waypoint.extend(self.ROBOT_EXP_START[self.ROBOT_JOINTS_INDEX['jointI1'] : ]) # jointI1 - jointT1 fixed
					last_wp = wps_df.loc[wps_df.last_valid_index()][:-2].values
					t_travel = self.estimateMoveTime(last_wp, waypoint)
					waypoint.extend(["wrist flexion", t_travel])
					waypoint = dict(zip(cols, waypoint))
					wps_df = pd.concat([wps_df, pd.DataFrame([waypoint])],  ignore_index=True)

				# thumb abduction
				for rh8d_wp in finger_waypoints:
					print("thumb abduction", len(wps_df))
					waypoint = self.ROBOT_EXP_START[ : self.JOINT5_IDX] # joint 1,2,3,4 fixed
					thumb_only = [joint5_wp, joint6_wp, *self.ROBOT_EXP_START[self.ROBOT_JOINTS_INDEX['joint7'] : self.ROBOT_JOINTS_INDEX['jointT0']], rh8d_wp] # jointT0 moving
					waypoint.extend(thumb_only)
					waypoint.extend(self.ROBOT_EXP_START[self.ROBOT_JOINTS_INDEX['jointT1'] : ]) # jointT1 fixed
					last_wp = wps_df.loc[wps_df.last_valid_index()][:-2].values
					t_travel = self.estimateMoveTime(last_wp, waypoint)
					waypoint.extend(["thumb abduction", t_travel])
					waypoint = dict(zip(cols, waypoint))
					wps_df = pd.concat([wps_df, pd.DataFrame([waypoint])],  ignore_index=True)

				# thumb flexion (abduction at max)
				for rh8d_wp in finger_waypoints:
					print("thumb flexion", len(wps_df))
					waypoint = self.ROBOT_EXP_START[ : self.JOINT5_IDX] # joint 1,2,3,4 fixed
					thumb_only = [joint5_wp, joint6_wp, *self.ROBOT_EXP_START[self.ROBOT_JOINTS_INDEX['joint7'] : self.ROBOT_JOINTS_INDEX['jointT0']], self.ROBOT_UP_LIM[self.ROBOT_JOINTS_INDEX['jointT0']], rh8d_wp] # jointT1 moving
					waypoint.extend(thumb_only)
					last_wp = wps_df.loc[wps_df.last_valid_index()][:-2].values
					t_travel = self.estimateMoveTime(last_wp, waypoint)
					waypoint.extend(["thumb flexion", t_travel])
					waypoint = dict(zip(cols, waypoint))
					wps_df = pd.concat([wps_df, pd.DataFrame([waypoint])],  ignore_index=True)

				# reverse
				inverted_seq = wps_df.tail(len(finger_waypoints)+1)
				inverted_seq = inverted_seq.iloc[::-1]
				wps_df = pd.concat([wps_df, inverted_seq], ignore_index=True)

				# fingers interleaved
				finger_start_cnt = len(wps_df)
				wp_cnt = len(self.INTERLEAVED_JOINTS)*[0]
				while wps_df.loc[wps_df.last_valid_index(), self.INTERLEAVED_JOINTS[-1]] < finger_waypoints[-1]:
					print("finger flexion", len(wps_df))
					last_waypoint =  wps_df.loc[wps_df.last_valid_index()]
					waypoint = last_waypoint.copy()

					for idx, joint in enumerate(self.INTERLEAVED_JOINTS):
						if (idx == 0 and last_waypoint[self.INTERLEAVED_JOINTS[0]]  < finger_waypoints[-1]) \
							or (last_waypoint[self.INTERLEAVED_JOINTS[idx-1]] >= interleaving_offset):
							if wp_cnt[idx] < len(finger_waypoints):
								waypoint[joint] = finger_waypoints[wp_cnt[idx]]
								wp_cnt[idx] += 1
							else:
								waypoint[joint] = finger_waypoints[-1]

					t_travel = self.estimateMoveTime(last_waypoint[:-2].values, waypoint[:-2].values)
					waypoint[-2] = "finger flexion"
					waypoint[-1] = t_travel

					wps_df = pd.concat([wps_df, pd.DataFrame([waypoint])],  ignore_index=True)		

				# reverse
				inverted_seq = wps_df.tail(len(wps_df) - finger_start_cnt)
				inverted_seq = inverted_seq.iloc[::-1]
				wps_df = pd.concat([wps_df, inverted_seq], ignore_index=True)

		# last waypoint is home
		wps_df = pd.concat([wps_df, pd.DataFrame([dict(zip(cols, home))])], ignore_index=True) 

		t_exp_sum = len(wps_df)*t_exp
		t_travel_sum = wps_df['t_travel'].sum()
		t_total = t_travel_sum+t_exp_sum
		t_total = str(datetime.timedelta(seconds=t_total))
		t_str = t_total.split('.')[0].replace(':','_')
		wps_df.to_json(os.path.join(WAYPOINT_PTH, f'waypoints_{len(wps_df)}_{int(robot_resolution_deg)}_{int(wrist_resolution_deg)}_{int(finger_resolution_deg)}_{t_str}.json'), orient="index", indent=1, double_precision=3)		
		print("\nTotal cnt:", len(wps_df), "\nestimated experiment time:", t_exp_sum, "s",  "\nestimated move time:", t_travel_sum, "s", "\nestimated time total:", t_total)
		
if __name__ == '__main__':
	MoveRobot.generateWaypointsSequential(30, 20, 20, 10)
