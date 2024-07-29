#!/usr/bin/env python3

# Imports
import math
import operator
import functools
import fractions
import dataclasses
import collections
from typing import Sequence, Optional, Tuple, Union, Any
import numpy as np
import cv2
import cv_bridge
import rospy
import tf2_ros
import sensor_msgs.msg
import geometry_msgs.msg
import open_manipulator_msgs.msg
import open_manipulator_msgs.srv
import dynamic_reconfigure.server
import camera_models
from nicol_rh8d.cfg import RH8DDatasetCollectorConfig

# Main function
def main():
	rospy.init_node('RH8D_dataset_collector')
	RH8DDatasetCollector(
		move_hand=rospy.get_param('~move_hand', False),
		hand_period=rospy.get_param('~hand_period', 1.3),
		hand_increment=rospy.get_param('~hand_increment', 0.1),
		min_pitch=rospy.get_param('~min_pitch', 1.0),
		max_pitch=rospy.get_param('~max_pitch', 0.2),
		max_yaw=rospy.get_param('~max_yaw', 0.8),
		detect_markers=rospy.get_param('~detect_markers', True),
		refmarker_csv=rospy.get_param('~refmarker_csv', ''),  # Note: Don't append '.csv'
		use_cameras=(rospy.get_param('~use_markers_camera', True), rospy.get_param('~use_depth', False)),
        camera_width=rospy.get_param('~camera_width', 1920),
		camera_height=rospy.get_param('~camera_height', 1080),
		record_visbag=rospy.get_param('~record_visbag', ''),
		debug=rospy.get_param('~debug', False),
	    depth_width=rospy.get_param('~depth_width', 1280),
		depth_height=rospy.get_param('~depth_height', 720),
		tf_listener=rospy.get_param('~use_tf', True),
	).run(rate_hz=rospy.get_param('~f_loop', 1.0))

@dataclasses.dataclass(eq=False)
class RefMarker:
	camera_id: int
	stamp: rospy.Time
	centrex: float
	centrey: float
	radius: float
	area: float
	aspect: float
	best_dist: float
	track_id: int = -1
	track_age: int = 1
	centrex_smooth: Optional[float] = None
	centrey_smooth: Optional[float] = None
	stability: float = 0.0
	saved_to_csv: int = 0
	centrex_csv: Optional[float] = None
	centrey_csv: Optional[float] = None
	csv_dist: float = math.inf

	def __repr__(self):
		return f'{self.__class__.__name__}(cam={self.camera_id}, track={self.track_id}, age={self.track_age}, aspect={self.aspect:.2f}, area={self.area:.0f}, radius={self.radius:.1f}, centre=({self.centrex:.1f}, {self.centrey:.1f}), dist={self.best_dist:.1f}, csv={self.csv_dist:.1f}, stable={self.stability:.1f})'

# Dataset collector class
class RH8DDatasetCollector():

	CAMERAS = ('marker_realsense/color/image_raw', 'marker_realsense/aligned_depth_to_color/image_raw')
	JOINT_SERVICE = '/right/open_manipulator_p/goal_joint_space_path'
	JOINT_STATES = '/right/open_manipulator_p/actuator_states'
	LABEL_THICKNESS = 3
	LABEL_FONTSIZE = 3.0
	
	camera_image_topics: Tuple[Optional[str], ...]
	camera_info_topics: Tuple[Optional[str], ...]
	camera_resolutions: Tuple[Optional[Tuple[int, int]], ...]
	tf_buffer: Optional[tf2_ros.Buffer]

	def __init__(self, 
			                move_hand: bool, 
                            hand_period: float, 
                            hand_increment: float, 
                            min_pitch: float, 
                            max_pitch: float, 
                            max_yaw: float, 
                            detect_markers: bool, 
                            refmarker_csv: str, 
                            use_cameras: bool, 
                            camera_width: int, 
                            camera_height: int, 
                            record_visbag: str, 
							debug: bool, 
                            depth_width: int, 
                            depth_height: int, 
							camera_force_copy: bool = False,
			                tf_listener: Union[bool, float] = False,
			                tf_camera_qsize: int = 15):

		self.move_hand = move_hand
		self.hand_period = max(hand_period, 1.3)
		self.hand_increment = hand_increment
		self.min_pitch = min_pitch
		self.max_pitch = max_pitch
		self.max_yaw = max_yaw
		self.detect_markers = detect_markers
		self.use_cameras = use_cameras if self.detect_markers else (False, False)
		self.camera_width = camera_width
		self.camera_height = camera_height
		if refmarker_csv.endswith('.csv'):
			refmarker_csv = refmarker_csv[:-4]
		self.refmarker_csv = [f'{refmarker_csv}_{camera}.csv' if refmarker_csv and use_camera else None for camera, use_camera in zip(self.CAMERAS, self.use_cameras)]
		self.tracks_csv = [f'{refmarker_csv}_{camera}_tracks.csv' if refmarker_csv and use_camera else None for camera, use_camera in zip(self.CAMERAS, self.use_cameras)]
		self.record_visbag = record_visbag
		self.debug = debug
		self.depth_width = depth_width
		self.depth_height = depth_height
		
		# Name to use for the demo (e.g. 'MyRobotDemo', default is class name)
		self.name = self.__class__.__name__
		rospy.loginfo(f"Initialising {self.name}")
		
		# Dynamic reconfigure config type to use (i.e. *Config)
		self.config_type = RH8DDatasetCollectorConfig
		rospy.loginfo(f"Using dynamic reconfigure with config {self.config_type.__name__}")
		self.config_server = self.config = None

		# Iterable of topics to subscribe to for camera images (None/empty = skip)
		self.camera_image_topics = tuple(f'/{camera}' if use_camera else None for camera, use_camera in zip(self.CAMERAS, self.use_cameras)),
		self.num_cameras = len(self.camera_image_topics)

		# Iterable of topics to subscribe to for camera info or None/empty to ignore (all ignored if False, all auto-constructed from image topic if True, length must match camera_image_topics if iterable provided)
		self.camera_info_topics = tuple((f'{rospy.names.resolve_name(image_topic).rsplit("/", maxsplit=1)[0]}/camera_info' if image_topic else None) for image_topic in self.camera_image_topics)
		if len(self.camera_info_topics) != self.num_cameras:
			raise ValueError(f"Must have equal number of camera image and info topics ({self.num_cameras} vs {len(self.camera_info_topics)})")
		
		# Iterable of target camera resolutions as (width, height) pairs or None to keep received resolution (None = Keep resolution for all, single value = use for all, length must match camera_image_topics if iterable provided)
		self.camera_resolutions = ((self.camera_width, self.camera_height), (self.depth_width, self.depth_height))
		if len(self.camera_resolutions) != self.num_cameras:
			raise ValueError(f"Must have equal number of camera image topics and resolutions ({self.num_cameras} vs {len(self.camera_resolutions)})")

		self.camera_stamp = [rospy.Time() for _ in range(self.num_cameras)]
		# Force the camera images to either be a copy or a rescaled version of the raw received image (in any case not the same object)
		self.camera_force_copy = camera_force_copy
		self.camera_cv_bridge = cv_bridge.CvBridge()

		for c, (image_topic, info_topic, resolution) in enumerate(zip(self.camera_image_topics, self.camera_info_topics, self.camera_resolutions)):
			if image_topic:
				rospy.loginfo(f"Camera {c} image: {image_topic}")
				if info_topic:
					rospy.loginfo(f"Camera {c} info:  {info_topic}")
				rospy.loginfo(f"Camera {c} size:  {f'{resolution[0]}x{resolution[1]}' if resolution is not None else 'Dynamic'}")

		# Whether to listen to TFs (if a strictly positive float then use this as the cache time)
		self.tf_cache_time = 5.0
		if isinstance(tf_listener, float):
			if tf_listener > 0:
				self.tf_cache_time = tf_listener
				tf_listener = True
			else:
				tf_listener = False

		# Maximum TF camera image queue size per camera (camera images that are awaiting their corresponding TF transform)
		self.tf_camera_qsize = max(tf_camera_qsize, 1)
		if tf_listener:
			self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration.from_sec(self.tf_cache_time))
			self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
			self.tf_image_queues = tuple(collections.deque(maxlen=self.tf_camera_qsize) for _ in range(self.num_cameras))
			rospy.loginfo(f"Started TF listener with {self.tf_cache_time:.1f}s cache time")
			if not self.tf_buffer.can_transform(target_frame='base_link', source_frame='world', time=rospy.Time.now(), timeout=rospy.Duration(secs=1)):
				rospy.logwarn("Cannot detect suitable TF transforms")
		else:
			self.tf_buffer = self.tf_listener = None

		self.srv_hand_pose = self.move_hand_timer = None
		self.move_hand_counter = -5
		self.num_pitches = max(abs(round((self.max_pitch - self.min_pitch) / self.hand_increment)) + 1, 2)
		self.num_yaws = max(abs(round(2 * self.max_yaw / self.hand_increment)) + 1, 2)
		self.pitches = tuple(((self.num_pitches - i - 1) * self.min_pitch + i * self.max_pitch) / (self.num_pitches - 1) for i in range(self.num_pitches))
		self.yaws = tuple((2 * j / (self.num_yaws - 1) - 1) * self.max_yaw for j in range(self.num_yaws))

		self.pub_visimg = None
		self.pub_visimg_enable = [True, True]
		self.next_track_ids = [0, 0]
		self.last_refmarkers = [[], []]
		self.prepared_run = False
		self.running = False

	def prepare_run(self):

		if self.detect_markers:
			rospy.loginfo("Detecting markers and publishing visualisation images")
			self.prepare_refmarker_csvs()
			self.pub_visimg = tuple(rospy.Publisher(f'~{camera}/visimg', sensor_msgs.msg.Image, queue_size=1) if use_camera else None for camera, use_camera in zip(self.CAMERAS, self.use_cameras))

		self.prepared_run = True

		if self.config_type is not None:
			self.config_server = dynamic_reconfigure.server.Server(self.config_type, self.config_callback)

		# camera_listeners = []
		# for c, (image_topic, info_topic, info_yaml, resolution) in enumerate(zip(self.camera_image_topics, self.camera_info_topics, self.camera_info_yamls, self.camera_resolutions)):
		# 	if not image_topic:
		# 		camera_listeners.append(None)
		# 	else:
		# 		if not info_topic or info_yaml:
		# 			info_data = camera_models.load_calibration_file(info_yaml) if info_yaml else None
					# camera_listener = rospy.Subscriber(image_topic, sensor_msgs.msg.Image, functools.partial(self.camera_callback, info_data=info_data, camera_id=c, resolution=resolution), queue_size=1, buff_size=40000000)
				# else:
					# sub_image = message_filters.Subscriber(image_topic, sensor_msgs.msg.Image, queue_size=1, buff_size=40000000)
					# sub_info = message_filters.Subscriber(info_topic, sensor_msgs.msg.CameraInfo)
					# camera_listener = message_filters.TimeSynchronizer((sub_image, sub_info), queue_size=3, reset=True)
					# camera_listener.registerCallback(functools.partial(self.camera_callback, camera_id=c, resolution=resolution))
		# 		camera_listeners.append(camera_listener)
		# 		rospy.loginfo(f"Subscribed to camera {c}")
		# self.camera_listeners = tuple(camera_listeners)

		if self.move_hand:
			rospy.loginfo("Moving hand to generate calibration data")
			self.srv_hand_pose = rospy.ServiceProxy(self.JOINT_SERVICE, open_manipulator_msgs.srv.SetJointPosition, persistent=True)
			rospy.loginfo("Initialising hand pose to start of scanning pattern")
			self.move_hand_to(index=0, move_time=1.5 * self.hand_period)
			self.move_hand_timer = rospy.Timer(period=rospy.Duration.from_sec(self.hand_period), callback=functools.partial(self.move_hand_callback, move_time=min(self.hand_period - 0.7, 1.0)))

	# noinspection PyUnusedLocal
	def move_hand_callback(self, event: Optional[rospy.timer.TimerEvent] = None, move_time=1.0):
		self.move_hand_counter += 1
		if self.move_hand_counter > 0:
			if self.move_hand_counter == 1:
				rospy.loginfo("Starting hand scanning pattern")
			self.move_hand_to(self.move_hand_counter, move_time=move_time)

	def move_hand_to(self, index, move_time):
		i, j = divmod(index, self.num_yaws)
		if i % 2 == 1:
			j = self.num_yaws - 1 - j
		if i >= self.num_pitches:
			pitch = yaw = 0.0
			move_time = 1.5 * self.hand_period
			self.move_hand_timer.shutdown()
			rospy.loginfo("Stopping hand scanning pattern")
			rospy.loginfo("Returning hand to neutral position")
		else:
			pitch = self.pitches[i]
			yaw = self.yaws[j]
		try:
			self.srv_hand_pose(open_manipulator_msgs.srv.SetJointPositionRequest(joint_position=open_manipulator_msgs.msg.JointPosition(position=(pitch, yaw)), path_time=move_time))
		except Exception as e:
			rospy.logerr_throttle_identical(20.0, f"{e.__class__.__name__}: {e}")
			
	# noinspection PyUnusedLocal
	def config_callback(self, config, level):
		self.config = config  # Atomic thread-safe simple assignment
		return config

	def reset_camera_history(self, camera_id: int):
		rospy.loginfo(f"Resetting camera {camera_id} marker history due to jump backwards in time")
		self.stop_refmarker_csv(camera_id)
		self.next_track_ids[camera_id] = 0
		self.last_refmarkers[camera_id] = []
		if self.record_visbag:
			self.pub_visimg_enable[camera_id] = False

	def prepare_refmarker_csvs(self):
		for camera_id, refmarker_csv in enumerate(self.refmarker_csv):
			if refmarker_csv:
				rospy.loginfo(f"Camera {camera_id} marker CSV: {refmarker_csv}")
				with open(refmarker_csv, 'w') as file:
					print("timestamp,track_id,centre_x,centre_y,img_width,img_height,tfrm_source_frame,tfrm_target_frame,tfrm_trans_x,tfrm_trans_y,tfrm_trans_z,tfrm_rot_x,tfrm_rot_y,tfrm_rot_z,tfrm_rot_w", file=file)
		for camera_id, tracks_csv in enumerate(self.tracks_csv):
			if tracks_csv:
				rospy.loginfo(f"Camera {camera_id} marker CSV: {tracks_csv}")
				with open(tracks_csv, 'w') as file:
					print("name,world_x,world_y,world_z,track_id", file=file)

	def append_refmarker_csv(self, camera_id: int, stamp: rospy.Time, img_size: tuple[int, int], camera_tfrm: geometry_msgs.msg.TransformStamped, refmarkers: Sequence[RefMarker]):
		if refmarkers:
			if tracks_csv := self.tracks_csv[camera_id]:
				with open(tracks_csv, 'a') as file:
					for refmarker in refmarkers:
						if refmarker.centrex_csv is None:
							print(f",,,,{refmarker.track_id}", file=file)
			if refmarker_csv := self.refmarker_csv[camera_id]:
				row_pre = f"{stamp.to_sec():.9f}"
				row_post = f"{img_size[0]},{img_size[1]},{camera_tfrm.child_frame_id},{camera_tfrm.header.frame_id},{camera_tfrm.transform.translation.x:.9f},{camera_tfrm.transform.translation.y:.9f},{camera_tfrm.transform.translation.z:.9f},{camera_tfrm.transform.rotation.x:.9f},{camera_tfrm.transform.rotation.y:.9f},{camera_tfrm.transform.rotation.z:.9f},{camera_tfrm.transform.rotation.w:.9f}"
				with open(refmarker_csv, 'a') as file:
					for refmarker in refmarkers:
						print(f"{row_pre},{refmarker.track_id},{refmarker.centrex:.3f},{refmarker.centrey:.3f},{row_post}", file=file)
						refmarker.saved_to_csv = 4
						refmarker.centrex_csv = refmarker.centrex
						refmarker.centrey_csv = refmarker.centrey

	def stop_refmarker_csv(self, camera_id):
		rospy.loginfo(f"Saved camera {camera_id} marker CSV: {self.refmarker_csv[camera_id]}")
		rospy.loginfo(f"Saved camera {camera_id} tracks CSV: {self.tracks_csv[camera_id]}")
		self.refmarker_csv[camera_id] = None
		self.tracks_csv[camera_id] = None

	def process_camera_image(self, camera_id, stamp, frame_id, img, raw_img, raw_ratio, camera_model, camera_tfrm):

		scalesq = (1920 * 1080) / (img.shape[0] * img.shape[1])  # Note: The cutoffs/constants used in this method are relative to the nominal calibration resolution of 1920x1440
		scale = math.sqrt(scalesq)

		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(3, 3))
		img_binary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img_binary = cv2.threshold(img_binary, thresh=self.config.binary_threshold, maxval=255, type=cv2.THRESH_BINARY_INV)[1]
		img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel, iterations=1)
		img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel, iterations=1)

		img_red_mask = np.where(img_binary[:, :, None], np.array((0, 0, 255), dtype=img.dtype)[None, None, :], img)
		cv2.addWeighted(src1=img, alpha=0.3, src2=img_red_mask, beta=0.7, gamma=0.0, dst=img)

		refmarkers = []
		for contour in cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
			(centrex, centrey), radius = cv2.minEnclosingCircle(contour)  # Note: Returned centre is in pixel coordinates, and radius is the maximum pixel distance from the returned centre to the centre of any pixel on the border of the contour
			area = cv2.contourArea(contour)
			width, height = cv2.minAreaRect(contour)[1]
			aspect = max(width, height) / min(width, height)
			border = 2 * radius
			if (
				(centrey + 0.5) * scale - 0.5 >= self.config.centrey_min and radius * scale < self.config.radius_max and
				self.config.area_min < area * scalesq < self.config.area_max and aspect < self.config.aspect_max and
				border < centrex + 0.5 < img.shape[1] - border and border < centrey + 0.5 < img.shape[0] - border
			):
				refmarkers.append(RefMarker(camera_id=camera_id, stamp=stamp, centrex=centrex, centrey=centrey, radius=radius, area=area, aspect=aspect, best_dist=math.inf))
		refmarkers.sort(key=operator.attrgetter('area'), reverse=True)

		possible_matches = []
		dist_thres = self.config.track_tol / scale
		last_refmarkers = self.last_refmarkers[camera_id]
		for refmarker in refmarkers:
			for last_refmarker in last_refmarkers:
				dist = math.sqrt((refmarker.centrex - last_refmarker.centrex) ** 2 + (refmarker.centrey - last_refmarker.centrey) ** 2)
				if dist < dist_thres:
					possible_matches.append((dist, refmarker, last_refmarker))
		possible_matches.sort(key=operator.itemgetter(0))
		self.last_refmarkers[camera_id] = refmarkers

		available_refmarkers = {refmarker: None for refmarker in refmarkers}
		available_last_refmarkers = set(last_refmarkers)
		smooth_cutoff_n = math.log(0.05) / self.config.smooth_ts
		for match in possible_matches:
			dist, refmarker, last_refmarker = match
			if refmarker in available_refmarkers and last_refmarker in available_last_refmarkers:
				del available_refmarkers[refmarker]
				available_last_refmarkers.remove(last_refmarker)
				refmarker.best_dist = dist
				refmarker.track_id = last_refmarker.track_id
				refmarker.track_age = last_refmarker.track_age + 1
				alpha = 1 - math.exp(smooth_cutoff_n * (refmarker.stamp - last_refmarker.stamp).to_sec())
				refmarker.centrex_smooth = last_refmarker.centrex_smooth + alpha * (0.5 * (refmarker.centrex + last_refmarker.centrex) - last_refmarker.centrex_smooth)
				refmarker.centrey_smooth = last_refmarker.centrey_smooth + alpha * (0.5 * (refmarker.centrey + last_refmarker.centrey) - last_refmarker.centrey_smooth)
				refmarker.stability = math.sqrt((refmarker.centrex - refmarker.centrex_smooth) ** 2 + (refmarker.centrey - refmarker.centrey_smooth) ** 2)
				if last_refmarker.saved_to_csv >= 1:
					refmarker.saved_to_csv = last_refmarker.saved_to_csv - 1
				refmarker.centrex_csv = last_refmarker.centrex_csv
				refmarker.centrey_csv = last_refmarker.centrey_csv

		next_track_id = self.next_track_ids[camera_id]
		for refmarker in available_refmarkers:
			refmarker.track_id = next_track_id
			refmarker.centrex_smooth = refmarker.centrex
			refmarker.centrey_smooth = refmarker.centrey
			next_track_id += 1
		self.next_track_ids[camera_id] = next_track_id

		refmarkers_stable = True
		csv_refmarkers = []
		for refmarker in refmarkers:
			if refmarker.track_age >= self.config.track_age_min:
				refmarkers_stable &= refmarker.stability * scale < self.config.stable_dist
				if refmarker.centrex_csv is not None:
					refmarker.csv_dist = math.sqrt((refmarker.centrex - refmarker.centrex_csv) ** 2 + (refmarker.centrey - refmarker.centrey_csv) ** 2)
				if refmarker.csv_dist * scale > self.config.debounce_dist:
					csv_refmarkers.append(refmarker)
		if refmarkers_stable:
			self.append_refmarker_csv(camera_id=camera_id, stamp=stamp, img_size=(img.shape[1], img.shape[0]), camera_tfrm=camera_tfrm, refmarkers=csv_refmarkers)

		track_text_scale = self.LABEL_FONTSIZE / scale
		for refmarker in refmarkers:
			if refmarker.track_age >= self.config.track_age_min:
				centre = (round(refmarker.centrex), round(refmarker.centrey))
				radius = round(refmarker.radius)
				if self.debug:
					cv2.circle(img, center=(round(refmarker.centrex_smooth), round(refmarker.centrey_smooth)), radius=radius, color=(204, 204, 0), thickness=self.LABEL_THICKNESS)
				cv2.circle(img, center=centre, radius=radius, color=(0, 204, 0) if refmarker.saved_to_csv > 0 else (204, 0, 0), thickness=self.LABEL_THICKNESS)
				track_text = str(refmarker.track_id)
				(track_text_width, track_text_height), track_text_baseline = cv2.getTextSize(text=track_text, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=track_text_scale, thickness=self.LABEL_THICKNESS)
				cv2.putText(
					img=img,
					text=track_text,
					org=(round(centre[0] + 0.5 * (self.LABEL_THICKNESS - track_text_width)), round(centre[1] - 0.5 * track_text_baseline - radius - self.LABEL_THICKNESS + 1)),
					fontFace=cv2.FONT_HERSHEY_PLAIN,
					fontScale=track_text_scale,
					color=(0, 128, 0) if refmarker.saved_to_csv > 0 else (128, 0, 0),
					thickness=self.LABEL_THICKNESS,
					lineType=cv2.LINE_AA,
				)
			if self.debug:
				if refmarker.track_age < self.config.track_age_min:
					cv2.circle(img, center=(round(refmarker.centrex), round(refmarker.centrey)), radius=round(refmarker.radius), color=(0, 204, 204), thickness=3)
				print(refmarker)
		if self.debug:
			print('-' * 80)

		if scale <= 1.51:  # Note: This makes sure that downsizing happens for 1280x960 (scale = 1.5) even with possible floating point errors, but not for smaller resolutions
			img = cv2.resize(img, dsize=(img.shape[1] // 2, img.shape[0] // 2))
		if self.pub_visimg_enable[camera_id]:
			self.pub_visimg[camera_id].publish(self.camera_cv_bridge.cv2_to_imgmsg(img, 'bgr8'))

	def preprocess_camera_image(self, image_data: sensor_msgs.msg.Image, info_data: Optional[sensor_msgs.msg.CameraInfo], camera_id: int, resolution: Optional[Tuple[int, int]]):
		# Called for each camera image that arrives for a particular camera

		stamp = image_data.header.stamp
		if stamp < self.camera_stamp[camera_id]:
			self.reset_camera_history(camera_id)
		self.camera_stamp[camera_id] = stamp
		frame_id = image_data.header.frame_id
		raw_img = self.camera_cv_bridge.imgmsg_to_cv2(image_data, 'bgr8')
		raw_img_size = (raw_img.shape[1], raw_img.shape[0])

		img_size = resolution if resolution is not None else raw_img_size
		raw_ratio = fractions.Fraction(numerator=raw_img_size[0], denominator=img_size[0])
		if raw_ratio != fractions.Fraction(numerator=raw_img_size[1], denominator=img_size[1]):
			raise ValueError(f"Raw image ({raw_img_size[0]}x{raw_img_size[1]}) and resized working image ({img_size[0]}x{img_size[1]}) must have same aspect ratio")
		elif raw_ratio < 1:
			rospy.logwarn_throttle_identical(86400, f"Raw image ({raw_img_size[0]}x{raw_img_size[1]}) is smaller than resized working image ({img_size[0]}x{img_size[1]})")
		if img_size != raw_img_size:
			img = cv2.resize(raw_img, dsize=img_size)
		else:
			img = raw_img.copy() if self.camera_force_copy else raw_img

		if info_data:
			if info_data.header.stamp.is_zero():
				info_data.header.stamp = stamp
			if not info_data.header.frame_id:
				info_data.header.frame_id = frame_id
		camera_model = info_data and camera_models.PinholeCameraModel(msg=info_data).change_roi_resolution(img_size)
		camera_image_data = dict(camera_id=camera_id, stamp=stamp, frame_id=frame_id, img=img, raw_img=raw_img, raw_ratio=raw_ratio, camera_model=camera_model, camera_tfrm=None)
		if not self.tf_buffer:
			self.process_camera_image(**camera_image_data)
		else:
			tf_image_queue = self.tf_image_queues[camera_id]
			if tf_image_queue and stamp < tf_image_queue[-1]['stamp']:
				tf_image_queue.clear()
			tf_image_queue.append(camera_image_data)
			num_to_clear = 0
			for i, cidata in enumerate(tf_image_queue, start=1):
				cidata_stamp = cidata['stamp']
				if (stamp - cidata_stamp).to_sec() <= self.tf_cache_time:
					try:
						# Attempts to retrieve the TF transform from the world frame to the required frame_id at the required timestamp
						# Note that the numeric translation/rotation contained in the returned transform is of the world csys relative to the camera csys, BUT nonetheless tf2_geometry_msgs.do_transform*() converts world coordinates to camera coordinates!
						cidata['camera_tfrm'] = self.tf_buffer.lookup_transform(target_frame=cidata['frame_id'], source_frame='world', time=cidata_stamp)
					except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException, rospy.ROSTimeMovedBackwardsException):
						continue
					self.process_camera_image(**cidata)
				num_to_clear = i
			for _ in range(num_to_clear):
				tf_image_queue.popleft()
				
	def run(self, rate_hz=45.0):
		if not self.prepared_run:
			self.prepare_run()
		
		rospy.loginfo(f"Running {self.name} at {rate_hz:.1f}Hz")
		rate = rospy.Rate(rate_hz, reset=True)
		self.running = True
		while not rospy.is_shutdown():
			self.step()
			try:
				rate.sleep()
			except rospy.ROSInterruptException:
				break
				
	def step(self):
		pass

# Run main function
if __name__ == "__main__":
	main()
# EOF
