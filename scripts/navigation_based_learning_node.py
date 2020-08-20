#!/usr/bin/env python
from __future__ import print_function
import roslib
roslib.load_manifest('obstacle_avoidance')
import rospy
import cv2
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from kobuki_msgs.msg import BumperEvent
from cv_bridge import CvBridge, CvBridgeError
from reinforcement_learning import *
from skimage.transform import resize
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Int8
from std_srvs.srv import Trigger
from actionlib_msgs.msg import GoalStatusArray
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelState
import csv
import os
import time
import copy
import random
import math

class cource_following_learning_node:
	def __init__(self):
		rospy.init_node('cource_following_learning_node', anonymous=True)
		self.action_num = rospy.get_param("/LiDAR_based_learning_node/action_num", 3)
		print("action_num: " + str(self.action_num))
		self.rl = reinforcement_learning(n_action = self.action_num)
		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
		self.bumper_sub = rospy.Subscriber("/mobile_base/events/bumper", BumperEvent, self.callback_bumper)
		self.vel_sub = rospy.Subscriber("/nav_vel", Twist, self.callback_vel)
		self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.callback_scan)
		self.status_sub = rospy.Subscriber("/move_base/status", GoalStatusArray, self.callback_loop)
		self.action_pub = rospy.Publisher("action", Int8, queue_size=1)
		self.nav_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
		self.action = 0
		self.reward = 0
		self.episode = 0
		self.count = 0
		self.status = 0
		self.loop_count = 0
		self.success = 0
		self.vel = Twist()
		self.cv_image = np.zeros((480,640,3), np.uint8)
		self.learning = True
		self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
		self.action_list = ['Front', 'Right', 'Left']
		self.path = 'data/result'
		self.previous_reset_time = 0
		self.start_time_s = rospy.get_time()
		os.makedirs(self.path + self.start_time)

		with open(self.path + self.start_time + '/' +  'reward.csv', 'w') as f:
			writer = csv.writer(f, lineterminator='\n')
			writer.writerow(['epsode', 'time(s)', 'reward'])

	def callback(self, data):
		try:
			self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

	def reset_simulation(self):
		self.action_pub.publish(0)
		rospy.wait_for_service('/gazebo/set_model_state')
		set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
		model_state = ModelState()
		model_state.model_name = 'mobile_base'
		model_state.pose.position.x = 0
		model_state.pose.position.y = 0
		model_state.pose.position.z = 0
		model_state.pose.orientation.x = 0
		model_state.pose.orientation.y = 0
		model_state.pose.orientation.z = 0.3 * (random.random() - 0.5)
		model_state.pose.orientation.w = 1
		model_state.twist.linear.x = 0
		model_state.twist.linear.y = 0
		model_state.twist.linear.z = 0
		model_state.twist.angular.x = 0
		model_state.twist.angular.y = 0
		model_state.twist.angular.z = 0
		model_state.reference_frame = 'world'
		try:
			set_model_state(model_state)
		except rospy.ServiceException as exc:
			print("Service did not process request: " + str(exc))

	def checkInsideRectangle(self, points, width, length, offset, angle):
		for point in points:
			x =   point[0] * math.cos(angle) + point[1] * math.sin(angle)
			y = - point[0] * math.sin(angle) + point[1] * math.cos(angle)
			if (-width/2 + offset) <= y <= (width/2 + offset) and 0 <= x <= length:
				return True
		return False
	

	def callback_scan(self, scan):
		collision = False
		points = []
		angle = scan.angle_min
		for distance in scan.ranges:
			if distance != float('inf') and not math.isnan(distance):
				points.append((distance * math.cos(angle), distance * math.sin(angle)))
			angle += scan.angle_increment

			if distance <= 0.2:
				collision = True

		if collision:
			self.callback_bumper(True)

	def callback_bumper(self, bumper):
		if rospy.get_time() - self.previous_reset_time < 1:
			return
		self.previous_reset_time = rospy.get_time()

		print("!!!!!!! RESET !!!!!!!")
		self.reset_simulation()

		if not self.learning:
			return

		self.reward = 0
		line = [str(self.episode), str(float(self.success) / self.count)]
		with open(self.path + self.start_time + '/' + 'reward.csv', 'a') as f:
			writer = csv.writer(f, lineterminator='\n')
			writer.writerow(line)
		print(" episode: " + str(self.episode) + ", success ratio: " + str(float(self.success)/self.count) + " " + str(self.count))
		self.count = 0
		self.success = 0
		self.episode += 1

	def callback_vel(self, data):
		self.vel = data
# action
		if 0.1 < self.vel.angular.z <= 0.2:
			self.action = 1
		elif -0.1 > self.vel.angular.z >= -0.2:
			self.action = 2
		elif self.vel.angular.z > 0.2:
			self.action = 3
		elif self.vel.angular.z < -0.2:
			self.action = 4
		else:
			self.action = 0

	def callback_loop(self, data):
		self.status = data.status_list[0]

		if self.status.status == 3:
			rospy.wait_for_service('start_wp_nav')
			try:
				service = rospy.ServiceProxy('start_wp_nav', Trigger)
				response = service()
			except rospy.ServiceException as e:
				print("Service call failed: %s" % e)
			self.loop_count += 1

	def loop(self):
		if self.cv_image.size != 640 * 480 * 3:
			return

		rospy.wait_for_service('/gazebo/get_model_state')
		get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
		try:
			previous_model_state = get_model_state('mobile_base', 'world')
		except rospy.ServiceException as exc:
			print("Service did not process request: " + str(exc))

		img = resize(self.cv_image, (48, 64), mode='constant')
		r, g, b = cv2.split(img)
		imgobj = np.asanyarray([r,g,b])

		ros_time = str(rospy.Time.now())

		if self.episode >= 30:
			self.learning = False

		if self.learning:
			if self.loop_count < 18:
				action = self.rl.act_and_trains(imgobj, self.reward)
				self.reward = 0 if action == self.action else -1
				if action == self.action:
					self.success += 1
				self.count += 1
#				print(self.loop_count)
			else:
				action = self.rl.stop_episode_and_train(imgobj, self.reward, False)
				self.reward = 0
				line = [str(self.episode), str(float(self.success) / self.count)]
				with open(self.path + self.start_time + '/' + 'reward.csv', 'a') as f:
					writer = csv.writer(f, lineterminator='\n')
					writer.writerow(line)
				print(" episode: " + str(self.episode) + ", success ratio: " + str(float(self.success)/self.count) + " " + str(self.count))
#				self.reset_simulation()
				self.count = 0
				self.success = 0
				self.loop_count = 0
				self.episode += 1

			self.nav_pub.publish(self.vel)

		else:
			self.action = self.rl.act(imgobj)
			print("TEST MODE")
			self.action_pub.publish(self.action)

		temp = copy.deepcopy(img)
		cv2.imshow("Resized Image", temp)
		cv2.waitKey(1)

if __name__ == '__main__':
	rg = cource_following_learning_node()
	DURATION = 0.1
	r = rospy.Rate(1 / DURATION)
	while not rospy.is_shutdown():
		rg.loop()
		r.sleep()
