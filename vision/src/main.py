#!/usr/bin/env python
"""Skeleton code for Lab 6
Course: EECS C106A, Fall 2019
Author: Amay Saxena

This file implements a ROS node that subscribes to topics for RGB images,
pointclouds, and camera calibration info, and uses the functions you
implemented to publish a segmented pointcloud to the topic /segmented_points.

Once you are confident in your implementation in image_segmentation.py and
pointcloud_segmentation.py, run this file to begin publishing a segmented
pointcloud.
"""

from __future__ import print_function
from collections import deque

from matplotlib.pyplot import axis

import rospy
import message_filters
import ros_numpy
import tf

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import Vector3

import numpy as np
import cv2

from cv_bridge import CvBridge

from image_segmentation import segment_image, discretize, show_image
from pointcloud_segmentation import segment_pointcloud

# for CNN Model
import torch
import torchvision
import torch.nn as nn  
import torch.optim as optim 
from unet import UNET  

def get_camera_matrix(camera_info_msg):
    # TODO: Return the camera intrinsic matrix as a 3x3 numpy array
    # by retreiving information from the CameraInfo ROS message.
    # Hint: numpy.reshape may be useful here.
    return np.reshape(camera_info_msg.K, (3, 3))

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    
def segment_patient(model, image):
    model.eval()
    with torch.no_grad():
        segmented_patient = torch.sigmoid(model(image))
        segmented_patient = (segmented_patient > 0.5).float()
    return segmented_patient

def isolate_object_of_interest(points, image, cam_matrix, trans, rot):
    segmented_image = segment_image(image)
    points = segment_pointcloud(points, segmented_image, cam_matrix, trans, rot)
    return points

def isolate_working_space(image):
    return segment_image(image, 2)
def isolate_section(points, image, cam_matrix, trans, rot):
    sections = discretize(image)
    
    for s in sections[1:]:
        # s is in the form of ((start, end), pixel values)
        # print(s)
        # check if this section is closed/sutured
        # use teachable machine for this part
        # if section is closed, skip it. Othewise, send the point cloud to baxter
        if True: # found the lowest unclosed skin
            # project the point cloud only in this section
            mask = image*0
            mask[s[0][0]: s[0][1],] = 1
            
            # create an image that only shows the region we are interested in
            section = image*mask
            # show_image(section)
            segmented_section = segment_image(section)
            # show_image(segmented_section)

            p = segment_pointcloud(points, segmented_section, cam_matrix, trans, rot)
            
            if len(p) > 0:
                p_temp = np.array([np.array(list(point)) for point in p])
                # poke_position = np.mean(p[:,0:3], axis=0)
                poke_position = np.mean(p_temp, axis=0)
                # poke_position = [tuple(poke_position)]
                print("got a point")
                # print(poke_position)
                # print(p)
                return (poke_position, p)
    print("got no points")
    return (None, None)

        
def check_closed():
    # edit this using the teachable machine model
    return False

def numpy_to_pc2_msg(points):
    return ros_numpy.msgify(PointCloud2, points, stamp=rospy.Time.now(),
        frame_id='camera_depth_optical_frame')

class PointcloudProcess:
    """
    Wraps the processing of a pointcloud from an input ros topic and publishing
    to another PointCloud2 topic.

    """
    def __init__(self, points_sub_topic, 
                       image_sub_topic,
                       cam_info_topic,
                       points_pub_topic,
                       poke_position_topic):

        self.num_steps = 0

        self.messages = deque([], 5)
        self.pointcloud_frame = None
        points_sub = message_filters.Subscriber(points_sub_topic, PointCloud2)
        image_sub = message_filters.Subscriber(image_sub_topic, Image)
        caminfo_sub = message_filters.Subscriber(cam_info_topic, CameraInfo)

        self._bridge = CvBridge()
        self.listener = tf.TransformListener()
        
        self.points_pub = rospy.Publisher(points_pub_topic, PointCloud2, queue_size=10)
        self.image_pub = rospy.Publisher('segmented_image', Image, queue_size=10)

        # additional topics
        self.seg_image_pub = rospy.Publisher('segmented_workspace', Image, queue_size=10)
        self.section_pub = rospy.Publisher('segmented_section', PointCloud2, queue_size=10)

        self.poking_pos_pub = rospy.Publisher('poke_position', Vector3, queue_size=10)

        ts = message_filters.ApproximateTimeSynchronizer([points_sub, image_sub, caminfo_sub],
                                                          10, 0.1, allow_headerless=True)
        
        ts.registerCallback(self.callback)

    def callback(self, points_msg, image, info):
    
        try:
            intrinsic_matrix = get_camera_matrix(info)            
            rgb_image = ros_numpy.numpify(image)
            points = ros_numpy.numpify(points_msg)
        except Exception as e:
            rospy.logerr(e)
            return
        self.num_steps += 1
        self.messages.appendleft((points, rgb_image, intrinsic_matrix))

    def publish_once_from_queue(self, model=None):
        if self.messages:
            points, image, info = self.messages.pop()

            try:

                trans, rot = self.listener.lookupTransform(
                                                       '/camera_color_optical_frame',
                                                       '/camera_depth_optical_frame',
                                                       rospy.Time(0))
                rot = tf.transformations.quaternion_matrix(rot)[:3, :3]
            except (tf.LookupException,
                    tf.ConnectivityException, 
                    tf.ExtrapolationException):
                return

            #1. This segments the work area; including table and patient
            work_area = isolate_working_space(image)
            #2. This makes the work area rgb
            work_area_rgb = cv2.cvtColor(work_area, cv2.COLOR_GRAY2RGB)
            #3. This publishes the segmented work_area
            msg = self._bridge.cv2_to_imgmsg(work_area_rgb*image, "rgb8")
            self.seg_image_pub.publish(msg)

            #4. This gets the work area -> segments patient using CNN and publishes pointclouds. 
            # This helps in ridding of noisy environment and will display only the patient and table
            # on Rviz.
            workspace_points = segment_pointcloud(points, work_area, info, np.array(trans), np.array(rot))
            points_msg = numpy_to_pc2_msg(workspace_points)
            self.points_pub.publish(points_msg)

            #5. Now from the workspace segment patient using the CNN model <commented out>
            # patient = segment_patient(model, work_area)
            # patient_points = segment_pointcloud(points, patient, info, np.array(trans), np.array(rot))
            # points_msg = numpy_to_pc2_msg(patient_points)
            # didn't publish this

            # Update
            print("Published segmented pointcloud at timestamp:",
                   points_msg.header.stamp.secs)
            
            # This publishes poking point on a topic. Planner can either use 
            # this or use the publish_tool on Rviz
            
            poking_point, section = isolate_section(points, image, info, 
                np.array(trans), np.array(rot))

            if section is not None:
                section_msg = numpy_to_pc2_msg(section)
                self.section_pub.publish(section_msg)

            if poking_point is not None:
                p_point = Vector3()
                p_point.x = poking_point[0]
                p_point.y = poking_point[1]
                p_point.z = poking_point[2]

                self.poking_pos_pub.publish(p_point)

def main():
    CAM_INFO_TOPIC = '/camera/color/camera_info'
    RGB_IMAGE_TOPIC = '/camera/color/image_raw'
    POINTS_TOPIC = '/camera/depth/color/points'
    POINTS_PUB_TOPIC = 'segmented_points'
    POKE_POSITION_TOPIC = "poking_point"

    rospy.init_node('realsense_listener')
    process = PointcloudProcess(POINTS_TOPIC, RGB_IMAGE_TOPIC,
                                CAM_INFO_TOPIC, POINTS_PUB_TOPIC,
                                POKE_POSITION_TOPIC)
    
    # we saw a good result for this but did get to test it on the lab machines 
    # because of torchvision and shortage of time. But we strongly beleive incorporating a 
    # CNN image segmenation model would be helpful
    model = UNET(in_channels=3, out_channels=1)
    # load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    # model.eval()

    r = rospy.Rate(100)
    while not rospy.is_shutdown():
        process.publish_once_from_queue(model)
        r.sleep()

if __name__ == '__main__':
    
    main()
