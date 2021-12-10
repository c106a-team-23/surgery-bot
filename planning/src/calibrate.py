#!/usr/bin/env python
"""
Calibration Script for Surgery Robot
Author: Leonard Wei
"""

import sys
import rospy
import tf2_ros

from moveit_msgs.msg import OrientationConstraint
from geometry_msgs.msg import PoseStamped

# from path_planner import PathPlanner

# def move_to_table():
#     planner = PathPlanner("left_arm")
#     goal = PoseStamped()
#     goal.header.frame_id = "base"

#     #x, y, and z position
#     goal.pose.position.x = 0.7
#     goal.pose.position.y = 0.1
#     goal.pose.position.z = 0.2

#     #Orientation as a quaternion
#     goal.pose.orientation.x = 0.0
#     goal.pose.orientation.y = -1.0
#     goal.pose.orientation.z = 0.0
#     goal.pose.orientation.w = 0.0

#     plan = planner.plan_to_pose(goal, [])

#     raw_input("Press <Enter> to move the left arm to table calibration location: ")
#     if not planner.execute_plan(plan):
#         raise Exception("Movement to table calibration location failed")

# def move_to_realsense():
#     planner = PathPlanner("left_arm")
#     goal = PoseStamped()
#     goal.header.frame_id = "base"

#     #x, y, and z position
#     goal.pose.position.x = 0.0
#     goal.pose.position.y = 0.0
#     goal.pose.position.z = 0.0

#     #Orientation as a quaternion
#     goal.pose.orientation.x = 0.0
#     goal.pose.orientation.y = -1.0
#     goal.pose.orientation.z = 0.0
#     goal.pose.orientation.w = 0.0

#     plan = planner.plan_to_pose(goal, [])

#     raw_input("Press <Enter> to move the left arm to realsense calibration location: ")
#     if not planner.execute_plan(plan):
#         raise Exception("Movement to realsense calibration location failed")

def get_transform(source_frame, target_frame):
    tfBuffer = tf2_ros.Buffer()
    tfListener = tf2_ros.TransformListener(tfBuffer)

    r = rospy.Rate(10) # 10 hz

    # Run until transform is found
    trans = None
    while not rospy.is_shutdown():
        print("Trying...")
        try:
            trans = tfBuffer.lookup_transform(target_frame, source_frame, rospy.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            pass
        else:
            print("Transform from {} to {} has been found!".format(source_frame, target_frame))
            print(trans.transform)
            return trans
        r.sleep()

def main():
    # Find table transform
    # move_to_table()
    table_transform = get_transform("table", "usb_cam")

    # Find realsense transform
    # move_to_realsense()
    realsense_transform = get_transform("camera_link", "left")
    raw_input("Calibration finished...\nPress <Enter> to continue...")

if __name__ == '__main__':
    rospy.init_node('moveit_node')
    main()