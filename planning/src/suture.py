#!/usr/bin/env python
"""
Suture Script for Surgery Robot
Author: Leonard Wei
"""
from baxter_interface import Limb, gripper

import rospy

from moveit_msgs.msg import OrientationConstraint
from geometry_msgs.msg import PoseStamped, Point

from path_planner import PathPlanner
from selector import Selector

class SurgeryRobot():
    """
    Surgery Robot Class
    """
    def __init__(self, needle_length):
        """
        Constructor

        Inputs:
        needle_length: Length of needle
        """
        self._needle_length = needle_length
        
        self._suture_selector = Selector("clicked_point")
        
        self._left_planner = PathPlanner("left_arm")
        self._right_planner = PathPlanner("right_arm")
        _pose = PoseStamped()
        _pose.header.frame_id = "reference/base"
        _pose.pose.position.x = 0.5
        _pose.pose.position.z = -0.35
        self._left_planner.add_box_obstacle([1, 2, 0.01], "left_table", _pose)
        self._left_gripper = gripper.Gripper('left')
        self._right_gripper = gripper.Gripper('right')

        # TODO set rest poses 
        self._left_rest_pose = PoseStamped()
        self._left_rest_pose.pose.position.x = 0.572
        self._left_rest_pose.pose.position.y = 0.179
        self._left_rest_pose.pose.position.z = 0.25
        self._left_rest_pose.pose.orientation.x = 0.5
        self._left_rest_pose.pose.orientation.y = 0.5
        self._left_rest_pose.pose.orientation.z = -0.5
        self._left_rest_pose.pose.orientation.w = 0.5

        self._right_rest_pose = PoseStamped()
        self._right_rest_pose.pose.position.x = 0.572
        self._right_rest_pose.pose.position.y = -0.179
        self._right_rest_pose.pose.position.z =  0.25
        self._left_rest_pose.pose.orientation.x = -0.5
        self._left_rest_pose.pose.orientation.y = 0.5
        self._left_rest_pose.pose.orientation.z = 0.5
        self._left_rest_pose.pose.orientation.w = 0.5

        self._left_calibrate_table_pose = PoseStamped()
        self._left_calibrate_table_pose.pose.position.x = 0.6
        self._left_calibrate_table_pose.pose.position.y = 0.0
        self._left_calibrate_table_pose.pose.position.z = 0.25
        self._left_calibrate_table_pose.pose.orientation.y = -1.0

        self._left_calibrate_camera_pose = PoseStamped()
        self._left_calibrate_camera_pose.pose.position.x = 0.659
        self._left_calibrate_camera_pose.pose.position.y = 0.246
        self._left_calibrate_camera_pose.pose.position.z = 0.158
        self._left_calibrate_camera_pose.pose.orientation.y = -1.0
    
    def move_arm(self, planner, pose, orient_const=[], prompt=False):
        """
        Moves arm

        Inputs:
        planner: arm planner
        pose: goal pose (geometry_msgs/PoseStamped)
        """

        limit = 50
        if prompt == True:
            raw_input("Press <Enter> to try to move...")
        while not rospy.is_shutdown() and limit > 0:
            try:
                goal = pose
                goal.header.frame_id = "base"

                plan = planner.plan_to_pose(goal, orient_const)

                if not planner.execute_plan(plan):
                    raise Exception("Execution failed")
                else:
                    return
            except Exception as e:
                limit -= 1
                print(e)
            else:
                limit -= 1
                break
        print("Execution failed")
        

    def move_to_rest(self):
        self.move_arm(self._left_planner, self._left_rest_pose)
        self.move_arm(self._right_planner, self._right_rest_pose)
    
    def calibrate_grippers(self):
        """
        Calibrate both grippers
        """
        self._left_gripper.calibrate()
        self._right_gripper.calibrate()

    def generate_entry_exit_points(self, entry_point):
        """
        Generates entry and exit points for suturing
        Inputs:
        entry_point: Selected entry point for suture
        Ouputs:
        Entry point for suture
        Exit point for suture
        """
        # Manual offset for pointcloud error
        entry_point.x -= 0.1
        entry_point.y 
        entry_point.z -= .15

        exit_point = Point()
        exit_point.x = entry_point.x
        exit_point.y = entry_point.y - self._needle_length
        exit_point.z = entry_point.z

        return entry_point, exit_point

    def suture_entry(self, entry_point):
        """
        Perform suture_entry
        Inputs:
        entry_point: point of entyr as PointStamped
        """

        # Pose of entry
        entry_pose = PoseStamped()
        entry_pose.pose.position.x = entry_point.x
        entry_pose.pose.position.y = entry_point.y + .35
        entry_pose.pose.position.z = entry_point.z
        entry_pose.pose.orientation.x = 0.7071068
        entry_pose.pose.orientation.y =  0.7071068

        # Pose of insert
        insert_pose = PoseStamped()
        insert_pose.pose.position.x = entry_point.x
        insert_pose.pose.position.y = entry_point.y + .1
        insert_pose.pose.position.z = entry_point.z
        insert_pose.pose.orientation.x = 0.7071068
        insert_pose.pose.orientation.y =  0.7071068 
        

        # Pose of release
        release_pose = PoseStamped()
        release_pose.pose.position.x = insert_pose.pose.position.x
        release_pose.pose.position.y = insert_pose.pose.position.y
        release_pose.pose.position.z = insert_pose.pose.position.z + .25
        release_pose.pose.orientation.x = 0.7071068
        release_pose.pose.orientation.y =  0.7071068 

        # Execute entry
        orien_const = OrientationConstraint()
        orien_const.link_name = "left_gripper"
        orien_const.header.frame_id = "base"
        orien_const.orientation.x = 0.7071068
        orien_const.orientation.y = 0.7071068
        orien_const.absolute_x_axis_tolerance = 0.1
        orien_const.absolute_y_axis_tolerance = 0.1
        orien_const.absolute_z_axis_tolerance = 0.1
        orien_const.weight = 1.0
        move_left_arm = lambda pose: self.move_arm(self._left_planner, pose, [orien_const])
        print("move left arm to suture")
        # move_left_arm(self._left_rest_pose)
        move_left_arm(entry_pose)
        print("moved left arm to suture")
        move_left_arm(insert_pose)
        print("insert needle")
        self._left_gripper.open()
        print("release needle")
        move_left_arm(release_pose)
        print("go to release position")
        move_left_arm(self._left_rest_pose)
        print("back to rest")

    def suture_exit(self, grip_point):
        """
        Perform suture exit
        Inputs:
        grip_point: point to grip needle
        """
        # TODO: Figure out correct poses
        # Pose of grip
        grip_pose = PoseStamped()
        grip_pose.pose.position.x = grip_point.x
        grip_pose.pose.position.y = grip_point.y + (2/3) * self._needle_length * 10
        grip_pose.pose.position.z = grip_point.z
        grip_pose.pose.orientation.x = 0.7071068
        grip_pose.pose.orientation.y =  0.7071068

        # Pose of exit
        exit_pose = PoseStamped()
        exit_pose.pose.position.x = grip_point.x
        exit_pose.pose.position.y = grip_point.y
        exit_pose.pose.position.z = grip_point.z
        exit_pose.pose.orientation.x = 0.7071068
        exit_pose.pose.orientation.y =  0.7071068

        # Pose of release
        release_pose = PoseStamped()
        release_pose.pose.position.x = self._right_rest_pose.pose.position.x
        release_pose.pose.position.y = self._right_rest_pose.pose.position.y - 0.1
        release_pose.pose.position.z = self._right_rest_pose.pose.position.z - 0.1
        release_pose.pose.orientation.x = 0.7071068
        release_pose.pose.orientation.y =  0.7071068

        # Execute exit
        orien_const = OrientationConstraint()
        orien_const.link_name = "left_gripper"
        orien_const.header.frame_id = "base"
        orien_const.orientation.x = 0.7071068
        orien_const.orientation.y = 0.7071068
        orien_const.absolute_x_axis_tolerance = 0.1
        orien_const.absolute_y_axis_tolerance = 0.1
        orien_const.absolute_z_axis_tolerance = 0.1
        orien_const.weight = 1.0
        move_right_arm = lambda pose: self.move_arm(self._left_planner, pose, [orien_const], prompt=True)
        move_right_arm(self._right_rest_pose)
        print("moved arm to rest position")
        move_right_arm(grip_pose)
        print("moved towards the needle")
        self._right_gripper.close()
        print("grip the needle")
        move_right_arm(exit_pose)
        print("pull the needle out horizontaly")
        move_right_arm(release_pose)
        print("moved to release pose")
        move_right_arm(self._right_rest_pose)
        print("moved arm to rest position")

    def suture_handoff(self):
        """
        Handoff needle from right to left gripper
        """
        self._left_gripper.close()
        self._right_gripper.open()

def main():
    rospy.init_node('moveit_node')
    robot = SurgeryRobot(0.6)
    
    # Calibrate Robot
    robot.calibrate_grippers()
    raw_input("Calibration succesful...\n \
               Please load patient and needle...\n \
               Press <Enter to continue... >")

    robot._left_gripper.close()
    
    # Query user for input 
    robot._left_gripper.close()
    robot._suture_selector.query()
    entry_points = robot._suture_selector.get_points()

    # Perform sutures
    for p in entry_points:
        entry_point, exit_point = robot.generate_entry_exit_points(p)
        robot.suture_entry(entry_point)
        robot.suture_exit(exit_point)
        robot.suture_handoff()
    robot.move_to_rest()

def test():
    rospy.init_node('moveit_node')
    robot = SurgeryRobot(0.6)
    
    # Query user for input 
    robot._suture_selector.query()
    entry_points = robot._suture_selector.get_points()

    orien_const = OrientationConstraint()
    orien_const.link_name = "left_gripper"
    orien_const.header.frame_id = "base"
    orien_const.orientation.x = 0.7071068
    orien_const.orientation.y = 0.7071068
    orien_const.absolute_x_axis_tolerance = 0.1
    orien_const.absolute_y_axis_tolerance = 0.1
    orien_const.absolute_z_axis_tolerance = 0.1
    orien_const.weight = 1.0

    for entry_point in entry_points:
        entry_pose = PoseStamped()
        entry_pose.pose.position.x = entry_point.x
        entry_pose.pose.position.y = entry_point.y
        entry_pose.pose.position.z = entry_point.z
        entry_pose.pose.orientation.x = 0.7071068
        entry_pose.pose.orientation.y =  0.7071068
        robot.move_arm(robot._left_planner, entry_pose, [orien_const], prompt=True)
    

if __name__ == '__main__':
    # main()
    test()