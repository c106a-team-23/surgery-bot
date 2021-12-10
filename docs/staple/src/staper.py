#!/usr/bin/env python

from baxter_interface import Limb, gripper

import rospy

from moveit_msgs.msg import OrientationConstraint
from geometry_msgs.msg import PoseStamped, Point

from path_planner import PathPlanner
from selector import Selector

class Stapler():
    def __init__(self, pick_point):
        self._left_planner = PathPlanner("left_arm")
        self._left_gripper = gripper.Gripper("left")
        self._suture_selector = Selector("clicked_point")
        self._pick_point = pick_point

    def move_arm(self, planner, pose):
        """
        Moves arm

        Inputs:
        planner: arm planner
        pose: goal pose (geometry_msgs/PoseStamped)
        """
        limit = 50
        raw_input("Press <Enter> to try to move...")
        while not rospy.is_shutdown() and limit > 0:
            try:
                goal = pose
                goal.header.frame_id = "reference/base"

                # Might have to edit this . . . 
                plan = planner.plan_to_pose(goal, [])

                if not planner.execute_plan(plan):
                    raise Exception("Execution failed")
                else:
                    return
            except Exception as e:
                limit -= 1
                print e
            else:
                limit -= 1
                break
        print("Execution failed")

    def calibrate_grippers(self):
        """
        Calibrate both grippers
        """
        self._left_gripper.calibrate()
        self._right_gripper.calibrate()

    def pick(self, pick_point):
        move_pose = PoseStamped()
        move_pose.pose.position.x = pick_point.x
        move_pose.pose.position.y = pick_point.y
        move_pose.pose.position.y = pick_point.z + 0.05
        move_pose.pose.orientation.y = 1.0

        grip_pose = PoseStamped()
        grip_pose.pose.position.x = pick_point.x
        grip_pose.pose.position.y = pick_point.y
        grip_pose.pose.position.y = pick_point.z
        grip_pose.pose.orientation.y = 1.0

        move_left_arm = lambda pose:self.move_arm(self._left_planner, pose)
        move_left_arm(move_pose)
        move_left_arm(grip_pose)
        self._left_gripper.close()
        move_left_arm(move_pose)

    def place(self, place_point):
        move_pose = PoseStamped()
        move_pose.pose.position.x = place_point.x
        move_pose.pose.position.y = place_point.y
        move_pose.pose.position.y = place_point.z + 0.05
        move_pose.pose.orientation.y = 1.0

        release_pose = PoseStamped()
        release_pose.pose.position.x = place_point.x
        release_pose.pose.position.y = place_point.y
        release_pose.pose.position.y = place_point.z
        release_pose.pose.orientation.y = 1.0

        move_left_arm = lambda pose:self.move_arm(self._left_planner, pose)
        move_left_arm(move_pose)
        move_left_arm(release_pose)
        self._left_gripper.open()
        move_left_arm(move_pose)

def main():
    rospy.init_node('moveit_node')
    pick_point = Point()
    pick_point.x = 0
    pick_point.y = 0
    pick_point.z = 0
    robot = Stapler(pick_point)

    # Calibrate Robot
    robot.calibrate_grippers()

    # Query user for input
    robot._suture_selector.query()
    place_points = robot._suture_selector.get_points()

    # Pick and place staples
    for p in place_points:
        robot.pick(robot._pick_point)
        robot.place(p)

if __name__ == '__main__':
    main()