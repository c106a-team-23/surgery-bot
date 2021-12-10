#!/usr/bin/env python
from geometry_msgs.msg import Vector3
import numpy as np
import rospy
from vision.srv import Poke  # Service type



def patrol_callback():    
    pub = rospy.Publisher(
        '/poke_position', Vector3, queue_size=50)
    cmd = Vector3()
    # get poke_position x, y, z

    # cmd.x = poke_x
    # cmd.y = poke_y
    # cmd.z = poke_z

    # Publish to cmd_vel at 5 Hz
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        pub.publish(cmd)  # Publish to cmd_vel
        rate.sleep()  # Sleep until 
    return cmd  # This line will never be reached

def patrol_server():
    # Initialize the server node for turtle1
    rospy.init_node('poke_positoin_server')
    # Register service
    rospy.Service(
        'Poke pos server',  # Service name
        Poke,  # Service type
        patrol_callback  # Service callback
    )
    rospy.loginfo('Running patrol server...')
    rospy.spin() # Spin the node until Ctrl-C


if __name__ == '__main__':
    patrol_server()
