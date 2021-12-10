#!/usr/bin/env python
import numpy as np
import rospy
from turtle_patrol.srv import Poke  # Import service type


def patrol_client():
    # Initialize the client node
    rospy.init_node('poke_position_client')
    # Wait until patrol service is ready
    rospy.wait_for_service('Poke pos server')
    try:
        # Acquire service proxy
        proxy = rospy.ServiceProxy('Poke pos server', Poke)
        rospy.loginfo('requested service')
        pos = proxy()
        
    except rospy.ServiceException as e:
        rospy.loginfo(e)


if __name__ == '__main__':
    patrol_client()
