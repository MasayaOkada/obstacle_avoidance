#!/usr/bin/env python

import rospy
from std_srvs.srv import Trigger

def call_service():
    rospy.wait_for_service("/start_waypoint_nav")

    try:
        service = rospy.ServiceProxy("/start_waypoint", Trigger)
        response = service()

    except rospy.Exception, e:
        print "Service call faild: %s" % e

if __name__ == "__main__":
    call_service()
