#!/usr/bin/env python3

import rospy

from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import struct
import socket
import pickle
import json

class BoundingLidarFusion:
    def __init__(self):
        self.lidar = None

        self.image_resolution = (640, 480)

        # Subscribe to bounding box server
        self.host_ip = "192.168.1.7"
        self.host_port = 5679

        self.trajectory_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.trajectory_socket.connect((self.host_ip, self.host_port))
        self.data = b""
        self.trajectories_goals = None
        # This might be wrong for what I am sending.
        # https://www.digitalocean.com/community/tutorials/python-struct-pack-unpack
        self.payload_size = struct.calcsize("Q")
        self.fusion = None

        # Subscribe to Lidar / Subscribe to camera inputs (which I will probably subscribe to the depth to be able to get distance)



        pass

    def get_trajectory(self):
        while len(self.data) < self.payload_size:
            packet = self.trajectory_socket.recv(4*1024) # 4K
            if not packet:
                break
            self.data += packet
        packed_msg_size = self.data[:self.payload_size]
        self.data = self.data[self.payload_size:]
        msg_size = struct.unpack("Q",packed_msg_size)[0]

        while len(self.data) < msg_size:
            self.data += self.trajectory_socket.recv(4*1024)
        trajectory_data = self.data[:msg_size]
        self.data  = self.data[msg_size:]
        trajectory_goal = pickle.loads(trajectory_data)

        return trajectory_goal
    
    def getLidar(self, msg):
        self.lidar = msg.data

    
    def coalesce_data(self):
        # First get the angles of the bounding boxes. Unless I use the depth camera, then this isn't necessary
        pass





    def bounding_box_lidar_fusion(self):
        pub = rospy.Publisher("fusionBoundingLidar", String, queue_size=10)
        rospy.init_node("bounding_lidar_fusion", anonymous=True)
        rate = rospy.Rate(200)
        
        # TODO: Find out what the lidar message is actually called
        rospy.Subscriber("laser", LaserScan, self.getLidar)

        while not rospy.is_shutdown():
            self.get_trajectory()
            if self.lidar is not None and self.trajectories_goals is not None:
                self.coalesce_data()
                rospy.loginfo(json.dumps(self.fusion))
                pub.publish(json.dumps(self.fusion))

            rate.sleep()
                




if __name__ == "__main__":
    pass