#!/usr/bin/env python3

import rospy
# Might not have the .msg ending part
from nav_msgs.msg import OccupancyGrid
import struct
import socket
import pickle


class Occupancy_Trajectory:
    def __init__(self):
        self.occupancy_grid = OccupancyGrid()
        self.new_occupancy_grid = OccupancyGrid()
        

        # Change this ip to be whatever device it will be running on
        self.host_ip = "192.168.1.7"
        self.host_port = 5679

        self.trajectory_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.trajectory_socket.connect((self.host_ip, self.host_port))

        self.data = b""
        # This might be wrong for what I am sending.
        # https://www.digitalocean.com/community/tutorials/python-struct-pack-unpack
        self.payload_size = struct.calcsize("Q")

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


    def get_occupancy_grid(self, msg):
        self.occupancy_grid = msg.data


    def occupancy_grid_trajectory(self):
        pub = rospy.Publisher("occupancy_grid_trajectory", OccupancyGrid, queue_size=10)    
        rospy.init_node("occupancy_trajectory_combiner", anonymous=True)
        rate = rospy.Rate(200) # 200 hz, or 0.5 times a second. How often should it create a new occupancy grid? Probably every 2-3 seconds?

        rospy.Subscriber("/map", OccupancyGrid, self.get_occupancy_grid)
        

        while not rospy.is_shutdown():


            rospy.loginfo(self.new_occupancy_grid)
            pub.publish(self.new_occupancy_grid)
            rate.sleep()



if __name__ == "__main__":
    try:
        occupancy_trajectory = Occupancy_Trajectory()
        occupancy_trajectory.occupancy_grid_trajectory()

    except rospy.ROSInterruptException:
        pass