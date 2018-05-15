#Adrian Rodriguez-Febres, Group 201?
#-No Partner

#!/usr/bin/env python3

''' Get a raw frame from camera and display in OpenCV
By press space, save the image from 001.bmp to ...
'''

import time
import sys
import cv2
import cozmo
import numpy as np
from numpy.linalg import inv
import threading
import math

from ar_markers.hamming.detect import detect_markers

from grid import CozGrid
from gui import GUIWindow
from particle import Particle, Robot
from setting import *
from particle_filter import *
from utils import *
from cozmo.util import degrees, distance_mm


# camera params
camK = np.matrix([[295, 0, 160], [0, 295, 120], [0, 0, 1]], dtype='float32')

#marker size in inches
marker_size = 3.5

# tmp cache
last_pose = cozmo.util.Pose(0,0,0,angle_z=cozmo.util.Angle(degrees=0))
flag_odom_init = False

# goal location for the robot to drive to, (x, y, theta)
goal = (6,10,0)

# map
Map_filename = "map_arena.json"
grid = CozGrid(Map_filename)
gui = GUIWindow(grid)



async def image_processing(robot):

    global camK, marker_size

    event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)

    # convert camera image to opencv format
    opencv_image = np.asarray(event.image)
    
    # detect markers
    markers = detect_markers(opencv_image, marker_size, camK)
    
    # show markers
    for marker in markers:
        marker.highlite_marker(opencv_image, draw_frame=True, camK=camK)
        #print("ID =", marker.id);
        #print(marker.contours);
    cv2.imshow("Markers", opencv_image)

    return markers

#calculate marker pose
def cvt_2Dmarker_measurements(ar_markers):
    
    marker2d_list = [];
    
    for m in ar_markers:
        R_1_2, J = cv2.Rodrigues(m.rvec)
        R_1_1p = np.matrix([[0,0,1], [0,-1,0], [1,0,0]])
        R_2_2p = np.matrix([[0,-1,0], [0,0,-1], [1,0,0]])
        R_2p_1p = np.matmul(np.matmul(inv(R_2_2p), inv(R_1_2)), R_1_1p)
        #print('\n', R_2p_1p)
        yaw = -math.atan2(R_2p_1p[2,0], R_2p_1p[0,0])
        
        x, y = m.tvec[2][0] + 0.5, -m.tvec[0][0]
        #print('x =', x, 'y =', y,'theta =', yaw)
        
        # remove any duplate markers
        dup_thresh = 2.0
        find_dup = False
        for m2d in marker2d_list:
            if grid_distance(m2d[0], m2d[1], x, y) < dup_thresh:
                find_dup = True
                break
        if not find_dup:
            marker2d_list.append((x,y,math.degrees(yaw)))

    return marker2d_list


#compute robot odometry based on past and current pose
def compute_odometry(curr_pose, cvt_inch=True):
    global last_pose, flag_odom_init
    last_x, last_y, last_h = last_pose.position.x, last_pose.position.y, \
        last_pose.rotation.angle_z.degrees
    curr_x, curr_y, curr_h = curr_pose.position.x, curr_pose.position.y, \
        curr_pose.rotation.angle_z.degrees
    
    dx, dy = rotate_point(curr_x-last_x, curr_y-last_y, -last_h)
    if cvt_inch:
        dx, dy = dx / 25.6, dy / 25.6

    return (dx, dy, diff_heading_deg(curr_h, last_h))

#particle filter functionality
class ParticleFilter:

    def __init__(self, grid):
        self.particles = Particle.create_random(PARTICLE_COUNT, grid)
        self.grid = grid

    def update(self, odom, r_marker_list):

        # ---------- Motion model update ----------
        self.particles = motion_update(self.particles, odom)

        # ---------- Sensor (markers) model update ----------
        self.particles = measurement_update(self.particles, r_marker_list, self.grid)

        # ---------- Show current state ----------
        # Try to find current best estimate for display
        m_x, m_y, m_h, m_confident = compute_mean_pose(self.particles)
        return (m_x, m_y, m_h, m_confident)


async def run(robot: cozmo.robot.Robot):

    global goal
    global flag_odom_init, last_pose
    global grid, gui

    # start streaming
    robot.camera.image_stream_enabled = True

    #start particle filter
    pf = ParticleFilter(grid)

    ###################
    objectiveLocated = False
    objective = cozmo.util.Pose(0,0,0,angle_z=cozmo.util.Angle(degrees=0))
    state = 0
    await robot.set_head_angle(degrees(5)).wait_for_completed()
    while(True):
        while(robot.is_picked_up):
            print("Picked up.")
            state = 0
            pf = ParticleFilter(grid)
            last_pose = robot.pose
            TimeToPutDown = time.time() + 5

            while time.time() < TimeToPutDown:
                await robot.drive_straight(cozmo.util.distance_mm(0), speed=cozmo.util.speed_mmps(100)).wait_for_completed()
            robot.stop_all_motors()
            print("I am down.")
        if(state == 0):
            markers = await image_processing(robot)
            markersWithDimensions = cvt_2Dmarker_measurements(markers)
            poseTuple = compute_odometry(robot.pose)
            objective = pf.update(poseTuple, markersWithDimensions)
            last_pose = robot.pose
            gui.show_particles(pf.particles)
            gui.show_mean(objective[0], objective[1], objective[2], objective[3])
            gui.updated.set()

            if(objective[3] == True):
                state = 1
            cv2.waitKey(1)
            await robot.drive_wheels(25,-25)
            await robot.drive_straight(cozmo.util.distance_mm(0), speed=cozmo.util.speed_mmps(100)).wait_for_completed()

        if(state == 1):
            print("OBJECTive FOUND, ENSURING:")
            print(objective)
            TimeRemaining = time.time() + 10
            while time.time() < TimeRemaining:
                markers = await image_processing(robot)
                cv2.waitKey(1)
                markersWithDimensions = cvt_2Dmarker_measurements(markers)
                poseTuple = compute_odometry(robot.pose)
                objective = pf.update(poseTuple, markersWithDimensions)
                last_pose = robot.pose

                if(objective[3] == True):
                    state = 2
                else:
                    state = 0
                gui.show_particles(pf.particles)
                gui.show_mean(objective[0], objective[1], objective[2], objective[3])
                gui.updated.set()
                await robot.drive_straight(cozmo.util.distance_mm(0), speed=cozmo.util.speed_mmps(100)).wait_for_completed()
            #state = 2
        if(state == 2):
            robot.stop_all_motors()
            goalX = goal[0]*25
            goalY = goal[1]*25
            goalAngle = goal[2]
            cozmoX , cozmoY, cozmoAngle = objective[0],objective[1], objective[2]
            cozmoX *= 25
            cozmoY *= 25
            toReturnRotation = 0

            await robot.turn_in_place(degrees(goalAngle - cozmoAngle)).wait_for_completed()


            await robot.drive_straight(cozmo.util.distance_mm(goalX - cozmoX), cozmo.util.speed_mmps(30)).wait_for_completed()    
            
            if(goalY - cozmoY > 0):
                toReturnRotation = 90
                await robot.turn_in_place(degrees(toReturnRotation)).wait_for_completed()
            else:
                toReturnRotation = -90
                await robot.turn_in_place(degrees(toReturnRotation)).wait_for_completed()

            await robot.drive_straight(cozmo.util.distance_mm(abs(goalY - cozmoY)), cozmo.util.speed_mmps(30)).wait_for_completed()    
            
            await robot.turn_in_place(degrees(-toReturnRotation)).wait_for_completed()
            state = -1
        if(state == -1):
            await robot.drive_straight(cozmo.util.distance_mm(0), speed=cozmo.util.speed_mmps(100)).wait_for_completed()

    ###################


class CozmoThread(threading.Thread):
    
    def __init__(self):
        threading.Thread.__init__(self, daemon=False)

    def run(self):
        cozmo.robot.Robot.drive_off_charger_on_connect = False  # Cozmo can stay on his charger
        cozmo.run_program(run, use_viewer=False)


if __name__ == '__main__':

    # cozmo thread
    cozmo_thread = CozmoThread()
    cozmo_thread.start()

    # init
    grid = CozGrid(Map_filename)
    gui = GUIWindow(grid)
    gui.start()

