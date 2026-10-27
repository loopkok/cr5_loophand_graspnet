#!/usr/bin/env python3
import numpy as np
from PIL import Image
import cv2
import socket
import pickle
import os
import json
import time
import threading
from scipy.spatial.transform import Rotation as R

from segment_process import segment_image

IP_ADDRESS = 'xx.xx.xx.xx' # replace with your server IP address

class GraspPlanner:
    def __init__(self):
        self.mask = None
        self.target_pose = None
        self.success = False

        # the home pose for object detection every time
        home_translation = [-0.45, 0, 0.7]
        rpy_deg = [-176, 0, 90]  # RPY angle value

        rpy_rad = np.array(rpy_deg) * np.pi / 180.0  # convert to radian
        rot_matrix = R.from_euler('xyz', rpy_rad).as_matrix()
        # flatten 3x3 matrix to list
        rot_list = rot_matrix.flatten().tolist()
        
        self.home_pose = {
            'translation': home_translation,  # unit: meter
            'rotation_matrix': rot_list,
            'score': 1.0  # set a fixed confidence score
        }
        
        # add execution status variables
        self.execution_completed = False
        self.is_executing = False
        self.wait_thread = None  # add waiting thread variable

    def capture_images(self):
        """capture images and automatically handle camera initialization and shutdown"""
        pass
        # try:
        #     # initialize camera
        #     pipeline = rs.pipeline()
        #     config = rs.config()
        #     config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
        #     config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
            
        #     # start camera
        #     pipeline.start(config)
        #     align = rs.align(rs.stream.color)
            
        #     # wait for camera to warm up
        #     rospy.sleep(1.0)
            
        #     # capture images
        #     frames = pipeline.wait_for_frames()
        #     aligned_frames = align.process(frames)
        #     depth_frame = aligned_frames.get_depth_frame()
        #     color_frame = aligned_frames.get_color_frame()
            
        #     self.depth_image = np.asanyarray(depth_frame.get_data())
        #     self.color_image = np.asanyarray(color_frame.get_data())
            
        #     # close camera
        #     pipeline.stop()
            
        #     return self.color_image, self.depth_image
            
        # except Exception as e:
        #     rospy.logerr(f"Error capturing images: {str(e)}")
        #     # ensure camera is closed in case of error
        #     if pipeline:
        #         try:
        #             pipeline.stop()
        #         except:
        #             pass
        #     return None, None

    def generate_mask(self, image, target_prompt,bbox):
        """generate mask and save box information"""
        try:
            # capture images
            # color_image, _ = self.capture_images()
            # if color_image is None:
            #     return False
            
            # call mask generation service
            self.mask = segment_image(image, target_prompt, bbox,output_mask='mask_test.png')
            
            if self.mask is None:
                print("No masks detected")
                return False 
            return True
            
        except Exception as e:
            print(f"Error in generate_mask: {str(e)}")
            return False

    def detect_grasp_pose(self):
        """detect grasp pose"""
        return True
        # try:
        #     # check if necessary data is available
        #     if self.box_info is None:
        #         print("No box information found. Please run generate_mask first.")
        #         return False
            
        #     if self.color_image is None or self.depth_image is None:
        #         print("No images found. Please run generate_mask first.")
        #         return False
            
        #     # process images
        #     x1, y1, x2, y2 = self.box_info['box']
        #     mask = np.zeros_like(self.depth_image, dtype=bool)
        #     mask[y1:y2, x1:x2] = True
            
        #     depth_masked = self.depth_image.copy()
        #     depth_masked[~mask] = 2000
            
        #     rgb_masked = self.color_image.copy()
        #     rgb_masked[~mask] = 0
            
        #     # convert to ROS message
        #     rgb_msg = self.bridge.cv2_to_imgmsg(rgb_masked, "bgr8")
        #     depth_msg = self.bridge.cv2_to_imgmsg(depth_masked)
            
        #     # call grasp service
        #     response = self.grasp_service(rgb_msg, depth_msg)
            
        #     if response.success:
        #         # save target pose
        #         self.target_pose = {
        #             'translation': response.translation,
        #             'rotation_matrix': response.rotation_matrix,
        #             'score': response.score
        #         }
        #         # print("grasp pose:", self.target_pose)
        #         print("Successfully detected grasp pose")
        #         return True
        #     else:
        #         print("Failed to get grasp pose")
        #         return False
                
        # except Exception as e:
        #     print(f"Error in detect_grasp_pose: {str(e)}")
        #     return False

    def detect_release_pose(self):
        """detect release pose"""
        return True
        # try:
        #     # check if necessary data is available
        #     if self.box_info is None:
        #         print("No box information found. Please run generate_mask first.")
        #         return False
            
        #     if self.color_image is None or self.depth_image is None:
        #         print("No images found. Please run generate_mask first.")
        #         return False
            
        #     # get the center point of the box
        #     x1, y1, x2, y2 = self.box_info['box']
        #     center_x = (x1 + x2) // 2
        #     center_y = (y1 + y2) // 2
            
        #     # get the depth information (unit: millimeter)
        #     depth = self.depth_image[center_y, center_x]
            
        #     # use fixed camera intrinsic parameters
        #     depth_intrin = rs.intrinsics()
        #     depth_intrin.width = 640
        #     depth_intrin.height = 480
        #     depth_intrin.ppx = 321.285  # principal point x
        #     depth_intrin.ppy = 237.486  # principal point y
        #     depth_intrin.fx = 383.970   # focal length x
        #     depth_intrin.fy = 383.970   # focal length y
        #     depth_intrin.model = rs.distortion.brown_conrady
        #     depth_intrin.coeffs = [0, 0, 0, 0, 0]  # distortion coefficients
            
        #     # convert pixel coordinates to camera coordinates (unit: meter)
        #     camera_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [center_x, center_y], depth)
            
        #     x = camera_point[0] / 1000.0  # convert to meter
        #     y = camera_point[1] / 1000.0
        #     z = camera_point[2] / 1000.0 
            
            
        #     rotation_matrix = [0, 1, 0,
        #                      0, 0, 1,
        #                      1, 0, 0]
            
        #     # save target pose
        #     self.target_pose = {
        #         'translation': tuple([x, y, z]),  # convert to tuple
        #         'rotation_matrix': rotation_matrix,
        #         'score': 1.0  # set a fixed confidence score
        #     }
            
        #     # add flag after successfully detecting release pose
        #     self.is_release_pose = True
            
        #     print("Successfully detected release pose")
        #     return True
                
        # except Exception as e:
        #     print(f"Error in detect_release_pose: {str(e)}")
        #     return False

    def move_to_target_pose(self):
        """send target pose to tf_convert"""
        return True
        # try:
        #     if self.target_pose is None:
        #         print("No target pose available. Please detect grasp pose first.")
        #         return False
            
        #     # connect to tf_convert and send data
        #     client_socket = None
        #     try:
        #         client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #         client_socket.settimeout(3.0)
        #         client_socket.connect(('10.7.145.140', 12347))
                
        #         # create command data
        #         command_data = {
        #             'pose': self.target_pose
        #         }
                
        #         # if release pose, add special flag
        #         if hasattr(self, 'is_release_pose') and self.is_release_pose:
        #             command_data['is_release'] = True
        #             command_data['release_height'] = 0.15 # release height (meter)
                
        #         data = pickle.dumps(command_data)
        #         client_socket.send(len(data).to_bytes(4, 'big'))
        #         client_socket.sendall(data)
                
        #         response_size = int.from_bytes(client_socket.recv(4), 'big')
        #         server_response = b''
        #         while len(server_response) < response_size:
        #             packet = client_socket.recv(response_size - len(server_response))
        #             if not packet:
        #                 raise Exception("Connection closed by server")
        #             server_response += packet
                
        #         # reset release pose flag
        #         if hasattr(self, 'is_release_pose'):
        #             self.is_release_pose = False
        #             print("Reset release pose flag")
                
        #         print("Successfully sent pose to tf_convert")
        #         return True
                
        #     except Exception as e:
        #         print(f"Failed to send pose to tf_convert: {e}")
        #         return False
        #     finally:
        #         if client_socket:
        #             client_socket.close()
                
        # except Exception as e:
        #     print(f"Error in move_to_target_pose: {str(e)}")
        #     return False

    def move_home(self):
        """move to initial pose"""
        return True
        # try:
        #     # connect to tf_convert and send data
        #     client_socket = None
        #     try:
        #         client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #         client_socket.settimeout(3.0)
        #         client_socket.connect(('10.7.145.140', 12347))
                
        #         # add flag indicating this is home pose
        #         home_command = {
        #             'is_home': True,
        #             'pose': self.home_pose
        #         }
                
        #         data = pickle.dumps(home_command)
        #         client_socket.send(len(data).to_bytes(4, 'big'))
        #         client_socket.sendall(data)
                
        #         response_size = int.from_bytes(client_socket.recv(4), 'big')
        #         server_response = b''
        #         while len(server_response) < response_size:
        #             packet = client_socket.recv(response_size - len(server_response))
        #             if not packet:
        #                 raise Exception("Connection closed by server")
        #             server_response += packet
                
        #         rospy.loginfo("Successfully sent home pose to tf_convert")
        #         return True
                
        #     except Exception as e:
        #         rospy.logerr(f"Failed to send home pose to tf_convert: {e}")
        #         return False
        #     finally:
        #         if client_socket:
        #             client_socket.close()
                
        # except Exception as e:
        #     rospy.logerr(f"Error in move_home: {str(e)}")
        #     return False

    def control_hand(self, command):
        """control hand"""
        return True
        # try:
        #     if command == 'grasp':
        #         # close hand
        #         response = self.hand_service(0, 0, 0, 0, 0, 0)
        #         success = response.angle_accepted
        #     elif command == 'release':
        #         # open hand
        #         response = self.hand_service(1000, 1000, 1000, 1000, 1000, 0)
        #         success = response.angle_accepted
            
        #     if success:
        #         print(f"Successfully executed hand {command}")
        #         return True
        #     else:
        #         print(f"Failed to execute hand {command}")
        #         return False
                
        # except Exception as e:
        #     print(f"Error controlling hand: {str(e)}")
        #     return False
        
    def pull(self):
        """execute pull action"""
        return True
        # try:
        #     # connect to tf_convert and send data
        #     client_socket = None
        #     try:
        #         client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #         client_socket.settimeout(3.0)
        #         client_socket.connect(('10.7.145.140', 12347))
                
        #         # create pull command
        #         pull_command = {
        #             'is_pull': True,  # mark this is a pull command
        #             'distance': -0.25   # move distance (meter)
        #         }
                
        #         data = pickle.dumps(pull_command)
        #         client_socket.send(len(data).to_bytes(4, 'big'))
        #         client_socket.sendall(data)
                
        #         response_size = int.from_bytes(client_socket.recv(4), 'big')
        #         server_response = b''
        #         while len(server_response) < response_size:
        #             packet = client_socket.recv(response_size - len(server_response))
        #             if not packet:
        #                 raise Exception("Connection closed by server")
        #             server_response += packet
                
        #         print("Successfully sent pull command to tf_convert")
        #         return True
                
        #     except Exception as e:
        #         print(f"Failed to send pull command to tf_convert: {e}")
        #         return False
        #     finally:
        #         if client_socket:
        #             client_socket.close()
                
        # except Exception as e:
        #     print(f"Error in pull: {str(e)}")
        #     return False

    def plan_process(self,command):
        # process command
        if command['type'] == 'generate_mask':
            success = self.generate_mask(command['image'], command['prompt'], command['bbox'])
        elif command['type'] == 'detect_grasp_pose':
            success = self.detect_grasp_pose()
        elif command['type'] == 'detect_release_pose':
            success = self.detect_release_pose()
        elif command['type'] == 'move_to_target_pose':
            success = self.move_to_target_pose()
        elif command['type'] == 'grasp':
            success = self.control_hand('grasp')
        elif command['type'] == 'release':
            success = self.control_hand('release')
        elif command['type'] == 'move_home':
            success = self.move_home()
        elif command['type'] == 'pull':
            success = self.pull()
        else:
            print(f"Unknown command type: {command['type']}")
        if (success == True):
            print(f"{command['type']}执行完毕")
        
        return  success
