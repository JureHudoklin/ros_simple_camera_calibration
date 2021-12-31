#!/usr/bin/env python2

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, TransformStamped
from simple_camera_calibration.srv import SimpleCameraCalibration, SimpleCameraCalibrationResponse
import tf2_ros as tf2
import utils


class CameraCalibrationService(object):
    def __init__(self, pattern_size):
        self.pattern_size = pattern_size

        self.bridge = CvBridge()
        self.objpoints = []
        self.imgpoints = []

        self.camera_calibration_service = rospy.Service('camera_calibration', SimpleCameraCalibration, self.camera_calibration_callback)

        # Subscribers
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.img_cb)
        self.camera_info_sub = rospy.Subscriber('/camera/rgb/camera_info', CameraInfo, self.info_cb)

    def img_cb(self, img):
        try:
            self.color_msg = msg
            self.color_img = self.bridge.imgmsg_to_cv2(self.color_msg,"bgr8")
        except CvBridgeError as e:
            print(e)
        return

    def info_cb(self, msg):
        """
        Callback for the camera information.
        ----------
        Args:
            msg {CameraInfo}: The camera information message.
        ----------
            self.K {numpy.array}: The camera matrix.
            self.D {numpy.array}: The distortion coefficients.
        """
        self.K = np.reshape(msg.K,(3,3))    # Camera matrix
        self.D = np.array(msg.D) # Distortion matrix. 5 for IntelRealsense, 8 for AzureKinect

    def camera_calibration_callback(self, req):

        response = SimpleCameraCalibrationResponse()
        # Get the image
        image = self.color_img
        image = cv2.undistort(image, self.K, self.D)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        # Make the hand base transfrom into a matrix
        hand_base_tf = req.hand_base_tf
        hand_base_tf_trans = np.array([hand_base_tf.transform.translation.x, hand_base_tf.transform.translation.y, hand_base_tf.transform.translation.z])
        hand_base_tf_rot = np.array([hand_base_tf.transform.rotation.x, hand_base_tf.transform.rotation.y, hand_base_tf.transform.rotation.z, hand_base_tf.transform.rotation.w])
        hand_base_tf_mat = utils.quat_trans_to_matrix(hand_base_tf_trans, hand_base_tf_rot)

        # Find the cheesboard corners
        objp = np.zeros((1, self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)

        retval, corners = cv2.findChessboardCorners(gray, self.pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        # Optimize the corners
        if retval == True:
            print("Found corners")
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            self.objpoints.append(objp)
            self.imgpoints.append(corners_2)
        else:
            print("Corners not found")
            response.success = False
            return response

        # Calibrate the camera
        retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
        hand_camera_tf = self.make_pose(rvecs[0], tvecs[0])
        hand_camera_tf_mat = utils.pose_to_matrix(hand_camera_tf)
        camera_hand_tf_mat = tf.transformations.inverse_matrix(hand_camera_tf_mat)

        camera_base_tf_mat = np.dot(camera_hand_tf_mat, hand_base_tf_mat)
        rospy.logwarn("Camera base tf: {}".format(camera_base_tf_mat))
        camera_base_tf = utils.matrix_to_pose(camera_base_tf_mat)

        # Make the camera calibration response
        response.success = True
        response.camera_base_tf.transform.translation = camera_base_tf.position
        response.camera_base_tf.transform.rotation = camera_base_tf.orientation
        response.camera_base_tf.header.frame_id = "camera_base_link"
        response.camera_base_tf.header.stamp = rospy.Time.now()

        return response


    def make_pose(self, rvec, tvec):
        """
        Given a rotation vector and a translation vector, returns a cheesboard Pose.
        ----------
        Args:
            id {int} -- id of the marker
            rvec {np.array} -- rotation vector of the marker
            tvec {np.array} -- translation vector of the marker
        ----------
        Returns:
            Pose -- Pose of the marker
        """
        marker_pose = Pose()
        tvec = np.squeeze(tvec)
        rvec = np.squeeze(rvec)

        r_mat = np.eye(3)
        cv2.Rodrigues(rvec, r_mat)
        tf_mat = np.eye(4)
        tf_mat[0:3,0:3] = r_mat

        quat = tf.transformations.quaternion_from_matrix(tf_mat)

        marker_pose.position.x = tvec[0]
        marker_pose.position.y = tvec[1]
        marker_pose.position.z = tvec[2]

        marker_pose.orientation.x = quat[0]
        marker_pose.orientation.y = quat[1]
        marker_pose.orientation.z = quat[2]
        marker_pose.orientation.w = quat[3]

        return marker_pose


if __name__ == '__main__':
    rospy.init_node('camera_calibration_service')

    camera_calibration_service = CameraCalibrationService((6,5))

    rospy.spin()