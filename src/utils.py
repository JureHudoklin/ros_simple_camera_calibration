from geometry_msgs.msg import Pose, Point, Quaternion
import numpy as np
import tf
import rospy
import math


def pose_to_matrix(pose):
    """
    Converts a Pose message to a 4x4 numpy matrix.
    """
    trans = tf.transformations.translation_matrix(
        (pose.position.x, pose.position.y, pose.position.z))
    rot = tf.transformations.quaternion_matrix(
        (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w))
    return np.dot(trans, rot)

def matrix_to_pose(matrix):
    """
    Converts a 4x4 numpy matrix to a Pose message.
    """
    return Pose(position=Point(*tf.transformations.translation_from_matrix(matrix)), orientation=Quaternion(*tf.transformations.quaternion_from_matrix(matrix)))

def matrix_to_quat_trans(matrix):
    """
    Convert a 4x4 numpy matrix to a quaternion and translation vector.
    """
    quat = tf.transformations.quaternion_from_matrix(matrix)
    trans = tf.transformations.translation_from_matrix(matrix)
    return trans, quat

def quat_trans_to_matrix(trans, quat):
    """
    Convert a translation and quaternion vector to a matrix.
    """
    trans_mtx = tf.transformations.translation_matrix(trans)
    rot_mtx = tf.transformations.quaternion_matrix(quat)
    combined = np.dot(trans_mtx, rot_mtx)
    return combined


def pose_to_quat_trans(pose):
    """
    Converts a Pose to a quaternion and translation vector
    ----------
    Args:
        pose {Pose}: ros geometry_msg.msg Pose
    ----------
    Returns:
        trans {np.array}: [t_x, t_y, t_z] transation vector
        quat {np.array}: [q_x, q_y, q_z, q_w] quaternion vector
    """
    quat = np.array([pose.orientation.x, pose.orientation.y,
              pose.orientation.z, pose.orientation.w])
    trans = np.array([pose.position.x,pose.position.y, pose.position.z])
    return trans, quat

def quat_trans_to_pose(trans, quat):
    """
    Converts a quaternion and translation vector to a Pose.
    """
    pose = Pose()
    pose.position.x = trans[0]
    pose.position.y = trans[1]
    pose.position.z = trans[2]
    pose.orientation.x = quat[0]
    pose.orientation.y = quat[1]
    pose.orientation.z = quat[2]
    pose.orientation.w = quat[3]
    return pose


def normalize_quaternion(v, tolerance=0.000001):
    mag2 = sum(n * n for n in v)
    if mag2 > tolerance:
        mag = math.sqrt(mag2)
        v = tuple(n / mag for n in v)
    return np.array(v)

def average_quaternions(quat_list,weights = None):
    """
    Average a list of quaternions.
    """
    if len(quat_list) == 0:
        return None
    elif len(quat_list) == 1:
        return quat_list[0]
    else:
        #quat_list = np.roll(np.array(quat_list),1,axis=1)
        if not weights is None:
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            quat_list = quat_list * weights[:, np.newaxis]
        Q_mtx = quat_list.T
        QQ_t = np.dot(Q_mtx, Q_mtx.T)
        w, v = np.linalg.eigh(QQ_t)
        max_eigen = np.argmax(w)
        max_eigen_vect = v[:, max_eigen]
        #max_eigen_vect = np.roll(max_eigen_vect, -1)
        max_eigen_vect = max_eigen_vect
        #print(max_eigen_vect, "Average QUATERNION")
        return max_eigen_vect
