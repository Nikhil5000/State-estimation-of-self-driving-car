import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rotations import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion

with open('data/pt1_data.pkl', 'rb') as file:
    data = pickle.load(file)

################################################################################################
# Each element of the data dictionary is stored as an item from the data dictionary:
#   gt: Data object containing ground truth. with the following fields:
#     a: Acceleration of the vehicle, in the inertial frame
#     v: Velocity of the vehicle, in the inertial frame
#     p: Position of the vehicle, in the inertial frame
#     alpha: Rotational acceleration of the vehicle, in the inertial frame
#     w: Rotational velocity of the vehicle, in the inertial frame
#     r: Rotational position of the vehicle, in Euler (XYZ) angles in the inertial frame
#     _t: Timestamp in ms.
#   imu_f: StampedData object with the imu specific force data (given in vehicle frame).
#     data: The actual data
#     t: Timestamps in ms.
#   imu_w: StampedData object with the imu rotational velocity (given in the vehicle frame).
#     data: The actual data
#     t: Timestamps in ms.
#   gnss: StampedData object with the GNSS data.
#     data: The actual data
#     t: Timestamps in ms.
#   lidar: StampedData object with the LIDAR data (positions only).
#     data: The actual data
#     t: Timestamps in ms.
################################################################################################
gt = data['gt']
imu_f = data['imu_f']
imu_w = data['imu_w']
gnss = data['gnss']
lidar = data['lidar']

################################################################################################

# Correct calibration rotation matrix, corresponding to Euler RPY angles (0.05, 0.05, 0.1).
C_li = np.array([
   [ 0.99376, -0.09722,  0.05466],
   [ 0.09971,  0.99401, -0.04475],
   [-0.04998,  0.04992,  0.9975 ]
])
t_i_li = np.array([0.5, 0.1, 0.5])

# Transform from the LIDAR frame to the vehicle (IMU) frame.
lidar.data = (C_li @ lidar.data.T).T + t_i_li

#### 2. Constants ##############################################################################

var_imu_f = 0.10
var_imu_w = 0.25
var_gnss  = 0.01
var_lidar = 1.00
################################################################################################
# We can also set up some constants that won't change for any iteration of our solver.
################################################################################################
g = np.array([0, 0, -9.81])  # gravity
l_jac = np.zeros([9, 6])
l_jac[3:, :] = np.eye(6)  # motion model noise jacobian
h_jac = np.zeros([3, 9])
h_jac[:, :3] = np.eye(3)  # measurement model jacobian

#### 3. Initial Values #########################################################################

################################################################################################
# Setting up initial values for UKF solver.
################################################################################################
p_est = np.zeros([imu_f.data.shape[0], 3])  # position estimates
v_est = np.zeros([imu_f.data.shape[0], 3])  # velocity estimates
q_est = np.zeros([imu_f.data.shape[0], 4])  # orientation estimates as quaternions
p_cov = np.zeros([imu_f.data.shape[0], 9, 9])  # covariance matrices at each timestep

# Set initial values.
p_est[0] = gt.p[0]
v_est[0] = gt.v[0]
q_est[0] = Quaternion(euler=gt.r[0]).to_numpy()
p_cov[0] = np.zeros(9)  # covariance of estimate
gnss_i  = 0
lidar_i = 0

#### 4. Measurement Update #####################################################################

def measurement_update(sensor_var, p_cov_check, y_k, p_check, v_check, q_check):
    # 3.1 Compute Kalman Gain
    I = np.identity(3)
    R = I * sensor_var
    K = p_cov_check.dot(h_jac.T).dot(np.linalg.inv(h_jac.dot(p_cov_check).dot(h_jac.T) + R))

    # 3.2 Compute error state
    error = K.dot(y_k - p_check)

    # 3.3 Correct predicted state
    p_del = error[:3]
    v_del = error[3:6]
    phi_del = error[6:]

    p_hat = p_check + p_del
    v_hat = v_check + v_del
    #q_hat = Quaternion(euler=phi_del).quat_mult_right(q_check)
    q_hat = Quaternion(euler=phi_del).quat_mult_left(q_check) 

    # 3.4 Compute corrected covariance
    p_cov_hat = (np.identity(9) - K.dot(h_jac)).dot(p_cov_check)

    return p_hat, v_hat, q_hat, p_cov_hat

#### 5. Main Filter Loop #######################################################################

for k in range(1, imu_f.data.shape[0]):  # start at 1 b/c we have initial prediction from gt
    delta_t = imu_f.t[k] - imu_f.t[k - 1]

    # 1. Update state with IMU inputs
    rotation_matrix = Quaternion(*q_est[k-1]).to_mat()

    # 1.1 Linearize the motion model and compute Jacobians
    p_est[k] = p_est[k-1] + delta_t*v_est[k-1] + (delta_t**2 / 2)*(rotation_matrix.dot(imu_f.data[k-1]) + g)
    v_est[k] = v_est[k-1] + delta_t*(rotation_matrix.dot(imu_f.data[k-1]) + g)
    q_est[k] = Quaternion(axis_angle=imu_w.data[k-1] * delta_t).quat_mult_right(q_est[k-1])

    # 2. Propagate uncertainty
    F = np.identity(9)
    Q = np.identity(6)
    F[:3, 3:6] = delta_t * np.identity(3)
    F[3:6, 6:] = -skew_symmetric(rotation_matrix.dot(imu_f.data[k-1])) * delta_t
    F[3:6, 6:] = -(rotation_matrix.dot(skew_symmetric(imu_f.data[k-1].reshape((3,1)))))
    Q[:3, :3] = var_imu_f * delta_t**2 * np.identity(3)
    Q[3:, 3:] = var_imu_w * delta_t**2 * np.identity(3)
    Q[:, :3] *= delta_t**2 * var_imu_f
    Q[:, -3:] *= delta_t**2 * var_imu_w
    p_cov[k] = F.dot(p_cov[k-1]).dot(F.T) + l_jac.dot(Q).dot(l_jac.T)

    # 3. Check availability of GNSS and LIDAR measurements
    for i in range(len(gnss.t)):
        if gnss.t[i] == imu_f.t[k-1]:
            p_est[k], v_est[k], q_est[k], p_cov[k] = measurement_update(var_gnss, p_cov[k], gnss.data[i].T, p_est[k], v_est[k], q_est[k])
    for i in range(len(lidar.t)):
        if lidar.t[i] == imu_f.t[k-1]:
            p_est[k], v_est[k], q_est[k], p_cov[k] = measurement_update(var_lidar, p_cov[k], lidar.data[i].T, p_est[k], v_est[k], q_est[k])
    for i in range(len(gnss.t)):
        if gnss.t[i] == imu_f.t[k-1]:
            p_est[k], v_est[k], q_est[k], p_cov[k] = measurement_update(var_gnss, p_cov[k], gnss.data[i].T, p_est[k], v_est[k], q_est[k])
    if lidar_i < lidar.t.shape[0] and lidar.t[lidar_i] == imu_f.t[k-1]:
        p_est[k], v_est[k], q_est[k], p_cov[k] = measurement_update(var_lidar, p_cov[k], lidar.data[lidar_i].T, p_est[k], v_est[k], q_est[k])
        lidar_i += 1
    if gnss_i < gnss.t.shape[0] and gnss.t[gnss_i] == imu_f.t[k-1]:
        p_est[k], v_est[k], q_est[k], p_cov[k] = measurement_update(var_gnss, p_cov[k], gnss.data[gnss_i].T, p_est[k], v_est[k], q_est[k])
        gnss_i += 1

#### 6. Results and Analysis ###################################################################

est_traj_fig = plt.figure()
ax = est_traj_fig.add_subplot(111, projection='3d')
ax.plot(p_est[:,0], p_est[:,1], p_est[:,2], label='Estimated')
ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2], label='Ground Truth')
ax.set_xlabel('Easting [m]')
ax.set_ylabel('Northing [m]')
ax.set_zlabel('Up [m]')
ax.set_title('Ground Truth and Estimated Trajectory')
ax.set_xlim(0, 200)
ax.set_ylim(0, 200)
ax.set_zlim(-2, 2)
ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
ax.set_yticks([0, 30, 60, 90, 120, 150, 180])
ax.set_zticks([-3, -2, -1, 0, 1, 2, 3])
ax.legend(loc=(0.62,0.77))
ax.view_init(elev=45, azim=-50)
plt.show()