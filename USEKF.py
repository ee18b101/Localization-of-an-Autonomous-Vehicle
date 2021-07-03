import csv
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import JulierSigmaPoints

data = pd.read_csv('GPS.csv')
data = pd.DataFrame(data).to_numpy()
time = data[:,0]
GPS = data[:,1:4]
rel_pos = data[:,4:7]
vel = data[:,7:10]
acc = data[:,10:13]
theta = data[:,13:16]
w_vel = data[:,16:19]
w_acc = data[:,19:22]

def convert_ground_frame(x, theta):
	R_alpha = np.array([[np.cos(theta[0]) , -np.sin(theta[0]), 0],
						[np.sin(theta[0]) , np.cos(theta[0]), 0],
						[0, 0, 1]])

	R_beta =  np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
						[0, 1, 0],
						[-np.sin(theta[1]), 0, np.cos(theta[1])]])

	R_gamma = np.array([[1, 0, 0],
						[0, np.cos(theta[2]), -np.sin(theta[2])],
						[0, np.sin(theta[2]), np.cos(theta[2])]])


	R_conversion = R_alpha.dot(R_beta).dot(R_gamma)
	x_converted = R_conversion.dot(x)
	return x_converted


# transform rel_pos to GPS frame
transform = np.array([[85787.7444144, 7397932.81266],
			[110979.170252,-4418056.86805],
			[1, 0]])

GPS[:,0] = GPS[:,0]*transform[0,0] + transform[0,1]
GPS[:,1] = GPS[:,1]*transform[1,0] + transform[1,1]
GPS[:,2] = GPS[:,2]*transform[2,0] + transform[2,1]

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(GPS[:,0], GPS[:,1], GPS[:,2], label='Converted GPS Coordinates')
ax.plot3D(rel_pos[:,0], rel_pos[:,1], rel_pos[:,2], label='Actual Coordinates')
plt.legend()
plt.show()

# variance setting
var_gps = 0.00001
var_imu_f = 0.01
var_imu_w = 0.01

# setting estimated state and setting initial values
x_est = np.zeros([len(time), 9])
p_cov = np.zeros([len(time), 9, 9])
speed = np.zeros((len(time), 3), dtype=float)

x_est[0,:3] = rel_pos[0]
x_est[0,3:6] = convert_ground_frame(vel[0], theta[0])
x_est[0,6:] = theta[0]
p_cov[0] = np.identity(9)/10000
speed[0] = vel[0]
k_global = 1

l_jac = np.zeros([9, 6])
l_jac[3:, :] = np.eye(6)  # motion model noise jacobian

# functions fx and hx
def convert_chassis_frame(x, k):
	R_alpha = np.array([[np.cos(x_est[k,6]) , -np.sin(x_est[k,6]), 0],
						[np.sin(x_est[k,6]) , np.cos(x_est[k,6]), 0],
						[0, 0, 1]])

	R_beta =  np.array([[np.cos(x_est[k,7]), 0, np.sin(x_est[k,7])],
						[0, 1, 0],
						[-np.sin(x_est[k,7]), 0, np.cos(x_est[k,7])]])

	R_gamma = np.array([[1, 0, 0],
						[0, np.cos(x_est[k,8]), -np.sin(x_est[k,8])],
						[0, np.sin(x_est[k,8]), np.cos(x_est[k,8])]])


	R_conversion = R_alpha.dot(R_beta).dot(R_gamma)
	x_converted = np.linalg.inv(R_conversion).dot(x)
	return x_converted

def wrap_to_pi(x):
	for i in range(len(x)):
		if x[i] > np.pi - 0.05:
			# print('wrap_to_pi+ ', x[i])
			x[i] -= 2*np.pi
		elif x[i] < -np.pi:
			# print('wrap_to_pi- ', x[i])
			x[i] += 2*np.pi
	return x

def fx(x, delta_t):
	x_check = np.zeros((9,), dtype=float)

	x_check[:3] = x[:3] + x[3:6]*delta_t + (delta_t**2)/2*(acc[k_global])
	x_check[3:6] = x[3:6] + delta_t*(acc[k_global])
	x_check[6] = x[6] + delta_t*w_vel[k_global,0] + (delta_t**2)/2*(w_acc[k_global,0])
	x_check[7] = x[7] + delta_t*w_vel[k_global,1] + (delta_t**2)/2*(w_acc[k_global,1]) + (x_check[6] - x[6])/(1.55252176405 + 4.673119614496461)*0.8104022289913551 + (speed[k_global, 2] - speed[k_global-1, 2])/25 - (speed[k_global, 0] - speed[k_global-1, 0])/27.777778*0.0087 - (speed[k_global,1] - speed[k_global-1, 1])/1.01*0.0004
	x_check[8] = x[8] + delta_t*w_vel[k_global,2] + (delta_t**2)/2*(w_acc[k_global,2]) + (x_check[6] - x[6])/(1.55252176405 + 4.673119614496461)*0.04589307822503154
	# x_check[6:] = wrap_to_pi(x_check[6:], x_est[k_global - 2, 6:])
	return x_check


def hx(x):
	return x[:3]

# Unscented Kalman Filter
dt = time[1] - time[0]
points = JulierSigmaPoints(9, kappa=-6)
kf = UnscentedKalmanFilter(dim_x=9, dim_z=3, dt=dt, fx=fx, hx=hx, points=points)

kf.x = x_est[0]
kf.P = p_cov[0]
kf.R = np.identity(3) * var_gps

for k in range(1, len(time)):
	# print(k)
	delta_t = time[k] - time[k-1]
	k_global = k

	Q = np.identity(6)
	Q[:, :3] *= delta_t**2 * var_imu_f
	Q[:, 3:] *= delta_t**2 * var_imu_w

	kf.Q = np.dot(np.dot(l_jac, Q), l_jac.T)

	kf.predict(dt=delta_t)

	kf.R = np.identity(3) * var_gps
	kf.update(GPS[k])


	x_est[k] = kf.x
	p_cov[k] = kf.P

	speed[k] = convert_chassis_frame(x_est[k,3:6].reshape(3,1), k).reshape(3,)


x_est[:,6] = wrap_to_pi(x_est[:,6])

plt.plot(time, rel_pos[:,0], label='actual')
plt.plot(time, x_est[:,0], label='estimated')
plt.title("posx")
plt.legend()
plt.show()

plt.plot(time, rel_pos[:,1], label='actual')
plt.plot(time, x_est[:,1], label='estimated')
plt.title("posy")
plt.legend()
plt.show()

plt.plot(time, rel_pos[:,2], label='actual')
plt.plot(time, x_est[:,2], label='estimated')
plt.title("posz")
plt.legend()
plt.show()


plt.plot(time, theta[:,0], label='actual')
plt.plot(time, x_est[:,6], label='estimated')
plt.title("yaw")
plt.legend()
plt.show()


plt.plot(time, theta[:,1], label='actual')
plt.plot(time, x_est[:,7], label='estimated')
plt.title("pitch")
plt.legend()
plt.show()


plt.plot(time, theta[:,2], label='actual')
plt.plot(time, x_est[:,8], label='estimated')
plt.title("roll")
plt.legend()
plt.show()


# plt.plot(time, vel[:,0], label='actual')
plt.plot(time, x_est[:,0+3], label='estimated')
plt.title("vx")
plt.legend()
plt.show()

# plt.plot(time, vel[:,1], label='actual')
plt.plot(time, x_est[:,1+3], label='estimated')
plt.title("vy")
plt.legend()
plt.show()

# plt.plot(time, vel[:,2], label='actual')
plt.plot(time, x_est[:,2+3], label='estimated')
plt.title("vz")
plt.legend()
plt.show()
