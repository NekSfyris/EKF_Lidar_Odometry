import pickle
import numpy as np
import matplotlib.pyplot as plt

'''
#To see what the pickle file contains do the following
objects = []
with (open('data/data.pickle', 'rb')) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
            
print(objects)
'''

with open('data.pickle', 'rb') as f:
    data = pickle.load(f)

t = data['t']  # timestamps [s]
#print(t)

x_init  = data['x_init'] # initial x position [m]
y_init  = data['y_init'] # initial y position [m]
th_init = data['th_init'] # initial theta position [rad]
#print(th_init)

# input signal
v  = data['v']  # translational velocity input [m/s]
om = data['om']  # rotational velocity input [rad/s]

# bearing and range measurements, LIDAR constants
b = data['b']  # bearing to each landmarks center in the frame attached to the laser [rad]
r = data['r']  # range measurements [m]
l = data['l']  # x,y positions of landmarks [m]
d = data['d']  # distance between robot center and laser rangefinder [m]




v_var = 0.01  # translation velocity variance - pickle gives a value of 0.004
om_var = 0.01  # rotational velocity variance - pickle gives a value of 0.008
#r_var = 0.1  # range measurements variance - pickle gives a value of 0.001
r_var = 0.01
#b_var = 0.1  # bearing measurement variance - pickle gives a value of 0.0005
b_var = 10

Q_km = np.diag([v_var, om_var]) # input noise covariance 
cov_y = np.diag([r_var, b_var])  # measurement noise covariance 

x_est = np.zeros([len(v), 3])  # estimated states, x, y, and theta
P_est = np.zeros([len(v), 3, 3])  # state covariance matrices

x_est[0] = np.array([x_init, y_init, th_init]) # initial state
P_est[0] = np.diag([1, 1, 0.1]) # initial state covariance

print(P_est)








# Wraps angle to (-pi,pi] range
def wraptopi(x):
    if x > np.pi:
        x = x - (np.floor(x / (2 * np.pi)) + 1) * 2 * np.pi
    elif x < -np.pi:
        x = x + (np.floor(x / (-2 * np.pi)) + 1) * 2 * np.pi
    return x



def measurement_update(lk, rk, bk, P_check, x_check):
    
    # 1. Compute measurement Jacobian
    x = x_check[0]
    y = x_check[1]
    th = wraptopi(x_check[2])

    #landmark x and y coordinates
    print("Landmark coordinates:", lk)
    lx = lk[0]
    ly = lk[1]
 
    #calculate expected range measurement
    #d is the distance between robot center and laser rangefinder [m]
    print("d = ", d)
    d_x = lx - x - d*np.cos(th)
    d_y = ly - y - d*np.sin(th)
    frac = d_x**2 + d_y**2  
    range_exp = np.sqrt( frac )
    
    
    H_k = np.array([[-d_x/range_exp, -d_y/range_exp, d * (d_x*np.sin(th) - d_y*np.cos(th))/range_exp],
                  [d_y/frac, -d_x/frac, -1 - d * (np.sin(th)*d_y + np.cos(th)*d_x)/frac]])
    H_k = H_k.reshape(2, 3)
    
    M_k = np.identity(2)

    
    print("H_k.shape = ", H_k.shape)
    print("H_k = \n", H_k)
    print("M_k.shape = ", M_k.shape)
    print("P_check.shape = ", P_check.shape)
    print("cov_y.shape = ", cov_y.shape)
    
    
    # 2. Compute Kalman Gain. Here is a 3x2 array
    K_k = P_check @ H_k.T @ np.linalg.inv(H_k @ P_check @ H_k.T + M_k @ cov_y @ M_k.T)
    print("K_k.shape = ", K_k.shape)
    
    # 3. Correct predicted state (remember to wrap the angles to [-pi,pi])
    phi = np.arctan2(d_y, d_x) - th
    y_k = np.array([[range_exp], [wraptopi(phi)]])
    y_k = y_k.reshape(2,1)
    y_measured = np.array([[rk], [wraptopi(bk)]])
    
    print("y_k.shape = ", y_k.shape)
    print("y_measured.shape = ", y_measured.shape)

    x_check = x_check + K_k @ (y_measured - y_k)
    
    print("x_check = \n", x_check)
    print("x_check.shape = ", x_check.shape)
    
    x_check[2] = wraptopi(x_check[2])


    # 4. Correct covariance
    P_check = (np.identity(3) - K_k @ H_k) @ P_check
    
    '''  '''

    return x_check, P_check




#### 5. Main Filter Loop #######################################################################

x_check = np.zeros(3)
flag_initial=1

for k in range(1, len(t)):  # start at 1 because we've set the initial prediciton

    delta_t = t[k] - t[k - 1]  # time step (difference between timestamps)

    # 1. Update state with odometry readings (remember to wrap the angles to [-pi,pi])
    
    
    #if it is the first time we run this loop, we use the initial state value
    if (flag_initial == 1):
        #copy initial state values by taking first row
        x_check = np.array(x_est[0, :]).reshape(3, 1)

        #copy initial covariance by taking first row
        P_check = P_est[0]
        
        flag_initial=0


    print(x_check)
    print(x_check.shape)
    theta = x_check[2]
    print(theta)
    
    
    #trying to create the motion model function
    A1 = np.array([[np.cos(wraptopi(theta)), 0], [np.sin(wraptopi(theta)), 0], [0, 1]], dtype='float')  
    A2 = np.array([[v[k-1]], [om[k-1]]])
    
    print("A1.shape = ", A1.shape)
    print("A2.shape = ", A2.shape)
    
    x_check = x_check + delta_t * np.matmul(A1, A2)
    x_check[2] = wraptopi(x_check[2])
    
    
    print("x_check = \n", x_check)
    print("x_check.shape = ", x_check.shape)
        

    # 2. Motion model jacobian with respect to last state
    # Our state funtion takes a 3 dimensional vector (x, y, theta) and spits out
    # a 3 dimensional one too. So the Jacobian is a 3x3 matrix.
    F_km = np.zeros([3, 3])
    F_km = np.array([[1, 0, -1 * delta_t * v[k-1] * np.sin(wraptopi(theta))], [0, 1, delta_t * v[k-1] * np.cos(wraptopi(theta))], [0, 0, 1]], dtype='float')
    print(F_km)
 

    # 3. Motion model jacobian with respect to noise
    L_km = np.zeros([3, 2])
    L_km = delta_t * A1
    print(L_km)
    
    
    # 4. Propagate uncertainty
    P_check = F_km @ P_check @ F_km.T + L_km @ Q_km @ L_km.T
    print(P_check)
    

    # 5. Update state estimate using available landmark measurements
    for i in range(len(r[k])):
        x_check, P_check = measurement_update(l[i], r[k, i], b[k, i], P_check, x_check)
   
    
    # Set final state predictions for timestep
    x_est[k, 0] = x_check[0]
    x_est[k, 1] = x_check[1]
    x_est[k, 2] = x_check[2]
    P_est[k, :, :] = P_check
    




e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(x_est[:, 0], x_est[:, 1])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Estimated trajectory')
plt.show()

e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(t[:], x_est[:, 2])
ax.set_xlabel('Time [s]')
ax.set_ylabel('theta [rad]')
ax.set_title('Estimated trajectory')
plt.show()
