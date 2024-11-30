import numpy as np
# Define all variances
var_x = 0.1556817383717473 # a mesurer
var_y = 0.10466897516520202# a mesurer
#var_angle = 1.321882373271498# a mesurer
var_vel = 1.061875959140749

# Compute Kalman parameters
var_vel_meas = var_vel / 2  # variance on velocity measurement 
var_vel = var_vel / 2  # variance on velocity state

var_x_meas = var_x/2     # variance on position x measurement 
var_x = var_x/2     # variance on position x state

var_y_meas = var_y/2     # variance on position y measurement
var_y = var_y/2     # variance on position y state

#var_angle_meas = var_angle/2      # variance on angle measurement
#var_angle = var_angle/2     # variance on angle state
var_angle = np.arctan2(var_y, var_x) # variance on angle state

Ts = 0.1

# Model matrices
# model
A = np.array([[1, Ts, 0, 0, 0, 0], 
              [0, 1, 0, 0, 0, 0], 
              [0, 0, 1, Ts, 0, 0], 
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, Ts],
              [0, 0, 0, 0, 0, 1]])
# model covariance
Q = np.array([[var_x, 0, 0, 0, 0, 0], 
              [0, var_vel, 0, 0, 0, 0], 
              [0, 0, var_y, 0, 0, 0], 
              [0, 0, 0, var_vel, 0, 0],
              [0, 0, 0, 0, var_angle, 0],
              [0, 0, 0, 0, 0, var_vel]])    
    
    
def kalman_filter(thytanic, camera) : #, x_est_prev, P_est_prev):
    """
    Kalman filter.
    
    Arguments:
    - thymio: Thymio object
    - camera : camera estimated positions
    - x_est_prev: previous estimated states of the Thymio
    - P_est_prev: previous estimated covariance of the states
    
    Returns:
    - x_est: new estimated states of the Thymio
    - P_est: new estimated covariance of the states
    """
    
    x_est_prev = thytanic.x_est
    P_est_prev = thytanic.P_est
    
    if camera is None: # No camera available, only use the Thymio's velocity and angle from the odometry
        #a = (angle)%360 
        #if(a >180): a = a - 180   # for keeping the angle between -180 and 180
        #x_est_prev[4] = x_est_prev[4] % 360 
        #if (x_est_prev[4] > 180): x_est_prev[4] = x_est_prev[4] - 180
        
        # calculate v_x, v_y and v_angular
        velocity = thytanic_velocity(thytanic, x_est_prev[4])
        y_true = np.array(velocity)
        # state to output matrix
        H = np.array([[0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 1]])
        # measurement covariance
        R = np.diag([var_vel_meas, var_vel_meas, var_vel_meas])
        
    else:
        #if abs(GPS[2] - x_est_a_priori[4]) > 45:######wtf ?
        #    x_est_a_priori[4] = GPS[2]
        #camera[2] = camera[2] % 360
        #if (camera[2] > 180): camera[2] = camera[2] - 180
        # calculate v_x, v_y and v_angular
        velocity = thytanic_velocity(thytanic, camera[2])#camera[2])
        y_true = np.array([camera[0],
                      velocity[0],
                      camera[1],
                      velocity[1],
                      #camera[2],
                      velocity[2]])
        # state to output matrix
        H = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      #[0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]])      
        # measurement covariance
        R = np.diag([var_x_meas, (var_vel_meas + var_angle_meas),  #+ var_angle_meas), 
                     var_y_meas, (var_vel_meas + var_angle_meas), #+ var_angle_meas), 
                      var_vel_meas]) #var_angle_meas , var_vel_meas])

        """
    # Filtering step
    K = np.dot(P_est, np.dot(H.T, np.linalg.inv(np.dot(H, np.dot(P_est, H.T)) + R))) #compute Kalman
    x_est = x_est_prev + K * (y - np.dot(H, x_est_prev)) #x_prev - K*I
    x_est[4] = x_est[4] % 360  # angle modulo 360

    P_est = P_est - np.dot(K, np.dot(H, P_est)) # P_est = (I-KH)P_est

    x_est = np.dot(A, x_est)
    x_est[4] = x_est[4] % 360 # angle modulo 360
    
    # Estimated covariance of the states
    P_est = np.dot(A, np.dot(P_est_prev, A.T)) + Q
    """
    # Prediction Step
    x_est_a_priori = np.dot(A, x_est_prev)  #new state
    #print("x_est_a_priori",x_est_a_priori)
    P_a_priori = np.dot(A, np.dot(P_est_prev, A.T)) + Q  #predicted covariance
    
    # Kalman Gain
    S = np.dot(H, np.dot(P_a_priori, H.T)) + R  #innovation step
    K = np.dot(P_a_priori, np.dot(H.T, np.linalg.inv(S)))  #Kalman gain

    # Update Step
    x_est = x_est_a_priori + np.dot(K, (y_true - np.dot(H, x_est_a_priori)))  #corrected state with innovation and Kalman gain
    #x_est[4] = x_est[4] % 360  # Wrap angle to [0, 360]
    #if (x_est_prev[4] > 180): x_est_prev[4] = x_est_prev[4] - 180
    P_est = np.dot((np.eye(len(K)) - np.dot(K, H)), P_a_priori)  #adjust covariance

    thytanic.x_est = x_est
    thytanic.P_est = P_est
    
    return x_est, P_est
"""
def thytanic_velocity(thymio, angle_thymio):
    
    Compute the velocity of the Thymio.
    
    Arguments:
    - thymio: Thymio object
    - angle_thymio: angle of the Thymio
    
    Returns:
    - velocity: velocity of the Thymio
    
    [speed_l, speed_r] = thymio.read_wheel_speed()

    if speed_r < 0 or speed_l < 0:
        # Turning
        if abs(speed_r) > abs(speed_l):
            # speed_r is max
            if speed_r < 0:
                # Turn clockwise, speed_l > 0
                angular_v = -2 * abs(speed_l) / thymio.axle_length * 180 / np.pi
                lin_velocity = -(abs(speed_r) - abs(speed_l))
            else:
                # Turn anticlockwise, speed_l < 0, speed_r > 0
                angular_v = 2 * abs(speed_l) / thymio.axle_length * 180 / np.pi
                lin_velocity = abs(speed_r) - abs(speed_l)
        else: 
            # speed_l is max
            if speed_l < 0:
                # Turn anticlockwise, speed_r > 0
                angular_v = 2 * abs(speed_r) / thymio.axle_length * 180 / np.pi
                lin_velocity = -(abs(speed_l) - abs(speed_l))
            else :
                # Turn clockwise, speed_r < 0, speed_l > 0
                angular_v = -2 * abs(speed_r) / thymio.axle_length * 180 / np.pi
                lin_velocity = abs(speed_l) - abs(speed_l)
    else:
        if speed_r > speed_l:
            # Turn anticlockwise
            angular_v = -abs(speed_r - speed_l) / thymio.axle_length * 180 / np.pi
            lin_velocity = speed_l
        else:
            # Turn clockwise
            angular_v = abs(speed_r - speed_l) / thymio.axle_length * 180 / np.pi
            lin_velocity = speed_r

    v_x = lin_velocity * np.cos(angle_thymio * np.pi / 180)
    v_y = lin_velocity * np.sin(angle_thymio * np.pi / 180)
    velocity = np.array([v_x, v_y, angular_v])

    return velocity

"""
def thytanic_velocity(thytanic, angle):
    """
    Calculates the velocity components of the Thytanic according to the angle.

    Arguments:
    - thytanic: Thytanic object with motor speeds and physical attributes
    - angle: Orientation angle of the Thytanic in degrees

    Returns:
    - velocity: A numpy array containing [v_x, v_y, angular_velocity]
    """
    speed_left, speed_right = (speed for speed in thytanic.read_wheel_speed())
    # Linear and angular velocity computations
    linear_velocity = abs(speed_left + speed_right) * thytanic.wheel_radius / 2
    angular_velocity = (speed_right - speed_left) * thytanic.wheel_radius / thytanic.axle_length
    # Decomposing linear velocity into components
    v_x = linear_velocity * np.cos(angle)#(np.radians(angle))
    v_y = -linear_velocity * np.sin(angle)##(np.radians(angle)) #################-
    velocity = np.array([v_x, v_y, angular_velocity])
    return velocity
