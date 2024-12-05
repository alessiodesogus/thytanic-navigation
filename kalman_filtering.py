import numpy as np

# Define all variances
var_x = 0.1556817383717473  # a mesurer
var_y = 0.10466897516520202  # a mesurer
var_angle = 1.321882373271498  # a mesurer
var_vel = 1.061875959140749

# Compute Kalman parameters
# Assuming that the variance is caused half by the model and half by the measurement
# This can be justified by plotting the real time data from odometry and the estimated ones
# And comparing the variance when there is camera or not to see if the variance increases exponentially or not
var_vel_meas = var_vel / 2  # variance on velocity measurement
var_vel = var_vel / 2  # variance on velocity state

var_x_meas = var_x   # variance on position x measurement
var_x = var_x  # variance on position x state

var_y_meas = var_y  # variance on position y measurement
var_y = var_y   # variance on position y state

var_angle_meas = var_angle  # variance on angle measurement
var_angle = np.arctan2(var_y, var_x)  # variance on angle state

Ts = 0.1

# Model matrix
A = np.array(
    [
        [1, Ts, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, Ts, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, Ts],
        [0, 0, 0, 0, 0, 1],
    ]
)
# model covariance matrix 
Q = np.diag([[var_x, var_vel, var_y, var_vel, var_angle, var_vel]])


def kalman_filter(thytanic, camera):
    """
    Kalman filter.

    Arguments:
    - thytanic: robot object
    - camera : camera estimated positions (x, y and theta)

    Returns:
    - thytanic object with new estimated states and covariance
    """
    
    x_est_prev = thytanic.x_est
    P_est_prev = thytanic.P_est

    if np.isnan(camera).any():  # No camera available, only use the Thymio's velocity and angle from the odometry
        print("Look at me, no camera!")
        # calculate v_x, v_y and v_angular
        v_x, v_y, v_angular = thytanic_velocity(thytanic, x_est_prev[4]) #  angle from previous estimation
        y_true = np.array([v_x, v_y, v_angular])
        # state to output matrix
        H = np.array([[0, 1, 0, 0, 0, 0], 
                      [0, 0, 0, 1, 0, 0], 
                      [0, 0, 0, 0, 0, 1]])
        # measurement covariance matrix
        R = np.diag([var_vel_meas, var_vel_meas, var_vel_meas])

    else:
        # calculate v_x, v_y and v_angular
        v_x, v_y, v_angular = thytanic_velocity(thytanic, camera[2]) # angle from camera
        y_true = np.array(
            [camera[0], v_x, camera[1], v_y, camera[2], v_angular]
        )
        # state to output matrix : we have access to all the measurements of the states
        H = np.identity(6) 
        # measurement covariance
        R = np.diag(
            [ var_x_meas,
                (var_vel_meas + var_angle_meas), # because we use theta to compute the v_x
                var_y_meas,
                (var_vel_meas + var_angle_meas), ## because we use theta to compute the v_y
                var_angle_meas,
                var_vel_meas]
        ) 

    # Prediction Step
    x_est_a_priori = np.dot(A, x_est_prev)  #new state from model
    P_est_a_priori = np.dot(A, np.dot(P_est_prev, A.T)) + Q  #predicted covariance with model
    
    #innovation : difference between the measurement and the prediction of model
    i = y_true - np.dot(H, x_est_a_priori)
    # Kalman
    S = np.dot(H, np.dot(P_est_a_priori, H.T)) + R  #covariance with measurement prediction 
    K = np.dot(P_est_a_priori, np.dot(H.T, np.linalg.inv(S)))  #Kalman gain : how much the predictions should be corrected based on the measurements

    # Update 
    x_est = x_est_a_priori + np.dot(K, i) #corrected state with innovation and Kalman gain
    P_est = P_est_a_priori - np.dot(K,np.dot(H, P_est_a_priori))

    thytanic.x_est = x_est
    thytanic.P_est = P_est

    return


def thytanic_velocity(thytanic, angle):
    """
    Calculates the velocity components of the Thytanic according to the angle from either the camera or the previous estimation.

    Arguments:
    - thytanic: Thytanic object
    - angle: Orientation angle of the Thytanic in radians

    Returns:
    - v_x : velocity in x direction
    - v_y : velocity in y direction
    - angular_velocity : of the thytanic
    """
    speed_left, speed_right = (speed for speed in thytanic.read_wheel_speed())
    
    # Linear and angular velocity computations according to the slides formulae
    linear_velocity = (
        abs(speed_left + speed_right)
        * thytanic.wheel_radius
        / thytanic.mm_per_pixel
        / 2
    )
    angular_velocity = (
        (speed_right - speed_left) * thytanic.wheel_radius / thytanic.axle_length
    )# no absolute value to have the sign of the turn
    
    # Decomposing linear velocity into x and ycomponents
    v_x = linear_velocity * np.cos(angle) 
    v_y = -linear_velocity * np.sin(angle)  # because y axis is pointing down
    return v_x, v_y, angular_velocity
