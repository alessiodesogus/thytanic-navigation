import numpy as np

# Define all variances
var_x = 0.011652641682748977  # a mesurer
var_y = 0.020724245655266565  # a mesurer
var_angle = 0.0008948119074700559  # a mesurer
var_vel = 4.079861111111112

# Compute Kalman parameters
var_vel_meas = var_vel / 2  # variance on velocity measurement
var_vel = var_vel / 2  # variance on velocity state

var_x_meas = var_x / 2  # variance on position x measurement
var_x = var_x / 2  # variance on position x state

var_y_meas = var_y / 2  # variance on position y measurement
var_y = var_y / 2  # variance on position y state

var_angle_meas = var_angle / 2  # variance on angle measurement
var_angle = np.arctan2(var_y, var_x)  # variance on angle state

Ts = 0.15

# Model matrices
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
# model covariance
Q = np.array(
    [
        [var_x, 0, 0, 0, 0, 0],
        [0, var_vel, 0, 0, 0, 0],
        [0, 0, var_y, 0, 0, 0],
        [0, 0, 0, var_vel, 0, 0],
        [0, 0, 0, 0, var_angle, 0],
        [0, 0, 0, 0, 0, var_vel],
    ]
)


def kalman_filter(thytanic, camera):
    """
    Kalman filter.

    Arguments:
    - thytanic: robot object
    - camera : camera estimated positions (x,y,theta)

    Returns:
    - thytanic : updated robot state estimation and covariance
    """

    x_est_prev = thytanic.x_est
    P_est_prev = thytanic.P_est

    if np.isnan(
        camera
    ).any():  # No camera available, only use the Thymio's velocity and angle from the odometry
        print("Look at me, no camera!")
        # calculate v_x, v_y and v_angular
        velocity = thytanic_velocity(thytanic, x_est_prev[4])
        y_true = np.array(velocity)
        # state to output matrix in case we don't have measurements of x, y and theta
        H = np.array([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1]])
        # measurement covariance
        R = np.diag([var_vel_meas, var_vel_meas, var_vel_meas])

    else:
        velocity = thytanic_velocity(thytanic, camera[2])
        y_true = np.array(
            [camera[0], velocity[0], camera[1], velocity[1], camera[2], velocity[2]]
        )
        # state to output matrix in case we can measure all states
        H = np.identity(6)
        # measurement covariance
        R = np.diag(
            [
                var_x_meas,
                (var_vel_meas + var_angle_meas),
                var_y_meas,
                (var_vel_meas + var_angle_meas),
                var_angle_meas,
                var_vel_meas,
            ]
        )

    # Prediction Step
    x_est_a_priori = np.dot(A, x_est_prev)  # new state
    P_est_a_priori = np.dot(A, np.dot(P_est_prev, A.T)) + Q  # predicted covariance

    # Kalman Gain
    i = y_true - np.dot(H, x_est_a_priori)  # innovation
    S = np.dot(H, np.dot(P_est_a_priori, H.T)) + R
    K = np.dot(P_est_a_priori, np.dot(H.T, np.linalg.inv(S)))  # Kalman gain

    # Update Step
    x_est = x_est_a_priori + np.dot(K, i)
    P_est = np.dot((np.eye(len(K)) - np.dot(K, H)), P_est_a_priori)  # adjust covariance

    thytanic.x_est = x_est
    thytanic.P_est = P_est

    return x_est, P_est


def thytanic_velocity(thytanic, angle):
    """
    Calculates the velocity components of the Thytanic according to the angle.

    Arguments:
    - thytanic: Thytanic object
    - angle: Orientation angle of the Thytanic in radians

    Returns:
    - velocity: A numpy array containing [v_x, v_y, angular_velocity]
    """
    speed_left, speed_right = (speed for speed in thytanic.read_wheel_speed())
    # Linear and angular velocity computations
    linear_velocity = (
        abs(speed_left + speed_right)
        * thytanic.wheel_radius
        / thytanic.mm_per_pixel
        / 2
    )
    angular_velocity = (
        (speed_right - speed_left) * thytanic.wheel_radius / thytanic.axle_length
    )
    # Decomposing linear velocity into components
    v_x = linear_velocity * np.cos(angle)
    v_y = -linear_velocity * np.sin(angle)
    velocity = np.array([v_x, v_y, angular_velocity])
    return velocity
