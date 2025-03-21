from enum import Enum
import numpy as np
from tdmclient import ClientAsync, aw
import time


class ThytanicState(Enum):
    """Define the operational states of the Thytanic."""

    GLOBAL_MOVEMENT = 0
    AVOIDING_OBSTACLE = 1
    STOP = 2


class ThytanicController:
    """Control the operation and movement of the Thytanic."""

    def __init__(self):
        """Initialize Thytanic attributes and state variables."""

        self.robot_node = None
        self.robot_client = None

        # State and speed parameters
        self.robot_state = ThytanicState.STOP
        self.normal_speed = 50  
        self.avoidance_turn_speed = 80  
        self.detection_threshold = 1500  
        self.obstacle_side = None
        self.conversion_factor = 0.48

        # Astofi controller parameters
        self.k_rho = 32  
        self.k_alpha = 36  
        # self.k_beta = -15 
        self.wheel_radius = 21
        self.axle_length = 95
        self.min_distance = 7  
        self.max_distance = 15  
        self.mm_per_pixel = 9.75  

    def establish_connection(self):
        """Connect to the Thytanic and lock it."""

        self.robot_client = ClientAsync()
        self.robot_node = aw(self.robot_client.wait_for_node())
        self.robot_client.process_waiting_messages()
        aw(self.robot_node.lock())

    def disconnect(self):
        """Stop the Thytanic and release the connection."""

        self.set_wheel_speed(0, 0)
        aw(self.robot_node.unlock())
        self.robot_node = None
        self.robot_client = None
        self.robot_state = ThytanicState.STOP

    def set_wheel_speed(self, left_speed, right_speed):
        """Assign speeds to the Thytanic's left and right wheels."""

        speed_config = {
            "motor.left.target": [int(left_speed)],
            "motor.right.target": [int(right_speed)],
        }
        aw(self.robot_node.set_variables(speed_config))

    def read_wheel_speed(self):
        """Read speed from the Thytanic."""

        aw(
            self.robot_node.wait_for_variables(
                {"motor.left.speed", "motor.right.speed"}
            )
        )
        speed_values = [
            self.robot_node.v.motor.left.speed
            * self.conversion_factor
            / self.mm_per_pixel,
            self.robot_node.v.motor.right.speed
            * self.conversion_factor
            / self.mm_per_pixel,
        ]
        aw(self.robot_client.sleep(0.0001))

        return speed_values

    def read_accelerometer(self):
        """Read accelerometer from the Thytanic."""

        aw(self.robot_node.wait_for_variables({"acc"}))
        """print("accelerometer:", self.robot_node.v.acc[0])
        print("accelerometer:", self.robot_node.v.acc[1])
        print("accelerometer:", self.robot_node.v.acc[2])"""
        return self.robot_node.v.acc

    def rotate_robot(self, direction, rotation_speed):
        """Rotate the Thytanic in the specified direction."""

        if direction == "BABOR":
            self.set_wheel_speed(-rotation_speed, rotation_speed)
        elif direction == "TRIBORD":
            self.set_wheel_speed(rotation_speed, -rotation_speed)

    def read_proximity_sensors(self):
        """Retrieve proximity sensor data from the Thytanic."""

        aw(self.robot_node.wait_for_variables({"prox.horizontal"}))
        sensor_values = self.robot_node.v.prox.horizontal[0:5]
        aw(self.robot_client.sleep(0.0001))
        return sensor_values

    def maneuver_around_obstacle(self, sensor_readings):
        """Perform a maneuver to avoid an obstacle based on sensor data."""

        # Turn right if the obstacle is on the left side
        if any(value > self.detection_threshold for value in sensor_readings[0:2]):
            self.rotate_robot("TRIBORD", self.avoidance_turn_speed)
            return "BABOR"

        # Turn left if the obstacle is on the right side
        elif any(value > self.detection_threshold for value in sensor_readings[2:5]):
            self.rotate_robot("BABOR", self.avoidance_turn_speed)
            return "TRIBORD"

    def update_robot_state(self):
        """Update the Thytanic's operational state and execute corresponding actions."""

        # Get sensor data
        sensor_data = self.read_proximity_sensors()
        is_obstacle_detected = any(
            value > self.detection_threshold for value in sensor_data
        )

        # Get speed from the controller
        # left_speed, right_speed = self.control_robot([0, 0, 0], [10, 10])

        if self.robot_state == ThytanicState.STOP:
            self.set_wheel_speed(0, 0)
            return

        elif (
            self.robot_state == ThytanicState.GLOBAL_MOVEMENT
            and not is_obstacle_detected
        ):
            self.control_robot()

        elif self.robot_state == ThytanicState.GLOBAL_MOVEMENT and is_obstacle_detected:
            self.control_robot()
            self.robot_state = ThytanicState.AVOIDING_OBSTACLE
            self.obstacle_side = self.maneuver_around_obstacle(sensor_data)
        elif (
            self.robot_state == ThytanicState.AVOIDING_OBSTACLE and is_obstacle_detected
        ):
            self.maneuver_around_obstacle(sensor_data)

        elif (
            self.robot_state == ThytanicState.AVOIDING_OBSTACLE
            and not is_obstacle_detected
        ):
            self.set_wheel_speed(100, 100)
            time.sleep(3)
            self.goal_idx += 2
            self.robot_state = ThytanicState.GLOBAL_MOVEMENT
            self.control_robot()

    def astolfi_control(self, state_est):
        """Compute control commands for the Thytanic."""

        x, y, theta = state_est
        # print("x", x, "y", y, "theta", theta)
        x_goal, y_goal = self.goal
        # print("goal index", self.goal_idx)

        # Compute polar coordinates relative to the goal
        delta_x = x_goal - x
        delta_y = y_goal - y
        # print("delta_x", delta_x, "delta_y", delta_y)
        rho = np.sqrt(delta_x**2 + delta_y**2)  # Distance to the goal
        alpha = -theta + np.arctan2(-delta_y, delta_x)  # Orientation to the goal
        # beta = -alpha -theta # Final orientation adjustment

        # print("alpha", alpha)
        # Normalize alpha and beta to [-pi, pi]
        if alpha > np.pi:
            alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
        if alpha < -np.pi:
            alpha = (alpha + 2 * np.pi) % (2 * np.pi)
        # beta = (beta + np.pi) % (2 * np.pi) - np.pi

        # Compute control law
        v = self.normal_speed
        if abs(alpha) > 0.1:
            v = self.k_rho * np.log(rho)
        omega = self.k_alpha * alpha  # + self.k_beta * beta

        return rho, v, omega

    def control_robot(self):
        """Main function to control the Thytanic."""

        state_est = [self.x_est[0], self.x_est[2], self.x_est[4]]  # x y theta
        # Compute control commands
        rho, v, omega = self.astolfi_control(state_est)
        # print("v", v, "omega", omega)

        # If close enough to the goal, stop
        if rho < self.min_distance:
            self.goal_idx += 1  # get new target
            return 0, 0

        """if rho > self.max_distance:
            self.goal_idx += 1  # get new
            print("POINTS SKIPPED")
            return 0, 0"""

        # Convert translational and rotational velocities to wheel speeds
        left_speed = (
            v * self.mm_per_pixel / self.conversion_factor
            - omega * self.axle_length / 2
        ) / self.wheel_radius
        right_speed = (
            v * self.mm_per_pixel / self.conversion_factor
            + omega * self.axle_length / 2
        ) / self.wheel_radius
        # print("left speed", left_speed)
        # print("right speed", right_speed)
        # get the speed infos to motors
        self.set_wheel_speed(left_speed, right_speed)

        return left_speed, right_speed
