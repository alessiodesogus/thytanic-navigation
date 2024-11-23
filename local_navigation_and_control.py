from enum import Enum
import numpy as np
from tdmclient import ClientAsync, aw

class ThytanicState(Enum):
    """Define the three operational states of the Thymio."""

    GLOBAL_MOVEMENT = 0
    AVOIDING_OBSTACLE = 1
    STOP = 2

class ThytanicController:
    """Control the operation and movement of the Thymio robot (Thytanic)."""

    def __init__(self):
        """Initialize robot attributes and state variables."""
        self.robot_node = None
        self.robot_client = None

        # State and speed parameters
        self.robot_state = ThytanicState.STOP
        self.normal_speed = 100  # NEED TO BE FINE-TUNED
        self.avoidance_turn_speed = 150  # NEED TO BE FINE-TUNED
        self.detection_threshold = 2000  # NEED TO BE FINE-TUNED
        self.obstacle_side = None

    def establish_connection(self):
        """Connect to the robot and lock it."""
        self.robot_client = ClientAsync()
        self.robot_node = aw(self.robot_client.wait_for_node())
        self.robot_client.process_waiting_messages()
        aw(self.robot_node.lock())

    def disconnect(self):
        """Stop the robot and release the connection."""
        self.set_wheel_speed(0, 0)
        aw(self.robot_node.unlock())
        self.robot_node = None
        self.robot_client = None
        self.robot_state = ThytanicState.STOP

    def set_wheel_speed(self, left_speed, right_speed):
        """
        Assign speeds to the robot's left and right wheels.

        Arguments:
        - left_speed: Speed for the left wheel.
        - right_speed: Speed for the right wheel.
        """
        speed_config = {
            'motor.left.target': [int(left_speed)],
            'motor.right.target': [int(right_speed)],
        }
        aw(self.robot_node.set_variables(speed_config))

    def rotate_robot(self, direction, rotation_speed):
        """
        Rotate the robot in the specified direction.

        Arguments:
        - direction: "LEFT" or "RIGHT".
        - rotation_speed: Speed of the rotation for both wheels.
        """
        if direction == "LEFT":
            self.set_wheel_speed(-rotation_speed, rotation_speed)
        elif direction == "RIGHT":
            self.set_wheel_speed(rotation_speed, -rotation_speed)

    def read_proximity_sensors(self):
        """
        Retrieve proximity sensor data from the robot.

        Returns:
        - sensor_values: An array of proximity sensor readings.
        """
        aw(self.robot_node.wait_for_variables({'prox.horizontal'}))
        sensor_values = self.robot_node.v.prox.horizontal[0:5]
        aw(self.robot_client.sleep(0.0001))
        return sensor_values

    def maneuver_around_obstacle(self, sensor_readings):
        """
        Perform a maneuver to avoid an obstacle based on sensor data.

        Arguments:
        - sensor_readings: Proximity sensor data.

        Returns:
        - obstacle_direction: The detected direction of the obstacle.
        """
        # Turn right if the obstacle is on the left side
        if any(value > self.detection_threshold for value in sensor_readings[0:2]):
            self.rotate_robot("RIGHT", self.avoidance_turn_speed)
            return "LEFT"

        # Turn left if the obstacle is on the right side
        elif any(value > self.detection_threshold for value in sensor_readings[2:5]):
            self.rotate_robot("LEFT", self.avoidance_turn_speed)
            return "RIGHT"

    def update_robot_state(self):
        """
        Update the robot's operational state and execute corresponding actions.
        """
        # Get sensor data
        sensor_data = self.read_proximity_sensors()
        is_obstacle_detected = any(value > self.detection_threshold for value in sensor_data)

        if self.robot_state == ThytanicState.STOP:
            self.set_wheel_speed(0, 0)
            return

        if self.robot_state == ThytanicState.GLOBAL_MOVEMENT and is_obstacle_detected:
            self.robot_state = ThytanicState.AVOIDING_OBSTACLE
            self.obstacle_side = self.maneuver_around_obstacle(sensor_data)

        elif self.robot_state == ThytanicState.AVOIDING_OBSTACLE and not is_obstacle_detected:
            self.robot_state = ThytanicState.GLOBAL_MOVEMENT
            self.set_wheel_speed(self.normal_speed, self.normal_speed)

        elif self.robot_state == ThytanicState.AVOIDING_OBSTACLE and is_obstacle_detected:
            self.maneuver_around_obstacle(sensor_data)
