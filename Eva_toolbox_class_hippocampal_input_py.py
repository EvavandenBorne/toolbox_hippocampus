# The input to the hippocampus given an environment

import numpy as np
from Eva_toolbox_global_functions_py import closest_location_index

class hippocampal_input:
    #
    def __init__(self, *args):
        input_attributes = args[0]
        # self.distance_matrix
        # self.trajectory_matrix
        # self.place_field_coordinates
        # self.sorted_indices_place_fields
        # self.active_neurons
        # self.place_fields
        if "number_locations" in input_attributes:
            self.number_locations = input_attributes["number_locations"]
        else:
            self.number_locations = 1000 # (100)
        if "place_fields_settings" in input_attributes:
            self.place_fields_settings = input_attributes["place_fields_settings"]
        else:
            self.place_fields_settings = {"type": "Von_Mises", "width": (2*np.pi)/20, "normalise": "yes"}
        if "fraction_active_neurons" in input_attributes:
            self.fraction_active_neurons = input_attributes["fraction_active_neurons"]
        else:
            self.fraction_active_neurons = 1./2. # (1.)
        self.location_coordinates = np.linspace(0., 2.*np.pi, self.number_locations)

    # Initialises distance matrix and then place fields
    def initialise_place_fields(self, number_neurons):
        minimum_coordinate = np.min(self.location_coordinates)
        maximum_coordinate = np.max(self.location_coordinates)
        self.place_field_coordinates = np.random.uniform(low=minimum_coordinate, high=maximum_coordinate, size=(number_neurons,))
        self.sorted_indices_place_fields = np.argsort(self.place_field_coordinates)
        grid_x, grid_y = np.meshgrid(self.location_coordinates, self.place_field_coordinates)
        self.distance_matrix = np.matrix(np.absolute(grid_x - grid_y))
        if self.place_fields_settings["type"] is "cosine":
            self.place_fields = np.cos(self.distance_matrix)
        elif self.place_fields_settings["type"] is "Von_Mises":
            sigma = self.place_fields_settings["width"]
            self.place_fields = np.exp(np.cos(self.distance_matrix)/sigma)
        elif self.place_fields_settings["type"] is "Gaussian":
            sigma = self.place_fields_settings["width"]
            self.place_fields = np.exp(-np.power((self.distance_matrix)/sigma,2)/2)
        if self.place_fields_settings["normalise"] is "yes":
            normalisation =  np.max(self.place_fields)
            self.place_fields = np.divide(self.place_fields, normalisation)
        self.active_neurons = np.random.binomial(n=1, p=self.fraction_active_neurons, size=(number_neurons,))
        self.place_fields = np.transpose(np.multiply(np.transpose(self.place_fields), self.active_neurons))
        return

    def generate_linear_trajectory_matrix(self, trajectory, number_locations, location_coordinates):
        number_time_points = len(trajectory)
        trajectory_matrix = np.zeros((number_locations, number_time_points), dtype=float)
        time_index = 0
        for location in trajectory:
            location_index = closest_location_index(location_coordinates, location)
            trajectory_matrix[location_index,time_index] = 1.
            time_index = time_index + 1
        return np.matrix(trajectory_matrix)

    def generate_Tmaze_trajectory_matrix(self, direction, number_locations_arm, time):
        time_step = time[1] - time[0]
        number_locations_body = self.number_locations - 2*number_locations_arm
        location_coordinates_body = self.location_coordinates[0:number_locations_body]
        location_coordinates_arm = self.location_coordinates[number_locations_body:number_locations_body+number_locations_arm]
        #rat_speed = (self.location_coordinates[self.number_locations - number_locations_arm] - self.location_coordinates[0])/np.float(time[-1] - time[0] + 1)
        #rat_speed = (number_locations_body + number_locations_arm)/np.float(time[-1] - time[0] + 1)
        rat_speed = (number_locations_body + number_locations_arm)/np.float(time[-1] - time[0] + 1)
        #rat_speed = (self.location_coordinates[self.number_locations - number_locations_arm] - self.location_coordinates[0] - 2*self.location_coordinates[1])/np.float(time[-1] - time[0] + 1) # SAME AS:
        #rat_speed = self.location_coordinates[self.number_locations - number_locations_arm - 2]/np.float(time[-1] - time[0] + 1) # EQUAL TO FOURTH
        end_time_body = int(round(number_locations_body/np.float(rat_speed)))
        end_time_arm = int(round(number_locations_arm/np.float(rat_speed)))
        #end_time_body = int(round((location_coordinates_body[-1] - location_coordinates_body[0])/np.float(rat_speed)))
        #end_time_arm = int(round((location_coordinates_arm[-1] - location_coordinates_arm[0])/np.float(rat_speed)))
        time_body = np.arange(0, end_time_body, time_step)
        #time_arm = np.arange(0, end_time_arm, time_step)
        time_arm = np.arange(end_time_body, end_time_body + end_time_arm, time_step)
        #trajectory_body = rat_speed*time_body
        #trajectory_arm = rat_speed*time_arm
        trajectory_body = 2*np.pi*rat_speed*time_body/np.float(self.number_locations)
        trajectory_arm = 2*np.pi*rat_speed*time_arm/np.float(self.number_locations)
        trajectory_matrix_body = self.generate_linear_trajectory_matrix(trajectory_body, number_locations_body, location_coordinates_body)
        trajectory_matrix_arm = self.generate_linear_trajectory_matrix(trajectory_arm, number_locations_arm, location_coordinates_arm)
        empty_arms_T1 = np.matrix(np.zeros((2*number_locations_arm, len(time_body)), dtype=float))
        empty_arm_T2 = np.matrix(np.zeros((number_locations_arm, len(time_arm)), dtype=float))
        empty_body_T2 = np.matrix(np.zeros((number_locations_body, len(time_arm)), dtype=float))
        trajectory_matrix_T1 = np.append(trajectory_matrix_body, empty_arms_T1, axis=0)
        if direction is "left":
            trajectory_matrix_T2 = np.append(empty_body_T2, trajectory_matrix_arm, axis=0)
            trajectory_matrix_T2 = np.append(trajectory_matrix_T2, empty_arm_T2, axis=0)
        elif direction is "right":
            trajectory_matrix_T2 = np.append(empty_body_T2, empty_arm_T2, axis=0)
            trajectory_matrix_T2 = np.append(trajectory_matrix_T2, trajectory_matrix_arm, axis=0)
        else:
            print 'Choose a valid direction ("left" or "right").'
            return
        trajectory_matrix = np.append(trajectory_matrix_T1, trajectory_matrix_T2, axis=1)
        return trajectory_matrix

    def generate_trajectory_matrix(self, time, trajectory_parameters):
        trajectory_type = trajectory_parameters["trajectory_type"]
        if trajectory_type == "linear":
            initial_position = trajectory_parameters["initial_position"]
            rat_speed = trajectory_parameters["rat_speed"]
            trajectory = np.mod(initial_position + rat_speed*time, 2*np.pi)
            self.trajectory_matrix = self.generate_linear_trajectory_matrix(trajectory, self.number_locations, self.location_coordinates)
        elif trajectory_type == "T-maze":
            left_direction_probability = trajectory_parameters["left_direction_probability"]
            left_direction = np.random.binomial(1, left_direction_probability)
            if left_direction == 1:
                direction = "left"
            elif left_direction == 0:
                direction = "right"
            number_locations_arm = trajectory_parameters["number_locations_arm"]
            self.trajectory_matrix = self.generate_Tmaze_trajectory_matrix(direction, number_locations_arm, time)
        else:
            print "Invalid trajectory type."
            return
        self.locations_trajectory = np.matrix(np.nonzero(self.trajectory_matrix.T)[1]).T # Needed for the rank correlation in the network class.
        return

