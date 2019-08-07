#

import numpy as np
import matplotlib.pyplot as plt
import warnings
import scipy.stats
import scipy.signal as signal
from Eva_toolbox_global_functions_py import activation_function
from Eva_toolbox_global_functions_py import permute_matrix
from Eva_toolbox_class_hippocampal_input_py import hippocampal_input

class network:
    #
    def __init__(self, **kwargs):
        self.activation_parameters = 1 # parameter(s) of the activation function
        self.connectivity_settings = {"random": "yes", "initial_weights_sd": 0.1}
        self.constant_current = -7. # unit?
        self.ex2in_connectivity_scaling = 0.1
        self.oscillatory_current = 8.
        self.place_current = 15.
        self.random_current = 0.
        self.decay_average = 1./70. # parameter of Exponential Moving Average
        self.depression_time = 800. # synaptic depression time constant in ms for the excitatory population
        self.environments = []
        self.history_average_firing_rate = []
        self.history_connectivity_matrix = []
        self.history_firing_rate = []
        self.history_inhibitory_firing_rate = []
        self.history_inhibitory_synaptic_depression = []
        self.history_synaptic_depression = []
        self.in2ex_connectivity_scaling = -0.001
        self.inhibitory_depression_time = 800. # synaptic depression time constant in ms for the inhibitory population
        self.inhibitory_membrane_time = 10. # membrane time constant in ms for the inhibitory population
        self.learning_rate = 5.*10**0
        self.membrane_time = 10. # membrane time constant in ms for the excitatory population
        self.number_inhibitory_neurons = 100 # 1
        self.number_neurons = 800 # (100)
        self.oscillatory_frequency = 10*10**-3 # input frequency in 1/ms
        self.stm_maximum = 10. # stm: short term memory
        self.stm_minimum = 1. # stm: short term memory
        self.stm_time_minus = 7.5 # stm: short term memory
        self.stm_time_plus = 7.5 # stm: short term memory
        self.synaptic_depletion = 0.8*10**-3 # fraction of utilized synaptic resources released by each spike
        self.initialised = 'no'
        #self.__dict__.update(kwargs)

    def create_new_environment(self, **kwargs):
        new_hippocampal_input = hippocampal_input(kwargs)
        new_hippocampal_input.initialise_place_fields(self.number_neurons)
        self.environments = self.environments + [new_hippocampal_input]
        return

    # This method resets the network to the initial configurations
    def initialise_network(self, distance_matrix, **kwargs):
        if int(distance_matrix.shape[0]) == int(self.number_neurons):
            if self.connectivity_settings["random"] is "yes":
                sd_connectivity = self.connectivity_settings["initial_weights_sd"]
                self.connectivity_matrix = sd_connectivity*np.matrix(np.random.randn(self.number_neurons, self.number_neurons))
                np.fill_diagonal(self.connectivity_matrix,0) # remove self connections
            else:
                self.connectivity_matrix = np.matrix(np.zeros((self.number_neurons, self.number_neurons), dtype=float))
            if "constant_inhibitory_connectivity" in kwargs:
                constant_inhibitory_connectivity = kwargs["constant_inhibitory_connectivity"]
                if constant_inhibitory_connectivity > 0:
                    warnings.warn("Warning: giving a positive constant connectivity could crash the network")
                self.fixed_connectivity_matrix = constant_inhibitory_connectivity*np.matrix(np.ones((self.number_neurons, self.number_neurons), dtype=float))
                np.fill_diagonal(self.fixed_connectivity_matrix, 0) # remove self connections
            else:
                self.fixed_connectivity_matrix = np.matrix(np.zeros((self.number_neurons, self.number_neurons), dtype=float))
            self.ex2in_connectivity_matrix = self.ex2in_connectivity_scaling*np.matrix(np.ones((self.number_inhibitory_neurons, self.number_neurons), dtype=float))
            self.in2ex_connectivity_matrix = self.in2ex_connectivity_scaling*np.matrix(np.ones((self.number_neurons, self.number_inhibitory_neurons), dtype=float))
            self.stm_connectvity_matrix = np.matrix(np.ones((self.number_neurons, self.number_neurons), dtype=float))
            #
# CHANGED
            #self.firing_rate = np.matrix(np.zeros((self.number_neurons, 1), dtype=float)) # OLD initialize with zeros
            self.firing_rate = np.matrix(np.random.normal(loc=0., scale=0.1, size=(self.number_neurons, 1))) # Initialize with random values
# END CHANGED
            self.neuron_highest_activation = np.matrix(np.zeros((1, 1), dtype=float)) # This will be sorted on location (place field) and used for the rank correlation
            self.average_firing_rate = np.matrix(np.zeros((self.number_neurons, 1), dtype=float))
            self.synaptic_depression = np.matrix(np.ones((self.number_neurons, 1), dtype=float))
            self.inhibitory_firing_rate = np.matrix(np.zeros((self.number_inhibitory_neurons, 1), dtype=float))
            self.inhibitory_synaptic_depression = np.matrix(np.ones((self.number_inhibitory_neurons, 1), dtype=float))
            self.initialised = "yes"
            return
        else:
            print "The size of the distance matrix is not consistent with the number of neurons"
            return

    def run_simulation(self, time_parameters, trajectory_parameters, environment_index, learning, spontaneous_activity, **kwargs):
        # MOVE TO OWN FUNCTION ????
        self.neuron_highest_activation = np.matrix(np.zeros((1, 1), dtype=float))
        sorted_indices = self.environments[environment_index].sorted_indices_place_fields
        #
        real_learning_rate = self.learning_rate
        if learning is "no":
            self.learning_rate = 0
        if "connectivity_scaling" in kwargs:
            recurrent_scaling = kwargs["connectivity_scaling"]
        else:
            recurrent_scaling = 1
        initial_time = time_parameters["initial_time"]
        end_time = time_parameters["end_time"]
        time_step = time_parameters["time_step"]
        time = np.arange(initial_time, end_time, time_step)
        self.environments[environment_index].generate_trajectory_matrix(time, trajectory_parameters)
        if self.initialised is "no":
            print "The network has not been initialised (use the 'initialise_network' method)"
            return
        else:
            for t in np.arange(initial_time, end_time, time_step):
                # Network dynamics
                if spontaneous_activity is "yes":
                    place_input = 0.
                elif spontaneous_activity is "no":
                    place_input = self.environments[environment_index].place_fields*self.environments[environment_index].trajectory_matrix[:,t]
                else:
                    print 'Spontaneous activity needs to be set to "yes" or "no"'
                    return
                #recurrent_input = (self.connectivity_matrix*np.multiply(self.firing_rate[:,-1], self.synaptic_depression[:,-1]))/self.number_neurons
# CHANGE
                #recurrent_input = ((    recurrent_scaling*    np.multiply(self.stm_connectvity_matrix, (self.connectivity_matrix+self.fixed_connectivity_matrix)))*np.multiply(self.firing_rate[:,-1], self.synaptic_depression[:,-1]))
                recurrent_input = ((np.multiply(self.stm_connectvity_matrix, (self.connectivity_matrix+self.fixed_connectivity_matrix)))*np.multiply(self.firing_rate[:,-1], self.synaptic_depression[:,-1]))
# END CHANGE
                external_input = self.constant_current + self.oscillatory_current*np.cos(2.*np.pi*self.oscillatory_frequency*t) + self.place_current*place_input
                random_input = self.random_current*np.random.normal(loc=0., scale=1., size=(self.number_neurons,1))
                ex2in_input = self.ex2in_connectivity_matrix*self.firing_rate[:,-1]
                in2ex_input = self.in2ex_connectivity_matrix*self.inhibitory_firing_rate[:,-1]
                new_firing_rate = self.firing_rate[:,-1] + (time_step/self.membrane_time)*(-self.firing_rate[:,-1] + activation_function(recurrent_input + external_input + random_input + in2ex_input, self.activation_parameters))
                sorted_new_firing_rate = new_firing_rate[sorted_indices]
                new_neuron_highest_activation = np.argmax(sorted_new_firing_rate)
                new_synaptic_depression = self.synaptic_depression[:,-1] + time_step*((1-self.synaptic_depression[:,-1])/self.depression_time - self.synaptic_depletion*np.multiply(self.synaptic_depression[:,-1], new_firing_rate))
                new_average_firing_rate = (1 - time_step*self.decay_average)*self.average_firing_rate[:,-1] + time_step*self.decay_average*new_firing_rate
                new_inhibitory_firing_rate = self.inhibitory_firing_rate[:,-1] + (time_step/self.inhibitory_membrane_time)*(-self.inhibitory_firing_rate[:,-1] + activation_function(ex2in_input, self.activation_parameters))
                new_inhibitory_synaptic_depression = self.inhibitory_synaptic_depression[:,-1] + time_step*((1-self.inhibitory_synaptic_depression[:,-1])/self.inhibitory_depression_time - self.synaptic_depletion*np.multiply(self.inhibitory_synaptic_depression[:,-1], new_inhibitory_firing_rate))
                self.firing_rate = np.c_[self.firing_rate, new_firing_rate]
#DEBUG????
                self.neuron_highest_activation = np.c_[self.neuron_highest_activation, new_neuron_highest_activation]
# END DEBUG
                self.synaptic_depression = np.c_[self.synaptic_depression, new_synaptic_depression]
                self.average_firing_rate = np.c_[self.average_firing_rate, new_average_firing_rate]
                self.inhibitory_firing_rate = np.c_[self.inhibitory_firing_rate, new_inhibitory_firing_rate]
                self.inhibitory_synaptic_depression = np.c_[self.inhibitory_synaptic_depression, new_inhibitory_synaptic_depression]
                #
                # Synaptic learning
# DEBUGGING
                demeaned_firing_rate = new_firing_rate - new_average_firing_rate
                #demeaned_firing_rate = new_firing_rate - np.mean(new_firing_rate)
                correlation_matrix = (demeaned_firing_rate)*np.transpose(demeaned_firing_rate)
                #standard_deviation = np.matrix(np.sqrt(np.diag(correlation_matrix)))
                #normalization_matrix = standard_deviation.T*standard_deviation
                #normalized_correlation_matrix = recurrent_scaling*np.divide(correlation_matrix, normalization_matrix) # ADDED RECURRENT SCALING

                #self.connectivity_matrix = (1. - time_step*self.learning_rate)*self.connectivity_matrix + time_step*self.learning_rate*correlation_matrix # decay term
                #self.connectivity_matrix = (1. - time_step*self.learning_rate)*self.connectivity_matrix + time_step*self.learning_rate*normalized_correlation_matrix
                #np.fill_diagonal(self.connectivity_matrix, 0) # remove self connections
# END DEBUGGING
                self.history_connectivity_matrix = self.history_connectivity_matrix + [self.connectivity_matrix]
#
                self.stm_connectvity_matrix = self.stm_connectvity_matrix + time_step*((1/self.stm_time_plus)*(self.stm_maximum - correlation_matrix) - (1/self.stm_time_minus)*(self.stm_connectvity_matrix - self.stm_minimum))
            self.learning_rate = real_learning_rate
            #
            #mean_connectivity_matrix = self.connectivity_matrix.mean()
            #self.connectivity_matrix = self.connectivity_matrix - mean_connectivity_matrix
            #np.fill_diagonal(self.connectivity_matrix, 0) # remove self connections
            #
            self.history_firing_rate = self.history_firing_rate + [self.firing_rate]
            self.history_average_firing_rate = self.history_average_firing_rate + [self.average_firing_rate]
            self.history_synaptic_depression = self.history_synaptic_depression + [self.synaptic_depression]
            self.history_inhibitory_firing_rate = self.history_inhibitory_firing_rate + [self.inhibitory_firing_rate]
            self.history_inhibitory_synaptic_depression = self.history_inhibitory_synaptic_depression + [self.inhibitory_synaptic_depression]
            #self.firing_rate = self.firing_rate[:,-1]
            self.firing_rate = np.matrix(np.zeros((self.number_neurons, 1), dtype=float))
            self.average_firing_rate = self.average_firing_rate[:,-1]
            #self.synaptic_depression = self.synaptic_depression[:,-1]
            self.synaptic_depression = np.matrix(np.ones((self.number_neurons, 1), dtype=float))
            self.inhibitory_firing_rate = np.matrix(np.zeros((self.number_inhibitory_neurons, 1), dtype=float))
            self.inhibitory_synaptic_depression = np.matrix(np.ones((self.number_inhibitory_neurons, 1), dtype=float))
            self.neuron_highest_activation = np.delete(self.neuron_highest_activation, 0, axis=1)
            self.size_connectivity = sum(sum(abs(self.connectivity_matrix)).T)
            return

    def compute_rank_correlation_activations(self, environment_index):
        rank_correlation_activations = scipy.stats.spearmanr(self.neuron_highest_activation.T, self.environments[environment_index].locations_trajectory)
        return rank_correlation_activations

    def update_connectivity(self, scaling, subtract_mean):
        normalized_covariance_matrix = np.matrix(np.corrcoef(np.array(self.history_firing_rate[-1])))
        self.connectivity_matrix = scaling*normalized_covariance_matrix
        if subtract_mean is "yes":
            self.fixed_connectivity_matrix = -self.connectivity_matrix.mean()*np.matrix(np.ones((self.number_neurons, self.number_neurons), dtype=float))
        np.fill_diagonal(self.connectivity_matrix, 0) # remove self connections
        return

    def clear_history(self, attribute_to_clear):
        if attribute_to_clear is "dynamic_variables":
            self.history_firing_rate = []
            self.history_average_firing_rate = []
            self.history_synaptic_depression = []
            self.history_inhibitory_firing_rate = []
            self.history_inhibitory_synaptic_depression = []
        elif attribute_to_clear is "history_connectivity_matrix":
            self.history_connectivity_matrix = []
        else:
            print "No valid attribute to clear has been chosen"
            return
        return

    # (some code duplication with "plot_firing_rate" & "plot_place_fields")
    def plot_connectivity(self, environment_index, only_active_neurons, **kwargs):
        # Sort
        if "connectivity_scaling" in kwargs:
            recurrent_scaling = kwargs["connectivity_scaling"]
        else:
            recurrent_scaling = 1
        if "include_fixed_connectivity" in kwargs and kwargs["include_fixed_connectivity"] is "yes":
            matrix_to_plot = recurrent_scaling*(self.connectivity_matrix+self.fixed_connectivity_matrix)
        else:
            matrix_to_plot = recurrent_scaling*self.connectivity_matrix
        sorted_indices = self.environments[environment_index].sorted_indices_place_fields
        sorted_connectivity_matrix = permute_matrix(matrix_to_plot, sorted_indices)
        if only_active_neurons is "yes":
            active = self.environments[environment_index].active_neurons==1
            active = active[sorted_indices]
            #sorted_indices = sorted_indices[active]
            sorted_connectivity_matrix = sorted_connectivity_matrix[active, :]
            sorted_connectivity_matrix = sorted_connectivity_matrix[:, active]
        # Plot
        plt.imshow(sorted_connectivity_matrix)
        plt.title("Connectivity matrix")
        plt.xlabel("neuron")
        plt.ylabel("neuron")
        plt.colorbar()
        return

    # (some code duplication with "plot_connectivity" & "plot_place_fields")
    def plot_firing_rate(self, environment_index, neuron_type, only_active_neurons, **kwargs):
        if "log_scale" in kwargs and kwargs["log_scale"] is "yes":
            plotting_function = lambda x: np.log(x)
        else:
            plotting_function = lambda x: x
        if "simulation_index" in kwargs:
            simulation_index = kwargs["simulation_index"]
        else:
            simulation_index = -1
        # SOME CODE DUPLICATION WITH "inhibitory"
        if neuron_type is "excitatory":
            # Sort
            sorted_indices = self.environments[environment_index].sorted_indices_place_fields
            sorted_firing_rate = plotting_function(self.history_firing_rate[simulation_index][sorted_indices,:])
            if only_active_neurons is "yes":
                active = self.environments[environment_index].active_neurons==1
                active = active[sorted_indices]
                sorted_firing_rate = sorted_firing_rate[active]
        # SOME CODE DUPLICATION WITH "excitatory"
        elif neuron_type is "inhibitory":
            sorted_firing_rate = plotting_function(self.history_inhibitory_firing_rate[simulation_index])
            if only_active_neurons is "yes":
                active = self.environments[environment_index].active_neurons==1
                sorted_firing_rate = sorted_firing_rate[active]
        else:
            print "Choose the neuron type you want to plot, either 'excitatory' or 'inhibitory'"
        # Plot
        plt.imshow(sorted_firing_rate)
        plt.title("Firing rate")
        plt.xlabel("time")
        plt.ylabel("neuron")
        if "ratio_axes" in kwargs:
            ratio_axes = kwargs["ratio_axes"]
        else:
            ratio_axes = 1
        plt.axes().set_aspect(ratio_axes)
        plt.colorbar()
        return

    # (CODE DUPLICATION)
    def plot_synaptic_depression(self, environment_index, neuron_type, only_active_neurons, **kwargs):
        # SOME CODE DUPLICATION WITH "inhibitory"
        if neuron_type is "excitatory":
            # Sort
            sorted_indices = self.environments[environment_index].sorted_indices_place_fields
            sorted_synaptic_depression = self.history_synaptic_depression[-1][sorted_indices,:]
            if only_active_neurons is "yes":
                active = self.environments[environment_index].active_neurons==1
                active = active[sorted_indices]
                sorted_synaptic_depression = sorted_synaptic_depression[active]
        # SOME CODE DUPLICATION WITH "excitatory"
        elif neuron_type is "inhibitory":
            #sorted_synaptic_depression = self.history_inhibitory_synaptic_depression[-1][sorted_indices,:]
            sorted_synaptic_depression = self.history_inhibitory_synaptic_depression[-1]
            if only_active_neurons is "yes":
                active = self.environments[environment_index].active_neurons==1
                sorted_synaptic_depression = sorted_synaptic_depression[active]
        else:
            print "Choose the neuron type you want to plot, either 'excitatory' or 'inhibitory'"
        # Plot
        plt.imshow(sorted_synaptic_depression)
        plt.title("Synaptic depression")
        plt.xlabel("time")
        plt.ylabel("neuron")
        if "ratio_axes" in kwargs:
            ratio_axes = kwargs["ratio_axes"]
        else:
            ratio_axes = 1
        plt.axes().set_aspect(ratio_axes)
        plt.colorbar()
        return

    # (some code duplication with "plot_connectivity" & "plot_firing_rate")
    def plot_place_fields(self, environment_index, only_active_neurons):
        # Sort
        sorted_indices = self.environments[environment_index].sorted_indices_place_fields
        sorted_place_fields = self.environments[environment_index].place_fields[sorted_indices,:]
        if only_active_neurons is "yes":
            active = self.environments[environment_index].active_neurons==1
            active = active[sorted_indices]
            sorted_place_fields = sorted_place_fields[active]
        # Plot
        plt.imshow(sorted_place_fields)
        plt.title("Place fields")
        plt.xlabel("location")
        plt.ylabel("neuron")
        plt.colorbar()
        return

# NEW CODE (NOT ON GITHUB)

    def get_highest_activations(self, environment_index):
        #self.neuron_highest_activation = np.matrix(np.zeros((1, 1), dtype=float))
        #sorted_indices = self.environments[environment_index].sorted_indices_place_fields
        return

    def decode_trajectory(self, environment_index):
        location_neuron_highest_activation = self.neuron_highest_activation.T*(self.environments[environment_index].number_locations/float(self.number_neurons))
        return location_neuron_highest_activation



    def plot_trajectory_and_decoded_trajectory(self, environment_index):
        location_neuron_highest_activation = self.decode_trajectory(environment_index)
# DEBUG????
        time_axis = np.arange(0, len(self.neuron_highest_activation.T), 1)
# END DEBUG
        # Plot
        plt.plot(time_axis, location_neuron_highest_activation, color='red')
        #plt.show()
        plt.plot(time_axis, self.environments[environment_index].locations_trajectory, color='blue')
        plt.show()
        return



    def compute_autocorrelation_highest_activation(self, environment_index):
        location_neuron_highest_activation = self.decode_trajectory(environment_index)
        self.autocorrelation_highest_activation = signal.correlate(location_neuron_highest_activation, location_neuron_highest_activation)
        self.autocorrelation_highest_activation = self.autocorrelation_highest_activation/max(self.autocorrelation_highest_activation)
        return self.autocorrelation_highest_activation

    # Done?
    def plot_autocorrelation_highest_activation(self):
        #time_axis = np.arange(0, len(self.neuron_highest_activation.T), 1)
        #plt.plot(time_axis, self.autocorrelation_highest_activation)
        plt.plot(self.autocorrelation_highest_activation)
        plt.show()
        return

# CHANGE PLOT FUNCTION TO PLOT EITHER CoM or HIGHEST ACTIVATION

    # plotting_function?
    def get_center_mass(self, environment_index, **kwargs):
        if "simulation_index" in kwargs:
            simulation_index = kwargs["simulation_index"]
        else:
            simulation_index = -1
        sorted_indices = self.environments[environment_index].sorted_indices_place_fields
        sorted_firing_rate = plotting_function(self.history_firing_rate[simulation_index][sorted_indices,:])
        location_number = np.linspace(0, self.environments[environment_index].number_locations, 1)
        center_mass = np.sum(np.multiply(sorted_firing_rate, location_number))/np.sum(sorted_firing_rate)
        return center_mass
