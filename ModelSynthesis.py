# https://paulmerrell.org/model-synthesis/

import random
from CModel import CModel
from Model import Model
import numpy as np
import copy
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

class ModelSynthesis():

    def __init__(self, example_model : Model, output_model_size : tuple, grow_from_initial_seed = False, base_mode_ground_layer = False):
        self.example_model = example_model
        self.output_model_size = output_model_size
        self.grow_from_initial_seed = grow_from_initial_seed
        # create empty output model
        # output_model_size[2] columns, output_model_size[1] row, output_model_size[0] depth
        self.base_output_model = Model(np.zeros(output_model_size, int))
        # change bottom rows of base output model to 1
        if base_mode_ground_layer:
            ones_row = np.ones((1, self.base_output_model.x_size), int)
            for depth_level in range(self.base_output_model.z_size):
                self.base_output_model.model[depth_level][-1] = ones_row



        self.print_model(self.base_output_model)

        # find max label number given by example model
        self.max_label = np.amax(self.example_model.model)
        # define all valid labels
        self.valid_labels = [i for i in range(self.max_label+1)]

        # create transition function with every transition marked as not allowed (will be updated with values from example model)
        self.transition_x = np.zeros((self.max_label+1, self.max_label+1), int)
        self.transition_y = np.zeros((self.max_label+1, self.max_label+1), int)
        self.transition_z = np.zeros((self.max_label+1, self.max_label+1), int)
        # self.transition_z = np.array([[[0 for i in range(self.max_label+1)] for j in range(self.max_label+1)] for k in range(self.max_label+1)])

        print("Example model")
        self.print_model(example_model)

        # calculate the transition function from the example model
        self.init_transition_function()
        self.valid_boundary_labels = self.find_boundary_constraints()

    def find_boundary_constraints(self):
        # find constraints for boundary of output model from the given example model
        valid_boundary_labels = {
            "bottom": [], 
            "top":  [],
            "left": [],
            "right":[],
            "front":[],
            "back": [],
            }

        # handles top and bottom rows
        # get all labels from the last/bottom row of every depth layer
        valid_boundary_labels["bottom"] = set.union(*[set(tuple(depth_layer[-1])) for depth_layer in self.example_model.model])
        # get all labels from the first/top row of every depth layer
        valid_boundary_labels["top"] = set.union(*[set(tuple(depth_layer[0])) for depth_layer in self.example_model.model])
        
        # handle left and right columns
        # rotate matrix so that the y-axes is the new x-axes
        working_model = np.rot90(self.example_model.model, axes=(1, 2))
        # get all labels from the first/left column of every depth layer (which is last row after rotation)
        valid_boundary_labels["left"] = set.union(*[set(tuple(depth_layer[-1])) for depth_layer in working_model])
        # get all labels from the last/right column of every depth layer (which is first row after rotation)
        valid_boundary_labels["right"] = set.union(*[set(tuple(depth_layer[0])) for depth_layer in working_model])

        # handle front and back planes
        # get all labels from the front matrix / plane
        valid_boundary_labels["front"] = set(self.example_model.model[0].flat)
        # get all labels from the back matrix / plane
        valid_boundary_labels["back"] = set(self.example_model.model[-1].flat) 
        
        return valid_boundary_labels

    # wrapper method
    def run(self, b_size = 4, zero_padding = False, plot_model = False):
        number_of_attempts = 20
        i = 0
        while(i < number_of_attempts):
            output_model = self.run_synthesis(b_size, zero_padding, plot_model)
            if output_model: 
                return output_model
            i += 1
            print("RESTARTING PROCESS")
        
        print("A consistent model could not be generated in time with given constraints. Try again")
        return None
    
    def run_synthesis(self, b_size = 4, zero_padding = False, plot_model = False):       
        # start with inputting the initial base output model (all zeros)
        input_model = self.base_output_model

        horizontal_step_size = int(b_size / 2)
        if horizontal_step_size + b_size > input_model.x_size:
            horizontal_step_size = 1
        horizontal_steps = int((input_model.x_size - b_size) / horizontal_step_size)
        if horizontal_steps < 0: horizontal_steps = 0

        vertical_step_size = int(b_size / 2)     
        if vertical_step_size + b_size > input_model.y_size:
            vertical_step_size = 1
        vertical_steps = int((input_model.y_size - b_size) / vertical_step_size)
        if vertical_steps < 0: vertical_steps = 0

        depth_step_size = int(b_size / 2)     
        if depth_step_size + b_size > input_model.z_size:
            depth_step_size = 1
        depth_steps = int((input_model.z_size - b_size) / depth_step_size)
        if depth_steps < 0: depth_steps = 0

        # Careful: Both vertices counted from 0 (array indexing)
        start_vertex = (0, input_model.y_size - b_size, 0)
        end_vertex = (0, start_vertex[0] + b_size - 1, b_size - 1)
        
        working_model : Model = None

        # Synthesize with various B regions to ensure that every vertex is at least updated once
        # depth direction
        for k in range(depth_steps + 1):  
            start_vertex = (k * depth_step_size, input_model.y_size - b_size, 0)
            end_vertex = (b_size - 1 + k * depth_step_size, start_vertex[1] + b_size - 1, b_size - 1)
            
            # column direction
            for j in range(horizontal_steps + 1):  
                start_vertex = (start_vertex[0], input_model.y_size - b_size, j * horizontal_step_size)
                end_vertex = (end_vertex[0], start_vertex[1] + b_size - 1, b_size - 1 + j * horizontal_step_size)
                
                used_start_vertex = start_vertex
                used_end_vertex = end_vertex
                # row direction
                for i in range(vertical_steps + 1):
                    if zero_padding:
                        used_start_vertex = list(start_vertex)
                        for index, value in enumerate(used_start_vertex):
                            if value == 0:
                                used_start_vertex[index] = 1
                        used_start_vertex = tuple(used_start_vertex)
                        
                        used_end_vertex = list(end_vertex)
                        for index, value in enumerate(used_end_vertex):
                            if value == self.output_model_size[index] - 1:
                                used_end_vertex[index] = value - 1
                        used_end_vertex = tuple(used_end_vertex)

                    # Synthesize the model with B region defined by start vertex and end vertex
                    working_model = self.synthesize_with_b(input_model, used_start_vertex, used_end_vertex)
                    if not working_model:
                        return None
                    # update start and end vertex for next B region
                    start_vertex = (start_vertex[0], start_vertex[1] - vertical_step_size, start_vertex[2])
                    end_vertex = (end_vertex[0], end_vertex[1] - vertical_step_size, end_vertex[2])  

        if working_model:
            print("SUCCESS")
            self.print_model(working_model)
            if plot_model:
                self.plot_model(working_model)
        else:
            print("Something went wrong. Check inputs")

        return working_model

    def synthesize_with_b(self, input_model : Model, start_vertex, end_vertex):
        working_model : Model = None
        working_model, b_vertices = self.remove_b(input_model, start_vertex, end_vertex)
        
        if not working_model:
            return None

        # init C(M)
        c_model = CModel(working_model.model, self.valid_labels, (self.transition_z, self.transition_x, self.transition_y), b_vertices, self.valid_boundary_labels)
        # C(M) calculation with initial B (where all vertices are removed (=-1) in the B region)
        valid_c_model = c_model.update_c_model()

        chosen_label, depth, row, col = None, None, None, None
        initial_update = True
        # store a copy of C(M) in case C(M) becomes inconsistent (then revert)
        legacy_c_model = copy.deepcopy(c_model)

        # condition that B is not the empty set
        while(True):
            # consistency check here because initial case must be also checked
            c_model_consistent = c_model.check_consistent()
            print("C MODEL CONSISTENT? ")
            print(c_model_consistent)
            # condition that c model is not the empty set
            if not c_model_consistent or not valid_c_model:
                print("MODEL INCONSISTENT --> Restoring legacy model")
                # restore the legacy model
                c_model = legacy_c_model
            
            elif not initial_update:
                self.apply_changes(working_model, b_vertices, (depth, row, col), chosen_label)
                # end the loop if all vertices of region B have been used
                if working_model.check_model_complete() or not b_vertices:
                    break
            
            # mark initial case as over
            initial_update = False

            # choose label from vertex of C(M), add this vertex to the working model (M) and update underlying c model accordingly
            
            depth, row, col = random.choice(b_vertices)
            valid_labels = c_model.c_model[depth][row][col]
            # if there are no valid labels then even restoring with the legacy model does not help --> restart whole process
            if not valid_labels:
                return None

            chosen_label = random.choice(valid_labels)           
            c_model.legacy_c_model = copy.deepcopy(c_model.c_model)
            c_model.c_model[depth][row][col] = [chosen_label]
            # mark vertex as changed --> propagate changes with next C(M) update
            c_model.u_t.append((depth,row,col))    

            # recalculate / update C(M)
            valid_c_model = c_model.update_c_model()
            # if chosen_label != 0 and self.grow_from_initial_seed:
            #     self.grow_from_seed(b_vertices, c_model, (depth, row, col))
        
        print("MODEL CONSISTENT: {}".format(self.check_model_consistent(working_model)))
        return working_model

    def grow_from_seed(self, b_vertices, c_model : CModel, seed_vertex):
        # grow from chosen seed with non zero labels wherever possible in adjacent neighborhood
        (depth, row, col) = seed_vertex
        seed_vertex_label = c_model.c_model[depth][row][col]
        # below
        v2 = tuple(np.add(seed_vertex, (0,1,0)))
        (depth2, row2, col2) = v2
        # check if v2 within boundaries
        if v2[1] < c_model.y_size and v2 in b_vertices:
            possible_labels = c_model.c_model[depth2][row2][col2]

            valid_labels = []
            # check if v2 is last element in column
            if v2[1] == c_model.y_size - 1:
                valid_labels = [label for label in possible_labels if label not in [0, seed_vertex_label]]
            else:
                valid_labels = [label for label in possible_labels if label not in [0]]
            # assign a random label from valid labels
            # check if not empty
            if valid_labels:
                chosen_label = random.choice(valid_labels)
                c_model.c_model[depth2][row2][col2] = [chosen_label]
                c_model.u_t.append(v2)
                # b_vertices.remove(v2)

        

    def apply_changes(self, working_model, b_vertices, vertex, label):
        working_model.model[vertex] = label
        # remove vertex from B (working model)
        b_vertices.remove(vertex)

    def remove_b(self, model : Model, start_vertex, end_vertex):
        max_depth_index = len(self.base_output_model.model) - 1
        max_row_index = len(self.base_output_model.model[0]) - 1
        max_col_index = len(self.base_output_model.model[0][0]) - 1
        
        # check if depth from given vertices are valid
        if start_vertex[0] < 0 or start_vertex[0] > max_depth_index or end_vertex[0] < 0 or end_vertex[0] > max_depth_index:
            return None, None   

        # check if rows from given vertices are valid
        if start_vertex[1] < 0 or start_vertex[1] > max_row_index or end_vertex[1] < 0 or end_vertex[1] > max_row_index:
            return None, None
        
        # check if columns from given vertices are valid
        if start_vertex[2] < 0 or start_vertex[2] > max_col_index or end_vertex[2] < 0 or end_vertex[2] > max_col_index:
            return None, None

        b_vertices = []
        # find all vertices in B region
        for k in range(start_vertex[0], end_vertex[0]+1):
            for i in range(start_vertex[1], end_vertex[1]+1):
                for j in range(start_vertex[2], end_vertex[2]+1):
                    b_vertices.append((k, i, j))

        # set label of all vertices within B to -1
        for k, depth_level in enumerate(model.model):
            for i, row in enumerate(depth_level):
                for j, col in enumerate(row):
                    if start_vertex[0] <= k <= end_vertex[0] and start_vertex[1] <= i <= end_vertex[1] and start_vertex[2] <= j <= end_vertex[2]:
                        model.model[k,i,j] = -1

        # return the model where all vertices in region B are removed (=-1) and all vertices of region B
        return model, b_vertices

    def check_model_consistent(self, model : Model):
        # check if the given model is consistent
        # consistent means that for every vertex the transition function with the given vertex and every surrounding vertex is 1
        for z in range(model.z_size):
            for y in range(model.y_size):
                for x in range(model.x_size):
                    if not self.check_transition_consistent(model, (z, y, x)):
                        print("MODEL INCONSISTENT")
                        return False

        print("Model is consistent")
        return True

    def check_transition_consistent(self, model : Model, v1):
        # check if given vertices (v1, v2) evaluate to true with the transition function
        # check if v1 and v2 are adjacent
        # calculate v2 as adjacent vertex
        # always only check in positive direction as checking in negative direction can be displayed as checking in positive direction

        k1 = model.model[v1]

        # check x direction
        # check with vertex after v1 in x direction
        v2 = tuple(np.add(v1, (0,0,1)))
        if v2[2] < model.x_size:
            k2 = model.model[v2]
            if not self.transition_x[k1, k2]:
                    return False

        # check y direction
        # check with vertex below v1 in y direction
        v2 =  tuple(np.add(v1, (0,1,0)))
        if v2[1] < model.y_size:
            k2 = model.model[v2]
            if not self.transition_y[k1, k2]:
                    return False

        # check z direction
        # check with vertex behind v1 in z direction
        v2 =  tuple(np.add(v1, (1,0,0)))
        if v2[0] < model.z_size:
            k2 = model.model[v2]
            if not self.transition_z[k1, k2]:
                    return False

        return True

    def init_transition_function(self):
        # update transition function with given example model
        for k1 in range(self.max_label+1):
            for k2 in range(self.max_label+1):
                # check x direction
                
                if(self.check_transition_in_model_direction(self.example_model, (k1, k2), 2)):
                    self.transition_x[k1, k2] = 1
                # check y direction
                if(self.check_transition_in_model_direction(self.example_model, (k1, k2), 1)):
                    self.transition_y[k1, k2] = 1
                # check z direction
                if(self.check_transition_in_model_direction(self.example_model, (k1, k2), 0)):
                    self.transition_z[k1, k2] = 1

        print("TRANSITIONS X")
        print(self.transition_x)
        print("TRANSITIONS Y")
        print(self.transition_y)
        print("TRANSITIONS Z")
        print(self.transition_z)
                
    def check_transition_in_model_direction(self, model : Model, transition : tuple, direction):
        # direction can be 0,1,2 --> Z,Y,X direction
    
        # check for transition in direction (left to right or top to bottom or front to back)
        local_model = model.model

        # if z direction then rotate matrix around y axis by 90 degree
        if direction == 0:
            local_model = np.rot90(local_model, 1, (0,2))

        k1, k2 = transition

        for depth_level in range(local_model.shape[0]):
            # if x direction nothing has to change
            table : np.ndarray = local_model[depth_level]
            # if y direction then rotate table by 90 degree --> x axis becomes y axis
            if direction == 1:
                table = np.rot90(table)

            for row_index in range(table.shape[0]):
                previous_value = -1
                for col_index in range(table.shape[1]):
                    if previous_value == k1:
                        if table[row_index, col_index] == k2:
                            return True
                        else:
                            previous_value = -1
                    
                    if table[row_index, col_index] == k1:
                        previous_value = k1
    
        return False

    def plot_model(self, working_model : Model):
        # plot the 2D model with colored rectangles
        color_list = ["black", "green", "blue", "yellow", "violet", "cyan", "gray", "brown"]

        horizontal_plot_number = int(working_model.z_size / 2)
        vertical_plot_number = int(working_model.z_size / horizontal_plot_number) 
        fig, axs = plt.subplots(vertical_plot_number, horizontal_plot_number, squeeze=False)

        vertical_index = 0
        horizontal_index = 0
        for i in range(working_model.z_size):
            for row in range(working_model.y_size):
                # row = working_model.y_size - 1 - inverse_row
                for col in range(working_model.x_size):
                    rect = Rectangle((col, row), 1, 1, color=color_list[working_model.model[i,working_model.y_size - 1 - row, col]])
                    axs[vertical_index, horizontal_index].add_patch(rect)
                    axs[vertical_index, horizontal_index].set_title('Axis [{},{}]'.format(vertical_index, horizontal_index))
                    axs[vertical_index, horizontal_index].set_xlim(0, working_model.x_size)
                    axs[vertical_index, horizontal_index].set_ylim(0, working_model.y_size)

            horizontal_index += 1
            if horizontal_index == horizontal_plot_number:
                horizontal_index = 0
                vertical_index += 1

        plt.show()

    def print_model(self, model : Model):
        print("Printing model: ")
        print(model.model)
        print("Finished printing model")

if __name__ == "__main__":
    # example_model = [
    #                     [
    #                         [0, 0, 0, 0],
    #                         [0, 2, 3, 0],
    #                         [0, 1, 0, 0],
    #                         [0, 0, 0, 0]
    #                     ],
    #                     [
    #                         [0, 2, 3, 0],
    #                         [0, 2, 3, 0],
    #                         [0, 1, 0, 0],
    #                         [0, 0, 0, 0]
    #                     ],
    #                     [
    #                         [1, 2, 3, 0],
    #                         [0, 2, 3, 0],
    #                         [0, 1, 0, 0],
    #                         [0, 0, 0, 0]
    #                     ],
    #                     # [
    #                     #     [1, 2, 3, 1],
    #                     #     [0, 2, 3, 0],
    #                     #     [0, 1, 0, 0],
    #                     #     [1, 0, 0, 1]
    #                     # ],
    #                     [
    #                         [0, 0, 0, 0],
    #                         [0, 2, 3, 0],
    #                         [0, 1, 0, 0],
    #                         [0, 0, 0, 0]
    #                     ]
    #                 ]
    
    # world like model with ground plane and basic structures
    example_model = [
                        [
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [1, 1, 1, 1]
                        ],
                        [
                            [0, 0, 0, 0],
                            [0, 3, 0, 0],
                            [5, 2, 0, 4],
                            [0, 2, 0, 0],
                            [1, 1, 1, 1]
                        ],
                        [
                            [0, 0, 0, 0],
                            [0, 3, 0, 0],
                            [0, 2, 0, 0],
                            [0, 2, 0, 0],
                            [1, 1, 1, 1]
                        ],
                        [
                            [0, 0, 0, 0],
                            [0, 3, 0, 0],
                            [0, 2, 0, 0],
                            [0, 2, 0, 0],
                            [1, 1, 1, 1]
                        ],
                        # [
                        #     [0, 0, 0, 0],
                        #     [1, 2, 3, 1],
                        #     [0, 2, 3, 0],
                        #     [0, 1, 0, 0],
                        #     [1, 0, 0, 1]
                        # ],
                        [
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [1, 1, 1, 1]
                        ]
                    ]

    model = Model(example_model)
    # depth, rows, columns
    model_synthesis_object = ModelSynthesis(model, (4, 8, 5), True, base_mode_ground_layer=True)
    model_synthesis_object.run(4, zero_padding=False, plot_model=True)