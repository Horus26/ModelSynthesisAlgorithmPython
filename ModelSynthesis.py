# https://paulmerrell.org/model-synthesis/

import random
from CModel import CModel
from Model import Model
import numpy as np
import copy
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

class ModelSynthesis():

    def __init__(self, example_model : Model, output_model_size : tuple):
        self.example_model = example_model
        # create empty output model
        # output_model_size[1] columns and output_model_size[0] row
        self.base_output_model = Model([[0 for i in range(output_model_size[1])] for j in range(output_model_size[0])])

        self.print_model(self.base_output_model)

        # find max label number given by example model
        self.max_label = np.amax(self.example_model.model)
        # define all valid labels
        self.valid_labels = [i for i in range(self.max_label+1)]

        # create transition function with every transition marked as not allowed (will be updated with values from example model)
        self.transition_x = np.array([[0 for i in range(self.max_label+1)] for j in range(self.max_label+1)])
        self.transition_y = np.array([[0 for i in range(self.max_label+1)] for j in range(self.max_label+1)])

        print("Example model")
        self.print_model(example_model)

        # calculate the transition function from the example model
        self.init_transition_function()

    def run_synthesis(self, b_size = 4):       
        # start with inputting the initial base output model (all zeros)
        input_model = self.base_output_model

        horizontal_step_size = int(b_size / 2)
        if horizontal_step_size + b_size > input_model.x_size:
            horizontal_step_size = 1
        horizontal_steps = int((input_model.x_size - b_size) / horizontal_step_size)

        vertical_step_size = int(b_size / 2)     
        if vertical_step_size + b_size > input_model.y_size:
            vertical_step_size = 1
        vertical_steps = int((input_model.y_size - b_size) / vertical_step_size)

        # Careful: Both vertices counted from 0 (array indexing)
        start_vertex = (input_model.y_size - b_size, 0)
        end_vertex = (start_vertex[0] + b_size - 1, b_size - 1)
        
        working_model : Model = None

        # Synthesize with various B regions to ensure that every vertex is at least updated once
        for j in range(horizontal_steps + 1):  
            start_vertex = (input_model.y_size - b_size, j * horizontal_step_size)
            end_vertex = (start_vertex[0] + b_size - 1, b_size - 1 + j * horizontal_step_size)
            for i in range(vertical_steps + 1):
                # Synthesize the model with B region defined by start vertex and end vertex
                working_model = self.synthesize_with_b(input_model, start_vertex, end_vertex)
                # update start and end vertex for next B region
                start_vertex = (start_vertex[0] - vertical_step_size, start_vertex[1])
                end_vertex = (end_vertex[0] - vertical_step_size, end_vertex[1])  

        if working_model:
            print("SUCCESS")
            self.print_model(working_model)
            self.plot_model(working_model)
        else:
            print("Something went wrong. Check inputs")

    def synthesize_with_b(self, input_model : Model, start_vertex, end_vertex):
        working_model : Model = None
        working_model, b_vertices = self.remove_b(input_model, start_vertex, end_vertex)
        
        if not working_model:
            return None

        # init C(M)
        c_model = CModel(working_model.model, self.valid_labels, (self.transition_x, self.transition_y), b_vertices)
        # C(M) calculation with initial B (where all vertices are removed (=-1) in the B region)
        c_model.update_c_model()

        chosen_label, row, col = None, None, None
        initial_update = True
        # store a copy of C(M) in case C(M) becomes inconsistent (then revert)
        c_model.legacy_c_model = copy.deepcopy(c_model.c_model)

        # condition that B is not the empty set
        while(True):
            # consistency check here because initial case must be also checked
            c_model_consistent = c_model.check_consistent()
            print("C MODEL CONSISTENT? ")
            print(c_model_consistent)
            # condition that c model is not the empty set
            if not c_model_consistent:
                print("MODEL INCONSISTENT --> Restoring legacy model")
                # restore the legacy model
                c_model.c_model = c_model.legacy_c_model
            
            elif not initial_update:
                self.apply_changes(working_model, b_vertices, (row, col), chosen_label)
                # end the loop if all vertices of region B have been used
                if working_model.check_model_complete() or not b_vertices:
                    break
            
            # mark initial case as over
            initial_update = False

            # choose label from vertex of C(M), add this vertex to the working model (M) and update underlying c model accordingly
            row, col = random.choice(b_vertices)
            chosen_label = random.choice(c_model.c_model[row][col])           
            c_model.legacy_c_model = copy.deepcopy(c_model.c_model)
            c_model.c_model[row][col] = [chosen_label]
            # mark vertex as changed --> propagate changes with next C(M) update
            c_model.u_t.append((row,col))    

            # recalculate / update C(M)
            c_model.update_c_model()
        
        return working_model

    def apply_changes(self, working_model, b_vertices, vertex, label):
        working_model.model[vertex] = label
        # remove vertex from B (working model)
        b_vertices.remove(vertex)

    def remove_b(self, model : Model, start_vertex, end_vertex):
        max_row_index = len(self.base_output_model.model) - 1
        max_col_index = len(self.base_output_model.model[0]) - 1
        # check if rows from given vertices are valid
        if start_vertex[0] < 0 or start_vertex[0] > max_row_index or end_vertex[0] < 0 or end_vertex[0] > max_row_index:
            return None, None
        
        # check if columns from given vertices are valid
        if start_vertex[1] < 0 or start_vertex[1] > max_col_index or end_vertex[1] < 0 or end_vertex[1] > max_col_index:
            return None, None

        b_vertices = []
        for i in range(start_vertex[0], end_vertex[0]+1):
            for j in range(start_vertex[1], end_vertex[1]+1):
                b_vertices.append((i,j))

        for i, row in enumerate(model.model):
            for j, col in enumerate(row):
                if start_vertex[0] <= i <= end_vertex[0] and start_vertex[1] <= j <= end_vertex[1]:
                    model.model[i,j] = -1
        # return the model where all vertices in region B are removed (=-1) and all vertices of region B
        return model, b_vertices

    def check_model_consistent(self, model : Model):
        # check if the given model is consistent
        # consistent means that for every vertex the transition function with the given vertex and every surrounding vertex is 1

        for y in range(model.y_size):
            for x in range(model.x_size):
                for y_2 in range(model.y_size):
                    for x_2 in range(model.x_size):
                        if self.check_transition_consistent(model, (y, x), (y_2, x_2)) == 0:
                            print("MODEL INCONSISTENT")
                            return False

        print("Model is consistent")
        return True

    def check_transition_consistent(self, model : Model, v1, v2):
        # check if given vertices (v1, v2) evaluate to true with the transition function
        
        # check if v1 and v2 are adjacent
        k1 = model.model[v1]
        k2 = model.model[v2]
        # if k1 is left and k2 is right
        if np.array_equal(v1, np.add(v2,(1,0))): return self.transition_x[k1, k2]
        # if k2 is left and k1 is right
        if np.array_equal(v1, np.subtract(v2,(1,0))): return self.transition_x[k2, k1]
        # if k1 is top and k2 is bottom
        if  np.array_equal(v1, np.add(v2,(0,1))): return self.transition_y[k1, k2]
        # if k2 is top and k1 is bottom
        if np.array_equal(v1, np.subtract(v2,(0,1))): return self.transition_y[k2, k1]

        return 1

    def init_transition_function(self):
        # update transition function with given example model
        for k1 in range(self.max_label+1):
            for k2 in range(self.max_label+1):
                # check x direction
                if(self.check_transition_in_model_direction(self.example_model, (k1, k2), False)):
                    self.transition_x[k1, k2] = 1
                # check y direction
                if(self.check_transition_in_model_direction(self.example_model, (k1, k2), True)):
                    self.transition_y[k1, k2] = 1

        print("TRANSITIONS X")
        print(self.transition_x)
        print("TRANSITIONS Y")
        print(self.transition_y)
                
    def check_transition_in_model_direction(self, model : Model, transition : tuple, transpose = False):
        # check for transition in direction (left to right or top to bottom)
        local_model = model.model
        if transpose:
            local_model = model.model.transpose()

        k1, k2 = transition
        for i in range(local_model.shape[0]):
            previous_value = -1
            for j in range(local_model.shape[1]):
                if previous_value == k1:
                    if local_model[i, j] == k2:
                        return True
                    else:
                        previous_value = -1
                
                if local_model[i, j] == k1:
                    previous_value = k1
    
        return False

    def plot_model(self, working_model : Model):
        # plot the 2D model with colored rectangles
        color_list = ["black", "green", "blue", "yellow", "violet", "gray", "cyan", "white"]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for row in range(working_model.y_size):
            # row = working_model.y_size - 1 - inverse_row
            for col in range(working_model.x_size):
                rect = Rectangle((col, row + 1), 1, 1, color=color_list[working_model.model[working_model.y_size - 1 - row, col]])
                ax.add_patch(rect)

        plt.xlim([0, working_model.x_size])
        plt.ylim([0, working_model.y_size])
        plt.show()

    def print_model(self, model : Model):
        print("Printing model: ")
        print(model.model)
        print("Finished printing model")

if __name__ == "__main__":
    example_model = [
                        [0, 0, 0, 0],
                        [0, 2, 3, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0]
                    ]
    
    model = Model(example_model)
    model_synthesis_object = ModelSynthesis(model, (8, 5))
    model_synthesis_object.run_synthesis()