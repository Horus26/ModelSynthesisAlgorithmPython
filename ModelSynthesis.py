# https://paulmerrell.org/model-synthesis/

import random
from CModel import CModel
from Model import Model
import numpy as np
import copy
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

class ModelSynthesis():

    def __init__(self, example_model : Model, output_model_size : tuple, grow_from_initial_seed = False, base_mode_ground_layer_value :int = None, boundary_constraints_location={}, apply_min_dimension_constraints = False):
        self.example_model = example_model
        self.output_model_size = output_model_size
        self.grow_from_initial_seed = grow_from_initial_seed
        # create empty output model
        # output_model_size[2] columns, output_model_size[1] row, output_model_size[0] depth
        self.base_output_model = Model(np.zeros(output_model_size, int))
        # change bottom rows of base output model to 1
        if base_mode_ground_layer_value:
            ground_layer_row = np.full((1, self.base_output_model.x_size), base_mode_ground_layer_value, int)
            # ones_row = np.ones((1, self.base_output_model.x_size), int)
            for depth_level in range(self.base_output_model.z_size):
                self.base_output_model.model[depth_level][-1] = ground_layer_row


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
        self.valid_boundary_labels = None
        if boundary_constraints_location:
            self.valid_boundary_labels = self.find_boundary_constraints(boundary_constraints_location)
        self.min_dimension_constraints = None
        if apply_min_dimension_constraints:
            self.min_dimension_constraints = self.find_min_dimension_constraints()

    def find_boundary_constraints(self, boundary_constraints_location):
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
        
        # reset valid boundary labels per location if needed
        valid_labels_set = set(self.valid_labels)
        if not boundary_constraints_location["top"]: valid_boundary_labels["top"] = valid_labels_set
        if not boundary_constraints_location["bottom"]: valid_boundary_labels["bottom"] = valid_labels_set
        if not boundary_constraints_location["left"]: valid_boundary_labels["left"] = valid_labels_set
        if not boundary_constraints_location["right"]: valid_boundary_labels["right"] = valid_labels_set
        if not boundary_constraints_location["front"]: valid_boundary_labels["front"] = valid_labels_set
        if not boundary_constraints_location["back"]: valid_boundary_labels["back"] = valid_labels_set


        return valid_boundary_labels

    def find_min_dimension_constraints(self):
        # find minimum dimension constraints, e.g: there must always be at least three 2 in a row if there is a 2 in that row
        # could be represented in the example model as: 0 2 2 0 0
        # this example constraint would not be enforced with this model: 0 2 2 0 0\ 0 0 0 0 0\ 0 2 0 0 0
        # find for every label in every direction the minimum amount of following labels with the same value
        min_dimension_size_per_label = []
        working_model = self.example_model.model

        for label_value in range(self.max_label + 1):
            min_dimension_size_per_label.append(
                {
                    "depth": 1, 
                    "width":  1,
                    "height": 1
                }
            )
            # check for dimension constraints for every label
            label_indices = np.argwhere(working_model == label_value)
            # TODO: USE LABEL INDICES (given by depth, row, col indices in separate arrays) TO CHECK FOR CONSTRAINTS
            sorted_indices_for_width = sorted(label_indices, key = lambda x: x[1])
            sorted_indices_for_height = sorted(label_indices, key = lambda x: x[2])
            sorted_indices_for_depth_layers = sorted(sorted_indices_for_width, key = lambda x: x[2])
            # for vertex in label_indices:
            #     pass

            row_counter = 1
            col_counter = 1
            depth_counter = 1
            last_max_count_row = 1
            last_max_count_depth = 1
            last_max_count_height = 1

            for i in range(1, len(label_indices)):
                
                # handle row direction
                previous_vertex = sorted_indices_for_width[i-1]
                current_vertex = sorted_indices_for_width[i]

                # check if adjacent within row
                temp_vertex = np.add(previous_vertex, (0,0,1))
                if np.array_equal(temp_vertex, current_vertex):
                    row_counter += 1
                else:
                    row_counter = 1
                if row_counter > last_max_count_row:
                    last_max_count_row = row_counter

                # handle depth direction
                previous_vertex = sorted_indices_for_depth_layers[i-1]
                current_vertex = sorted_indices_for_depth_layers[i]
                # check if adjacent within depth layers
                temp_vertex = np.add(previous_vertex, (1,0,0))
                if np.array_equal(temp_vertex, current_vertex):
                    depth_counter += 1
                else:
                    depth_counter = 1
                if depth_counter > last_max_count_depth:
                    last_max_count_depth = depth_counter

                # handle height direction
                previous_vertex = sorted_indices_for_height[i-1]
                current_vertex = sorted_indices_for_height[i]
                # check if adjacent within depth layers
                temp_vertex = np.add(previous_vertex, (0,1,0))
                if np.array_equal(temp_vertex, current_vertex):
                    col_counter += 1
                else:
                    col_counter = 1
                if col_counter > last_max_count_height:
                        last_max_count_height = col_counter
                
                min_dimension_size_per_label[-1]["width"] = last_max_count_row
                min_dimension_size_per_label[-1]["depth"] = last_max_count_depth
                min_dimension_size_per_label[-1]["height"] = last_max_count_height

        return min_dimension_size_per_label

    # wrapper method
    def run(self, b_size = 4, plot_model = False):
        # in case of plot model the given output depth must be even
        if plot_model and self.output_model_size[0] % 2:
            print("As plot model was true the output model size in depth direction must be even")
            return None
        
        number_of_attempts = 20
        i = 0
        while(i < number_of_attempts):
            print("Trying Synthesis ({}/{})".format(i, number_of_attempts))
            output_model = self.run_synthesis(b_size, plot_model)
            if output_model: 
                print("A working model was found!")
                return output_model
            i += 1
            print("RESTARTING PROCESS")
        
        print("A consistent model could not be generated in time with given constraints. Try again")
        return None
    
    def run_synthesis(self, b_size = 4, plot_model = False):       
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
        working_model_list = []
        inconsistent_model_count = 0

        # check if given b size is too large for output model with given size
        if any([b_size >= x for x in (input_model.z_size, input_model.y_size, input_model.y_size)]):
            print("The B region size is too large compared to the output model size")
            return None

        # Synthesize with various B regions to ensure that every vertex is at least updated once    
        # do the synthesize procedure with moving B over all of the model more than once
        max_repetitions = 3
        # +1 because the initial position is not respected within the various step sizes
        max_b_movements = max_repetitions * (depth_steps+1) * (horizontal_steps+1) * (vertical_steps+1)
        iteration = 1

        # repetitions of moving B all over the model
        for repetition in range(max_repetitions):

            # depth direction
            for k in range(depth_steps + 1):  
                start_vertex = (k * depth_step_size, input_model.y_size - b_size, 0)
                end_vertex = (b_size - 1 + k * depth_step_size, start_vertex[1] + b_size - 1, b_size - 1)
                
                # column direction
                for j in range(horizontal_steps + 1):  
                    start_vertex = (start_vertex[0], input_model.y_size - b_size, j * horizontal_step_size)
                    end_vertex = (end_vertex[0], start_vertex[1] + b_size - 1, b_size - 1 + j * horizontal_step_size)
                    
                    # row direction
                    for i in range(vertical_steps + 1):
                        used_start_vertex = (start_vertex[0], start_vertex[1] - vertical_step_size * i, start_vertex[2])
                        used_end_vertex = (end_vertex[0], end_vertex[1] - vertical_step_size * i, end_vertex[2])

                        print("B was moved: Iteration ({}/{})".format(iteration, max_b_movements))
                        iteration += 1
                        # Synthesize the model with B region defined by start vertex and end vertex
                        input_model_copy = copy.deepcopy(input_model.model)
                        working_model = self.synthesize_with_b(copy.deepcopy(input_model), used_start_vertex, used_end_vertex)
                        if not working_model or not self.check_model_consistent(working_model):
                            input_model.model = input_model_copy
                            inconsistent_model_count += 1
                            # return None
                        else:
                            input_model = working_model
                            working_model_list.append(working_model)
                            # print("MODEL CONSISTENT: {}".format(self.check_model_consistent(working_model)))

        if working_model_list:
            working_model = working_model_list[-1]
            print("Created a model")
            print("There were {} model created that are valid".format(len(working_model_list)))
            print("And {} models were inconsistent".format(inconsistent_model_count))
            self.print_model(working_model)
            model_consistent = self.check_model_consistent(working_model)
            print("MODEL CONSISTENT: {}".format(model_consistent))
            if not model_consistent:
                return None
            if plot_model:
                self.plot_model(working_model)
        else:
            print("Something went wrong. Check inputs")
            return None

        return working_model

    def synthesize_with_b(self, input_model : Model, start_vertex, end_vertex):
        working_model : Model = None
        working_model, b_vertices = self.remove_b(input_model, start_vertex, end_vertex)
        
        if not working_model:
            return None

        # init C(M)
        transition_rules = (self.transition_z, self.transition_x, self.transition_y)
        c_model = CModel(working_model.model, self.valid_labels, transition_rules, copy.deepcopy(b_vertices), self.valid_boundary_labels)
        # C(M) calculation with initial B (where all vertices are removed (=-1) in the B region)
        valid_c_model = c_model.update_c_model()
        # model is invalid if there are vertices with no label options
        if not valid_c_model:
            return None

        chosen_label, chosen_vertex = None, None
        initial_update = True
        # store a copy of C(M) in case C(M) becomes inconsistent (then revert)
        legacy_c_model = copy.deepcopy(c_model)

        chosen_invalid_vertex_and_labels = []
        possible_vertex_and_label_combinations = self.get_vertex_and_label_combinations(b_vertices, c_model)
        c_model_consistent = c_model.check_consistent()
        # condition that B is not the empty set
        while(True):
            # condition that c model is not the empty set
            if not c_model_consistent or not valid_c_model:
                # print("MODEL INCONSISTENT --> Restoring legacy model")
                # restore the legacy model
                # store the vertex label combination that made the model inconsistent^
                if initial_update:
                    return None

                chosen_invalid_vertex_and_labels.append((chosen_vertex, chosen_label))
                c_model = legacy_c_model
            
            elif not initial_update:
                self.apply_changes(working_model, b_vertices, chosen_vertex, chosen_label)
                possible_vertex_and_label_combinations.clear()
                possible_vertex_and_label_combinations = self.get_vertex_and_label_combinations(b_vertices, c_model)
                chosen_invalid_vertex_and_labels.clear()
                # end the loop if all vertices of region B have been used
                if working_model.check_model_complete() and not b_vertices:
                    break
            
            # mark initial case as over
            initial_update = False

            # choose label from vertex of C(M), add this vertex to the working model (M) and update underlying c model accordingly
            c_model.legacy_c_model = copy.deepcopy(c_model.c_model)

            # prev_vertex = copy.deepcopy((depth,row,col))
            # prev_label = copy.deepcopy(chosen_label)
            # temp = copy.deepcopy(possible_vertex_and_label_combinations)

            # if there are no valid labels then even restoring with the legacy model does not help --> restart whole process
            if possible_vertex_and_label_combinations is None or not possible_vertex_and_label_combinations:
                return None

            # remove combinations that made the model inconsistent
            if chosen_invalid_vertex_and_labels:
                for invalid_combination in chosen_invalid_vertex_and_labels:
                    # all invalid combinations will eventually be removed from possible_vertex_and_label_combinations but not from chosen_invalid_vertex_and_labels until changes are applied
                    if invalid_combination in possible_vertex_and_label_combinations:
                        possible_vertex_and_label_combinations.remove(invalid_combination)

            # the list can be empty after removing invalid combinations
            if not possible_vertex_and_label_combinations:
                return None

            vertex_and_label = random.choice(possible_vertex_and_label_combinations)          
            chosen_vertex, chosen_label = vertex_and_label
            c_model.set_with_tuple(chosen_vertex, [chosen_label])
            # mark vertex as changed --> propagate changes with next C(M) update
            c_model.u_t.append(chosen_vertex)    

            # recalculate / update C(M)
            valid_c_model = c_model.update_c_model()
            c_model_consistent = c_model.check_consistent()
            # print("C MODEL CONSISTENT? ")
            # print(c_model_consistent)
            if valid_c_model and c_model_consistent and self.min_dimension_constraints:
                # apply dimension constraints
                changed_vertices : set = self.apply_dim_constraints(c_model, chosen_vertex, chosen_label)
                if changed_vertices is not None:
                    while(changed_vertices):
                        chosen_changed_vertex = changed_vertices.pop()
                        # (chosen_depth, chosen_row, chosen_col) = changed_vertices.pop()
                        chosen_changed_label = c_model.get_from_tuple(chosen_changed_vertex)[0]
                        new_changed_vertices : set = self.apply_dim_constraints(c_model, chosen_changed_vertex, chosen_changed_label)
                        c_model_copy = copy.deepcopy(c_model.c_model)
                        valid_c_model = c_model.update_c_model()
                        if not valid_c_model:
                            c_model.c_model = c_model_copy
                            c_model.remove_with_tuple(chosen_changed_vertex, chosen_changed_label)
                            # c_model is again valid after resetting
                            valid_c_model = True
                            if not c_model.get_from_tuple(chosen_changed_vertex):
                                # a consistent model cannot be created
                                return None 
                        
                        if new_changed_vertices:
                            changed_vertices.update(new_changed_vertices)
                        
                else:
                    # dimension constraints cant be satisfied --> restart process
                    valid_c_model = False
        
        return working_model

    def get_vertex_and_label_combinations(self, b_vertices, c_model : CModel):
        # find all combinations of given vertices with their respective possible labels (at vertex position in c model)
        possible_combinations = []
        
        for vertex in b_vertices:
            valid_labels = c_model.get_from_tuple(vertex)
            # check if there exist valid labels (if not then vertex has no possible labels --> [])
            if not valid_labels:
                return None
            for label in valid_labels:
                possible_combinations.append((vertex, label))

        return possible_combinations

    def choose_vertex_and_label_random(self, b_vertices, c_model : CModel):
        vertex = random.choice(b_vertices)
        valid_labels = c_model.get_from_tuple(vertex)
        if not valid_labels:
            return None

        chosen_label = random.choice(valid_labels) 
        return vertex, chosen_label

    def find_adjacent_vertices_with_dim_constraints(self, c_model, vertex, label, positive_offset):
        # check for giving vertex if all dim constraints can be fulfilled
        negative_offset = tuple([-value for value in positive_offset])
        # get the counts in positive and negative direction
        count_positive_list, fixed_labels_right = self.check_label_in_direction(c_model, vertex, label, positive_offset)
        count_negative_list, fixed_labels_left = self.check_label_in_direction(c_model, vertex, label, negative_offset)
        # enforce label onto vertices if minimum dimension constraint can only be achieved by enforcing
        fixed_labels_sum = fixed_labels_left + fixed_labels_right  

        return (count_positive_list, count_negative_list), fixed_labels_sum

    def check_dim_constraints_viable(self, c_model, vertex, label):
        dim_constraints_viable = True

        width_constraint = self.min_dimension_constraints[label]["width"]
        height_constraint = self.min_dimension_constraints[label]["height"]
        depth_constraint = self.min_dimension_constraints[label]["depth"]

        # x direction
        (count_positive_list, count_negative_list), fixed_labels_sum = self.find_adjacent_vertices_with_dim_constraints(c_model, vertex, label, (0,0,1))
        neighbor_vertex_sum = len(count_negative_list) + len(count_positive_list)
        if not (width_constraint >= 1 and neighbor_vertex_sum + 1 >= width_constraint):
            dim_constraints_viable = False

        # y direction
        (count_positive_list, count_negative_list), fixed_labels_sum = self.find_adjacent_vertices_with_dim_constraints(c_model, vertex, label, (0,1,0))
        neighbor_vertex_sum = len(count_negative_list) + len(count_positive_list)
        if not (height_constraint >= 1 and neighbor_vertex_sum + 1 >= height_constraint):
            dim_constraints_viable = False

        # z direction
        (count_positive_list, count_negative_list), fixed_labels_sum = self.find_adjacent_vertices_with_dim_constraints(c_model, vertex, label, (1,0,0))
        neighbor_vertex_sum = len(count_negative_list) + len(count_positive_list)
        if not (depth_constraint >= 1 and neighbor_vertex_sum + 1 >= depth_constraint):
            dim_constraints_viable = False

        return dim_constraints_viable

    def apply_dim_constraint_in_direction(self, c_model : CModel, seed_vertex, label, direction_constraint, positive_offset):
        changed_vertices = set()
        
        # if the constraint is 1 then it is already fulfilled by given vertex
        if direction_constraint <= 1:
            return changed_vertices

        (count_positive_list, count_negative_list), fixed_labels_sum = self.find_adjacent_vertices_with_dim_constraints(c_model, seed_vertex, label, positive_offset)
        # enforce label onto vertices if minimum dimension constraint can only be achieved by enforcing
        neighbor_vertex_sum = len(count_negative_list) + len(count_positive_list)
        # fixed_labels_sum = fixed_labels_left + fixed_labels_right
        if neighbor_vertex_sum + 1 >= direction_constraint: 
            # check if there are not enough fixed labels set or the adjacent neighbors are do not have fixed labels (fixed = only one label)
            if fixed_labels_sum + 1 < direction_constraint or ((count_negative_list and len(c_model.get_from_tuple(count_negative_list[0])) > 1) or (count_positive_list and len(c_model.get_from_tuple(count_positive_list[0])) > 1)):
                neighbor_vertex_list = []

                # set as many labels as needed to ensure the constraint is fulfilled
                i = 0
                while i < (direction_constraint-1 - fixed_labels_sum):
                   
                    neighbor_vertex_list.clear()
                    # get adjacent vertices that have labels according to the constraint
                    if count_negative_list: neighbor_vertex_list.append(count_negative_list[0])
                    if count_positive_list: neighbor_vertex_list.append(count_positive_list[0]) 

                    # check if there even exist vertices that can fullfil the constraint
                    if neighbor_vertex_list:
                        chosen_neighbor = random.choice(neighbor_vertex_list)

                        # if the chosen neighbor already contains only one element then it does not count
                        if len(c_model.get_from_tuple(chosen_neighbor)) == 1:
                            # remove from correct side of vertex as the adjacent neighbors have changed or the neighbor vertex led to problems with min dimension constraint
                            if chosen_neighbor in count_negative_list: count_negative_list.remove(chosen_neighbor)
                            else: count_positive_list.remove(chosen_neighbor)
                            continue
                            
                        previous_labels = c_model.get_from_tuple(chosen_neighbor)
                        c_model.set_with_tuple(chosen_neighbor, [label])
                        # check if c_model vertices around the chosen neighbor can fulfill min dimension constraint
                        if self.check_dim_constraints_viable(c_model, chosen_neighbor, label):
                            c_model.u_t.append(chosen_neighbor)
                            changed_vertices.add(chosen_neighbor)
                        else:
                            # reset labels as min dimension constraint can not be fulfilled
                            c_model.set_with_tuple(chosen_neighbor, previous_labels)
                            i -= 1

                        # remove from correct side of vertex as the adjacent neighbors have changed or the neighbor vertex led to problems with min dimension constraint
                        if chosen_neighbor in count_negative_list: count_negative_list.remove(chosen_neighbor)
                        else: count_positive_list.remove(chosen_neighbor)

                        i+=1
                    else:
                        # constraint cant be satisfied --> failure
                        return None    
        else:
                # constraint cant be satisfied --> failure
                return None     

        return changed_vertices         

    def check_min_dim_constraints_0_label(self, c_model : CModel, seed_vertex):
        # if a 0 is the chosen label then check if min dimension constraint is viable for surrounding neighbor vertices
        
        changed_vertices = set()
        range_list = [-1,1]
        c_model_shape = (c_model.z_size, c_model.y_size, c_model.x_size)
        for depth_offset in range_list:
            for row_offset in range_list:
                    for col_offset in range_list:
                        offset = (depth_offset, row_offset, col_offset)
                        neighbor_vertex = tuple(np.add(seed_vertex, offset))
                        # check if neighbor vertex is within c_model
                        if all(first_value < second_value and first_value >= 0 for first_value, second_value in zip(neighbor_vertex, c_model_shape)):
                            neighbor_labels = set(c_model.get_from_tuple(neighbor_vertex))
                            for neighbor_label in neighbor_labels:
                                invalid_labels = []
                                if neighbor_label != 0 and not self.check_dim_constraints_viable(c_model, neighbor_vertex, neighbor_label):
                                    invalid_labels.append(neighbor_label)
                            new_neighbor_labels = set([label for label in neighbor_labels if label not in invalid_labels])
                            # check if there are remaining labels
                            if not new_neighbor_labels:
                                return None
                            # check if the labels have changed
                            if new_neighbor_labels.difference(neighbor_labels):
                                c_model.set_with_tuple(neighbor_vertex, new_neighbor_labels)
                                changed_vertices.add(neighbor_vertex)
                
        return changed_vertices

    def apply_dim_constraints(self, c_model : CModel, seed_vertex, label):
        changed_vertices = set()
        
        # if a 0 was set then check if min dimension constraint viable for surrounding neighbor vertices
        if label == 0:
            changed_vertices = self.check_min_dim_constraints_0_label(c_model, seed_vertex)
            return changed_vertices

        
        width_constraint = self.min_dimension_constraints[label]["width"]
        height_constraint = self.min_dimension_constraints[label]["height"]
        depth_constraint = self.min_dimension_constraints[label]["depth"]

        changed_vertices_width_direction = self.apply_dim_constraint_in_direction(c_model, seed_vertex, label, width_constraint, (0,0,1))
        if changed_vertices_width_direction is None:
            return None
        changed_vertices.update(changed_vertices_width_direction)

        changed_vertices_height_direction = self.apply_dim_constraint_in_direction(c_model, seed_vertex, label, height_constraint, (0,1,0))
        if changed_vertices_height_direction is None:
            return None
        changed_vertices.update(changed_vertices_height_direction)

        changed_vertices_depth_direction = self.apply_dim_constraint_in_direction(c_model, seed_vertex, label, depth_constraint, (1,0,0))
        if changed_vertices_depth_direction is None:
            return None
        changed_vertices.update(changed_vertices_depth_direction)          
                        
        return changed_vertices
  
    def check_label_in_direction(self, c_model : CModel, previous_vertex, previous_label, vertex_offset, count_vertex_list : list = None, fixed_labels = 0):
        # check how many labels can be theoretically set in the model in the given direction from previous vertex (recursive)
        
        # vertex_offset defines the offset to the next vertex to check
        if count_vertex_list is None:
            count_vertex_list = []
            fixed_labels = 0

        vertex = tuple(np.add(previous_vertex, vertex_offset))
        # check if next vertex is within the model
        if any(first_value >= second_value or first_value < 0 for first_value, second_value in zip(vertex, self.output_model_size)):
            return count_vertex_list, fixed_labels
        
        if previous_label in c_model.get_from_tuple(vertex):
            count_vertex_list.append(vertex)
            if len(c_model.get_from_tuple(vertex)) == 1:
                fixed_labels += 1
            return self.check_label_in_direction(c_model, vertex, previous_label, vertex_offset, count_vertex_list, fixed_labels)
        
        return count_vertex_list, fixed_labels

    def grow_from_seed(self, b_vertices, c_model : CModel, seed_vertex):
        # grow from chosen seed with non zero labels wherever possible in adjacent neighborhood
        seed_vertex_label = c_model.get_from_tuple(seed_vertex)
        # below
        v2 = tuple(np.add(seed_vertex, (0,1,0)))
        # check if v2 within boundaries
        if v2[1] < c_model.y_size and v2 in b_vertices:
            possible_labels = c_model.get_from_tuple(v2)

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
                c_model.set_with_tuple(v2, [chosen_label])
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
                        # print("MODEL INCONSISTENT")
                        return False

        # print("Model is consistent")
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
                            [4, 4, 4, 4]
                        ],
                        [
                            [0, 0, 0, 0],
                            [0, 3, 3, 0],
                            [5, 2, 2, 6],
                            [0, 2, 2, 0],
                            [4, 4, 4, 4]
                        ],
                        [
                            [0, 0, 0, 0],
                            [0, 3, 3, 0],
                            [0, 2, 2, 0],
                            [0, 2, 2, 0],
                            [4, 4, 4, 4]
                        ],
                        [
                            [0, 0, 0, 0],
                            [0, 3, 3, 0],
                            [0, 2, 2, 0],
                            [0, 2, 2, 0],
                            [4, 4, 4, 4]
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
                            [4, 4, 4, 4]
                        ]
                    ]

    model = Model(example_model)
    boundary_constraints_location = {
        "top": True,
        "bottom": True,
        "left": True,
        "right": True,
        "front": True,
        "back": True
    }
    # depth, rows, columns
    model_synthesis_object = ModelSynthesis(model, (6, 8, 20), grow_from_initial_seed=False, base_mode_ground_layer_value=4, boundary_constraints_location = boundary_constraints_location, apply_min_dimension_constraints = True)
    model_synthesis_object.run(4, plot_model=True)