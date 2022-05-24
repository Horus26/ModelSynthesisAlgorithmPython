import numpy as np
from Model import Model
import random

class CModel():
    # class that holds a C(M)
    # C(M) is the model M but with every possible/allowed label per vertex
    # therefore every vertex has a list of labels
    
    def __init__(self, model_list, valid_label_list, transition_rules, b_vertices):
        # B model part with neighbors
        self.incomplete_b_model = model_list
        if not isinstance(model_list, np.ndarray):
            self.incomplete_b_model = np.array(model_list)

        self.valid_label_list = valid_label_list

        # cache valid labels for transition in z direction
        valid_transitions_z_indices = np.where(transition_rules[0] == 1)
        self.valid_transitions_z = list(zip(valid_transitions_z_indices[0], valid_transitions_z_indices[1]))

        # cache valid labels for transition in x direction
        valid_transitions_x_indices = np.where(transition_rules[1] == 1)
        self.valid_transitions_x = list(zip(valid_transitions_x_indices[0], valid_transitions_x_indices[1]))

        # cache valid labels for transition in y direction
        valid_transitions_y_indices = np.where(transition_rules[2] == 1)
        self.valid_transitions_y = list(zip(valid_transitions_y_indices[0], valid_transitions_y_indices[1]))

        # initialize C(M) with all possible labels in every vertex
        self.c_model = [[[self.valid_label_list for i in range(self.incomplete_b_model.shape[2])] for j in range(self.incomplete_b_model.shape[1])] for k in range(self.incomplete_b_model.shape[0])]
        self.x_size = len(model_list[0][0])
        self.y_size = len(model_list[0]) 
        self.z_size = len(model_list) 

        self.legacy_c_model = None


        # # initialize with the one random vertex that has been defined in the incomplete Model
        # valid_indices = np.where(self.incomplete_b_model != -1)
        # # set that contains every vertex v=(row, col) that has been changed but the neighbors have not been updated yet
        # self.u_t = list(zip(valid_indices[0], valid_indices[1]))
        self._set_fixed_values_from_model_in_c_model()
        self._find_initial_changed_vertices(b_vertices)

    def _find_initial_changed_vertices(self, b_vertices):
        # find all vertices outside of B (all values in B have -1)
        start_vertex = b_vertices[0]
        end_vertex = b_vertices[-1]
        depth_count, row_count, col_count = self.incomplete_b_model.shape

        changed_vertices = []

        # depth layers
        for depth_level in range(start_vertex[0], end_vertex[0] + 1):
            # top rows
            if start_vertex[1] > 0:
                changed_vertices.extend([(depth_level, start_vertex[1]-1, i) for i in range(start_vertex[2], end_vertex[2]+1)])

            # bottom rows
            if end_vertex[1] < row_count-1:
                changed_vertices.extend([(depth_level, end_vertex[1] + 1, i) for i in range(start_vertex[2], end_vertex[2]+1)])

            # left columns
            if start_vertex[2] > 0:
                changed_vertices.extend([(depth_level, i, start_vertex[2] - 1) for i in range(start_vertex[1], end_vertex[1]+1)])

            # right columns
            if end_vertex[2] < col_count-1:
                changed_vertices.extend([(depth_level, i, end_vertex[2] + 1) for i in range(start_vertex[1], end_vertex[1]+1)])

        # add vertices in front and back of b
        if start_vertex[0] > 0:
            for row_index in range(start_vertex[1], end_vertex[1]+1):
                for col_index in range(start_vertex[2], end_vertex[2]+1):
                    changed_vertices.append((start_vertex[0] - 1, row_index, col_index))
        if end_vertex[0] < depth_count - 1:
            for row_index in range(start_vertex[1], end_vertex[1]+1):
                for col_index in range(start_vertex[2], end_vertex[2]+1):
                    changed_vertices.append((end_vertex[0] + 1, row_index, col_index))


        self.u_t = changed_vertices
        
    def _set_fixed_values_from_model_in_c_model(self):
        # create C0(M) by finding the changed vertex from the incomplete model and updating C(M)
        for depth_level, depth_layer in enumerate(self.incomplete_b_model):
            for row, row_values in enumerate(depth_layer):
                for col, label in enumerate(row_values):
                    if label != -1:
                        self.c_model[depth_level][row][col] = [label]

        # print(self.c_model)

    def update_c_model(self):
        # # copy the current c model in case the propagation results in an inconsistent c model
        # self.legacy_c_model = self.c_model.copy()
        while(self.u_t):
            if not self.c_model:
                return False
            if not self.propagate_c_model_changes():
                return False
            # print("Remaining u_t: {}".format(self.u_t))
        return True


    def propagate_c_model_changes(self):
        # select random vertex from u_t
        vertex = random.choice(self.u_t)
        depth_level, row, col = vertex

        # remove selected vertex from u_t
        self.u_t.remove(vertex)

        # print("Propagating changes from vertex: {}".format(vertex))
        # propagate labels from this vertex in the directions: top, left, right, bottom
        possible_chosen_vertex_labels = self.c_model[depth_level][row][col]

        # case vertex to update is to the right
        if col < len(self.c_model[0][0]) - 1:
            neighbor_labels_set = set(self.c_model[depth_level][row][col + 1])
            rules_set = {x[1] for x in self.valid_transitions_x if x[0] in possible_chosen_vertex_labels}
            # update neighbor by intersecting the neighbors set with rules set
            intersection_set = neighbor_labels_set.intersection(rules_set)
            # update in cmodel if there are changes
            if neighbor_labels_set != intersection_set:            
                self.c_model[depth_level][row][col + 1] = list(intersection_set)
                self.u_t.append((depth_level, row, col+1))

            # check if neighbor set is empty --> then stop 
            if not intersection_set:
                return False

        # case vertex to update is to the left
        if col > 0:
            neighbor_labels_set = set(self.c_model[depth_level][row][col - 1])
            rules_set = {x[0] for x in self.valid_transitions_x if x[1] in possible_chosen_vertex_labels}
            # update neighbor by intersecting the neighbors set with rules set
            intersection_set = neighbor_labels_set.intersection(rules_set)
            # update in cmodel if there are changes
            if neighbor_labels_set != intersection_set:
                self.c_model[depth_level][row][col - 1] = list(intersection_set)
                self.u_t.append((depth_level, row, col-1))
            # check if neighbor set is empty --> then stop 
            if not intersection_set:
                return False

        # case vertex to update is below
        if row < len(self.c_model[0]) - 1:
            neighbor_labels_set = set(self.c_model[depth_level][row + 1][col])
            rules_set = {x[1] for x in self.valid_transitions_y if x[0] in possible_chosen_vertex_labels}
            # update neighbor by intersecting the neighbors set with rules set
            intersection_set = neighbor_labels_set.intersection(rules_set)
            # update in cmodel if there are changes
            if neighbor_labels_set != intersection_set:            
                self.c_model[depth_level][row + 1][col] = list(intersection_set)
                self.u_t.append((depth_level, row + 1, col))
            # check if neighbor set is empty --> then stop 
            if not intersection_set:
                return False
                
        # case vertex to update is above
        if row > 0:
            neighbor_labels_set = set(self.c_model[depth_level][row - 1][col])
            rules_set = {x[0] for x in self.valid_transitions_y if x[1] in possible_chosen_vertex_labels}
            # update neighbor by intersecting the neighbors set with rules set
            intersection_set = neighbor_labels_set.intersection(rules_set)
            # update in cmodel if there are changes
            if neighbor_labels_set != intersection_set:            
                self.c_model[depth_level][row - 1][col] = list(intersection_set)
                self.u_t.append((depth_level, row - 1, col))
            # check if neighbor set is empty --> then stop 
            if not intersection_set:
                return False

        # case vertex to update is in front
        if depth_level > 0:
            neighbor_labels_set = set(self.c_model[depth_level-1][row][col])
            rules_set = {x[0] for x in self.valid_transitions_z if x[1] in possible_chosen_vertex_labels}
            # update neighbor by intersecting the neighbors set with rules set
            intersection_set = neighbor_labels_set.intersection(rules_set)
            # update in cmodel if there are changes
            if neighbor_labels_set != intersection_set:            
                self.c_model[depth_level-1][row][col] = list(intersection_set)
                self.u_t.append((depth_level - 1, row, col))
            # check if neighbor set is empty --> then stop 
            if not intersection_set:
                return False

        # case vertex to update is behind
        if depth_level < len(self.c_model) - 1:
            neighbor_labels_set = set(self.c_model[depth_level + 1][row][col])
            rules_set = {x[1] for x in self.valid_transitions_z if x[0] in possible_chosen_vertex_labels}
            # update neighbor by intersecting the neighbors set with rules set
            intersection_set = neighbor_labels_set.intersection(rules_set)
            # update in cmodel if there are changes
            if neighbor_labels_set != intersection_set:            
                self.c_model[depth_level + 1][row][col] = list(intersection_set)
                self.u_t.append((depth_level + 1, row, col))
            # check if neighbor set is empty --> then stop 
            if not intersection_set:
                return False

        print("CMODEL UPDATED")
        # self.print_model()
        return True

    def check_consistent(self):
        # the c model is inconsistent if there is a vertex with no possible labels
        for depth_layer in self.c_model:
            for row_values in depth_layer:
                for possible_labels in row_values:
                    if not possible_labels:
                        return False

        return True

    def print_model(self):
        np_cmodel = np.array(self.c_model)
        for k, depth_layer in enumerate(np_cmodel):
            for i, row_values in enumerate(depth_layer):
                np_cmodel[k][i] = np.array(row_values)
        print(np_cmodel)
