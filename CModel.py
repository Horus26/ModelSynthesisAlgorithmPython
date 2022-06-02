import numpy as np
import random

class CModel():
    # class that holds a C(M)
    # C(M) is the model M but with every possible/allowed label per vertex
    # therefore every vertex has a list of labels
    
    def __init__(self, model_list, valid_label_list, transition_rules, b_vertices, valid_boundary_labels):
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

        # legacy model for resetting if c_model becomes inconsistent
        self.legacy_c_model = None

        # set that contains every vertex v=(row, col) that has been changed but the neighbors have not been updated yet
        self.u_t = None

        # respect already fixed value from the final output model
        self._set_fixed_values_from_model_in_c_model()

        # find initial changed vertices and set them to self.u_t
        self.border_u_t, self.u_t = self._find_initial_changed_vertices(b_vertices)

        # apply boundary constraints if given
        if valid_boundary_labels:
            self._use_boundary_constraints(valid_boundary_labels)

    def _find_initial_changed_vertices(self, b_vertices):
        # find all vertices outside of B (all values in B have -1)
        # and add those border vertices to changed vertices
        # also add a string indicating in which direction the region B is relative to the border vertex
        start_vertex = b_vertices[0]
        end_vertex = b_vertices[-1]
        depth_count, row_count, col_count = self.incomplete_b_model.shape

        changed_vertices = []

        # depth layers
        for depth_level in range(start_vertex[0], end_vertex[0] + 1):
            # top rows
            if start_vertex[1] > 0:
                changed_vertices.extend([((depth_level, start_vertex[1]-1, i), "bottom") for i in range(start_vertex[2], end_vertex[2]+1)])

            # bottom rows
            if end_vertex[1] < row_count-1:
                changed_vertices.extend([((depth_level, end_vertex[1] + 1, i), "top") for i in range(start_vertex[2], end_vertex[2]+1)])

            # left columns
            if start_vertex[2] > 0:
                changed_vertices.extend([((depth_level, i, start_vertex[2] - 1), "right") for i in range(start_vertex[1], end_vertex[1]+1)])

            # right columns
            if end_vertex[2] < col_count-1:
                changed_vertices.extend([((depth_level, i, end_vertex[2] + 1), "left") for i in range(start_vertex[1], end_vertex[1]+1)])

        # add vertices in front and back of b
        if start_vertex[0] > 0:
            for row_index in range(start_vertex[1], end_vertex[1]+1):
                for col_index in range(start_vertex[2], end_vertex[2]+1):
                    changed_vertices.append(((start_vertex[0] - 1, row_index, col_index), "back"))
        if end_vertex[0] < depth_count - 1:
            for row_index in range(start_vertex[1], end_vertex[1]+1):
                for col_index in range(start_vertex[2], end_vertex[2]+1):
                    changed_vertices.append(((end_vertex[0] + 1, row_index, col_index), "front"))


        return changed_vertices, b_vertices
        
    def _set_fixed_values_from_model_in_c_model(self):
        # create C0(M) by finding the changed vertex from the incomplete model and updating C(M)
        for depth_level, depth_layer in enumerate(self.incomplete_b_model):
            for row, row_values in enumerate(depth_layer):
                for col, label in enumerate(row_values):
                    if label != -1:
                        self.c_model[depth_level][row][col] = [label]

        # print(self.c_model)

    # apply boundary constraints to generate c model and therefore constrain possible labels on the boundaries
    def _use_boundary_constraints(self, valid_boundary_labels : set):

        for depth_level, depth_layer in enumerate(self.c_model):

            # change top rows
            for col_count, col_labels in enumerate(depth_layer[0]):
                top_boundary_labels_set = set(col_labels)
                intersection_set = valid_boundary_labels["top"].intersection(top_boundary_labels_set)
                self.c_model[depth_level][0][col_count] = list(intersection_set)

            # change bottom rows
            for col_count, col_labels in enumerate(depth_layer[-1]):
                bottom_boundary_labels_set = set(col_labels)
                intersection_set = valid_boundary_labels["bottom"].intersection(bottom_boundary_labels_set)
                self.c_model[depth_level][-1][col_count] = list(intersection_set)


            # add placeholder elements (list length > 1) so that numpy rotation does not convert the array of lists to numpy arrays ...
            depth_layer[0][0].append(-2)
            depth_layer[0][0].append(-3)
            # rotate depth_layer by 90 degree --> x axis becomes y axis
            table = np.rot90(depth_layer)
            depth_layer[0][0].remove(-2)
            depth_layer[0][0].remove(-3)


            # change left columns
            for col_count, col_labels in enumerate(table[-1]):
                left_boundary_labels_set = set(col_labels)
                intersection_set = valid_boundary_labels["left"].intersection(left_boundary_labels_set)
                table[-1][col_count] = list(intersection_set)

            # change right columns
            for col_count, col_labels in enumerate(table[0]):
                right_boundary_labels_set = set(col_labels)
                intersection_set = valid_boundary_labels["right"].intersection(right_boundary_labels_set)
                table[0][col_count] = list(intersection_set)

            # write updated table / 2d matrix back to the c model
            self.c_model[depth_level] = np.rot90(table, -1)


        # change front matrix / plane
        for row_count, row in enumerate(self.c_model[0]):
            for col_count, col_labels in enumerate(row):
                front_boundary_labels_set = set(col_labels)
                intersection_set = valid_boundary_labels["front"].intersection(front_boundary_labels_set)
                self.c_model[0][row_count][col_count] = list(intersection_set)

        # change back matrix / plane
        for row_count, row in enumerate(self.c_model[-1]):
            for col_count, col_labels in enumerate(row):
                back_boundary_labels_set = set(col_labels)
                intersection_set = valid_boundary_labels["back"].intersection(back_boundary_labels_set)
                self.c_model[-1][row_count][col_count] = list(intersection_set)

    def update_c_model(self):
        # # copy the current c model in case the propagation results in an inconsistent c model
        # self.legacy_c_model = self.c_model.copy()
        while(self.u_t or self.border_u_t):
            if not self.c_model:
                return False
            if not self.propagate_c_model_changes():
                return False
            # print("Remaining u_t: {}".format(self.u_t))
        return True

    def propagate_c_model_changes(self):
        # directions in which to propagate from chosen vertex (differs for border vertices)
        propagate_directions = {
            "top": True,
            "bottom": True,
            "left": True,
            "right": True,
            "front": True,
            "back": True
        }
        
        # select random vertex from u_t
        vertex = None
        # first propagate from border inwards as starting with vertices inside b can lead to inconsistent models fast
        if self.border_u_t:
            vertex, propagate_direction = random.choice(self.border_u_t)
            self.border_u_t.remove((vertex, propagate_direction))
            # changes only need to be propagated from the border vertex in direction of B
            propagate_directions = {
                "top": False,
                "bottom": False,
                "left": False,
                "right": False,
                "front": False,
                "back": False
            }
            propagate_directions[propagate_direction] = True
        else:
            vertex = random.choice(self.u_t)
            # remove selected vertex from u_t
            self.u_t.remove(vertex)
       
       
        depth_level, row, col = vertex

        # print("Propagating changes from vertex: {}".format(vertex))
        # propagate labels from this vertex in the directions: top, left, right, bottom
        chosen_vertex_possible_labels = self.c_model[depth_level][row][col]

        # case vertex to update is to the right
        if propagate_directions["right"] and col < len(self.c_model[0][0]) - 1:
            neighbor_labels_set = set(self.c_model[depth_level][row][col + 1])
            rules_set = {x[1] for x in self.valid_transitions_x if x[0] in chosen_vertex_possible_labels}
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
        if propagate_directions["left"] and col > 0:
            neighbor_labels_set = set(self.c_model[depth_level][row][col - 1])
            rules_set = {x[0] for x in self.valid_transitions_x if x[1] in chosen_vertex_possible_labels}
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
        if propagate_directions["bottom"] and row < len(self.c_model[0]) - 1:
            neighbor_labels_set = set(self.c_model[depth_level][row + 1][col])
            rules_set = {x[1] for x in self.valid_transitions_y if x[0] in chosen_vertex_possible_labels}
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
        if propagate_directions["top"] and row > 0:
            neighbor_labels_set = set(self.c_model[depth_level][row - 1][col])
            rules_set = {x[0] for x in self.valid_transitions_y if x[1] in chosen_vertex_possible_labels}
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
        if propagate_directions["front"] and depth_level > 0:
            neighbor_labels_set = set(self.c_model[depth_level-1][row][col])
            rules_set = {x[0] for x in self.valid_transitions_z if x[1] in chosen_vertex_possible_labels}
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
        if propagate_directions["back"] and depth_level < len(self.c_model) - 1:
            neighbor_labels_set = set(self.c_model[depth_level + 1][row][col])
            rules_set = {x[1] for x in self.valid_transitions_z if x[0] in chosen_vertex_possible_labels}
            # update neighbor by intersecting the neighbors set with rules set
            intersection_set = neighbor_labels_set.intersection(rules_set)
            # update in cmodel if there are changes
            if neighbor_labels_set != intersection_set:            
                self.c_model[depth_level + 1][row][col] = list(intersection_set)
                self.u_t.append((depth_level + 1, row, col))
            # check if neighbor set is empty --> then stop 
            if not intersection_set:
                return False

        # print("CMODEL UPDATED")
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

    def get_from_tuple(self, vertex_tuple : tuple):
        depth, row, col = vertex_tuple
        return self.c_model[depth][row][col]
