import numpy as np

class Model():
    # class that holds a model
    # this model has only one value for every vertex
    def __init__(self, model_list, model_2d = False):
        self.model = model_list
        if not isinstance(model_list, np.ndarray):
            self.model = np.array(model_list)
        # depth, rows, columns
        self.z_size = len(model_list)
        self.y_size = len(model_list[0])
        if model_2d:
            self.x_size = len(model_list[0][0])
            self.z_size = 1
        else:
            self.x_size = len(model_list[0])

    def check_model_complete(self):
        # a model is complete if every vertex has only one valid label (invalid label = -1)
        for depth_level in self.model:
            for row in depth_level:
                for label in row:
                    if label < 0:
                        return False
        
        return True
