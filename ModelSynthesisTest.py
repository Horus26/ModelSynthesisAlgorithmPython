import unittest
from CModel import CModel
from ModelSynthesis import ModelSynthesis
from Model import Model
import numpy as np

class ModelSynthesisTest(unittest.TestCase):
    def setUp(self):
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
                    [
                        [0, 0, 0, 0],
                        [0, 3, 3, 0],
                        [0, 2, 2, 0],
                        [0, 2, 2, 0],
                        [4, 4, 4, 4]
                    ],
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [4, 4, 4, 4]
                    ]
                ]

        self.model = Model(example_model)
        self.boundary_constraints_location = {
            "top": False,
            "bottom": False,
            "left": False,
            "right": False,
            "front": False,
            "back": False
        }

        # depth, rows, columns
        self.output_model_size = (6, 8, 5)
        self.grow_from_initial_seed = False
        self.base_mode_ground_layer_value = None
        self.apply_min_dimension_constraints = False
        self.b_size = 3
        self.zero_padding = False
        self.plot_model = False
        self.model_synthesis_object_default = ModelSynthesis(self.model, self.output_model_size)
         
    
    def test_output_model_size_matches_given_output_model_size(self):
        model_synthesis_object = ModelSynthesis(self.model, self.output_model_size, grow_from_initial_seed=self.grow_from_initial_seed, base_mode_ground_layer_value=self.base_mode_ground_layer_value, boundary_constraints_location = self.boundary_constraints_location, apply_min_dimension_constraints = self.apply_min_dimension_constraints)
        output_model : Model= model_synthesis_object.run(b_size=self.b_size)
        self.assertEqual(output_model.model.shape, self.output_model_size,
                                'incorrect output model size')

    def test_find_boundary_constraints(self):
        example_model = [
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [3, 2, 0, 5],
                        [0, 1, 0, 0],
                        [4, 4, 4, 4]
                    ],
                    [
                        [0, 0, 0, 0],
                        [0, 3, 3, 0],
                        [0, 2, 2, 0],
                        [0, 2, 2, 0],
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
        model_synthesis_object = ModelSynthesis(model, self.output_model_size)
        boundary_constraints = model_synthesis_object.find_boundary_constraints(boundary_constraints_location)
        
        boundary_constraints_compare = {
            "top": {0},
            "bottom": {4},
            "left": {0,3,4},
            "right": {0,4,5},
            "front": {0,1,2,3,4,5},
            "back": {0,2,3,4}
        }
        for key in boundary_constraints_compare:
            self.assertEqual(boundary_constraints_compare[key], boundary_constraints[key])
    
    def test_transition_function(self):
        simple_example_model = [
                        [
                            [0, 0, 0, 0],
                            [0, 2, 3, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 0]
                        ],
                        [
                            [0, 0, 0, 0],
                            [0, 5, 6, 0],
                            [0, 4, 0, 0],
                            [0, 0, 0, 0]
                        ],
                        [
                            [0, 0, 0, 0],
                            [0, 2, 3, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 0]
                        ]
                    ]
        simple_model = Model(simple_example_model)
        simple_output_model = ModelSynthesis(simple_model, self.output_model_size)
        transition_x = np.array([
            [1, 1, 1, 0, 1, 1, 0],
            [1, 0, 0 ,0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0]
        ])
        transition_y = np.array([
            [1, 0, 1, 1, 0, 1, 1],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 0]
        ])
        transition_z = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        np.testing.assert_array_equal(simple_output_model.transition_x, transition_x)
        np.testing.assert_array_equal(simple_output_model.transition_y, transition_y)
        np.testing.assert_array_equal(simple_output_model.transition_z, transition_z)

    def test_find_min_dimension_constraints(self):
        example_model = [
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [3, 2, 0, 5],
                        [0, 1, 0, 0],
                        [4, 4, 4, 4]
                    ],
                    [
                        [0, 0, 0, 0],
                        [0, 3, 3, 0],
                        [0, 2, 2, 0],
                        [0, 2, 2, 0],
                        [4, 4, 4, 4]
                    ]
                ]
        model = Model(example_model)

        dimension_constraints = []
        for i in range(6):
            dimension_constraints.append(
                {
                    "depth": 1, 
                    "width":  1,
                    "height": 1
                }
            )
        dimension_constraints[0]["depth"] = 2
        dimension_constraints[0]["width"] = 4
        dimension_constraints[0]["height"] = 4

        dimension_constraints[1]["depth"] = 1
        dimension_constraints[1]["width"] = 1
        dimension_constraints[1]["height"] = 1

        dimension_constraints[2]["depth"] = 2
        dimension_constraints[2]["width"] = 2
        dimension_constraints[2]["height"] = 2

        dimension_constraints[3]["depth"] = 1
        dimension_constraints[3]["width"] = 2
        dimension_constraints[3]["height"] = 1

        dimension_constraints[4]["depth"] = 2
        dimension_constraints[4]["width"] = 4
        dimension_constraints[4]["height"] = 1

        dimension_constraints[5]["depth"] = 1
        dimension_constraints[5]["width"] = 1
        dimension_constraints[5]["height"] = 1     
        
        model_synthesis_object = ModelSynthesis(model, self.output_model_size)
        min_dimension_constraints = model_synthesis_object.find_min_dimension_constraints()
        
        for i, dimension in enumerate(dimension_constraints):
            self.assertEqual(dimension["width"], min_dimension_constraints[i]["width"])
            self.assertEqual(dimension["height"], min_dimension_constraints[i]["height"])
            self.assertEqual(dimension["depth"], min_dimension_constraints[i]["depth"])

    # def test_check_min_dim_constraints_0_label(self):
    #     example_model = [
    #         [[0, 0, 0],
    #         [0, 0, 0]],
            
    #         [[0, 1, 0],
    #         [0, 1, 0]],
            
    #         [[0, 1, 0],
    #         [0, 1, 0]],

    #         [[0, 0, 0],
    #         [0, 0, 0]],
    #     ]

    #     model = Model(example_model)
    #     model_synthesis_object = ModelSynthesis(model, (6,1, 3))
    #     transition_rules = (model_synthesis_object.transition_z, model_synthesis_object.transition_x, model_synthesis_object.transition_y)
    #     c_model = CModel(model.model, [0,1], transition_rules, set_fixed_values=False)
    #     # TODO: FINISH METHOD


if __name__ == '__main__':
    unittest.main()