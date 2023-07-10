import numpy as np
import pickle

raycastValues = pickle.load(open("raycast_values", "rb"))
solutions = pickle.load(open("solutions", "rb"))

print(solutions.dtype)