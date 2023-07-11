from math import pi, cos, sin
from Vector import Vector

class Matrix:
    def __init__(self, matrix_array) -> None:
        self.x1 = matrix_array[0][0]
        self.x2 = matrix_array[0][1]
        self.y1 = matrix_array[1][0]
        self.y2 = matrix_array[1][1]

    @staticmethod
    def generate_rotation_matrix(angle):
        angle_in_rad = (angle*pi)/180
        rotation_matrix = Matrix([[cos(angle_in_rad), -sin(angle_in_rad)], [sin(angle_in_rad), cos(angle_in_rad)]])
        
        return rotation_matrix
    
    def __mul__(self, other):
        if type(other) == Vector:
            vector_to_return = Vector(0, 0)
            vector_to_return.x = self.x1*other.x + self.x2*other.y
            vector_to_return.y = self.y1*other.x + self.y2*other.y

            return vector_to_return