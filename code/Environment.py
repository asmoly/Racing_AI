import pickle
import math
import numpy as np
from cv2 import imread, cvtColor, COLOR_BGR2GRAY

from Graphics import Graphics
from Car import Car
from Matrix import Matrix
from Vector import Vector

class Environment:
    actions_to_command = [[1, 0],    [-1, 0],
                          [1, 1],    [1, -1],
                          [-1, 1],   [-1, -1],
                          [1, 0.5],  [1, -0.5],
                          [-1, 0.5], [-1, -0.5],
                          [0, 1],    [0, -1],
                          [0, 0.5],  [0, -0.5],
                          [0, 0]]
    
    def __init__(self, path_to_gates, path_to_track, window_dimensions, car) -> None:
        self.window = Graphics(path_to_track, window_dimensions)
        self.car = car
        
        self.track_array = imread(path_to_track)
        self.track_array = cvtColor(self.track_array, COLOR_BGR2GRAY)

        self.gates = pickle.load(open(path_to_gates, "rb"))
        for i in range (0, len(self.gates)):
            self.gates[i] = [self.gates[i], 0]

    def step(self, delta_time, action):
        controls = Environment.actions_to_command[action]
        acceleration = controls[0]
        steering = controls[1]

        self.car.update_position(acceleration, steering, delta_time)
        
        raycasts, raycasts_to_draw = self.get_raycast_values([90, -90, 0, 10, 20, 30, 50, 70, -10, -20, -30, -50, -70])
        self.check_for_collision_with_wall()

        self.window.draw_raycasts(raycasts_to_draw)
        self.window.draw_car(self.car, "red", (10, 20))
        #self.window.draw_gates(self.gates)
        self.window.update_window()

    def get_raycast_values(self, raycast_angles):
        raycast_results = []
        raycasts_to_draw = []

        for angle in raycast_angles:
            angle += self.car.rotation
            angle_in_rad = (angle*math.pi)/180

            pixel_check_pos = [0, 0]

            hit = False
            counter = 0
            while hit == False and counter < 1000:
                pixel_check_pos = [int(self.car.position.x + math.cos(angle_in_rad)*counter), int(self.car.position.y + math.sin(angle_in_rad)*counter)]

                if self.track_array[pixel_check_pos[1], pixel_check_pos[0]] != 255:
                    hit = True

                counter += 1

            side = 0 # 0: inside, 1: outside
            if self.track_array[pixel_check_pos[1], pixel_check_pos[0]] == 0:
                side = 1

            raycasts_to_draw.append([[self.car.position.x, self.car.position.y], pixel_check_pos, side])
            
            length_of_raycast = math.sqrt((pixel_check_pos[0] - self.car.position.x)*(pixel_check_pos[0] - self.car.position.x) + (pixel_check_pos[1] - self.car.position.y)*(pixel_check_pos[1] - self.car.position.y)) 
            raycast_results.append([length_of_raycast, side])

        return raycast_results, raycasts_to_draw
    
    def reset(self):
        self.car.reset_position()

    def check_for_collision_with_wall(self):
        car_vertices = self.car.get_vertices(20, 10)
        for i in range (0, 4):
            if self.track_array[int(car_vertices[i].y), int(car_vertices[i].x)] != 255:
                self.reset()
                return True