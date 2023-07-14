import pickle
import math
import numpy as np
from cv2 import imread, cvtColor, COLOR_BGR2GRAY
from time import time

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
    
    def __init__(self, path_to_track, window_dimensions, car, car_size) -> None:
        path_to_track_image = f"{path_to_track}/race_track.png"
        path_to_gates = f"{path_to_track}/race_track_gates"
        path_to_finish_line = f"{path_to_track}/race_track_finish_line"
        
        self.window = Graphics(path_to_track_image, window_dimensions)
        self.car = car
        self.car_size = car_size
        
        self.track_array = imread(path_to_track_image)
        self.track_array = cvtColor(self.track_array, COLOR_BGR2GRAY)

        self.raycasts_to_draw = []

        self.gates = pickle.load(open(path_to_gates, "rb"))
        for i in range (0, len(self.gates)):
            self.gates[i] = [self.gates[i], 0]

        self.finish_line = pickle.load(open(path_to_finish_line, "rb"))

        self.window.draw_gates(self.gates, self.finish_line)

        self.best_lap_time = 100000
        self.lap_start_time = time()
        self.on_finish_line = False

        self.still_counter = 0

        self.state = 0
        self.step(1, len(Environment.actions_to_command) - 1)

    def step(self, delta_time, action):
        controls = Environment.actions_to_command[action]
        acceleration = controls[0]
        steering = controls[1]

        self.car.update_position(acceleration, steering, delta_time)
        
        reward = 0
        done = False

        raycasts, self.raycasts_to_draw = self.get_raycast_values([90, -90, 0, 10, 20, 30, 50, 70, -10, -20, -30, -50, -70])

        # 1 is right direction, 0 is wrong direction
        direction = 0
        if raycasts[0][1] == 1 and raycasts[1][1] == 0:
            direction = 1
            if self.car.speed > 1:
                reward += 1*(self.car.speed/self.car.max_speed)
        else:
            reward -= 1

        if self.car.speed <= 1:
            reward -= 1
            self.still_counter += 1

        if self.still_counter >= 100:
            reward -= 20
            done = True
            self.still_counter = 0

        gate_collision = self.check_for_collision_with_gates()

        if gate_collision == True:
            reward += 5
        elif gate_collision == "finish":
            reward += 20
            done = True
        elif gate_collision == "finish new best":
            reward += 50
            done = True

        wall_collision = self.check_for_collision_with_wall()

        if wall_collision == True:
            reward -= 25
            done = True

        raycasts_as_array = np.array(raycasts)
        # Normalizing raycast length, 1000 is max length
        raycasts_as_array[:, 0] = raycasts_as_array[:, 0]/1000
        raycasts_as_array = raycasts_as_array.reshape((len(raycasts)*2))

        state = np.zeros((raycasts_as_array.shape[0] + 1))
        state[0:raycasts_as_array.shape[0]] = raycasts_as_array
        state[raycasts_as_array.shape[0]] = self.car.speed

        self.state = state

        return state, reward, done

    def check_for_collision_with_gates(self):
        car_vertices = self.car.get_vertices(self.car_size[0], self.car_size[1])
        sides = [[car_vertices[0].to_tuple(), car_vertices[1].to_tuple()], [car_vertices[1].to_tuple(), car_vertices[2].to_tuple()], [car_vertices[2].to_tuple(), car_vertices[3].to_tuple()], [car_vertices[3].to_tuple(), car_vertices[0].to_tuple()]]

        touched_finish_line = False
        for side in sides:
            line_overlap, point_of_overlap = Environment.line_overlap(side, self.finish_line)
            if line_overlap == True:
                touched_finish_line = True

        if touched_finish_line == True:
            hit_all_gates = True
            for gate in self.gates:
                if gate[1] == 0:
                    hit_all_gates = False
            
            if hit_all_gates == True:
                lap_time = time() - self.lap_start_time
                if self.on_finish_line == False:
                    new_best = False
                    if lap_time < self.best_lap_time:
                        self.best_lap_time = lap_time
                        self.window.update_lap_time(self.best_lap_time)
                        new_best = True

                    self.lap_start_time = time()
                    #self.reset_gates()
                    self.on_finish_line = True
                      
                    if new_best == True:
                        return "finish new best"  
                    
                    return "finish"
        else:
            self.on_finish_line = False

        for i in range (0, len(self.gates)):
            if self.gates[i][1] == 0:
                for side in sides:
                    line_overlap, point_of_overlap = Environment.line_overlap(side, (self.gates[i][0][0], self.gates[i][0][1]))
                    if line_overlap == True:
                        if self.gates[i][1] == 0:
                            self.gates[i][1] = 1
                            self.window.update_gate(i, 1, self.gates[i])
                            
                            return True

        return False

    def render(self, draw_raycasts=True):
        if draw_raycasts == True:
            self.window.draw_raycasts(self.raycasts_to_draw)

        self.window.draw_car(self.car, "red", self.car_size)
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
        self.reset_gates()

        self.lap_start_time = time()

        self.step(1, len(Environment.actions_to_command) - 1)

        return self.state

    def reset_gates(self):
        for i in range (0, len(self.gates)):
            self.gates[i][1] = 0
            self.window.update_gate(i, 0, self.gates[i])

    def check_for_collision_with_wall(self):
        car_vertices = self.car.get_vertices(self.car_size[0], self.car_size[1])
        for i in range (0, 4):
            if self.track_array[int(car_vertices[i].y), int(car_vertices[i].x)] != 255:
                self.reset()
                return True
            
        return False

    @staticmethod
    def in_range(a, b, value):
        if value >= min(a, b) and value <= max(a, b):
            return True
        
        return False

    @staticmethod        
    def line_overlap(line_a, line_b):
        line_a_slope = 0
        line_b_slope = 0

        line_a_intercept = 0
        line_b_intercept = 0

        if line_a[1][0] != line_a[0][0]:
            line_a_slope = (line_a[1][1] - line_a[0][1])/(line_a[1][0] - line_a[0][0])
            line_a_intercept = -line_a[0][0]*line_a_slope + line_a[0][1]
        else:
            # Means it is verticle
            line_a_slope = "nan"

        if line_b[1][0] != line_b[0][0]:
            line_b_slope = (line_b[1][1] - line_b[0][1])/(line_b[1][0] - line_b[0][0])
            line_b_intercept = -line_b[0][0]*line_b_slope + line_b[0][1]
        else:
            line_b_slope = "nan"

        point_of_intercept = Vector(0, 0)

        if line_a_slope == line_b_slope:
            return False, point_of_intercept

        if line_a_slope == "nan":
            point_of_intercept.x = line_a[0][0]
            point_of_intercept.y = line_b_slope*point_of_intercept.x + line_b_intercept
        elif line_b_slope == "nan":
            point_of_intercept.x = line_b[0][0]
            point_of_intercept.y = line_a_slope*point_of_intercept.x + line_a_intercept
        else:
            point_of_intercept.x = (line_b_intercept - line_a_intercept)/(line_a_slope - line_b_slope)
            point_of_intercept.y = line_a_slope*point_of_intercept.x + line_a_intercept

        if (Environment.in_range(line_a[0][0], line_a[1][0], point_of_intercept.x) and Environment.in_range(line_a[0][1], line_a[1][1], point_of_intercept.y)) and (Environment.in_range(line_b[0][0], line_b[1][0], point_of_intercept.x) and Environment.in_range(line_b[0][1], line_b[1][1], point_of_intercept.y)):
            return True, point_of_intercept
        
        return False, point_of_intercept