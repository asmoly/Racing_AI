from time import time
from keyboard import is_pressed

from Graphics import Graphics
from Car import Car
from Vector import Vector
from Matrix import Matrix

WINDOW_SIZE = (1000, 1000)

CAR_SIZE = (10, 20)
CAR_STARTING_POSITION = (400, 920)

def test(window, car):
    delta_time = 0
    
    while True:
        start_time = time()
        
        acceleration = 0
        steering = 0

        if is_pressed("w"):
            acceleration = 1
        elif is_pressed("s"):
            acceleration = -1

        if is_pressed("a"):
            steering = -1
        if is_pressed("d"):
            steering = 1

        car.update_position(acceleration, steering, delta_time)

        window.draw_car(car, "red", CAR_SIZE)
        window.update_window()

        delta_time = time() - start_time

def train(window, car):
    delta_time = 0
    
    while True:
        start_time = time()
        
        # Do environment stuff here

        acceleration = 0
        steering = 0

        car.update_position(acceleration, steering, delta_time)

        window.draw_car(car, "red", CAR_SIZE)
        window.update_window()

        delta_time = time() - start_time

def main():
    window = Graphics("race_track.png", WINDOW_SIZE)
    car = Car(max_speed=300, acceleration=50, brake_force=120, turning_speed=160, drag=40, starting_position=CAR_STARTING_POSITION, steering_to_speed_ratio=0.5)

    test(window, car)

if __name__== "__main__":
    main()