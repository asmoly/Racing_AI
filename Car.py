import math
import numpy as np

class Car:
    def __init__(self, track, position) -> None:
        self.position = position
        self.rotation = 0
        self.speed = 0

        self.acceleration = 40
        self.brakingPower = 50
        self.turningSpeed = 120
        self.maxSpeed = 150

        self.vertices = [0, 0, 0, 0]

        self.track = track

    def accelerate(self, deltaTime):
        self.speed += self.acceleration*deltaTime

        if self.speed > self.maxSpeed:
            self.speed = self.maxSpeed
        

    def brake(self, deltaTime):
        self.speed -= self.brakingPower*deltaTime

        if self.speed < 0:
            self.speed = 0

    def turn_left(self, deltaTime):
        self.rotation -= self.turningSpeed*deltaTime

    def turn_right(self, deltaTime):
        self.rotation += self.turningSpeed*deltaTime

    def get_raycast_values(self):
        raycastAngles = [0, 10, 20, 35, 50, 70, 90, -10, -20, -35, -50, -70, -90]
        raycastResult = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            
        for i in range (0, len(raycastAngles)):
            raycastAngles[i] = raycastAngles[i] + self.rotation
            angleinRad = (raycastAngles[i]*math.pi)/180

            hit, counter = False, 0
            while hit == False and counter < 1000:
                if self.track[round(self.position[1] + math.sin(angleinRad)*counter)][round(self.position[0] + math.cos(angleinRad)*counter)] != 255:
                    hit = True
                else:
                    counter += 1

            raycastResult[i] = counter
        
        return raycastResult


    def update_vertices(self, deltaTime):
        carRotationInRad = self.rotation*math.pi/180
        
        self.position[0] += math.cos(carRotationInRad)*self.speed*deltaTime
        self.position[1] += math.sin(carRotationInRad)*self.speed*deltaTime

        self.vertices[0] = [10, -5]
        self.vertices[1] = [10, 5]
        self.vertices[2] = [-10, 5]
        self.vertices[3] = [-10, -5]

        for i in range (0, 4):
            self.vertices[i] = [self.vertices[i][0]*math.cos(carRotationInRad) - self.vertices[i][1]*math.sin(carRotationInRad),
                                self.vertices[i][0]*math.sin(carRotationInRad) + self.vertices[i][1]*math.cos(carRotationInRad)]

        self.vertices[0] = [self.vertices[0][0] + self.position[0], self.vertices[0][1] + self.position[1]]
        self.vertices[1] = [self.vertices[1][0] + self.position[0], self.vertices[1][1] + self.position[1]]
        self.vertices[2] = [self.vertices[2][0] + self.position[0], self.vertices[2][1] + self.position[1]]
        self.vertices[3] = [self.vertices[3][0] + self.position[0], self.vertices[3][1] + self.position[1]]

        return self.vertices