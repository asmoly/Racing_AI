from tkinter import *

from Vector import Vector
from Car import Car

class Graphics:
    def __init__(self, path_to_track, window_dimensions) -> None:
        self.root = Tk()
        self.main_screen = Canvas(width=window_dimensions[0], height=window_dimensions[1])
        self.main_screen.pack()

        track_image = PhotoImage(file=path_to_track)
        self.root.track_image = track_image
        self.main_screen.create_image(0, 0, image=track_image, anchor=NW)

        self.car_sprite = 0
        self.gates = []
        self.raycasts = []

    def draw_car(self, car, color, car_size):
        self.main_screen.delete(self.car_sprite)
        
        car_vertices = car.get_vertices(car_size[0], car_size[1])
        self.car_sprite = self.main_screen.create_polygon(car_vertices[0].x, car_vertices[0].y,
                                                          car_vertices[1].x, car_vertices[1].y,
                                                          car_vertices[2].x, car_vertices[2].y,
                                                          car_vertices[3].x, car_vertices[3].y,
                                                          fill=color)

    def draw_gates(self, gates):
        self.gates = []
        
        for i in range (0, len(self.gates)):
            self.main_screen.delete(self.gates[i])
        
        for gate in gates:
            color = "red"
            if gate[1] == 1:
                color = "green"

            self.gates.append(self.main_screen.create_line(gate[0][0][0], gate[0][0][1], gate[0][1][0], gate[0][1][1], fill=color, width=3))

    def draw_raycasts(self, raycasts):
        for i in range (0, len(self.raycasts)):
            self.main_screen.delete(self.raycasts[i])
        
        for raycast in raycasts:
            self.raycasts.append(self.main_screen.create_line(raycast[0][0], raycast[0][1], raycast[1][0], raycast[1][1], width=3, fill="blue"))

    def update_window(self):
        self.root.update_idletasks()
        self.root.update()