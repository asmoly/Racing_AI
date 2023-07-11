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

    def draw_car(self, car, color, car_size):
        self.main_screen.delete(self.car_sprite)
        
        car_vertices = car.get_vertices(car_size[0], car_size[1])
        self.car_sprite = self.main_screen.create_polygon(car_vertices[0].x, car_vertices[0].y,
                                                          car_vertices[1].x, car_vertices[1].y,
                                                          car_vertices[2].x, car_vertices[2].y,
                                                          car_vertices[3].x, car_vertices[3].y,
                                                          fill=color)

    def update_window(self):
        self.root.update_idletasks()
        self.root.update()