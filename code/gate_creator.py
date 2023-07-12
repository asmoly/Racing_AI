import pickle
from tkinter import *
from PIL import ImageTk as itk
from PIL import Image

TRACK_PATH = "race_track_1"
WINDOW_SIZE = Image.open(f"{TRACK_PATH}/race_track.png").size

mouse_pos = [0, 0]

gate_coords = [[0, 0], [0, 0]]
gates = []

adding_finish_line = False
finish_line_coords = [[0, 0], [0, 0]]

def motion(event):
    global mouse_pos
    x, y = event.x, event.y
    
    mouse_pos[0] = x
    mouse_pos[1] = y

def add_finish_line():
    print("adding finish line")

    global finish_line_coords
    global adding_finish_line

    adding_finish_line = True
    finish_line_coords = [[0, 0], [0, 0]]

def mouse_clicked(event):
    global image_screen
    global gate_coords
    global gates
    global adding_finish_line
    global finish_line_coords

    if mouse_pos[0] > 0 and mouse_pos[0] < WINDOW_SIZE[0] and mouse_pos[1] > 0 and mouse_pos[1] < WINDOW_SIZE[1]:
        if adding_finish_line == True:
            if finish_line_coords[0] == [0, 0]:
                finish_line_coords[0][0] = mouse_pos[0]
                finish_line_coords[0][1] = mouse_pos[1]
            else:
                finish_line_coords[1][0] = mouse_pos[0]
                finish_line_coords[1][1] = mouse_pos[1]
                
                image_screen.create_line(finish_line_coords[0][0], finish_line_coords[0][1], finish_line_coords[1][0], finish_line_coords[1][1], fill="blue", width=7)
                adding_finish_line = False
                gate_coords = [[0, 0], [0, 0]]
        else:
            if gate_coords[0] == [0, 0]:
                gate_coords[0][0] = mouse_pos[0]
                gate_coords[0][1] = mouse_pos[1]
            else:
                gate_coords[1][0] = mouse_pos[0]
                gate_coords[1][1] = mouse_pos[1]
                
                image_screen.create_line(gate_coords[0][0], gate_coords[0][1], gate_coords[1][0], gate_coords[1][1], fill="green", width=5)
                gates.append(gate_coords)

                gate_coords = [[0, 0], [0, 0]]

def save():
    global gates
    global finish_line_coords

    pickle.dump(gates, open(f"{TRACK_PATH}/race_track_gates", "wb"))
    pickle.dump(finish_line_coords, open(f"{TRACK_PATH}/race_track_finish_line", "wb"))
    print("Saved")


root = Tk()

image_screen = Canvas(width=WINDOW_SIZE[0], height=WINDOW_SIZE[1])
image_screen.grid(row=0, column=0, columnspan=2)

track_image = PhotoImage(file=f"{TRACK_PATH}/race_track.png")
root.track_image = track_image
image_screen.create_image(0, 0, image=track_image, anchor=NW)

save_button = Button(root, text="Save", bd=5, command=save)
save_button.grid(row=1, column=1)

finish_line_button = Button(root, text="Add Finish Line", bd=5, command=add_finish_line)
finish_line_button.grid(row=1, column=0)

root.bind('<Motion>', motion)
root.bind("<Button-1>", mouse_clicked)
root.mainloop()