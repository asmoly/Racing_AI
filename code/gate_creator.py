import pickle
from tkinter import *
from PIL import ImageTk as itk
from PIL import Image

IMAGE_PATH = "race_track.png"

mouse_pos = [0, 0]

gate_coords = [[0, 0], [0, 0]]
gates = []

WINDOW_SIZE = Image.open(IMAGE_PATH).size

def motion(event):
    global mouse_pos
    x, y = event.x, event.y
    
    mouse_pos[0] = x
    mouse_pos[1] = y

def mouse_clicked(event):
    global image_screen
    global gate_coords
    global gates

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
    pickle.dump(gates, open(f"{IMAGE_PATH}_Gates", "wb"))
    print("Saved")


root = Tk()

image_screen = Canvas(width=WINDOW_SIZE[0], height=WINDOW_SIZE[1])
image_screen.grid(row=0, column=0, columnspan=2)

track_image = PhotoImage(file=IMAGE_PATH)
root.track_image = track_image
image_screen.create_image(0, 0, image=track_image, anchor=NW)

save_button = Button(root, text="Save", bd=5, command=save)
save_button.grid(row=1, column=1)

root.bind('<Motion>', motion)
root.bind("<Button-1>", mouse_clicked)
root.mainloop()