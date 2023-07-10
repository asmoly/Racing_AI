import numpy as np
import cv2
import keyboard
import threading
import pickle
import argparse

from tkinter import *

from Car import *
from AI_Model import *

def collect_data():
    time.sleep(10)
    
    batches = 200
    batchSize = 128

    raycastData = np.empty((batches*batchSize, 13))
    targets = np.empty((batches*batchSize, 3))

    for i in range (0, batches*batchSize):
        raycasts = car.get_raycast_values()
        carState = [acceleration, steeringRight, steeringLeft]

        raycastData[i] = np.array(raycasts)
        targets[i] = np.array(carState)

        time.sleep(0.01)
        print("Collecting Data #", i)

    save_data(raycastData, targets)

def save_data(raycastData, targets):
    print("saving data")
    pickle.dump(raycastData, open("raycast_values", "wb"))
    pickle.dump(targets, open("solutions", "wb"))
    print("data collection complete")

def load_track():
    trackImage = cv2.imread(trackName)
    trackImage = cv2.cvtColor(trackImage, cv2.COLOR_BGR2GRAY)
    trackAsArray = np.asarray(trackImage)
    return trackAsArray

def master_timer():
    global root
    global mainScreen
    global carSprite
    global playerCarSprite
    global device

    global car
    global playerCar
    global racing_AI_model

    global acceleration
    global steeringRight
    global steeringLeft

    global deltaTime
    global raceStarted

    startTime = time()
    mainScreen.delete(carSprite, playerCarSprite)
    
    carVertices = car.update_vertices(deltaTime)
    carSprite = mainScreen.create_polygon(carVertices[0][0], carVertices[0][1], 
                                          carVertices[1][0], carVertices[1][1],
                                          carVertices[2][0], carVertices[2][1],
                                          carVertices[3][0], carVertices[3][1],
                                          fill="red")

    playerCarVertices = playerCar.update_vertices(deltaTime)
    playerCarSprite = mainScreen.create_polygon(playerCarVertices[0][0], playerCarVertices[0][1], 
                                          playerCarVertices[1][0], playerCarVertices[1][1],
                                          playerCarVertices[2][0], playerCarVertices[2][1],
                                          playerCarVertices[3][0], playerCarVertices[3][1],
                                          fill="green")

    if keyboard.is_pressed("w"):
        playerCar.accelerate(deltaTime)
        acceleration = 1
    else:
        playerCar.brake(deltaTime)
        acceleration = 0

    if keyboard.is_pressed("d"):
        playerCar.turn_right(deltaTime)
        steeringRight = 1
        steeringLeft = 0
    elif keyboard.is_pressed("a"):
        playerCar.turn_left(deltaTime)
        steeringRight = 0
        steeringLeft = 1
    else:
        steeringRight = 0
        steeringLeft = 0

    raycastVals = car.get_raycast_values()
    raycastVals = np.array(raycastVals)
    raycastVals = torch.Tensor(raycastVals).to(device)
    
    with torch.no_grad():
        result = racing_AI_model(raycastVals).cpu()
        print(result)
          
        if result[0] > 0.5:
            car.accelerate(deltaTime)
        if result[1] > 0.5:
            car.turn_right(deltaTime)
        if result[2] > 0.5:
            car.turn_left(deltaTime)

    endTime = time()
    deltaTime = (endTime - startTime)

    root.update_idletasks()
    root.update()

    if raceStarted == False:
        sleep(3)
        raceStarted = True

    root.after(0, master_timer)

def test():
    global root
    global mainScreen
    global carSprite
    global playerCarSprite
    global device

    global car
    global playerCar
    global racing_AI_model

    global acceleration
    global steeringRight
    global steeringLeft

    global deltaTime
    global raceStarted

    acceleration = 0
    steeringRight = 0
    steeringLeft = 0

    #threading.Thread(target=collect_data).start()

    deltaTime = 0
    raceStarted = False

    car = Car(load_track(), carStartingPos)
    car.maxSpeed = 120
    playerCar = Car(load_track(), [carStartingPos[0], carStartingPos[1] + 20])
    playerCar.maxSpeed = 120

    racing_AI_model = load_model(device)

    root = Tk()

    mainScreen = Canvas(width=800, height=600)
    mainScreen.pack()

    trackImage = PhotoImage(file=trackName)
    mainScreen.create_image(0, 0, image=trackImage, anchor=NW)
    
    carSprite = 0
    playerCarSprite = 0

    master_timer()
    root.mainloop()  

def main():
    global device
    global trackName
    global carStartingPos

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, help="train or test", default="test")
    parser.add_argument('-t', '--track', type=str, help="1 or 2", default="1")
    args = parser.parse_args()
    
    print("Mode: ", args.mode)
    print("Track: ", args.track)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trackName = ""
    carStartingPos = [0, 0]

    if args.track == "1":
        trackName = "race_track.png"
        carStartingPos = [400, 500]
    elif args.track == "2":
        trackName = "race_track2.png"
        carStartingPos = [400, 420]

    if args.mode == "test":
        test()
    elif args.mode == "train":
        train(device)

if __name__ == "__main__":
    main()