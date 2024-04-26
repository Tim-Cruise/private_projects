import numpy as np
import cv2
from PIL import ImageGrab, Image
import time 
from threading import Thread
from fishing.fishing_agent import FishingAgent

class MainAgent:
    """
    A central controller class that manages the automation of tasks within a simulated environment, specifically for fishing.

    Attributes:
        agents (list): A list to hold agent instances for different tasks.
        fishing_thread (threading.Thread): A thread for running fishing operations in parallel.
        cur_img (numpy.ndarray): The current screenshot from the game as a numpy array.
        cur_imgHSV (numpy.ndarray): The current screenshot converted to HSV color space.
        zone (str): The current game zone, default is "Ashenvale".
        time (str): The current time setting in the game, default is "Day".

    Methods:
        __init__(): Initializes the MainAgent with default settings and empty properties for images and threads.
    """
    def __init__(self) -> None:
        self.agents = []
        self.fishing_thread = None

        self.cur_img = None
        self.cur_imgHSV = None

        self.zone = "Ashenvale"
        self.time = "Day"



def update_screen(agent):
    """
    Captures the screen at regular intervals, processes the images, and updates the current image properties of the agent.

    Parameters:
        agent (MainAgent): The main agent instance that will have its screen updated.

    Runs indefinitely until 'q' is pressed. Handles screen capturing, cropping, conversion, and performance reporting.
    """
    t0 = time.time()
    fps_report_delay = 5
    fps_report_time = time.time()
    crop_coordinates = (300, 300, 1600, 900)
    while True:
        agent.cur_img = ImageGrab.grab()
        agent.cur_img = agent.cur_img.crop(crop_coordinates)
        agent.cur_img = np.array(agent.cur_img)
        agent.cur_img = cv2.cvtColor(agent.cur_img, cv2.COLOR_RGB2BGR)
        agent.cur_imgHSV = cv2.cvtColor(agent.cur_img, cv2.COLOR_BGR2HSV)


        # cv2.imshow("computer vison", agent.cur_img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        ex_time = time.time() - t0
        if time.time() - fps_report_time >= fps_report_delay:
            print("FPS: " + str(1/(ex_time)))
            fps_report_time = time.time()
        t0 = time.time()
        time.sleep(0.005)

def print_menu():
    print("Enter a command: ")
    print("\tS\tStart the main agent. ")
    print("\tZ\tSet zone.")
    print("\tF\tStart the fishing agent.")
    print("\tQ\tQuit the wowzer")


if __name__ == "__main__":
    main_agent = MainAgent()

    print_menu()
    while True:
        user_input = input()
        user_input = str.lower(user_input).strip()

        if user_input == "s":
            update_screen_thred = Thread(target=update_screen, args=(main_agent, ), name="update screen thread", daemon= True)
            update_screen_thred.start()
            print("Thread started")

            
        elif user_input == "z":
            pass

        elif user_input == "f":
            fishing_agent = FishingAgent(main_agent)
            fishing_agent.run()
        
        elif user_input == "q":
            cv2.destroyAllWindows()
            break

        else:
            print("input error")
            print_menu()
    
    print("Done.")

    

 