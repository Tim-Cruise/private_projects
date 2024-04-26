import cv2
import numpy as np
import pyautogui
import time
from random import randint
import matplotlib.pyplot as plt



class FishingAgent():
    """
    A class that automates the fishing process in a game using image recognition and control automation.

    Attributes:
        main_agent: A reference to the main agent or controller that manages the game state and interactions.
        fishing_target (numpy.ndarray): The template image used to identify the fishing lure in the game's screenshot.
        fishing_thread: A threading.Thread object used for running the fishing process asynchronously (if implemented).

    Methods:
        __init__(main_agent): Initializes the FishingAgent class with a reference to the main agent.
        cast_lure(): Simulates the casting of a fishing lure by sending a keyboard command and waits for the lure to settle.
        find_lure(): Searches the current game screenshot for the fishing lure using template matching and initiates lure watching.
        move_to_lure(max_loc): Moves the mouse cursor to the identified lure location with a smooth animation.
        watch_lure(max_loc): Monitors changes in pixel values at the lure's location to detect a bite and handles fishing timeouts.
        pull_line(): Simulates the action of pulling the fishing line by sending mouse and keyboard commands.
        run(): Continuously casts the lure and handles the fishing process in a loop, suitable for autonomous operation.

    """
    def __init__(self, main_agent) -> None:
        self.main_agent = main_agent
        self.fishing_target = cv2.imread("fishing/assets/test_target.png")
        self.fishing_thread = None


    def cast_lure(self):
        """
        Simulates the casting of a fishing lure by pressing a designated keyboard key and then pauses to allow the lure to settle before initiating the lure finding process.
        """
        print("Casting...")
        pyautogui.press("1")
        time.sleep(2)
        self.find_lure()


    def find_lure(self):
        """
        Uses template matching to locate the fishing lure in the current game screenshot stored in the main agent. Upon finding the lure, it proceeds to monitor the lure.
        
        Notes:
            This function will print the progress of matching and only executes if there is a current image loaded in the main agent.
        """
        if self.main_agent.cur_img is not None:
            cur_img = self.main_agent.cur_img
            print("matching images")
            method = eval("cv2.TM_CCOEFF")
            lure_location = cv2.matchTemplate(cur_img, 
                                            self.fishing_target, 
                                            method)
            print("Making array")
            lure_location_array = np.array(lure_location)

            _, _, _, max_loc = cv2.minMaxLoc(lure_location_array)

            # print(max_loc)
            #self.move_to_lure(max_loc)
            self.watch_lure(max_loc)

        # cv2.imshow("match templet", lure_location_array)
        # cv2.waitKey(1)


    def move_to_lure(self, max_loc):
        """
        Moves the mouse cursor smoothly to the location of the fishing lure identified by template matching.
        
        Parameters:
            max_loc (tuple): The (x, y) coordinates of the best match location found by template matching.
        """
        pyautogui.moveTo(max_loc[0], max_loc[1], 0.6, pyautogui.easeOutQuad)
        time.sleep(0.8)
        self.pull_line()


    def watch_lure(self, max_loc):
        """
        Monitors the specified location for changes indicating a fish bite. Breaks from monitoring on specific pixel value conditions or after a timeout period.
        
        Parameters:
            max_loc (tuple): The (x, y) coordinates where the lure has settled, to be monitored for activity.
        """
        watch_time = time.time()
        while True:
            pixel = self.main_agent.cur_imgHSV[max_loc[1], max_loc[0]]
            pixel_2 = pixel
            pixel_check = pixel_2[0] - 1
            pixel_check_2 = pixel_2[0] + 1
            print(f"pixel värde = {pixel}, pixel 2 värde = {pixel_2}")
            print(f"pixelcheck 1 = {pixel_check}, pixel check 2 = {pixel_check_2}")

            if self.main_agent.zone == "Ashenvale" and self.main_agent.time == "Day":
                # Ett av värdena att tuna för att få det att fungera korrekt
                # if pixel[0] <= pixel_check or pixel[0] >= pixel_check_2:
                if pixel[0] >= 108:
                    print("Bite detected!")
                    break

            if time.time() - watch_time >= 30:
                print("Fishing timeout")
                break
        self.move_to_lure(max_loc)
        


    def pull_line(self):
        """
        Simulates the action of pulling the fishing line by executing a right mouse click after a short delay. This method is typically called when a fish bite is detected.
        """
        print("Pull line!")
        # pyautogui.keyDown('shift')
        time.sleep(0.005)
        print("Clicking mouse")
        pyautogui.click(button='right')
        time.sleep(0.010)
        # pyautogui.keyUp('shift')
    

    def run(self):
        """
        Continuously initiates the process of casting and finding the fishing lure, allowing the agent to operate in a loop until manually stopped.
        """
        while True:
            self.cast_lure()  
            time.sleep(5)



