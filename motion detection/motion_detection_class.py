import cv2
import numpy as np
import matplotlib.pyplot as plt
from  datetime import datetime


class MotionDetect:

    def __init__(self, video_path) -> None:
        """
        Initialize MotionDetect object.

        Parameters:
            video_path (str): The path to the video file for motion detection.
        """
        self.cap = cv2.VideoCapture(video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    def display(img,cmap=None):
        """
        Display an image.

        Parameters:
            img: The image to be displayed.
            cmap: Colormap for displaying the image. Defaults to None.
        """
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        ax.imshow(img,cmap)
        plt.show()


    def get_writer(self):
        """
        Get a video writer for saving motion detected video.
        """
        date_and_time = datetime.now()
        time = date_and_time.strftime("%H:%M:%S")
        time= time.replace(":","_") 
        self.writer = cv2.VideoWriter("motion_detected_"+time+".mp4", 
                                        cv2.VideoWriter_fourcc(*'VIDX'),
                                        int(self.cap.get(cv2.CAP_PROP_FPS)),
                                        (self.width, self.height))
        

    def detect_motion(self):
        """
        Detect motion in the video and save motion detected frames as a separate video.
        """
        number_of_frames = 70
        minimum_area = 100
        color = (255, 255, 0)
        thickness = 2
        previous_contours_length = 0
        pauses_between_movement_count = 0
        maximum_pauses_allowed = 10
        motion = False
        video_recording = False
        frames_in_recording = 0
        try:
            for index in range(number_of_frames):
                ret, frame1 = self.cap.read()
                ret2, frame2 = self.cap.read()

                difference = cv2.absdiff(frame1, frame2)

                gray_image = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

                blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
                _, thresh = cv2.threshold(blur, 20, 255,
                                        cv2.THRESH_BINARY)
                
                dilated = cv2.dilate(thresh, None, iterations = 3)

                contours, _ = cv2.findContours(dilated,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
                
                contours = [contour for contour in contours \
                            if cv2.contourArea(contour)> minimum_area]
                
                for contour in contours:
                    (x, y, width, height) = cv2.boundingRect(contour)
                    cv2.rectangle(frame1, (x, y), (x + width, y + height),
                                color, thickness)
                    motion = True

                if previous_contours_length == 0:
                    previous_contours_length = len(contours)
                    print("No motion")
                    pauses_between_movement_count += 1

                elif len(contours) is not previous_contours_length:
                    print("Motion detected")
                    previous_contours_length = len(contours)
                    pauses_between_movement_count = 0

                else:
                    pauses_between_movement_count += 1 

                if pauses_between_movement_count == maximum_pauses_allowed:
                    print("Stop recording")
                    self.writer.release()
                    pauses_between_movement_count = 0
                    motion = False
                    video_recording = False
                    frames_in_recording = 0

                current_frame = frame1.copy()
                if motion:
                    if not video_recording:
                        video_recording = True
                        self.writer.write(current_frame)
                        print("Started recording")
                        frames_in_recording += 1
                    elif (frames_in_recording > number_of_frames):
                        self.writer.release()
                        print("Stopped recording")
                        frames_in_recording = 0
                        motion = False
                        video_recording = False
                    else:
                        self.writer.write(current_frame)
                        frames_in_recording += 1
                frame1 = frame2.copy()
                _, frame2 = self.cap.read()
            self.cap.release()
            cv2.destroyAllWindows()
        except:
            print("tom ruta")

if __name__ == "__main__":
    motion_detection = MotionDetect("PersonRunning.mp4")
    motion_detection.get_writer()
    motion_detection.detect_motion()