import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


class ImageFinder:
    """
    A class for finding and extracting frames from a video that match a given mask image.

    Attributes:
        None

    Methods:
        find_video_by_mask(original_video_path, mask_image_path):
            Finds frames in a video that match a given mask image and saves them as individual images.
            
            Args:
                original_video_path (str): The file path to the original video.
                mask_image_path (str): The file path to the mask image.

            Returns:
                None
    """
    def __init__(self) -> None:
        pass


    def  find_video_by_mask(self, original_video_path ,mask_image_path):
        """
        Finds frames in a video that match a given mask image and saves them as individual images.

        Args:
            original_video_path (str): The file path to the original video.
            mask_image_path (str): The file path to the mask image.

        Returns:
            None
        """
        CONFIDENCE_THRESHOLD = 0.80
        MIN_MATCH_COUNT = 70
        cap = cv2.VideoCapture(original_video_path)
        mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(mask_image, None)
        frame_count = 1
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        frame_save = 0

        while True:
            
            ret, frame = cap.read()
            if not ret:
                break
            
            og_frame = frame
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp2, des2 = orb.detectAndCompute(gray_frame, None)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            # Kontrollera antal match
            if len(matches) > MIN_MATCH_COUNT:
                confidence = 1 - matches[0].distance / 100
                if confidence > CONFIDENCE_THRESHOLD:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                    M, _  = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if M is not None:
                        h, w = mask_image.shape
                        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                        dst = cv2.perspectiveTransform(pts, M)
                        frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

                        frame_save += 1
                        
                        if frame_save == 40:
                            x, y, w, h = cv2.boundingRect(np.int32(dst))
                            cropped_frame = og_frame[y:y + h, x:x + w]

                            frame_count += 1
                            frame_filename = os.path.join("target", f"frame_{frame_count}.png")
                            cv2.imwrite(frame_filename, cropped_frame)
                            frame_save = 0

            cv2.imshow("frame", og_frame)
            if cv2.waitKey(1) & 0XFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        

if __name__ == "__main__":
    test = ImageFinder()
    test.find_video_by_mask("img/cars_on_highway.mp4", "img/Car_mask.jpg")
