import numpy as np
import cv2
import matplotlib.pyplot as plt

class ImageAreaFinder():
    """
    ImageAreaFinder class for identifying and analyzing shapes in an image.

    Parameters:
    - image (str): Path to the input image.

    Methods:
    - display(img, cmap=None): Display the input image with an optional colormap.
    - get_gaussianblur(img): Apply Gaussian blur to the input image.
    - get_threshold(img): Apply thresholding to the input image.
    - get_equalizhist(): Apply histogram equalization to the grayscale image.
    - find_contours(): Find contours in the image and display them.
    - find_edges(img): Find edges in the input image using the Canny edge detector.
    - find_corners(): Find corners in the thresholded image using the Shi-Tomasi corner detector.
    - create_area_list(): Identify shapes and calculate their areas, displaying the results.

    Attributes:
    - image (numpy.ndarray): Original image read from the input file.
    - image_og (numpy.ndarray): Copy of the original image.
    - image_gs (numpy.ndarray): Grayscale version of the original image.
    - gauss_image (numpy.ndarray): Image after applying Gaussian blur.
    - thresholded_image (numpy.ndarray): Image after applying thresholding.
    - image_gs_eq (numpy.ndarray): Image after histogram equalization.
    - contours (list): List of contours found in the image.
    - detected_edges_image (numpy.ndarray): Image with detected edges.
    - draw_contours (numpy.ndarray): Image with contours drawn on it.
    """

    def __init__(self, image) -> None:
        """
        Initialize the ImageAreaFinder instance.

        Parameters:
        - image (str): Path to the input image.
        """
        self.image = cv2.imread(image)
        self.image_og = cv2.imread(image)
        self.image_gs = cv2.imread(image, 0)


    def display(self, img,cmap=None):
        """
        Display the input image.

        Parameters:
        - img (numpy.ndarray): Image to be displayed.
        - cmap (str, optional): Colormap for displaying the image.
        """
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        ax.imshow(img,cmap)


    def get_gaussianblur(self, img):
        """
        Apply Gaussian blur to the input image.

        Parameters:
        - img (numpy.ndarray): Image to be blurred.
        """
        gauss_image = cv2.GaussianBlur(img, (3,3), 0)
        self.gauss_image = gauss_image
        


    def get_threshold(self, img):
        """
        Apply thresholding to the input image.

        Parameters:
        - img (numpy.ndarray): Image to be thresholded.

        Returns:
        - numpy.ndarray: Thresholded image.
        """
        _, thresholded_image = cv2.threshold(img, round(self.image_gs.mean()), 255, cv2.THRESH_BINARY)
        self.thresholded_image = thresholded_image
        return self.thresholded_image


    def get_equalizhist(self):
        """Apply histogram equalization to the grayscale image."""
        self.image_gs_eq = cv2.equalizeHist(self.image_gs)


    def find_contours(self):
        """
        Find contours in the image and display them.

        Returns:
        - tuple: Contours and image with contours drawn on it.
        """
        #self.get_gaussianblur(self.image_gs)
        #self.find_edges(self.image_gs)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(self.image_gs,(6,6), kernel, iterations = 1)
        ret, thres = cv2.threshold(dilated, round(self.image_gs.mean()), 255, 1)
        contours, _ = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        draw_contours = cv2.drawContours(self.image,
                                         contours,
                                         -1,
                                         (0,255,0),
                                         5)
        self.contours = contours
        print("-"*25)
        print(len(self.contours))
        print("-"*25)
        self.draw_contours = draw_contours
        return self.contours, self.draw_contours
    

    def find_edges(self, img):
        """
        Find edges in the input image using the Canny edge detector.

        Parameters:
        - img (numpy.ndarray): Image to find edges in.
        """
        detected_edges_image = cv2.Canny(img, self.image_gs.mean(), 160, L2gradient=True)
        self.detected_edges_image = detected_edges_image
        
    
    def find_corners(self):
        """
        Find corners in the thresholded image using the Shi-Tomasi corner detector.

        Returns:
        - numpy.ndarray: Array of corners.
        """
        image = self.get_threshold(self.image_gs)
        max_corner = 20
        quality_level = 0.01
        min_distance = 20
        corners = cv2.goodFeaturesToTrack(image, 
                                        max_corner,
                                        quality_level,
                                        min_distance)
        corners = np.int0(corners)
        return corners


    def create_area_list(self):
        """Identify shapes and calculate their areas, displaying the results."""
        shapes_and_area = []
        #corners = self.find_corners()
        self.find_contours()

        # for corner in corners:
        #     x, y = corner.ravel()
        #     x = int(x)
        #     y = int(y)
        #     cv2.circle(self.image, (x,y), 3, (255,0,0), -1)
        
        
        for contour in self.contours:
            epsilon = 0.05 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            area = cv2.contourArea(contour)
            num_corners = len(approx)
            shape_name = "unknown"
            
            if num_corners == 2:
                shape_name = "cirkel"
            elif num_corners == 3:
                shape_name = "triangel"
            elif num_corners == 4:
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity > 0.8:
                    shape_name = "cirkel"
                else:
                    shape_name = "fyrkant"
            elif num_corners == 5:
                shape_name = "pentagong"
            elif num_corners == 6:
                shape_name = "hexagong"
            print(shape_name)
            print(num_corners)

            shapes_and_area.append({"Shape": shape_name, "Area": area})
        print("-"*25)
        self.display(self.image)
        plt.show()
        sorted_shapes = sorted(shapes_and_area, key=lambda x: x['Area'], reverse=True)
        filtered_shapes = [shape for shape in sorted_shapes if shape['Shape'] != 'unknown']

        for item in filtered_shapes:
            print(item)
            


test = ImageAreaFinder("shapes.jpg")
test.create_area_list()
