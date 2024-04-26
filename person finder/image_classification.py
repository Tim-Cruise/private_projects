import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

'''
@misc{bochkovskiy2020yolov4,
      title={YOLOv4: Optimal Speed and Accuracy of Object Detection},
      author={Alexey Bochkovskiy and Chien-Yao Wang and Hong-Yuan Mark Liao},
      year={2020},
      eprint={2004.10934},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
'''

class ImageClassification():
    """
    ImageClassification class for performing object classification on images or video.

    This class utilizes the YOLOv4 object detection model to identify and classify
    objects in either a video stream or a static image.

    Attributes:
    - source: str or int
        Video source (file path or camera index).
    - cap: cv2.VideoCapture
        VideoCapture object for handling video input.
    - video_width: int
        Width of the video frames.
    - video_height: int
        Height of the video frames.
    - image: ndarray
        Input image for object classification.
    - image_og: ndarray
        Original copy of the input image.
    - image_gs: ndarray
        Grayscale version of the input image.
    """
    def __init__(self, image:None, video_source:None) -> None:
        """
        Initialize ImageClassification object.

        Parameters:
        - image: str, optional
            File path of the image for object classification.
        - video_source: str, optional
            File path or camera index for video object classification.
        """
        self.source = video_source
        self.cap = cv2.VideoCapture(self.source)
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.image = cv2.imread(image)
        self.image_og = cv2.imread(image)
        self.image_gs = cv2.imread(image, 0)
        

    def display(img,cmap=None):
        """
        Display the given image.

        Parameters:
        - img: ndarray
            Image to be displayed.
        - cmap: str, optional
            Colormap to be used for displaying the image.
        """
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        ax.imshow(img,cmap)

    def create_model(self):
        """
        Create and return the YOLOv4 object detection model.

        Returns:
        - model: cv2.dnn_Net
            YOLOv4 object detection model.
        """
        weights = "yolov4.weights"
        config = "yolov4.cfg"
        model = cv2.dnn.readNetFromDarknet(config, weights)
        return model
    

    def object_classification_video(self):
        """
        Perform object classification on a video source.

        Displays real-time object classification on the video source.
        """
        labels_file = "coco.names"
        model = self.create_model()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            blob = cv2.dnn.blobFromImage(frame,
                                        1/255.0,
                                        (608, 608),
                                        swapRB=True,
                                        crop=False)

            ln = model.getLayerNames()
            ln = [ln[i - 1] for i in model.getUnconnectedOutLayers()]

            model.setInput(blob)

            layer_output = model.forward(ln)

            height, width, color = frame.shape
            output_folder = "detected_objects"
            boxes = []
            confidence_score = []
            classIDs = []
            number = 0
            folder_names = {"Human": [0,1],
                        "Vehicles":[2,9],
                        "Animal":[15,24],
                        "Sport and lifestyle":[25,39],
                        "Kitchen stuff":[40,46],
                        "Food":[47,56],
                        "In house things":[57,80]
                        }
            classes = open(labels_file).read().strip().split("\n")

            for output in layer_output:
                for detected in output:
                    scores = detected[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    if confidence > 0.9:
                        box = detected[0:4] * np.array([width, height, width, height])
                        (center_x, center_y, box_width, box_height) = box.astype("int")
                        x = int(center_x - box_width/2)
                        y = int(center_y - box_height/2)
                        boxes.append([x, y, int(box_width), int(box_height)])
                        confidence_score.append(float(confidence))
                        classIDs.append(classID)
                        object_name = classes[classID]
                        object_index = classID + 1

                        folder_name = None
                        for key, value in folder_names.items():
                            if value[0] <= object_index <= value[1]:
                                folder_name = key
                                break
                        if folder_name is not None:
                            folder_path = os.path.join(output_folder, folder_name)
                            os.makedirs(folder_path, exist_ok=True)
                            number += 1
                            image_filename = f"{object_name}_{number}.jpg"
                            image_path = os.path.join(folder_path, image_filename)
                            try:
                                cv2.imwrite(image_path, frame[y:y+int(box_height), x:x+int(box_width)])
                            except:
                                print(f"Could not save this file: {object_name}")

            score_threshold = 0.9
            non_max_suppression_threshold = 0.9

            found_boxes = cv2.dnn.NMSBoxes(boxes,
                                        confidence_score,
                                        score_threshold,
                                        non_max_suppression_threshold)

            color = (255, 255, 0)
            thickness = 2
            font_scale = 1

            if len(found_boxes) > 0:
                for index in found_boxes.flatten():
                    (x, y) = (boxes[index][0], boxes[index][1])
                    (width, height) = (boxes[index][2], boxes[index][3])
                    cv2.rectangle(frame, (x, y), (x+width, y+height), color, thickness)
                    label = classes[classIDs[index]]
                    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
                    confidence = "{:.2f}".format(confidence_score[index])
                    cv2.putText(frame, confidence, (x + 75, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

            cv2.imshow("Object Classification", frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


    def object_classification_image(self):
        """
        Perform object classification on a static image.

        Displays the object classification results on the static image.
        Returns the processed image.

        Returns:
        - output_image: ndarray
            Processed image with object classification results.
        """
        labels_file = "coco.names"
        model = self.create_model()
        blob = cv2.dnn.blobFromImage(self.image,
                                 1/255.0,
                                 (608,608),
                                 swapRB= True,
                                 crop= False)
        
        ln = model.getLayerNames()
        ln = [ln[i - 1] for i in model.getUnconnectedOutLayers()]

        model.setInput(blob)

        layer_output = model.forward(ln)

        height, width, color = self.image.shape
        output_folder="detected_objects"
        boxes = []
        confidence_score = []
        classIDs = []
        number = 0
        folder_names = {"Human": [0,1],
                        "Vehicles":[2,9],
                        "Animal":[15,24],
                        "Sport and lifestyle":[25,39],
                        "Kitchen stuff":[40,46],
                        "Food":[47,56],
                        "In house things":[57,80]
                        }
        classes = open(labels_file).read().strip().split("\n")
        
        for output in layer_output:
            for detected in output:
                scores = detected[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > 0.9:
                    box = detected[0:4] * np.array([width, height, width, height])
                    (center_x, center_y, box_width, box_height) = box.astype("int")
                    x = int(center_x - box_width/2)
                    y = int(center_y - box_height/2)
                    boxes.append([x,y,int(box_width), int(box_height)])
                    confidence_score.append(float(confidence))
                    classIDs.append(classID)
                    object_name = classes[classID]
                    object_index = classID +1 

                    folder_name = None
                    for key, value in folder_names.items():
                        if value[0] <= object_index <= value[1]:
                            folder_name = key
                            break
                    if folder_name is not None:
                        folder_path = os.path.join(output_folder, folder_name)
                        os.makedirs(folder_path, exist_ok=True)
                        number += 1
                        image_filename = f"{object_name}_{number}.jpg"
                        image_path = os.path.join(folder_path, image_filename)
                        try:
                            cv2.imwrite(image_path, self.image[y:y+int(box_height), x:x+int(box_width)])
                        except:
                            print(f"gick inte att spara denna filen{object_name}")

        score_threshold = 0.9
        non_max_suppression_threshold = 0.9

        found_boxes = cv2.dnn.NMSBoxes(boxes,
                                   confidence_score,
                                   score_threshold,
                                   non_max_suppression_threshold)
        
        color = (255, 255, 0)
        thickness = 2
        font_scale = 1

        if len(found_boxes) > 0:
            for index in found_boxes.flatten():
                (x,y) = (boxes[index][0], boxes[index][1])
                (width, height) = (boxes[index][2], boxes[index][3])
                cv2.rectangle(self.image, (x, y), (x+width, y+height), color, thickness)
                label = classes[classIDs[index]]
                #print(label)
                cv2.putText(self.image, label, (x,y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
                confidence = "{:.2f}".format(confidence_score[index])
                cv2.putText(self.image, confidence, (x + 75, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

        output_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        plt.imshow(output_image)
        plt.show()
        return output_image


if __name__=="__main__":
    test = ImageClassification(image=None, video_source="YogaClass.mp4")
    out_image = test.object_classification_video()