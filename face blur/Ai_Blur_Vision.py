import cv2
from mtcnn import MTCNN

class FaceDetector:
    def __init__(self):
        """
        Initializes the FaceDetector object with an MTCNN detector.
        """
        self.detector = MTCNN()

    def detect_faces(self, image_path):
        """
        Detects faces in the provided image and applies blur effect on the detected faces.
        
        Args:
            image_path (str): Path to the image file.
        
        Returns:
            numpy.ndarray: Image array with detected faces and blur effect applied.
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.detect_faces(image)
        for result in results:
            bounding_box = result['box']
            confidence = result['confidence']
            x, y, w, h = bounding_box

            # Beräkna ny större bounding box
            padding = 20  # Storlek på marginalen runt ansiktet
            x_new = max(x - padding, 0)  # Se till att nya koordinaten inte går utanför bildens gränser
            y_new = max(y - padding, 0)
            w_new = w + 2 * padding
            h_new = h + 2 * padding

            # Applicera Gaussian Blur på den utvidgade ansiktsregionen
            face_region = image[y_new:y_new+h_new, x_new:x_new+w_new]
            blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
            image[y_new:y_new+h_new, x_new:x_new+w_new] = blurred_face
            
            # Skapa textsträng för confidence och rita det på bilden
            text = f"{confidence*100:.2f}%"
            cv2.putText(image, text, 
                        (bounding_box[0], bounding_box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return image


if __name__ == "__main__":
    # Exempel på användning
    detector = FaceDetector()
    detected_image = detector.detect_faces("meeting.jpg")
    cv2.imshow("Detected Faces", detected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()