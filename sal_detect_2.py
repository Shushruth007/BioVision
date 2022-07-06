import numpy as np
import cv2
import time
import sys

# Dependency Files
CLASS_NAMES = "YOLO\classes.names"
YOLO_CONFG = "YOLO\\yolov3bv.cfg"
YOLO_WEIGHTS = "YOLO\\yolov3_last.weights"

class SaliencyDetector:

    def __init__(self, input_file):
        self.input_file = input_file

    def set_up_network(self):
        # Loading COCO class labels from file
        # Opening file
        # Pay attention! If you're using Windows, yours path might looks like:
        # r'yolo-coco-data\coco.names'
        # or:
        # 'yolo-coco-data\\coco.names'
        with open(CLASS_NAMES) as f:
            # Getting labels reading every line
            # and putting them into the list
            self.labels = [line.strip() for line in f]

        # Loading trained YOLO v3 Objects Detector
        # with the help of 'dnn' library from OpenCV
        # Pay attention! If you're using Windows, yours paths might look like:
        # r'yolo-coco-data\yolov3.cfg'
        # r'yolo-coco-data\yolov3.weights'
        # or:
        # 'yolo-coco-data\\yolov3.cfg'
        # 'yolo-coco-data\\yolov3.weights'
        self.network = cv2.dnn.readNetFromDarknet(YOLO_CONFG, YOLO_WEIGHTS)

        # Getting list with names of all layers from YOLO v3 network
        self.layers_names_all = self.network.getLayerNames()

        # Check point
        # print()
        # print(layers_names_all)

        # Getting only output layers' names that we need from YOLO v3 algorithm
        # with function that returns indexes of layers with unconnected outputs
        self.layers_names_output = \
            [self.layers_names_all[i[0] - 1] for i in self.network.getUnconnectedOutLayers()]

    
    def yolo3(self):
        # Setting minimum probability to eliminate weak predictions
        probability_minimum = 0.5

        # Setting threshold for filtering weak bounding boxes
        # with non-maximum suppression
        threshold = 0.3

        """
        Start of:
        Reading input image
        """

        # Reading image with OpenCV library
        # In this way image is opened already as numpy array
        # WARNING! OpenCV by default reads images in BGR format
        image_BGR = self.current_frame
        image_BGR2 = self.current_sal.copy()

        # Check point
        # Showing image shape
        print()
        print('Image shape:', image_BGR.shape)  # tuple of (466, 700, 3)

        # Getting spatial dimension of input image
        h, w = image_BGR.shape[:2]  # Slicing from tuple only first two elements

        # Check point
        # Showing height an width of image
        print('Image height={0} and width={1}'.format(h, w))  # 466 700

        """
        End of: 
        Reading input image
        """

        """
        Start of:
        Getting blob from input image
        """

        # Getting blob from input image
        # The 'cv2.dnn.blobFromImage' function returns 4-dimensional blob
        # from input image after mean subtraction, normalizing, and RB channels swapping
        # Resulted shape has number of images, number of channels, width and height
        # E.G.:
        # blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
        blob = cv2.dnn.blobFromImage(image_BGR, 1 / 255.0, (416, 416),
                                    swapRB=True, crop=False)

        # Check point
        print('Blob shape:', blob.shape)  # (1, 3, 416, 416)

        """
        End of:
        Getting blob from input image
        """

        """
        Start of:
        Loading YOLO v3 network
        """


        # Generating colours for representing every detected object
        # with function randint(low, high=None, size=None, dtype='l')
        colours = np.random.randint(0, 255, size=(1, 3), dtype='uint8')

        # Check point
        # print()
        # print(type(colours))  # <class 'numpy.ndarray'>
        # print(colours.shape)  # (80, 3)
        # print(colours[0])  # [172  10 127]

        """
        End of:
        Loading YOLO v3 network
        """

        """
        Start of:
        Implementing Forward pass
        """

        # Implementing forward pass with our blob and only through output layers
        # Calculating at the same time, needed time for forward pass
        self.network.setInput(blob)  # setting blob as input to the network
        start = time.time()
        output_from_network = self.network.forward(self.layers_names_output)
        end = time.time()

        # Showing spent time for forward pass
        print()
        print('Objects Detection took {:.5f} seconds'.format(end - start))

        """
        End of:
        Implementing Forward pass
        """

        """
        Start of:
        Getting bounding boxes
        """

        # Preparing lists for detected bounding boxes,
        # obtained confidences and class's number
        self.curr_bounding_boxes = []
        self.curr_confidences = []
        class_numbers = []

        # Going through all output layers after feed forward pass
        for result in output_from_network:
            # Going through all detections from current output layer
            for detected_objects in result:
                # Getting 80 classes' probabilities for current detected object
                scores = detected_objects[5:]
                # Getting index of the class with the maximum value of probability
                class_current = np.argmax(scores)
                # Getting value of probability for defined class
                confidence_current = scores[class_current]

                # # Check point
                # # Every 'detected_objects' numpy array has first 4 numbers with
                # # bounding box coordinates and rest 80 with probabilities for every class
                # print(detected_objects.shape)  # (85,)

                # Eliminating weak predictions with minimum probability
                if confidence_current > probability_minimum:
                    # Scaling bounding box coordinates to the initial image size
                    # YOLO data format keeps coordinates for center of bounding box
                    # and its current width and height
                    # That is why we can just multiply them elementwise
                    # to the width and height
                    # of the original image and in this way get coordinates for center
                    # of bounding box, its width and height for original image
                    box_current = detected_objects[0:4] * np.array([w, h, w, h])

                    # Now, from YOLO data format, we can get top left corner coordinates
                    # that are x_min and y_min
                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))

                    # Adding results into prepared lists
                    self.curr_bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                    print(self.curr_bounding_boxes)
                    self.curr_confidences.append(float(confidence_current))
                    class_numbers.append(class_current)

        """
        End of:
        Getting bounding boxes
        """

        """
        Start of:
        Non-maximum suppression
        """

        # Implementing non-maximum suppression of given bounding boxes
        # With this technique we exclude some of bounding boxes if their
        # corresponding confidences are low or there is another
        # bounding box for this region with higher confidence

        # It is needed to make sure that data type of the boxes is 'int'
        # and data type of the confidences is 'float'
        # https://github.com/opencv/opencv/issues/12789
        results = cv2.dnn.NMSBoxes(self.curr_bounding_boxes, self.curr_confidences,
                                probability_minimum, threshold)

        """
        End of:
        Non-maximum suppression
        """

        """
        Start of:
        Drawing bounding boxes and labels
        """

        # Defining counter for detected objects
        counter = 1

        # Checking if there is at least one detected object after non-maximum suppression
        if len(results) > 0:
            # Going through indexes of results
            for i in results.flatten():
                # Showing labels of the detected objects
                print('Object {0}: {1}'.format(counter, self.labels[int(class_numbers[i])]))

                # Incrementing counter
                counter += 1

                # Getting current bounding box coordinates,
                # its width and height
                x_min, y_min = self.curr_bounding_boxes[i][0], self.curr_bounding_boxes[i][1]
                box_width, box_height = self.curr_bounding_boxes[i][2], self.curr_bounding_boxes[i][3]

                # Preparing colour for current bounding box
                # and converting from numpy array to list
                colour_box_current = colours[class_numbers[i]].tolist()

                # # # Check point
                # print(type(colour_box_current))  # <class 'list'>
                # print(colour_box_current)  # [172 , 10, 127]

                # Drawing bounding box on the original image
                cv2.rectangle(image_BGR2, (x_min, y_min),
                            (x_min + box_width, y_min + box_height),
                            colour_box_current, 2)

                # Preparing text with label and confidence for current bounding box
                text_box_current = '{}: {:.4f}'.format(self.labels[int(class_numbers[i])],
                                                    self.curr_confidences[i]) + str(self.curr_bounding_boxes)

                # Putting text with label and confidence on the original image
                cv2.putText(image_BGR2, text_box_current, (x_min, y_min - 5),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)

        # Comparing how many objects where before non-maximum suppression
        # and left after
        print()
        print('Total objects been detected:', len(self.curr_bounding_boxes))
        print('Number of objects left after non-maximum suppression:', counter - 1)

        """
        End of:
        Drawing bounding boxes and labels
        """

        # Saving resulted image in jpg format by OpenCV function
        # that uses extension to choose format to save with
        self.curr_detected_objects = image_BGR2
        self.curr_objects_after_nms = results


    def get_most_salient(self): # frame, prev_frame, list, boxes
        new_frame = self.current_frame.copy()
        gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        
        frameDelta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        list_of_scores = []
        boxes = self.curr_bounding_boxes
        if len(self.curr_objects_after_nms) > 0:
            for i in self.curr_objects_after_nms.flatten():
                x_min, y_min, box_width, box_height = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
                area = box_width*box_height
                change_patch = np.sum(thresh[y_min:y_min+box_height, x_min:x_min+box_width])/area
                list_of_scores.append(change_patch)
            
            max_score = np.argmax(list_of_scores)
            x, y, width, height = boxes[max_score][0], boxes[max_score][1], boxes[max_score][2], boxes[max_score][3]
            cv2.rectangle(new_frame, (x, y),(x + width, y + height),(0,0,255), 2)

        self.most_salient = new_frame
        self.prev_frame = gray
    
    def get_objects_yolo(self):
        vid = cv2.VideoCapture(self.input_file)
        frame_width = int(vid.get(3))
        frame_height = int(vid.get(4))  
        size = (frame_width, frame_height)
        saliency = None
        obj_dect = cv2.createBackgroundSubtractorKNN()
        result_vid = cv2.VideoWriter('result.avi', 
                                cv2.VideoWriter_fourcc(*'MJPG'),
                                10, size)
        self.set_up_network()
        count = 0
        prev_frame = None
        while True:
            isTrue, self.current_frame = vid.read()
            count += 1
            if prev_frame is None:
                gray1 = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
                gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)
                self.prev_frame = gray1


            if (count % 40 == 0):
                blank = np.zeros(self.current_frame.shape, dtype='uint8')
                saliency = cv2.saliency.StaticSaliencyFineGrained_create()
                (success, saliencyMap) = saliency.computeSaliency(self.current_frame)
                saliencyMap = (saliencyMap * 255).astype("uint8")
                self.current_sal = cv2.merge((saliencyMap,saliencyMap,saliencyMap))
                self.yolo3()
                print(self.curr_objects_after_nms)
                self.get_most_salient()

                cv2.imshow("Original", self.current_frame)
                cv2.imshow("Saliency", self.current_sal)
                cv2.imshow("Result", self.curr_detected_objects)
                cv2.imshow("Most Salient", self.most_salient)
                result_vid.write(self.most_salient)

            key = cv2.waitKey(1) & 0xFF

            if ((key == ord("q")) or (isTrue == False)):
                break
        vid.release()
        result_vid.release()
        cv2.destroyAllWindows()
        print("Done")


sal_detect = SaliencyDetector(sys.argv[1])
sal_detect.get_objects_yolo()