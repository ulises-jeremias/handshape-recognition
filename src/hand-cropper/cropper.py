mport tensorflow as tf
import cv2
import numpy as np
from tensorflow.python.client import session

class Cropper:

    def __init__(self, model_dir = "./models/saved_model.pb", confidence = 0.9):
        self.confidence = confidence
        self.cache_img = None
        self.cache_indexes = None
        self.cache_name = None
        with tf.io.gfile.GFile(model_dir, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")
        self.graph = graph
        self.inputs = graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = graph.get_tensor_by_name('detection_scores:0')
        detection_classes = graph.get_tensor_by_name('detection_classes:0')
        num_detections = graph.get_tensor_by_name('num_detections:0')
        self.outputs = [detection_boxes, detection_scores, detection_classes, num_detections]
    
    def crop_images(self, images, size = (64,64), return_index = True, name = None):
        if name is not None and self.cache_name is not None and name == self.cache_name:
            if return_index:
                return self.cache_img, self.cache_indexes
            else:
                return self.cache_img

        all_boxes = []
        all_scores = []
        all_classes = []
        all_num = []
        with session.Session(graph=self.graph) as sess:
            predict = tf.keras.backend.function(inputs=self.inputs, outputs=self.outputs)
            for i in range(len(images)):
                (boxes, scores, classes, num) = predict(images[i])
                np.squeeze(boxes), np.squeeze(scores)
                all_boxes = all_boxes + [boxes.tolist()]
                all_scores = all_scores + [scores.tolist()]
                all_classes = all_classes + [classes.tolist()]
                all_num = all_num + [num.tolist()]                
        new_images = []
        if return_index:
            new_indexes = []
        for i in range(len(all_scores)):
            score = np.squeeze(all_scores[i])
            max_i = score.argmax()
            if (score[max_i] > self.confidence):
                box = np.squeeze(all_boxes[i])[max_i]
                img = np.squeeze(images[i])
                img = self.crop_box(img, box)
                img = self.resize(img, size)
                new_images.append(img)
                if return_index:
                    new_indexes.append(i)
        if name is not None:
            self.cache_img = new_images
            self.cache_indexes = new_indexes
            self.cache_name = name
        if return_index:
            return new_images, new_indexes
        else:
            return new_images

    def crop_box(self, img, box, offset = 0.05):
        shape = img.shape
        min_y = int(((box[0] - offset) if box[0] > offset else 0)   * shape[0])
        min_x = int(((box[1] - offset) if box[1] > offset else 0) * shape[1])
        max_y = int(((box[2] + offset) if box[2] + offset < 1 else 1) * shape[0])
        max_x = int(((box[3] + offset) if box[3] + offset < 1 else 1) * shape[1])
        return img[int(min_y):int(max_y), int(min_x):int(max_x)]

    def resize(self, img, size):
        new_width, new_height = size
        height, width, channels = img.shape
        ratio = min(new_width/width, new_height/height)
        new_x = int(width * ratio)
        new_y = int(height * ratio)
        resized_img = cv2.resize(img, (new_x,new_y))
        new_img = np.zeros((new_width, new_height, 3))
        x_offset = int((new_width - new_x) / 2)
        y_offset = int((new_height - new_y) / 2)
        new_img[y_offset:y_offset+new_y, x_offset:x_offset+new_x] = resized_img
        return new_img

