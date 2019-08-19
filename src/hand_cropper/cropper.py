"""Cropper"""

import numpy as np
import tensorflow as tf
from tensorflow.python.client import session
from src.utils.images import resize, crop_box

class Cropper:
    """Cropper definition"""

    def __init__(self, model_dir="./models/saved_model.pb", confidence=0.9):
        self.confidence = confidence
        self.cache_img = None
        self.cache_indexes = None
        self.cache_name = None
        with tf.io.gfile.GFile(model_dir, "rb") as file:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(file.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")
        self.graph = graph
        self.inputs = graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = graph.get_tensor_by_name('detection_scores:0')
        detection_classes = graph.get_tensor_by_name('detection_classes:0')
        num_detections = graph.get_tensor_by_name('num_detections:0')
        self.outputs = [detection_boxes, detection_scores, detection_classes, num_detections]

    def crop_images(self, images, size=(64, 64), return_index=True, name=None):
        """
            Crops images

            Returns: cropped images
        """

        if name is not None and self.cache_name is not None and name == self.cache_name:
            if return_index:
                return self.cache_img, self.cache_indexes

            return self.cache_img

        all_boxes = []
        all_scores = []
        all_classes = []
        all_num = []
        with session.Session(graph=self.graph) as sess:
            predict = tf.keras.backend.function(inputs=self.inputs, outputs=self.outputs)
            for i in enumerate(images):
                (boxes, scores, classes, num) = predict(images[i])
                boxes, scores = np.squeeze(boxes), np.squeeze(scores)
                all_boxes = all_boxes + [boxes.tolist()]
                all_scores = all_scores + [scores.tolist()]
                all_classes = all_classes + [classes.tolist()]
                all_num = all_num + [num.tolist()]
        new_images = []
        if return_index:
            new_indexes = []
        for i in enumerate(all_scores):
            score = np.squeeze(all_scores[i])
            max_i = score.argmax()
            if score[max_i] > self.confidence:
                box = np.squeeze(all_boxes[i])[max_i]
                img = np.squeeze(images[i])
                img = crop_box(img, box)
                img = resize(img, size)
                new_images.append(img)
                if return_index:
                    new_indexes.append(i)
        if name is not None:
            self.cache_img = new_images
            self.cache_indexes = new_indexes
            self.cache_name = name

        if return_index:
            return new_images, new_indexes

        return new_images
