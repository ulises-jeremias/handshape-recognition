import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import handshape_datasets as hd
from src.hand_cropper.cropper import Cropper
from src.utils.model_selection import train_test_split_balanced

class DataLoader(object):
    def __init__(self, data, n_classes, n_way, n_support, n_query, x_dim):
        self.data = data
        self.n_way = n_way
        self.n_classes = n_classes
        self.n_support = n_support
        self.n_query = n_query
        self.x_dim = x_dim

    def get_next_episode(self):
        n_examples = self.data.shape[1]
        w, h, c = self.x_dim
        support = np.zeros([self.n_way, self.n_support, w, h, c], dtype=np.float32)
        query = np.zeros([self.n_way, self.n_query, w, h, c], dtype=np.float32)
        classes_ep = np.random.permutation(self.n_classes)[:self.n_way]

        for i, i_class in enumerate(classes_ep):
            selected = np.random.permutation(n_examples)[:self.n_support + self.n_query]
            support[i] = self.data[i_class, selected[:self.n_support]]
            query[i] = self.data[i_class, selected[self.n_support:]]

        return support, query

def load_rwth(data_dir, config, splits):
    """
    Load rwth dataset.

    Args:
        data_dir (str): path of the directory with 'splits', 'data' subdirs.
        config (dict): general dict with program settings.
        splits (list): list of strings 'train'|'val'|'test'

    Returns (dict): dictionary with keys as splits and values as tf.Dataset

    """

    DATASET_NAME = "rwth"
    DATASET_PATH = "/develop/data/rwth/data"

    data = hd.load(DATASET_NAME, DATASET_PATH)
    features = data[0]
    classes = data[1]['y']

    good_min = 20
    good_classes = []

    for i in range(len(classes)):
        images = features[np.equal(i, classes)]
        if len(images) >= good_min:
            good_classes = good_classes + [i]

    good_x = features[np.in1d(classes, good_classes)]
    good_y = classes[np.in1d(classes, good_classes)]
    my_dict = dict(zip(np.unique(good_y), range(len(np.unique(good_y)))))
    good_y = np.vectorize(my_dict.get)(good_y)

    features, classes = good_x, good_y

    if config['data.crop']:
        print('cropping')
        config['model.x_dim'] = '64,64,3'
        if '_cropper' not in config or ('_cropper' in config and config['_cropper'] is None):
            config['_cropper'] = Cropper(confidence=0.9, model_dir="src/hand_cropper/models/saved_model.pb")
        print('Cropper created')
        features, classes = config['_cropper'].crop_dataset(features, classes, size=(64, 64), use_cropped=config['data.use_cropped'], good_min=good_min, dataset_name=config['data.dataset'])
        print('dataset cropped')

    uniqueClasses = np.unique(classes)

    x_train, x_test, y_train, y_test = train_test_split_balanced(features,
                                                                 classes,
                                                                 train_size=config['data.train_size'],
                                                                 test_size=config['data.test_size'])

    x_train, x_test = x_train / 255.0, x_test / 255.0

    _, amountPerTrain = np.unique(y_train, return_counts=True)
    _, amountPerTest = np.unique(y_test, return_counts=True)

    train_datagen_args = dict(featurewise_center=True,
                              featurewise_std_normalization=True,
                              rotation_range=config['data.rotation_range'],
                              width_shift_range=config['data.width_shift_range'],
                              height_shift_range=config['data.height_shift_range'],
                              horizontal_flip=config['data.horizontal_flip'],
                              fill_mode='constant',
                              cval=0)
    train_datagen = ImageDataGenerator(train_datagen_args)
    train_datagen.fit(x_train)

    test_datagen_args = dict(featurewise_center=True,
                             featurewise_std_normalization=True,
                             fill_mode='constant',
                             cval=0)
    test_datagen = ImageDataGenerator(test_datagen_args)
    test_datagen.fit(x_train)

    w, h, c = list(map(int, config['model.x_dim'].split(',')))

    ret = {}
    for split in splits:
        # n_way (number of classes per episode)
        if split in ['val', 'test']:
            n_way = config['data.test_way']
        else:
            n_way = config['data.train_way']

        # n_support (number of support examples per class)
        if split in ['val', 'test']:
            n_support = config['data.test_support']
        else:
            n_support = config['data.train_support']

        # n_query (number of query examples per class)
        if split in ['val', 'test']:
            n_query = config['data.test_query']
        else:
            n_query = config['data.train_query']

        if split in ['val', 'test']:
            y = y_test
            x = x_test
            dg = train_datagen
            dg_args = train_datagen_args
        else:
            y = y_train
            x = x_train
            dg = test_datagen
            dg_args = test_datagen_args

        amountPerClass = amountPerTest if split in ['val', 'test'] else amountPerTrain

        i = np.argsort(y)
        x = x[i, :, :, :]
        
        for index in i:
            x[index, :, :, :] = dg.apply_transform(x[index], dg_args)

        data = np.reshape(x, (len(uniqueClasses), amountPerClass[0], w, h, c))

        data_loader = DataLoader(data,
                                 n_classes=len(uniqueClasses),
                                 n_way=n_way,
                                 n_support=n_support,
                                 n_query=n_query,
                                 x_dim=(w, h, c))

        ret[split] = data_loader

    return ret
