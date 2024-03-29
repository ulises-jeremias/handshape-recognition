#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf

import os, sys, getopt
import json
from datetime import datetime

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.handcropper.cropper import Cropper

from .datasets import load
from densenet import densenet_model

print(tf.__version__)

class DenseNetTrainer:
    def __init__(self, checkpoints = False, log_freq = 1, save_freq = 40):
        # log
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.checkpoints = checkpoints
        self.cropper = None
        self.models_directory = 'models/'
        self.results_directory = 'results/'
        self.config_directory = 'config/'
        self.general_directory = "./results/"
        self.summary_file = self.general_directory + 'summary.csv'

        # create summary file if not exists
        if not os.path.exists(self.general_directory):
            os.makedirs(self.general_directory)
        if not os.path.exists(self.summary_file):
            file = open(self.summary_file, 'w')
            file.write("datetime, model, config, min_loss, min_loss_accuracy\n")
            file.close()
        
        # initial config for model reutilization
        self.initial_config = None

    def load_model(self, n_classes, image_shape, growth_rate, nb_layers, reduction):
        config = [n_classes, image_shape, growth_rate, nb_layers, reduction]
        if config == self.initial_config:
            self.model.set_weights(self.initial_weights)
        else:
            self.model = densenet_model(classes=n_classes, shape=image_shape, growth_rate=growth_rate, 
                                        nb_layers=nb_layers, reduction=reduction)
            self.initial_weights = self.model.get_weights()
            self.initial_config = config
            
    def train(self, dataset_name = "lsa16", rotation_range = 10, width_shift_range = 0.10,
              height_shift_range = 0.10, horizontal_flip = True, growth_rate = 64,
              nb_layers = [6,12], reduction = 0.0, lr = 0.001, epochs = 400,
              max_patience = 25, batch_size= 16, weight_classes = False,
              random_split = False, train_size=None, test_size=None,
              crop = False, use_cropped = True, good_min=15):
        
        save_directory = self.general_directory + "{}/dense-net/".format(dataset_name)
        date = datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
        identifier = "{}-growth-{}-densenet-{}".format(
            '-'.join([str(i) for i in nb_layers]),
            growth_rate, 
            dataset_name) + date
        
        x, y = load(dataset_name)

        if crop:
            print("cropping")
            # if cropper class is already created use it, otherwise create it
            if self.cropper is None:
                self.cropper = Cropper(confidence = 0.9, model_dir="src/hand_cropper/models/saved_model.pb")
            x, y = self.cropper.crop_dataset(x, y, size=(64, 64), dataset_name=dataset_name, use_cropped=use_cropped, good_min=good_min)
            print('dataset cropped')
        
        image_shape = np.shape(x)[1:]

        x_train, x_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            train_size=train_size,
                                                            test_size=test_size,
                                                            random_state=None if random_split else 42,
                                                            stratify=y)
        x_train, x_test = x_train / 255.0, x_test / 255.0

        n_classes = len(np.unique(y))

        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            horizontal_flip=horizontal_flip,
            fill_mode='constant',
            cval=0)
        datagen.fit(x_train)

        test_datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            fill_mode='constant',
            cval=0)
        test_datagen.fit(x_train)

        print("data loaded")

        self.load_model(n_classes, image_shape, growth_rate, nb_layers, reduction)

        print("model created")

        if weight_classes:
            class_weights = compute_class_weight('balanced', np.unique(y), y)
            loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

            def weightedLoss(originalLossFunc, weightsList):

                @tf.function
                def lossFunc(true, pred):

                    axis = -1 #if channels last 
                    #axis=  1 #if channels first

                    #argmax returns the index of the element with the greatest value
                    #done in the class axis, it returns the class index    
                    classSelectors = tf.argmax(true, axis=axis, output_type=tf.int32) 

                    #considering weights are ordered by class, for each class
                    #true(1) if the class index is equal to the weight index   
                    classSelectors = [tf.equal(i, classSelectors) for i in range(len(weightsList))]

                    #casting boolean to float for calculations  
                    #each tensor in the list contains 1 where ground true class is equal to its index 
                    #if you sum all these, you will get a tensor full of ones. 
                    classSelectors = [tf.cast(x, tf.float32) for x in classSelectors]

                    #for each of the selections above, multiply their respective weight
                    weights = [sel * w for sel,w in zip(classSelectors, weightsList)] 

                    #sums all the selections
                    #result is a tensor with the respective weight for each element in predictions
                    weightMultiplier = weights[0]
                    for i in range(1, len(weights)):
                        weightMultiplier = weightMultiplier + weights[i]


                    #make sure your originalLossFunc only collapses the class axis
                    #you need the other axes intact to multiply the weights tensor
                    loss = originalLossFunc(true,pred) 
                    loss = loss * weightMultiplier

                    return loss
                return lossFunc
            loss_object = weightedLoss(loss_object, class_weights)
        else:
            loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

        optimizer = tf.keras.optimizers.Adam()

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        @tf.function
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                predictions = self.model(tf.cast(images, tf.float32), training=True)
                loss = loss_object(labels, predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            train_loss(loss)
            train_accuracy(labels, predictions)

        @tf.function
        def test_step(images, labels):
            predictions = self.model(tf.cast(images, tf.float32), training=False)
            t_loss = loss_object(labels, predictions)

            test_loss(t_loss)
            test_accuracy(labels, predictions)

        # create summary writers
        train_summary_writer = tf.summary.create_file_writer(save_directory + 'summaries/train/' + identifier)
        test_summary_writer = tf.summary.create_file_writer(save_directory +  'summaries/test/' + identifier)

        # create data generators
        train_gen =  datagen.flow(x_train, y_train, batch_size=batch_size)
        test_gen = test_datagen.flow(x_test, y_test, batch_size=batch_size, shuffle=False)

        print("starting training")

        min_loss = 100
        min_loss_acc = 0
        patience = 0
        results = 'epoch,loss,accuracy,test_loss,test_accuracy\n'

        for epoch in range(epochs):
            batches = 0
            for images, labels in train_gen:
                train_step(images, labels)
                batches += 1
                if batches >= len(x_train) / 32:
                    # we need to break the loop by hand because
                    # the generator loops indefinitely
                    break

            batches = 0
            for test_images, test_labels in test_gen:
                test_step(test_images, test_labels)
                batches += 1
                if batches >= len(x_test) / 32:
                    # we need to break the loop by hand because
                    # the generator loops indefinitely
                    break

            if (epoch % self.log_freq == 0):
                results += '{},{},{},{},{}\n'.format(epoch,
                                    train_loss.result(),
                                    train_accuracy.result()*100,
                                    test_loss.result(),
                                    test_accuracy.result()*100)
                print ('Epoch: {}, Train Loss: {}, Train Acc:{}, Test Loss: {}, Test Acc: {}'.format(epoch,
                                    train_loss.result(),
                                    train_accuracy.result()*100,
                                    test_loss.result(),
                                    test_accuracy.result()*100))

                if (test_loss.result() < min_loss):    
                    if not os.path.exists(save_directory + self.models_directory):
                        os.makedirs(save_directory + self.models_directory)
                    # serialize weights to HDF5
                    self.model.save_weights(save_directory + self.models_directory + "best{}.h5".format(identifier))
                    min_loss = test_loss.result()
                    min_loss_acc = test_accuracy.result()
                    patience = 0
                else:
                    patience += 1

                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', train_loss.result(), step=epoch)
                    tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
                    train_loss.reset_states()           
                    train_accuracy.reset_states()           

                with test_summary_writer.as_default():
                    tf.summary.scalar('loss', test_loss.result(), step=epoch)
                    tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)
                    test_loss.reset_states()           
                    test_accuracy.reset_states()   

            if self.checkpoints and epoch % self.save_freq == 0:
                if not os.path.exists(save_directory + self.models_directory):
                    os.makedirs(save_directory + self.models_directory)
                # serialize weights to HDF5
                self.model.save_weights(save_directory + self.models_directory+"{}_epoch{}.h5".format(identifier,epoch))

            if patience >= max_patience:
                break
                
        config = {
            'data.dataset_name': dataset_name, 
            'data.rotation_range': rotation_range, 
            'data.width_shift_range': width_shift_range, 
            'data.height_shift_range': height_shift_range, 
            'data.horizontal_flip': horizontal_flip, 
            'data.test_size': test_size, 
            'data.train_size': train_size, 
            'model.growth_rate': growth_rate, 
            'model.nb_layers': nb_layers, 
            'model.reduction': reduction, 
            'train.lr': lr, 
            'train.epochs': epochs, 
            'train.max_patience': max_patience, 
            'train.batch_size': batch_size, 
        }
        self.log(results, config, date, min_loss, min_loss_acc, save_directory, identifier)
        
    def log(self, results, config, date, min_loss, min_loss_acc, save_directory, identifier):
        if not os.path.exists(save_directory + self.results_directory):
            os.makedirs(save_directory + self.results_directory)
        file = open(save_directory + self.results_directory + 'results-'+ identifier + '.csv','w') 
        file.write(results) 
        file.close()

        if not os.path.exists(save_directory + self.config_directory):
            os.makedirs(save_directory + self.config_directory)

        file = open(save_directory + self.config_directory + identifier + '.json', 'w')
        file.write(json.dumps(config, indent=2))
        file.close()

        file = open(self.summary_file, 'a+')
        summary = "{}, {}, dense-net, {}, {}, {}\n".format(date,
                                                           config['data.dataset_name'],
                                                           save_directory + self.config_directory + identifier + '.json',
                                                           min_loss,
                                                           min_loss_acc)
        file.write(summary)
        file.close()

def train_densenet(dataset_name = "rwth", rotation_range = 10, width_shift_range = 0.10,
          height_shift_range = 0.10, horizontal_flip = True, growth_rate = 128,
          nb_layers = [6,12], reduction = 0.0, lr = 0.001, epochs = 400,
          max_patience = 25, batch_size= 16, checkpoints = False, weight_classes = False,
          train_size=None, test_size=None, crop = False, use_cropped = True, good_min=15):

    # log
    log_freq = 1
    save_freq = 40
    models_directory = 'models/'
    results_directory = 'results/'
    config_directory = 'config/'

    general_directory = "./results/"
    save_directory = general_directory + "{}/dense-net/".format(dataset_name)
    results = 'epoch,loss,accuracy,test_loss,test_accuracy\n'

    date = datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
    identifier = "{}-growth-{}-densenet-{}".format(
        '-'.join([str(i) for i in nb_layers]),
        growth_rate, 
        dataset_name) + date

    summary_file = general_directory + 'summary.csv'

    # create summary file if not exists

    if not os.path.exists(general_directory):
        os.makedirs(general_directory)
    if not os.path.exists(summary_file):
        file = open(summary_file, 'w')
        file.write("datetime, model, config, min_loss, min_loss_accuracy\n")
        file.close()

    print("hyperparameters set")
    #print(tf.test.is_gpu_available())

    x, y = load(dataset_name)

    if crop:
        print("cropping")
        cropper = Cropper(confidence = 0.9, model_dir="src/hand_cropper/models/saved_model.pb")
        x, y = cropper.crop_dataset(x, y, size=(64, 64), use_cropped=use_cropped, good_min=good_min)
        print('dataset cropped')
    
    image_shape = np.shape(x)[1:]

    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        train_size=train_size,
                                                        test_size=test_size,
                                                        random_state=42,
                                                        stratify=y)
    x_train, x_test = x_train / 255.0, x_test / 255.0

    n_classes = len(np.unique(y))

    if weight_classes:
        class_weights = compute_class_weight('balanced', 
                                            np.unique(y),
                                            y)
    
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        horizontal_flip=horizontal_flip,
        fill_mode='constant',
        cval=0)
    datagen.fit(x_train)

    test_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        fill_mode='constant',
        cval=0)
    test_datagen.fit(x_train)

    print("data loaded")

    model = densenet_model(classes=n_classes, shape=image_shape, growth_rate=growth_rate, nb_layers=nb_layers, reduction=reduction)

    print("model created")

    if weight_classes:
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

        def weightedLoss(originalLossFunc, weightsList):

            @tf.function
            def lossFunc(true, pred):

                axis = -1 #if channels last 
                #axis=  1 #if channels first

                #argmax returns the index of the element with the greatest value
                #done in the class axis, it returns the class index    
                classSelectors = tf.argmax(true, axis=axis, output_type=tf.int32) 

                #considering weights are ordered by class, for each class
                #true(1) if the class index is equal to the weight index   
                classSelectors = [tf.equal(i, classSelectors) for i in range(len(weightsList))]

                #casting boolean to float for calculations  
                #each tensor in the list contains 1 where ground true class is equal to its index 
                #if you sum all these, you will get a tensor full of ones. 
                classSelectors = [tf.cast(x, tf.float32) for x in classSelectors]

                #for each of the selections above, multiply their respective weight
                weights = [sel * w for sel,w in zip(classSelectors, weightsList)] 

                #sums all the selections
                #result is a tensor with the respective weight for each element in predictions
                weightMultiplier = weights[0]
                for i in range(1, len(weights)):
                    weightMultiplier = weightMultiplier + weights[i]


                #make sure your originalLossFunc only collapses the class axis
                #you need the other axes intact to multiply the weights tensor
                loss = originalLossFunc(true,pred) 
                loss = loss * weightMultiplier

                return loss
            return lossFunc
        loss_object = weightedLoss(loss_object, class_weights)
    else:
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(tf.cast(images, tf.float32), training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = model(tf.cast(images, tf.float32), training=False)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    # create summary writers
    train_summary_writer = tf.summary.create_file_writer(save_directory + 'summaries/train/' + identifier)
    test_summary_writer = tf.summary.create_file_writer(save_directory +  'summaries/test/' + identifier)

    # create data generators
    train_gen =  datagen.flow(x_train, y_train, batch_size=batch_size)
    test_gen = test_datagen.flow(x_test, y_test, batch_size=batch_size, shuffle=False)

    print("starting training")

    min_loss = 100
    min_loss_acc = 0
    patience = 0

    for epoch in range(epochs):
        batches = 0
        for images, labels in train_gen:
            train_step(images, labels)
            batches += 1
            if batches >= len(x_train) / 32:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break

        batches = 0
        for test_images, test_labels in test_gen:
            test_step(test_images, test_labels)
            batches += 1
            if batches >= len(x_test) / 32:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break

        if (epoch % log_freq == 0):
            results += '{},{},{},{},{}\n'.format(epoch,
                                train_loss.result(),
                                train_accuracy.result()*100,
                                test_loss.result(),
                                test_accuracy.result()*100)
            print ('Epoch: {}, Train Loss: {}, Train Acc:{}, Test Loss: {}, Test Acc: {}'.format(epoch,
                                train_loss.result(),
                                train_accuracy.result()*100,
                                test_loss.result(),
                                test_accuracy.result()*100))

            if (test_loss.result() < min_loss):    
                if not os.path.exists(save_directory + models_directory):
                    os.makedirs(save_directory + models_directory)
                # serialize weights to HDF5
                model.save_weights(save_directory + models_directory + "best{}.h5".format(identifier))
                min_loss = test_loss.result()
                min_loss_acc = test_accuracy.result()
                patience = 0
            else:
                patience += 1

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
                train_loss.reset_states()           
                train_accuracy.reset_states()           

            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)
                test_loss.reset_states()           
                test_accuracy.reset_states()   
                
        if checkpoints and epoch % save_freq == 0:
            if not os.path.exists(save_directory + models_directory):
                os.makedirs(save_directory + models_directory)
            # serialize weights to HDF5
            model.save_weights(save_directory + models_directory+"{}_epoch{}.h5".format(identifier,epoch))
            
        if patience >= max_patience:
            break

    if not os.path.exists(save_directory + results_directory):
        os.makedirs(save_directory + results_directory)
    file = open(save_directory + results_directory + 'results-'+ identifier + '.csv','w') 
    file.write(results) 
    file.close()

    if not os.path.exists(save_directory + config_directory):
        os.makedirs(save_directory + config_directory)

    config = {
        'data.dataset_name': dataset_name, 
        'data.rotation_range': rotation_range, 
        'data.width_shift_range': width_shift_range, 
        'data.height_shift_range': height_shift_range, 
        'data.horizontal_flip': horizontal_flip, 
        'data.test_size': test_size, 
        'data.train_size': train_size, 
        'model.growth_rate': growth_rate, 
        'model.nb_layers': nb_layers, 
        'model.reduction': reduction, 
        'train.lr': lr, 
        'train.epochs': epochs, 
        'train.max_patience': max_patience, 
        'train.batch_size': batch_size, 
    }

    file = open(save_directory + config_directory + identifier + '.json', 'w')
    file.write(json.dumps(config, indent=2))
    file.close()

    file = open(summary_file, 'a+')
    summary = "{}, {}, dense-net, {}, {}, {}\n".format(date,
                                                       dataset_name,
                                                       save_directory + config_directory + identifier + '.json',
                                                       min_loss,
                                                       min_loss_acc)
    file.write(summary)
    file.close()