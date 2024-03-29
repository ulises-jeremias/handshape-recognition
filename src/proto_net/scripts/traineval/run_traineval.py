import os
import time
import configparser
import numpy as np
from datetime import datetime

from src.proto_net.scripts.train.train_setup import train as train_protonet
from src.proto_net.scripts.eval.eval_setup import eval as eval_protonet

configs = {
    'lsa16': {
        'data.train_way': [5],
        'data.test_way': [5],
        # done (1, 1, 1, 1) in previous experiments
        'data.support_query': [(5, 5, 5, 5)],
        # 'data.train_size': [0.33, 0.5, 0.64, 0.75],
        'data.train_size': [0.75],
        'data.test_size': [0.25],

        #'data.rotation_range': [0, 25], 
        #'data.width_shift_range': [0.1], 
        #'data.height_shift_range': [0.1],
        #'data.horizontal_flip': [True, False],
        'data.args': [(0, 0, 0, False), (10, 0.2, 0.2, True)],

        'model.type': ['expr'],
        'model.nb_layers': [4],
        'model.nb_filters': [64],

        'train.lr': [0.001]
    },
    'ciarp': {
        'data.train_way': [5],
        'data.test_way': [5],
        # done (1, 1, 1, 1) in previous experiments
        'data.support_query': [(5, 5, 5, 5)],
        #'data.train_size': [0.33, 0.5, 0.64, 0.75],
        'data.train_size': [0.75],
        'data.test_size': [0.25],

        #'data.rotation_range': [0, 25], 
        #'data.width_shift_range': [0.1], 
        #'data.height_shift_range': [0.1],
        #'data.horizontal_flip': [True, False],
        'data.args': [(0, 0, 0, False), (10, 0.2, 0.2, True)],

        'model.type': ['expr'],
        'model.nb_layers': [4],
        'model.nb_filters': [64],

        'train.lr': [0.001]
    },
    'rwth': {
        'data.train_way': [20],
        'data.test_way': [5],
        'data.support_query': [(5, 5, 5, 5)],
        # 'data.train_size': [0.33, 0.5, 0.64, 0.75],
        'data.train_size': [0.75],
        'data.test_size': [0.25],
        
        #'data.rotation_range': [0, 25], 
        #'data.width_shift_range': [0.1], 
        #'data.height_shift_range': [0.1],
        #'data.horizontal_flip': [True, False],
        'data.args': [(0, 0, 0, False), (10, 0.2, 0.2, True)],

        'model.type': ['expr'],
        'model.nb_layers': [4],
        'model.nb_filters': [64],

        'train.lr': [0.001]
    }
}


def preprocess_config(c):
    conf_dict = {}
    int_params = ['data.train_way', 'data.test_way', 'data.train_support',
                  'data.test_support', 'data.train_query', 'data.test_query',
                  'data.episodes', 'data.gpu', 'data.cuda', 'model.z_dim', 
                  'train.epochs', 'train.patience']
    float_params = ['train.lr', 'data.rotation_range',
                    'data.width_shift_range', 'data.height_shift_range']
    for param in c:
        if param in int_params:
            conf_dict[param] = int(c[param])
        elif param in float_params:
            conf_dict[param] = float(c[param])
        else:
            conf_dict[param] = c[param]
    return conf_dict

eval_summary_file = f"results/eval_summary.csv"

# create summary file if not exists
if not os.path.exists(eval_summary_file):
    file = open(eval_summary_file, 'w')
    file.write("datetime, model, config, min_loss, min_loss_accuracy\n")
    file.close()

for dataset in ['rwth']:
    config_from_file = configparser.ConfigParser()
    config_from_file.read("./src/proto_net/config/config_{}.conf".format(dataset))

    ds_config = configs[dataset]

    for train_size in ds_config['data.train_size']:
        for test_size in ds_config['data.test_size']:
            for train_way in ds_config['data.train_way']:
                for test_way in ds_config['data.test_way']:
                    for train_support, train_query, test_support, test_query in ds_config['data.support_query']:
                        for rotation_range, width_shift_range, height_shift_range, horizontal_flip in ds_config['data.args']:
                            for model_type in ds_config['model.type']:
                                for nb_layers in ds_config['model.nb_layers']:
                                    for nb_filters in ds_config['model.nb_filters']:
                                        for lr in ds_config['train.lr']:
                                            try:
                                                custom_params = {
                                                    'data.train_way': train_way,
                                                    'data.train_support': train_support,
                                                    'data.train_query': train_query,
                                                    'data.test_way': test_way,
                                                    'data.test_support': test_support,
                                                    'data.test_query': test_query,
                                                    'data.train_size': train_size,
                                                    'data.test_size': test_size,

                                                    'data.rotation_range': rotation_range, 
                                                    'data.width_shift_range': width_shift_range,
                                                    'data.height_shift_range': height_shift_range,
                                                    'data.horizontal_flip': horizontal_flip, 
                                                    
                                                    'data.crop': True,
                                                    'data.use_cropped': True,

                                                    'model.type': model_type,
                                                    'model.nb_layers': nb_layers,
                                                    'model.nb_filters': nb_filters,

                                                    'train.lr': lr,
                                                }

                                                now = datetime.now()
                                                now_as_str = now.strftime('%Y_%m_%d-%H:%M:%S')

                                                preprocessed_config = preprocess_config({ **config_from_file['TRAIN'], **custom_params })
                                                train_protonet(preprocessed_config)

                                                preprocessed_config = {
                                                    **preprocessed_config,
                                                    # TODO: Select eval config
                                                    'data.crop': True,
                                                    'data.use_cropped': True,
                                                    'data.episodes': 1000,
                                                }

                                                losses = []
                                                accuracies = []

                                                for i in range(10):
                                                    loss, acc = eval_protonet(preprocessed_config)
                                                    print("Evalutation #{} finished".format(i))
                                                    losses.append(loss)
                                                    accuracies.append(acc)

                                                loss_avg = np.average(losses)
                                                acc_avg = np.average(accuracies)

                                                print("loss: {}, accuracy: {}".format(loss_avg, acc_avg))

                                                file = open(eval_summary_file, 'a+') 
                                                summary = "{}, {}, proto-net, {}, {}, {}\n".format(now_as_str,
                                                                                                   preprocessed_config['data.dataset'],
                                                                                                   preprocessed_config['model.save_path'],
                                                                                                   loss_avg,
                                                                                                   acc_avg)
                                                file.write(summary)
                                            except Exception as e:
                                                print(e)

