from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import os

import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils
from keras.callbacks import TensorBoard
from keras.utils.vis_utils import plot_model


def main():
    print("create model")
    C = config.Config()
    from keras_frcnn import resnet as nn
    C.network = 'resnet50'
    C.base_net_weights = nn.get_weight_path()

    if K.image_dim_ordering() == 'th':
        input_shape_img = (3, None, None)
    else:
        input_shape_img = (None, None, 3)
    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(None, 4))

    shared_layers = nn.nn_base(img_input, trainable=True)

    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn = nn.rpn(shared_layers, num_anchors) # this completes the rpn model

    classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=26, trainable=True)
    classifier_branch2 = nn.classifier_branch2(shared_layers, roi_input, C.num_rois, nb_classes=26, trainable=True)#harmeet. What is nb_classes ?

    model_rpn = Model(img_input, rpn[:2])
    model_classifier = Model([img_input, roi_input], classifier)
    model_classifier_branch2 = Model([img_input, roi_input], classifier_branch2)

    # this is a model that holds both the RPN and the classifier, used to load/save weights for the models
    model_all = Model([img_input, roi_input], rpn[:2] + classifier + classifier_branch2)

    plot_model(model_rpn, to_file='model_rpn.png', show_shapes=True, show_layer_names=True)
    plot_model(model_classifier, to_file='model_classifier.png', show_shapes=True, show_layer_names=True)
    plot_model(model_classifier_branch2, to_file='model_classifier_branch2.png', show_shapes=True, show_layer_names=True)
    plot_model(model_all, to_file='model_all.png', show_shapes=True, show_layer_names=True)
    try:
        # load_weights by name
        # some keras application model does not containing name
        # for this kinds of model, we need to re-construct model with naming
        print('loading weights from {}'.format(C.base_net_weights))
        model_rpn.load_weights(C.base_net_weights, by_name=True)
        model_classifier.load_weights(C.base_net_weights, by_name=True)
    except:
        print('Could not load pretrained model weights. Weights can be found in the keras application folder \
            https://github.com/fchollet/keras/tree/master/keras/applications')

    optimizer = Adam(lr=1e-5)
    optimizer_classifier = Adam(lr=1e-5)
    model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
    model_classifier.compile(optimizer=optimizer_classifier,
                             loss=[losses.class_loss_cls, losses.class_loss_regr(26 - 1)],
                             metrics={'dense_class_{}'.format(26): 'accuracy'})
    model_all.compile(optimizer='sgd', loss='mae')



if __name__ == "__main__":
    main()