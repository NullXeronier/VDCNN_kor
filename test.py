from vdcnn import *
import tensorflow as tf
import re, json
from data_helper import sentence_to_index_morphs
import absl

if __name__ == "__main__":
    # PATH = 'models'

    with open('vdcnn_model.json', 'r') as fp:
        json.load(fp)


    model = VDCNN(num_classes=y_test.shape[1],
                  depth=FLAGS.depth,
                  sequence_length=FLAGS.sequence_length,
                  shortcut=FLAGS.shortcut,
                  pool_type=FLAGS.pool_type,
                  sorted=FLAGS.sorted,
                  use_bias=FLAGS.use_bias)