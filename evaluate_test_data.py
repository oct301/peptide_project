import time
import tensorflow as tf
from model import *


feature_description = {
            'sequence': tf.io.VarLenFeature(tf.int64),
            'mz': tf.io.VarLenFeature(tf.int64),
            }

def parse_function(example_proto):
    parsed_example = tf.io.parse_single_example(example_proto,feature_description)
    mz = parsed_example['mz'].values
    sequence = parsed_example['sequence'].values
    return mz, sequence

path_theoretical_train_data='./theoretical_preprocessed_train_data.tfrecords'
path_theoretical_test_data='./theoretical_preprocessed_test_data.tfrecords'

path_test_data = tf.data.TFRecordDataset(path_theoretical_test_data)

test_dataset = (path_test_data
                .map(parse_function)
                .shuffle(100000)
                .prefetch(tf.data.AUTOTUNE)

# valid_loss, valid_accuracy = evaluate_aminoacid_level(valid_dataset)
# print(f'\tValid | Loss {valid_loss:.3f}, Accuracy {valid_accuracy:.3f}')

# accuracy_test_data = evaluate_peptide_level(test_dataset.take(200))
# print(f'Accuracy of test data for peptide level : {accuracy_test_data:.4f}')