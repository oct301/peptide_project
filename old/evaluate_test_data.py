import time
import tensorflow as tf

from model import *
from theoretical_train import *


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
                .prefetch(tf.data.AUTOTUNE))

'''
d_model : input(embedding), ouput 차원
num_layers : 인코더, 디코더 층
num_heads : 멀티헤드 수
d_ff : feedforward 차원 
'''
D_MODEL = 64
NUM_LAYERS = 2
NUM_HEADS = 2
DFF = 128
DROPOUT_RATE = 0.2

learning_rate = CustomSchedule(D_MODEL)
optimizer = tf.keras.optimizers.Adam(learning_rate,
                                     beta_1=0.9,
                                     beta_2=0.98,
                                     epsilon=1e-9)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

transformer = Transformer(
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dff=DFF,
    input_vocab_size=600000,
    target_vocab_size=30,
    positional_encoding_input = 1000,
    positional_encoding_target = 50,
    dropout_rate=DROPOUT_RATE)

#save checkpoint
checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')


valid_loss, valid_accuracy = evaluate_aminoacid_level(path_test_data)
print(f'\tValid | Loss {valid_loss:.3f}, Accuracy {valid_accuracy:.3f}')

# accuracy_test_data = evaluate_peptide_level(test_dataset.take(200))
# print(f'Accuracy of test data for peptide level : {accuracy_test_data:.4f}')