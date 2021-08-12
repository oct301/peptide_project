import time

from model import *

feature_description = {
            'sequence': tf.io.VarLenFeature(tf.int64),
            'intensity': tf.io.VarLenFeature(tf.int64),
            'mz': tf.io.VarLenFeature(tf.int64),
            }

def parse_function(example_proto):
    parsed_example = tf.io.parse_single_example(example_proto,feature_description)
    mz = parsed_example['mz'].values
    intensity = parsed_example['intensity'].values
    sequence = parsed_example['sequence'].values
    return mz, intensity, sequence

path_real_train_data='./real_preprocessed_train_data.tfrecords'
path_real_valid_data='./real_preprocessed_valid_data.tfrecords'

#size_real_train_dataset = 300000
size_train_dataset = 256105
real_train_dataset = tf.data.TFRecordDataset(path_real_train_data)#.take(size_real_train_dataset)
real_valid_dataset = tf.data.TFRecordDataset(path_real_valid_data).map(parse_function)

#Set batchs
BATCH_SIZE = 64
NUM_BATCHS = int(size_train_dataset/BATCH_SIZE)
real_train_batches = (real_train_dataset
                 .map(parse_function)
                 .padded_batch(BATCH_SIZE)
                 .prefetch(tf.data.AUTOTUNE))


def create_masks(input, target):
    enc_padding_mask = create_padding_mask(input)
    dec_padding_mask = create_padding_mask(input)
    look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])
    dec_target_padding_mask = create_padding_mask(target)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def accuracy_function(real, pred):
  accuracies = tf.equal(real, tf.argmax(pred, axis=2))

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')


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
    input_vocab_size=500000,
    target_vocab_size=30,
    positional_encoding_input = 1000,
    positional_encoding_target = 50,
    dropout_rate=DROPOUT_RATE)

#save checkpoint
checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

