import tensorflow as tf
import time
import pickle

from transformer.model import Transformer
from transformer.model import ModifiedTransformer
from transformer.model import create_padding_mask
from transformer.model import create_look_ahead_mask
from transformer.optimizer import CustomSchedule

feature_description = {
            'sequence': tf.io.VarLenFeature(tf.int64),
            'mz': tf.io.VarLenFeature(tf.int64),
            }

def parse_function(example_proto):
    parsed_example = tf.io.parse_single_example(example_proto,feature_description)
    mz = parsed_example['mz'].values
    sequence = parsed_example['sequence'].values
    return mz, sequence

path_theoretical_test_data = './data/theoretical_preprocessed_test_data.tfrecords'
path_real_test_data='./data/real_preprocessed_test_data.tfrecords'

theoretical_test_data = tf.data.TFRecordDataset(path_real_test_dataset).map(parse_function)
real_test_dataset = tf.data.TFRecordDataset(path_real_test_dataset).map(parse_function).shard(num_shards=10, index=0)



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
    positional_encoding_input = 2000,
    positional_encoding_target = 50,
    dropout_rate=DROPOUT_RATE)

modified_transformer = ModifiedTransformer(
    num_layers=4,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dff=DFF,
    intensity_vocab_size=12000,
    target_vocab_size=30,
    dropout_rate=DROPOUT_RATE)

def create_masks(input, target):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(input)
    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(input)
    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
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

#save checkpoint
checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

def evaluate_peptide_level(dataset, max_length = 50):
    cnt_total =0
    cnt_correct = 0
    for mz, sequence in dataset:
        cnt_total+=1
        if(cnt_total%10 == 0):
            print(cnt_total, cnt_correct/(cnt_total-1))

        encoder_input = tf.convert_to_tensor([mz])
        start, end = 1,2
        output = tf.convert_to_tensor([start],dtype=tf.int64)
        output = tf.expand_dims(output, 0)

        for i in range(max_length):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, output)
            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = transformer(encoder_input,
                                                         output,
                                                         False,
                                                         enc_padding_mask,
                                                         combined_mask,
                                                         dec_padding_mask)
            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.argmax(predictions, axis=-1)

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

            # return the result if the predicted_id is equal to the end token
            if predicted_id == end:
                if tf.reduce_all(output == sequence):
                    cnt_correct+=1
                break

    return cnt_correct/cnt_total


print(f'Accuracy of test data(real) for peptide level : {evaluate_peptide_level(test_dataset):.4f}')

