import tensorflow as tf
import time

from model import *
from transfer_model import *

feature_description1 = {
    'sequence': tf.io.VarLenFeature(tf.int64),
    'mz': tf.io.VarLenFeature(tf.int64),
}


def parse_function1(example_proto):
    parsed_example = tf.io.parse_single_example(example_proto, feature_description1)
    mz = parsed_example['mz'].values
    sequence = parsed_example['sequence'].values
    return mz, sequence


path_theoretical_train_data = './theoretical_preprocessed_train_data.tfrecords'
path_theoretical_valid_data = './theoretical_preprocessed_valid_data.tfrecords'

size_theoretical_train_dataset = 300000
theoretical_train_dataset = tf.data.TFRecordDataset(path_theoretical_train_data).take(size_theoretical_train_dataset)
theoretical_valid_dataset = tf.data.TFRecordDataset(path_theoretical_valid_data).map(parse_function1)

# Set batchs
BATCH_SIZE = 64
NUM_BATCHS = int(size_theoretical_train_dataset / BATCH_SIZE)
theoretical_train_batches = (theoretical_train_dataset
                             .map(parse_function1)
                             .padded_batch(BATCH_SIZE)
                             .prefetch(tf.data.AUTOTUNE))

feature_description2 = {
    'sequence': tf.io.VarLenFeature(tf.int64),
    'intensity': tf.io.VarLenFeature(tf.int64),
    'mz': tf.io.VarLenFeature(tf.int64),
}


def parse_function2(example_proto):
    parsed_example = tf.io.parse_single_example(example_proto, feature_description2)
    mz = parsed_example['mz'].values
    intensity = parsed_example['intensity'].values
    sequence = parsed_example['sequence'].values
    return mz, intensity, sequence


path_real_train_data = './real_preprocessed_train_data.tfrecords'
path_real_valid_data = './real_preprocessed_valid_data.tfrecords'

size_real_train_dataset = 300000
real_train_dataset = tf.data.TFRecordDataset(path_real_train_data).take(size_real_train_dataset)
real_valid_dataset = tf.data.TFRecordDataset(path_real_valid_data).map(parse_function2)

# Set batchs
BATCH_SIZE = 64
NUM_BATCHS2 = int(size_real_train_dataset / BATCH_SIZE)
real_train_batches = (real_train_dataset
                      .map(parse_function2)
                      .padded_batch(BATCH_SIZE)
                      .prefetch(tf.data.AUTOTUNE))


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

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


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

modified_transformer = ModifiedTransformer(
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dff=DFF,
    intensity_vocab_size=12000,
    target_vocab_size=30,
    positional_encoding_target = 50,
    dropout_rate=DROPOUT_RATE)

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


@tf.function(input_signature=train_step_signature)
def train_step(input, target):
    target_input = target[:, :-1]
    target_real = target[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(input, target_input)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(input, target_input,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)

        loss = loss_function(target_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(target_real, predictions))


train_step_signature_transfer = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


# @tf.function(input_signature=train_step_signature_transfer)
# def train_step_transfer_learn(input, intensity, target):
#     target_input = target[:, :-1]
#     target_real = target[:, 1:]
#     enc_padding_mask, combined_mask, dec_padding_mask = create_masks(input, target_input)
#
#     with tf.GradientTape() as tape:
#         enc1_output = encoder1(input, False, enc_padding_mask)
#
#         '''
#         전처리하기
#         dense로 intensity 적용 ?
#
#         '''
#         tf.reshape(intensity, [BATCH_SIZE, len(enc1_output[1]), 1])
#         enc1_output = tf.concat([enc1_output, intensity], axis=3)
#         enc2_output = encoder2(enc1_output, intensity, True, enc_padding_mask)
#
#         predictions, _ = decoder(target_input,
#                                  enc2_output,
#                                  False,
#                                  combined_mask,
#                                  dec_padding_mask)
#
#         loss = loss_function(target_real, predictions)
#
#     gradients = tape.gradient(loss, encoder2.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, encoder2.trainable_variables))
#
#     train_loss(loss)
#     train_accuracy(accuracy_function(target_real, predictions))


# def evaluate_aminoacid_level(dataset):
#     batch_size = 200
#     num_batchs = 0
#     accuracy = 0
#     loss = 0
#     dataset_batchs = dataset.padded_batch(batch_size = batch_size, drop_remainder=True)
#
#     for batch, (input, target) in enumerate(dataset_batchs):
#         num_batchs = batch+1
#
#         target_input = target[:, :-1]
#         target_real = target[:, 1:]
#         enc_padding_mask, combined_mask, dec_padding_mask = create_masks(input, target_input)
#
#         predictions, _ = transformer(input, target_input,
#                                    False,
#                                    enc_padding_mask,
#                                    combined_mask,
#                                    dec_padding_mask)
#         loss += loss_function(target_real, predictions)
#         accuracy += accuracy_function(target_real, predictions)
#
#     return loss/num_batchs, accuracy/num_batchs


for inp, intensity, tar in real_train_batches:
    #print(intensity)
    target_input = tar[:, :-1]
    target_real = tar[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, target_input)

    encoder1_output = transformer.encoder(inp, False, enc_padding_mask)
    decoder1_output, _ = transformer.decoder(target_input, encoder1_output, False, combined_mask,
                                                        dec_padding_mask)
    print(encoder1_output.shape)
    print(decoder1_output.shape)
    predict, _ = modified_transformer(encoder1_output, decoder1_output, intensity, BATCH_SIZE, D_MODEL, True,
                                      enc_padding_mask, combined_mask, dec_padding_mask)
    print(predict.shape)
    # enc1_output = transformer.encoder(inp, False, enc_padding_mask)
    # print(enc1_output.shape)
    # intensity = tf.reshape(intensity, [BATCH_SIZE, len(enc1_output[1]), 1])
    # print(intensity.shape)
    # #intensity = tf.repeat(intensity, repeats=D_MODEL, axis=2)
    # intensity = tf.cast(intensity, tf.float32)
    # enc1_output1 = tf.concat([enc1_output, intensity], axis=2)
    # print(enc1_output1.shape)
    # enc1_output1 = tf.keras.layers.Dense(D_MODEL)(enc1_output1)
    # print(enc1_output1.shape)
    break