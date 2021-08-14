import time

from model import *
from transfer_model import *

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

def evaluate_aminoacid_level_finefuning(dataset):
    batch_size = 200
    num_batchs = 0
    accuracy = 0
    loss = 0
    dataset_batchs = dataset.padded_batch(batch_size = batch_size, drop_remainder=True)

    for batch, (input, intensity, target) in enumerate(dataset_batchs):
        num_batchs = batch+1
        target_input = target[:, :-1]
        target_real = target[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(input, target_input)

        encoder1_output = pretrained_transformer.encoder(input, False, enc_padding_mask)
        encoder2_output = modified_transformer.encoder(encoder1_output, intensity, BATCH_SIZE, D_MODEL, True,
                                                       enc_padding_mask)

        decoder1_output, _ = pretrained_transformer.decoder(target_input, encoder1_output, False, combined_mask,
                                                            dec_padding_mask)
        decoder1_output_layer = tf.Keras.layers.Dense(30)(decoder1_output)
        decoder2_output, _ = modified_transformer.decoder(target_input, encoder2_output, True, combined_mask,
                                                          dec_padding_mask)
        decoder2_output_layer = tf.Keras.layers.Dense(30)(decoder1_output_layer)

        loss = loss_function(target_real, decoder2_output_layer)

        loss += loss_function(target_real, decoder2_output_layer)
        accuracy += accuracy_function(target_real, decoder2_output_layer)

    return loss/num_batchs, accuracy/num_batchs


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


pretrained_transformer = Transformer(
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dff=DFF,
    input_vocab_size=600000,
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

#save pretrain checkpoint
checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(transformer=pretrained_transformer,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

#save finetuning checkpoint
checkpoint_path_finetuning = "./checkpoints/finetuning"
ckpt_finetuning = tf.train.Checkpoint(transformer=modified_transformer,
                           optimizer=optimizer)
ckpt_manager_finetuning = tf.train.CheckpointManager(ckpt_finetuning, checkpoint_path_finetuning, max_to_keep=10)
if ckpt_manager_finetuning.latest_checkpoint:
    ckpt_finetuning.restore(ckpt_manager_finetuning.latest_checkpoint)
    print('Latest checkpoint restored!!')

train_step_signature_finetuning = [
  tf.TensorSpec(shape=(None, None), dtype=tf.int64),
  tf.TensorSpec(shape=(None, None), dtype=tf.int64),
  tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function(input_signature=train_step_signature_finetuning)
def train_step_finetuning(input, intensity, target):

    target_input = target[:, :-1]
    target_real = target[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(input, target_input)

    with tf.GradientTape() as tape:
        #final_layer = tf.Keras.layers.Dense(30)
        encoder1_output = pretrained_transformer.encoder(input, False, enc_padding_mask)
        encoder2_output = modified_transformer.encoder(encoder1_output, intensity, BATCH_SIZE, D_MODEL, True, enc_padding_mask)

        decoder1_output, _ = pretrained_transformer.decoder(target_input, encoder1_output, False, combined_mask, dec_padding_mask)
        decoder1_output_layer = tf.Keras.layers.Dense(30)(decoder1_output)
        decoder2_output, _ = modified_transformer.decoder(target_input, encoder2_output, True, combined_mask, dec_padding_mask)
        decoder2_output_layer = tf.Keras.layers.Dense(30)(decoder1_output_layer)

        loss = loss_function(target_real, decoder2_output_layer)


    gradients = tape.gradient(loss, modified_transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, modified_transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(target_real, decoder2_output_layer))


epoch = 0
while True:
    start = time.time()
    train_loss.reset_states()
    train_accuracy.reset_states()

    for batch, (input, intensity, target) in enumerate(real_train_batches):
        train_step_finetuning(input, intensity, target)

        print('\r',
              f'Epoch {epoch + 1} | batch {batch + 1}/{NUM_BATCHS} Loss {train_loss.result():.3f} Accuracy {train_accuracy.result():.4f}',
              end='')

    print('\r', f'Epoch {epoch + 1} : Time {time.time() - start:.2f}s')
    if (epoch + 1) % 5 == 0:
        ckpt_path_finetuning = ckpt_manager_finetuning.save()
        print('\r', f'Saving checkpoint for epoch {epoch + 1} at {ckpt_path_finetuning}')

    print(f'\tTrain | Loss {train_loss.result():.3f}, Accuracy {train_accuracy.result():.3f}')
    valid_loss, valid_accuracy = evaluate_aminoacid_level_finefuning(real_valid_dataset)
    print(f'\tValid | Loss {valid_loss:.3f}, Accuracy {valid_accuracy:.3f}')

    epoch += 1