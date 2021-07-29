import time
import tensorflow as tf
#import Transformer as tfr
import tensor_Transformer as ttfr

feature_description = {
            'sequence': tf.io.VarLenFeature(tf.int64),
            'mz': tf.io.VarLenFeature(tf.int64),
            }

def parse_function(example_proto):
    parsed_example = tf.io.parse_single_example(example_proto,feature_description)
    mz = parsed_example['mz'].values
    sequence = parsed_example['sequence'].values
    return mz, sequence

path_train_data='./preprocessed_train_data.tfrecords'

train_dataset = tf.data.TFRecordDataset(path_train_data)

BATCH_SIZE = 64
train_batches = (train_dataset
                 .map(parse_function)
                 .padded_batch(BATCH_SIZE)
                 .prefetch(tf.data.AUTOTUNE))

learning_rate = ttfr.CustomSchedule(ttfr.dmodel)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

#체크포인트 설정
checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(transformer=ttfr.transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')


# 학습하기
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]
@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = ttfr.create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = ttfr.transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        loss = ttfr.loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, ttfr.transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, ttfr.transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(ttfr.accuracy_function(tar_real, predictions))

EPOCHS=20
#실행
for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    batch=0
    for (inp, outp) in train_batches:
        train_step(inp, outp)

        print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        batch += 1

    #if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

    print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

    print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')