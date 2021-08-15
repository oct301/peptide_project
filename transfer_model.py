import tensorflow as tf
from model import *

class Encoder2(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff,
               intensity_vocab_size, dropout_rate=0.1):
    super(Encoder2, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(intensity_vocab_size, self.d_model)

    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate)
                       for _ in range(num_layers)]

    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x, intensity, BATCH_SIZE, D_MODEL, training, mask):

    # adding embedding and position encoding.
    # intensity = self.embedding(intensity)  # (batch_size, intensity_seq_len, d_model)
    # x += intensity

    intensity = tf.reshape(intensity, [BATCH_SIZE, len(x[1]), 1])
    # intensity = tf.repeat(intensity, repeats=D_MODEL, axis=2)
    intensity = tf.cast(intensity, tf.float32)
    x = tf.concat([x, intensity], axis=2)
    x = tf.keras.layers.Dense(D_MODEL)(x)

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)

    return x  # (batch_size, input_seq_len, d_model)
class Decoder2(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff,
               target_vocab_size, maximum_position_encoding, dropout_rate=0.1):

    super(Decoder2, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate)
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x, enc_output, training,
           look_ahead_mask, padding_mask):

    #seq_len = tf.shape(x)[1]
    attention_weights = {}

    # x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
    # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    # x += self.pos_encoding[:, :seq_len, :]
    #
    # x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask)

      attention_weights[f'decoder_layer{i+1}_block1'] = block1
      attention_weights[f'decoder_layer{i+1}_block2'] = block2

    # x.shape == (batch_size, target_seq_len, d_model)
    return x, attention_weights

class ModifiedTransformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,
               intensity_vocab_size, target_vocab_size,
               positional_encoding_target, dropout_rate=0.1):
        super(ModifiedTransformer, self).__init__()

        self.encoder = Encoder2(num_layers, d_model, num_heads, dff,
                              intensity_vocab_size, dropout_rate)

        self.decoder = Decoder2(num_layers, d_model, num_heads, dff,
                             target_vocab_size, positional_encoding_target, dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, encoder_input, decoder_input, intensity, BATCH_SIZE, D_MODEL, training, enc_padding_mask,
           look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(encoder_input, intensity, BATCH_SIZE, D_MODEL, training, enc_padding_mask)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(decoder_input, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights