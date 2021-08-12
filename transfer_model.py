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

  def call(self, x, intensity, training, mask):

    # adding embedding and position encoding.
    intensity = self.embedding(intensity)  # (batch_size, intensity_seq_len, d_model)
    x += intensity

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)

    return x  # (batch_size, input_seq_len, d_model)


class ModifiedTransformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff,
               intensity_vocab_size, target_vocab_size,
               positional_encoding_target, dropout_rate=0.1):
      super(ModifiedTransformer, self).__init__()

      self.encoder = Encoder2(num_layers, d_model, num_heads, dff,
                              intensity_vocab_size, dropout_rate)

      self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                             target_vocab_size, positional_encoding_target, dropout_rate)

      self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, input, intensity, target, training, enc_padding_mask,
           look_ahead_mask, dec_padding_mask):
      enc_output = self.encoder(input, intensity, training, enc_padding_mask)  # (batch_size, input_seq_len, d_model)

      # dec_output.shape == (batch_size, tar_seq_len, d_model)
      dec_output, attention_weights = self.decoder(
          target, enc_output, training, look_ahead_mask, dec_padding_mask)

      final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

      return final_output, attention_weights