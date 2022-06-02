import tensorflow as tf
import numpy as np


x = np.array([[1,2,3],
            [2,3,4]])
intensity = np.array([1,2])

res = tf.concat([x,intensity], axis=1)
print(res)

