from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf
import os
from scipy import misc 
import cv2

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate = 0.001
num_steps = 2000
batch_size = 64

image_dim = 784
hidden_dim = 521
latent_dim = 2

###########################################

# Obtain images paths
path = "frames/"

imagepaths = list()
walk = os.walk(path).__next__()

for sample in walk[2]:
    if sample.endswith(".png"):
        imagepaths.append(os.path.join(path, sample))

images = []
for img in imagepaths:
    i = misc.imread(img, flatten=True).flatten()
    i = i/255
    images.append(i)

# images = np.asarray(images)
# dataset = tf.data.Dataset.from_tensor_slices(images).batch(batch_size).repeat()

def create_batches():
    while (True):
        for i in range(0,len(images),batch_size):
            yield(images[i:i+batch_size])

iter = create_batches()

########################################################

def glo_rot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0]/2.))

weights = {
    'encoder_h1':  tf.Variable(glo_rot_init([image_dim, hidden_dim])),
    'z_mean': tf.Variable(glo_rot_init([hidden_dim, latent_dim])),
    'z_std': tf.Variable(glo_rot_init([hidden_dim, latent_dim])),
    'decoder_h1': tf.Variable(glo_rot_init([latent_dim, hidden_dim])),
    'decoder_out': tf.Variable(glo_rot_init([hidden_dim, image_dim]))
}

biases = {
    'encoder_b1': tf.Variable(glo_rot_init([hidden_dim])),
    'z_mean': tf.Variable(glo_rot_init([latent_dim])),
    'z_std': tf.Variable(glo_rot_init([latent_dim])),
    'decoder_b1': tf.Variable(glo_rot_init([hidden_dim])),
    'decoder_out': tf.Variable(glo_rot_init([image_dim]))
}

input_image = tf.placeholder(tf.float32, shape=[None, image_dim])
encoder = tf.matmul(input_image, weights['encoder_h1']) + biases['encoder_b1']
encoder = tf.nn.tanh(encoder)
z_mean = tf.matmul(encoder, weights['z_mean']) + biases['z_mean']
z_std = tf.matmul(encoder, weights['z_std']) + biases['z_std']

eps = tf.random_normal(tf.shape(z_std), dtype=tf.float32, mean=0., stddev=1.0, name='epsilon')
z = z_mean + tf.exp(z_std/2) * eps

decoder = tf.matmul(z, weights['decoder_h1']) + biases['decoder_b1']
decoder = tf.nn.tanh(decoder)
decoder = tf.matmul(decoder, weights['decoder_out']) + biases['decoder_out']
decoder = tf.nn.sigmoid(decoder)

def vae_loss(x_reconstructed, x_true):
    encode_decode_loss = x_true * tf.log(x_reconstructed) + (1 - x_true) * tf.log(1 - x_reconstructed)
    encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)
    kl_div_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
    kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
    return tf.reduce_mean(encode_decode_loss + kl_div_loss)

loss_op = vae_loss(decoder, input_image)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.1)
train_op = optimizer.minimize(loss_op)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1, num_steps+1):
    batch_x = iter.__next__()
    # batch_x, _ = mnist.train.next_batch(batch_size)
    feed_dict = {input_image: batch_x}
    _, l = sess.run([train_op, loss_op], feed_dict=feed_dict)
    if i % 1000 == 0 or i == 1:
        print('Step %i, Loss %f' % (i, l))

##########################

noise_input = tf.placeholder(tf.float32, shape=[None, latent_dim])

decoder = tf.matmul(noise_input, weights['decoder_h1']) + biases['decoder_b1']
decoder = tf.nn.tanh(decoder)
decoder = tf.matmul(decoder, weights['decoder_out']) + biases['decoder_out']
decoder = tf.nn.sigmoid(decoder)

n = 20
x_axis = np.linspace(-3, 3, n)
y_axis = np.linspace(-3, 3, n)

canvas = np.empty((28 *n, 28 * n))
for i, yi in enumerate(x_axis):
    for j, xi in enumerate(y_axis):
        z_mu = np.array([[xi, yi]] * batch_size)
        x_mean = sess.run(decoder, feed_dict={noise_input: z_mu})
        canvas[(n - i - 1) * 28:(n - i) * 28, j * 28:(j + 1) * 28] = x_mean[0].reshape(28, 28)

# plt.figure(figsize=(8, 10))
# Xi, Yi = np.meshgrid(x_axis, y_axis)
plt.imshow(canvas, origin="upper", cmap="gray")
plt.show()


##################

canvas = np.empty((28*5, 28*2))

input_image = tf.placeholder(tf.float32, shape=[None, image_dim])
encoder = tf.matmul(input_image, weights['encoder_h1']) + biases['encoder_b1']
encoder = tf.nn.tanh(encoder)
z_mean = tf.matmul(encoder, weights['z_mean']) + biases['z_mean']
z_std = tf.matmul(encoder, weights['z_std']) + biases['z_std']

eps = tf.random_normal(tf.shape(z_std), dtype=tf.float32, mean=0., stddev=1.0, name='epsilon')
z = z_mean + tf.exp(z_std/2) * eps

decoder = tf.matmul(z, weights['decoder_h1']) + biases['decoder_b1']
decoder = tf.nn.tanh(decoder)
decoder = tf.matmul(decoder, weights['decoder_out']) + biases['decoder_out']
decoder = tf.nn.sigmoid(decoder)

path = "vr_on/"

imagepaths = list()
walk = os.walk(path).__next__()

for sample in walk[2]:
    if sample.endswith(".png"):
        imagepaths.append(os.path.join(path, sample))

images = []
canvas = np.zeros(28*2)
for img in imagepaths:
    i = misc.imread(img, flatten=True).flatten().reshape(1, image_dim)
    i = i/255
    imgcp = i
    canvas
    o = sess.run(decoder, feed_dict={input_image: i})
    out = o.reshape(28, 28)
    canvas = np.vstack((canvas, np.hstack((i.reshape(28, 28), out))))

plt.imshow(canvas, origin="upper", cmap="gray")
plt.show()
