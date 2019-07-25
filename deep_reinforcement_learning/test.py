import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


#X and Z will have the real data and generated data respectlvely

X = tf.placeholder(tf.float32, shape=[None, 784])
Z = tf.placeholder(tf.float32, shape=[None, 100])

def show_images(images):
    images = np.reshape(images, [images.shape[0], -1])
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    return


def generator(z):
    with tf.variable_scope("generator"):
        init = tf.contrib.layers.xavier_initializer()
        h1 = tf.layers.dense(inputs=z, units=128, activation=tf.nn.relu, kernel_initializer=init, use_bias=True)
        out = tf.layers.dense(inputs=h1, units=784, activation=tf.nn.tanh, kernel_initializer=init, use_bias=True)

        return out


def discriminator(x):
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        init = tf.contrib.layers.xavier_initializer()
        h1 = tf.layers.dense(inputs=x, units=128, activation=tf.nn.relu, kernel_initializer=init, use_bias=True)
        logits = tf.layers.dense(inputs=h1, units=1, kernel_initializer=init, use_bias=True)

        return logits

def sample_Z(r, c):
    return np.random.uniform(-1., 1., size=[r, c])

G_sample = generator(Z)
logits_real = discriminator(X)
logits_fake = discriminator(G_sample)
print(G_sample.shape, X.shape)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real),
                                                                     logits=logits_real))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_fake),
                                                                     logits=logits_fake))
D_loss = D_loss_real +D_loss_fake

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_fake),
                                                                logits=logits_fake))

D_real = tf.nn.sigmoid(logits_real)
D_fake = tf.nn.sigmoid(logits_fake)

D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))

D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

D_solver = tf.train.AdamOptimizer(learning_rate=1e-3,beta1=0.5).minimize(D_loss, var_list=D_vars)
G_solver = tf.train.AdamOptimizer(learning_rate=1e-3,beta1=0.5).minimize(G_loss, var_list=G_vars)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

print("Initial generated images")
samples = sess.run(G_sample, feed_dict={Z: sample_Z(128, 100)}) # To generate the
fig = show_images(samples[:16])
plt.show()
print()

for it in range(50000):

    if it % 1000 == 0:
        samples = sess.run(G_sample, feed_dict={Z: sample_Z(128, 100)})
        fig = show_images(samples[:16])
        plt.show()
        print()

    x, _ = mnist.train.next_batch(128)
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: x, Z: sample_Z(128, 100)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(128, 100)})

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'.format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()