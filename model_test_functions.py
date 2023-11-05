import tensorflow as tf
from working_memory_model import WorkingMemoryModel
import random
from tensorflow.keras import activations, losses


def make_flashing_mod_game(num_flashes, mod, period = 4):
    flashes = []
    labels = []
    for i in range(num_flashes):
        flashes += [[1.0, random.random(), random.random(), random.random()] for _ in range(period)] + [[0.0, random.random(), random.random(), random.random()] for _ in range(period)]
        label = [0.0 for _ in range(mod)]
        label[i%mod] = 1.0
        labels += [label[:] for _ in range(period*2)]

    inputs = tf.stack(flashes)
    label = tf.stack(labels)
    return inputs, label

mse = losses.MeanSquaredError()
def training_loop(model, min_flashes, max_flashes, do_print=False):
    num_flashes = random.randrange(min_flashes, max_flashes)
    #print(num_flashes)
    inputs, label = make_flashing_mod_game(num_flashes, 3)
    with tf.GradientTape() as tape:
        result = model(inputs)
        if do_print:
            print(tf.round(result * 100) / 100, label)
        loss = mse(label, result)
        grads = tape.gradient(loss, model.trainable_weights())
    #print(grads)
    model.apply_gradients(model.trainable_weights(), grads)
    model.reset_memory()
    return loss