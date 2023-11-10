import tensorflow as tf
from working_memory_model import WorkingMemoryModel
import random
from tensorflow.keras import activations, losses


def make_flashing_mod_game(num_flashes, mod, period = 4):
    flashes = []
    labels = []

    # averaging period
    # flashes += [[0.0, random.random(), random.random(), random.random()] for _ in range(period)] + [[0.0, random.random(), random.random(), random.random()] for _ in range(period)]
    # label = [0.0 for _ in range(mod)]
    # labels += [label[:] for _ in range(period*2)]


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


def make_association_game(num_bits, comb_num = 2, learning_length = 8, gap_length = 8, recall_length = 8, do_comb = None, ban_comb = []):
    if do_comb:
        chosen = do_comb
    else:
        chosen = random.sample(range(num_bits), k=comb_num)
        while set(chosen) in ban_comb:
            chosen = random.sample(range(num_bits), k=comb_num)
    
    trigger = random.choice(chosen)
    inputs = [[1 if i in chosen else 0 for i in range(num_bits+1)] for _ in range(learning_length)] + [([random.random() for _ in range(num_bits)]+[1]) for _ in range(gap_length)] + [[1 if i == trigger else 0 for i in range(num_bits+1)] for _ in range(recall_length)]
    labels = [[1 if i in chosen else 0 for i in range(num_bits)] for _ in range(learning_length)] + [([0 for _ in range(num_bits)]) for _ in range(gap_length)] + [[1 if i in chosen else 0 for i in range(num_bits)] for _ in range(recall_length)]
    inputs = tf.stack(inputs)
    labels = tf.stack(labels)
    return inputs, labels

def training_loop_association(model, num_bits, comb_num = 2, learning_length = 8, gap_length = 8, recall_length = 8, do_comb = None, ban_comb = [], do_print=False, do_train=True):
    
    inputs, label = make_association_game(num_bits, comb_num, learning_length, gap_length, recall_length, do_comb, ban_comb)
    with tf.GradientTape() as tape:
        result = model(inputs)
        if do_print:
            print(inputs, tf.round(result * 100) / 100, label)
        loss = mse(label, result)
        grads = tape.gradient(loss, model.trainable_weights())
    #print(grads)
    if do_train:
        model.apply_gradients(model.trainable_weights(), grads)
    model.reset_memory()
    return loss