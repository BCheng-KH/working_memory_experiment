from inner_memory_layer import InnerMemoryLayer
import tensorflow as tf
from tensorflow.keras import layers, optimizers, Model

class WorkingMemoryModel(Model):
    def __init__(self, shape, learning_rate, do_initial_relu = True, do_final_sigmoid = True, do_hebbian_learning = True):
        super().__init__()
        self.shape = shape
        self.all_layers = []
        self.memory_layers = []
        self.do_hebbian_learning = do_hebbian_learning
        if do_initial_relu:
            self.all_layers.append(layers.Dense(shape[0], activation='relu'))
        else:
            layer = InnerMemoryLayer(shape[0])
            self.all_layers.append(layer)
            self.memory_layers.append(layer)
        for units in shape[1:-1]:
            layer = InnerMemoryLayer(units)
            self.all_layers.append(layer)
            self.memory_layers.append(layer)
        if do_final_sigmoid:
            layer = InnerMemoryLayer(shape[-1], activation='sigmoid')
            self.all_layers.append(layer)
            self.memory_layers.append(layer)
        else:
            layer = InnerMemoryLayer(shape[-1])
            self.all_layers.append(layer)
            self.memory_layers.append(layer)
        self.optimizer = optimizers.Adam(learning_rate=learning_rate)
    def call(self, inputs):
        if self.do_hebbian_learning:
            outputs = []
            for inp in tf.unstack(inputs):
                out = tf.stack([inp])
                prev_layer = None
                for layer in self.all_layers:
                    if isinstance(prev_layer, InnerMemoryLayer) and isinstance(layer, InnerMemoryLayer):
                        out = layer(out, prev_layer)
                    else:
                        out = layer(out)
                    prev_layer = layer
                outputs.append(tf.unstack(out)[0])
            outputs = tf.stack(outputs)
            return outputs
        else:
            out = inputs
            for layer in self.all_layers:
                out = layer(out)
            return out
    def trainable_weights(self):
        output = []
        for layer in self.all_layers:
            output += layer.trainable_weights
        return output
    def apply_gradients(self, weights, grads):
        self.optimizer.apply_gradients(zip(grads, weights))
    def reset_memory(self):
        for layer in self.memory_layers:
            layer.reset_memory()
    def get_memory(self):
        memory = []
        for layer in self.memory_layers:
            memory.append(layer.get_memory())
        return memory
    def set_memory(self, memory):
        for i, layer in enumerate(self.memory_layers):
            layer.set_memory(memory[i])
    def update_learning_rate(self, learning_rate):
        self.optimizer = optimizers.Adam(learning_rate=learning_rate)
    def empty_build(self, input):
        self.call(input)
        self.reset_memory()