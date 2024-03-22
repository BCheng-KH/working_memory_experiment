import tensorflow as tf
from tensorflow.keras import activations, backend
import tensorflow.keras as keras
from tensorflow.keras.layers import Layer
from math import pi, sqrt

z_p_approx_constant = sqrt(8/pi)

def z_p_approx(z):
    return tf.math.sigmoid(z*z_p_approx_constant)

class InnerMemoryLayer(Layer):
    def __init__(self, units, activation='relu', use_bias = True, **kwargs):
        super().__init__(activity_regularizer=None, **kwargs)
        self.units = int(units) if not isinstance(units, int) else units
        if self.units < 0:
            raise ValueError(
                "Received an invalid value for `units`, expected "
                f"a positive integer. Received: units={units}"
            )
        self.activation = activations.get(activation)
        self.use_bias = use_bias
    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or backend.floatx())
        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to a Dense layer "
                "should be defined. Found None. "
                f"Full input shape received: {input_shape}"
            )
        
        ### augmentation ###

        # initialize all trainable variables

        weight_init_limit = tf.math.sqrt(6 / (last_dim + self.units))
        weight_init = tf.random_uniform_initializer(-weight_init_limit, weight_init_limit)
        self.weight = tf.Variable(name="weight",
            initial_value=weight_init(shape=(self.units, last_dim),
            dtype=dtype),
            trainable=True)
        
        augment_init = tf.random_normal_initializer()
        self.augment = tf.Variable(name="augment",
            initial_value=augment_init(shape=(self.units, last_dim),
            dtype=dtype),
            trainable=True)
        if self.use_bias:
            bias_init = tf.zeros_initializer()
            self.bias = tf.Variable(name="bias",
                initial_value=bias_init(shape=(self.units,),
                dtype=dtype),
                trainable=True)

        factory_init = tf.random_normal_initializer()
        self.factory_upper = tf.Variable(name="factory_upper",
            initial_value=factory_init(shape=(self.units, last_dim),
            dtype=dtype),
            trainable=True)
        self.factory_lower = tf.Variable(name="factory_lower",
            initial_value=factory_init(shape=(self.units, last_dim),
            dtype=dtype),
            trainable=True)
        
        # initializing untrainable values

        self.resource = tf.Variable(name="resource", initial_value=tf.ones(shape=(self.units, last_dim),
            dtype=dtype), dtype=dtype, trainable=False)
        
        self.factory = tf.Variable(name="factory", initial_value=tf.ones(shape=(self.units, last_dim),
            dtype=dtype)*0.5, dtype=dtype, trainable=False)
        
        ### Hebbian ###

        self.ra_mean_activation = tf.Variable(name="ra_mean_activation", initial_value=tf.ones(shape=(self.units,),
            dtype=dtype)*0.5, dtype=dtype, trainable=False)
        self.ra_variance = tf.Variable(name="ra_variance", initial_value=tf.ones(shape=(self.units,),
            dtype=dtype), dtype=dtype, trainable=False)
        self.ra_h_factor = tf.Variable(name="ra_h_factor", initial_value=tf.zeros(shape=(self.units, last_dim),
            dtype=dtype), dtype=dtype, trainable=False)
        self.smoothing_factor = 0.1
    
    
        

    def call(self, inputs, prev_layer = None):

        outputs = []
        for inp in tf.unstack(inputs):



            # calculate factory upper and lower bounds
            factory_upper = tf.math.sigmoid(self.factory_upper)/2
            factory_lower = factory_upper*tf.math.sigmoid(self.factory_lower)

            # calculate activity between 0 and 1
            activity = 1-tf.math.exp(-1*tf.math.exp(self.augment)*inp)

            # calculate output = activity * resource * weight + bias
            out = activity*(self.weight + (tf.math.abs(self.weight)*self.ra_h_factor))*self.resource
            if self.use_bias:
                output = tf.reduce_sum(out, axis=1)+self.bias
            else:
                output = tf.reduce_sum(out, axis=1)
            outputs.append(output)

            # calculate restoration factor
            restoration = factory_lower + self.factory*(factory_upper - factory_lower)

            # deplete resource based on activity
            new_resource = self.resource*(1-activity)

            # replenish resource based on restoration factor
            new_resource = new_resource + (1-new_resource)*restoration

            # deplete factory based on restoration, aka, factory will always return to baseline at 0.1 the rate resource is restored
            new_factory = self.factory*(1-0.1*restoration)

            # elevate factory based on activity
            new_factory = new_factory + (1-new_factory)*activity


            # set the current resource and factory to the new resource and factory for the next iteration
            self.factory = new_factory
            self.resource = new_resource

            self.ra_mean_activation = self.ra_mean_activation + (output - self.ra_mean_activation)*self.smoothing_factor
            self.ra_variance = self.ra_variance + (((output - self.ra_mean_activation)**2)-self.ra_variance)*self.smoothing_factor

            if prev_layer:
                self.ra_h_factor = self.ra_h_factor + ((tf.expand_dims((1-2*z_p_approx((output-self.ra_mean_activation)/tf.math.sqrt(self.ra_variance))), -1)*(1-2*z_p_approx((inp-prev_layer.ra_mean_activation)/tf.math.sqrt(prev_layer.ra_variance))))-self.ra_h_factor) * self.smoothing_factor

        
        # assemble the output into a single tensor
        outputs = tf.stack(outputs)

        if self.activation is not None:
            outputs = self.activation(outputs)


        return outputs
    
    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                "The last dimension of the input shape of a Dense layer "
                "should be defined. Found None. "
                f"Received: input_shape={input_shape}"
            )
        return input_shape[:-1].concatenate(self.units)
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "activation": activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "weight": self.weight,
                "augment": self.augment,
                "bias": self.bias,
                "factory_upper": self.factory_upper,
                "factory_lower": self.factory_lower,
                "resource": self.resource,
                "factory": self.factory,
                "ra_mean_activation": self.ra_mean_activation,
                "ra_variance": self.ra_variance,
                "ra_h_factor": self.ra_h_factor,
                "smoothing_factor": self.smoothing_factor,


            }
        )
        return config
        
    def reset_augmentation_memory(self):
        self.resource = tf.ones_like(self.resource)
        
        self.factory = tf.ones_like(self.factory)*0.5
    def reset_hebbian_memory(self):
        self.ra_mean_activation = tf.ones_like(self.ra_mean_activation)*0.5
        self.ra_variance = tf.ones_like(self.ra_variance)
        self.ra_h_factor = tf.zeros_like(self.ra_h_factor)
    def reset_memory(self):
        self.reset_augmentation_memory()
        self.reset_hebbian_memory
    def get_memory(self):
        return (self.resource, self.factory, self.ra_mean_activation, self.ra_variance, self.ra_h_factor)
    def set_memory(self, memory):
        self.resource, self.factory, self.ra_mean_activation, self.ra_variance, self.ra_h_factor = memory