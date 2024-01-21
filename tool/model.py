import jax.numpy as jnp
from jax import random

def shallow_network(activation):
    """
        SNN model, returns a handle of model function.
    """
    
    def model(params, x):
        """
            Input x must have only 1 dimension.
        """
        w1, b1 = params[0]
        w2 = params[1]
        linear1 = jnp.dot(w1, x) + b1
        hidden = activation(linear1)
        linear2 = jnp.dot(w2, hidden)
        return jnp.reshape(linear2, ())

    return model


def deep_fc_network(activation):
    """
        DNN model, fully connected, returns a handle of model function.
    """
    
    def model(params, x):
        """
            Input x must have only 1 dimension.
        """
        hidden = x
        for w, b in params[:-1]:
            outputs = jnp.dot(w, hidden) + b
            hidden = activation(outputs)
        linear_w, linear_b = params[-1]
        return jnp.reshape(jnp.dot(linear_w, hidden) + linear_b, ())
    
    return model
    

def random_layer_params(m, n, generator, key, scale=1e-1, shift=0.):
    w_key, b_key = random.split(key)
    return scale * generator(w_key, (n, m)) + shift, scale * generator(b_key, (n,)) + shift

def normal_init(sizes, key):
    keys = random.split(key, len(sizes))
    generator = random.normal
    params = [random_layer_params(m, n, generator, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]
    return [(params[0][0],params[0][1]),(params[1][0])]

def xavier_init(sizes, key):
    keys = random.split(key, len(sizes))
    generator = random.uniform
    scales = [2*jnp.sqrt(6/(sum(sizes[:-1]))), 2*jnp.sqrt(6/(sum(sizes[1:])))]
    shifts = [-jnp.sqrt(6/(sum(sizes[:-1]))), -jnp.sqrt(6/(sum(sizes[1:])))]
    params = [random_layer_params(m, n, generator, k, s1, s2) \
              for m, n, k, s1, s2 in zip(sizes[:-1], sizes[1:], keys, scales, shifts)]
    return [(params[0][0],params[0][1]),(params[1][0])]

def normal_init_mlayer(sizes, key):
    keys = random.split(key, len(sizes))
    generator = random.normal
    return [random_layer_params(m, n, generator, k, 2*1e-1) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def xavier_init_mlayer(sizes, key):
    keys = random.split(key, len(sizes))
    generator = random.uniform
    scales = [2*jnp.sqrt(6/(sum(sizes[:-1]))), 2*jnp.sqrt(6/(sum(sizes[1:])))]
    shifts = [-jnp.sqrt(6/(sum(sizes[:-1]))), -jnp.sqrt(6/(sum(sizes[1:])))]
    return [random_layer_params(m, n, generator, k, s1, s2) \
              for m, n, k, s1, s2 in zip(sizes[:-1], sizes[1:], keys, scales, shifts)]