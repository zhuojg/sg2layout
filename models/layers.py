import math
import tensorflow as tf


def build_mlp(dim_list, activation='relu', batch_norm='none', dropout=0, final_nonlinearity=True):
    mlp = tf.keras.Sequential()
    for i in range(len(dim_list) - 1):
        dim_in, dim_out = dim_list[i], dim_list[i + 1]

        if i == 0:
            mlp.add(tf.keras.Input(shape=(dim_in,)))

        mlp.add(tf.keras.layers.Dense(dim_out))
        is_final_layer = (i == len(dim_list) - 2)

        if not is_final_layer or final_nonlinearity:
            if batch_norm == 'batch':
                mlp.add(tf.keras.layers.BatchNormalization())
            
            if activation == 'relu':
                mlp.add(tf.keras.layers.ReLU())
            elif activation == 'leakyrelu':
                mlp.add(tf.keras.layers.LeakyReLU())
        if dropout > 0:
            mlp.add(tf.keras.layers.Dropout(rate=dropout))

    return mlp