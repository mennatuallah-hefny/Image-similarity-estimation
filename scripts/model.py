import tensorflow as tf
from keras import layers, Model
from keras.applications import resnet

target_shape = (200, 200)

class DistanceLayer(layers.Layer):
    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)

def create_embedding_model():
    base_cnn = resnet.ResNet50(weights="imagenet", input_shape=target_shape + (3,), include_top=False)
    for layer in base_cnn.layers:
        layer.trainable = False if layer.name != "conv5_block1_out" else True

    flatten = layers.Flatten()(base_cnn.output)
    dense1 = layers.Dense(512, activation="relu")(flatten)
    dense2 = layers.Dense(256, activation="relu")(dense1)
    output = layers.Dense(256)(dense2)
    
    return Model(base_cnn.input, output, name="Embedding")

def create_siamese_network(embedding):
    anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
    positive_input = layers.Input(name="positive", shape=target_shape + (3,))
    negative_input = layers.Input(name="negative", shape=target_shape + (3,))
    
    distances = DistanceLayer()(
        embedding(resnet.preprocess_input(anchor_input)),
        embedding(resnet.preprocess_input(positive_input)),
        embedding(resnet.preprocess_input(negative_input)),
    )
    
    return Model(inputs=[anchor_input, positive_input, negative_input], outputs=distances)
