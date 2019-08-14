import tensorflow as tf

layer_node = [10, 20, 5, 6, 18]
layer_size = len(layer_node)

def get_weight(input, shape, regularizer):
    weight = tf.Variable(tf.random_normal(shape, seed=1, stddev=1))
    bias = tf.Variable(tf.constant(shape=shape[1]))
    output = tf.nn.relu(conv2d(input, weight) + bias)
    tf.add_to_collection('losses', regularizer(weight))
    return output


input_demon = layer_node[0]
def train(input, regularizer):
    for i in range(1, layer_size):
        output_demon = layer_size[i]
        shape = [input_demon, output_demon]
        output = get_weight(input, shape, regularizer)
        input = output
        input_demon = output_demon
    return input

