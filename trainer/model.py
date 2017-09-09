import argparse
import tensorflow as tf

def create_model():
	"""Create a model to be used by tasks"""
	# parser = argparse.ArgumentParse()

	# parser.add_argument()
	return Model()


def read_data(train_data_paths):
    filename_queue = tf.train.string_input_producer(train_data_paths, num_epochs=3)
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,
    	features={
    		'image_raw': tf.FixedLenFeature([], tf.string),
    		'label': tf.FixedLenFeature([], tf.string)
    	}
    )

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label_raw = tf.decode_raw(features['label'], tf.float64)

    image.set_shape([784])

    image = tf.cast(image, tf.float32)  #* (1/255) 

    label_raw.set_shape([10])

    label = tf.cast(label_raw, tf.float32)
    # label = label_raw

    image_batch, labels_batch = tf.train.shuffle_batch([image, label], batch_size=100,
    	capacity=20000, min_after_dequeue=1)

    return image_batch, labels_batch

class GraphMode(object):
    TRAIN = 0
    EVAL = 1
    PREDICTION = 2

class Model(object):
    """A model"""
    def __init__(self):
    	self.tensors = None

    def build_linear(self, images, labels):
        full_layer = tf.contrib.layers.flatten(images)

    def build_cnn(self, images, labels):
        conv1 = conv_layer(images, shape=[5,5,1,32])
        conv1_pool = max_pool_2x2(conv1)

        conv2 = conv_layer(conv1, shape=[5,5,32,64])
        conv2_pool = max_pool_2x2(conv2)

        conv2_flat = tf.contrib.layers.flatten(conv2_pool)

        # conv2_flat = tf.contrib.layers.flatten(image_2d)

        # full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))

        # keep_prob = tf.placeholder(tf.float32)
        # keep_prob = 0.5
        # full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

        logits = full_layer(conv2_flat, 10)

        return logits

    def build_graph(self, data_paths, graph_mode):
        tensors = Tensors()

        # always use a data pipeline
        images, labels = read_data(data_paths)

        # reshape to 28 X 28
        images = tf.reshape(images, [-1, 28, 28, 1])

    	logits = self.build_cnn(images, labels)
        softmax = tf.nn.softmax(logits)

        # prediction
        prediction = tf.argmax(softmax, 1)
        tensors.predictions = [prediction, softmax]
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

        loss_mean = tf.reduce_mean(loss)

        # for training
        tensors.train = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
        tensors.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.accuracy_updates, self.accuracy_op = calculate_accuracy(logits, labels)
        # if graph_mode == GraphMode.EVAL:
        tf.summary.scalar('accuracy', self.accuracy_op)
        # tf.summary.scalar('acc updates', self.accuracy_updates)

        # if graph_mode == GraphMode.EVAL:
        tf.summary.scalar('loss', loss_mean)

        tensors.metric_updates = self.accuracy_updates
        tensors.metric_values = [loss_mean, self.accuracy_op] + self.accuracy_updates

        return tensors

    def build_train_graph(self, data_paths):
        return self.build_graph(data_paths, GraphMode.TRAIN)

    def build_eval_graph(self, data_paths):
        return self.build_graph(data_paths, GraphMode.EVAL)

class Tensors(object):
  """Holder of base tensors used for training model using common task."""

  def __init__(self):
    self.examples = None
    self.train = None
    self.global_step = None
    self.metric_updates = []
    self.metric_values = []
    self.keys = None
    self.predictions = []
    self.input_jpeg = None


def weight_variable(shape):
    print("weight", shape)
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    print("bias", shape)
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],
                          strides=[1,2,2,1], padding="SAME")

def conv_layer(input, shape):
    w = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, w)+b)

def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    print("in_size", in_size)
    w = weight_variable([in_size, size])
    print(in_size, size)
    b = bias_variable([size])
    return tf.matmul(input, w) + b

def loss(loss_value):
    """Calculates aggregated mean loss."""
    total_loss = tf.Variable(0.0, False)
    loss_count = tf.Variable(0, False)
    total_loss_update = tf.assign_add(total_loss, loss_value)
    loss_count_update = tf.assign_add(loss_count, 1)
    loss_op = total_loss / tf.cast(loss_count, tf.float32)
    return [total_loss_update, loss_count_update], loss_op

def calculate_accuracy(logits, labels):
    """Calculates aggregated accuracy."""

    prediction = tf.argmax(logits, 1)
    observed = tf.argmax(labels, 1)

    is_correct = tf.equal(prediction, observed)
    # labels1 = tf.cast(labels, tf.int32)

    # is_correct = tf.nn.in_top_k(logits, labels1, 1)
    correct = tf.reduce_sum(tf.cast(is_correct, tf.int32))
    incorrect = tf.reduce_sum(tf.cast(tf.logical_not(is_correct), tf.int32))
    
    correct_count = tf.Variable(0, False)
    incorrect_count = tf.Variable(0, False)
    
    correct_count_update = tf.assign_add(correct_count, correct)
    incorrect_count_update = tf.assign_add(incorrect_count, incorrect)
    
    accuracy_op = tf.cast(correct_count, tf.float32) / tf.cast(
      correct_count + incorrect_count, tf.float32)
    
    return [correct_count_update, incorrect_count_update], accuracy_op

# def loss(logits, labels):
#   """Calculates the loss from the logits and the labels.
#   Args:
#     logits: Logits tensor, float - [batch_size, NUM_CLASSES].
#     labels: Labels tensor, int32 - [batch_size].
#   Returns:
#     loss: Loss tensor of type float.
#   """
#   labels = tf.to_int64(labels)
#   cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
#       logits=logits, labels=labels, name='xentropy')
#   return tf.reduce_mean(cross_entropy, name='xentropy_mean')

def format_metric_values(self, metric_values):
    """Formats metric values - used for logging purpose."""

    # Early in training, metric_values may actually be None.
    loss_str = 'N/A'
    accuracy_str = 'N/A'
    try:
        loss_str = '%.3f' % metric_values[0]
        accuracy_str = '%.3f' % metric_values[1]
    except (TypeError, IndexError):
        pass

    return '%s, %s' % (loss_str, accuracy_str)  