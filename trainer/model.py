import argparse
import tensorflow as tf

def create_model():
	"""Create a model to be used by tasks"""
	# parser = argparse.ArgumentParse()

	# parser.add_argument()
	return Model()

class GraphMode(object):
    TRAIN = 0
    EVAL = 1
    PREDICTION = 2

class Model(object):
    """Tensorflow model"""

    def __init__(self):
    	self.tensors = None

    def read_data(self, datapaths):
        """Create a pipeline loading data

        Args:
            datapaths: a  list of file paths of TFRecord files
        """
        filename_queue = tf.train.string_input_producer(datapaths, num_epochs=1)
        
        reader = tf.TFRecordReader()
        key, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(serialized_example,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string)
            }
        )

        # read images 
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image.set_shape([784])
        image = tf.cast(image, tf.float32) # * (1/255) 

        # read labels
        label = tf.decode_raw(features['label'], tf.float64)
        label.set_shape([10])
        label = tf.cast(label, tf.float32)

        image_batch, label_batch = tf.train.shuffle_batch([image, label], 
            batch_size=100,
            capacity=4000, 
            min_after_dequeue=1000)

        return image_batch, label_batch

    def add_operations(self, images, labels, keep_prob=1.0):
        """Add operations of a CNN model

        Args:
            images: 2d images
            labels: labels
            keep_prob: keep probability
        """
        conv1 = conv_layer(images, shape=[5,5,1,32])
        conv1_pool = max_pool_2x2(conv1)

        conv2 = conv_layer(conv1, shape=[5,5,32,64])
        conv2_pool = max_pool_2x2(conv2)

        conv2_flat = tf.contrib.layers.flatten(images)
        full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))

        full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

        logits = full_layer(full1_drop, 10)

        return logits

    def build_graph(self, data_paths, graph_mode):
        """Build a graph 
        Args:
            data_paths: a list of files of input
            graph_mode: graph mode for train, test or prediction
        """

        tensors = Tensors()
        images, labels = self.read_data(data_paths)
        images = tf.reshape(images, [-1, 28, 28, 1])

        keep_prob = 1.0
        # if graph_mode == GraphMode.TRAIN:
        #     keep_prob = 0.5

    	logits = self.add_operations(images, labels, keep_prob=keep_prob)
        
        # for training
        if graph_mode == GraphMode.TRAIN:
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss_mean = tf.reduce_mean(loss)
            tensors.train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
            tensors.global_step = tf.Variable(0, name='global_step', trainable=False)
            tf.summary.scalar('loss', loss_mean)

        # for evaluation
        self.accuracy_updates, self.accuracy_op = calculate_accuracy(logits, labels)
        tf.summary.scalar('accuracy', self.accuracy_op)

        tensors.metric_updates = self.accuracy_updates
        tensors.metric_values = [self.accuracy_op] + self.accuracy_updates

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
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
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
    w = weight_variable([in_size, size])
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