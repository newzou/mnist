import argparse

def create_model():
	"""Create a model to be used by tasks"""
	parser = argparse.ArgumentParse()

	parser.add_argument()
	return None


def read_data(train_data_paths):
    filename_queue = tf.train.string_input_producer([filename], num_epochs=10)
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
    	capacity=2000, min_after_dequeue=1000)

    return images, labels

class Model(object):
    """A model"""
    def __init__(self):
    	self.tensors = None

    def build_cnn(self, images, labels):
        conv1 = conv_layer(image_2d, shape=[5,5,1,32])
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

    def build_graph(self, train_data_paths):
        images, labels = read_data(train_data_paths)

        # reshape to 28 X 28
        images = tf.reshape(images, [-1, 28, 28, 1])

    	logits = self.build_cnn(images, labels)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

        loss_mean = tf.reduce_mean(loss)

        train_op = tf.train.AdamOptimizer().minimize(loss)

        tensors = train_op
        return tensors

    # def build_train_graph(self, train_data_paths):
    #     return self.build_graph(train_data_paths)