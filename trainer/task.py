import os
import tensorflow as tf 
import model as modellib
import argparse

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


def train():
    save_dir = './save_dir1'

    filename = os.path.join(save_dir, 'train.tfrecord')
    filename_queue = tf.train.string_input_producer([filename], num_epochs=10)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # features = tf.parse_single_example(serialized_example,
    # 	features={'image_raw': tf.FixedLenFeature([], tf.string),'label': tf.FixedLenFeature([], tf.int64)})

    features = tf.parse_single_example(serialized_example,
    	features={
    		'image_raw': tf.FixedLenFeature([], tf.string),
    		'label': tf.FixedLenFeature([], tf.string)
    	}
    )

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label_raw = tf.decode_raw(features['label'], tf.float64)
    # label_raw = tf.cast(features['label'], tf.int64)

    # label_raw.set_shape([10])
    image.set_shape([784])

    image = tf.cast(image, tf.float32)  #* (1/255) 

    label_raw.set_shape([10])

    label = tf.cast(label_raw, tf.float32)
    # label = label_raw

    image_batch, labels_batch = tf.train.shuffle_batch([image, label], batch_size=100,
    	capacity=2000, min_after_dequeue=1000)

    image_2d = tf.reshape(image_batch, [-1, 28, 28, 1])


    # labels_2d = tf.reshape(labels_batch, [-1, 10])

    conv1 = conv_layer(image_2d, shape=[5,5,1,32])
    onv1_pool = max_pool_2x2(conv1)

    conv2 = conv_layer(conv1, shape=[5,5,32,64])
    conv2_pool = max_pool_2x2(conv2)

    conv2_flat = tf.contrib.layers.flatten(conv2_pool)

    # conv2_flat = tf.contrib.layers.flatten(image_2d)

    # full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))

    # keep_prob = tf.placeholder(tf.float32)
    # keep_prob = 0.5
    # full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

    y_pred = full_layer(conv2_flat, 10)

    print("y_pred", y_pred.shape, y_pred.dtype)
    print("label_batch", labels_batch.shape)
    # print(labels_2d.shape)

    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=labels_batch)

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=labels_batch)

    # loss = tf.reduce_mean(loss)
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=labels_batch))

    # correct_prediction=tf.equal( tf.argmax(y_pred, 1), labels_batch)

    	# pred = tf.argmax(y_conv,1)

    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    loss_mean = tf.reduce_mean(loss)

    train_op = tf.train.AdamOptimizer().minimize(loss)

    sess = tf.Session()

    init = tf.global_variables_initializer()

    sess.run(init)

    init = tf.local_variables_initializer()

    sess.run(init)

    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
    	step = 0
    	while not coord.should_stop():
    		step += 1
    		# print("step: ", step)
    		sess.run([train_op])
    		if step % 500 == 0:
    			# print(sess.run([tf.reduce_sum(y_pred)]))

    			loss_mean_val, acc = sess.run([loss_mean, loss_mean])
    			print(step, loss_mean_val, acc)

    except tf.errors.OutOfRangeError:
    	print("done")
    finally:
    	coord.request_stop()

    coord.join(threads)

    sess.close()

class Trainer(object):
    """Performs model training and optionally evaluation"""

    def __init__(self, args, model):
        self.args = args
        self.model = model

    def run_training(self):
        self.train_data_file = os.path.join(self.args.train_data_path, 'train.tfrecord')
        print(self.train_data_file)

def run(model, argv):
    """run a training"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_data_path',
        type=str,
        action='append',
        help='The path to the training data files.')
    parser.add_argument(
        '--output_path',
        type=str,
        action='append',
        help='The path to which checkpoints and other outputs shoudl be saved')

    args, _ = parser.parse_known_args(argv)

    dispatch(model, args)

def dispatch(args, model):
    Trainer(args, model).run_training()    


def main(_):
    # cnn =  modellib.create_model()
    # run(cnn, argv)
    pass

if __name__ == '__main__':
    tf.app.run()
    print(flags)