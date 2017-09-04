from __future__ import print_function
import os
import tensorflow as tf 
from tensorflow.contrib.learn.python.learn.datasets import mnist

tmp_dir='./tmp'

datasets = mnist.read_data_sets(tmp_dir, dtype=tf.uint8, reshape=False, validation_size=1000, one_hot=True)

data_splits = ['train', 'test', 'validation']

save_dir = './mnistdata'

for i in range(len(data_splits)):
	dataset = datasets[i]

	filename=os.path.join(save_dir, data_splits[i] + '.tfrecord')

	writer = tf.python_io.TFRecordWriter(filename)

	for index in range(dataset.images.shape[0]):
		# print(dataset.images[index].dtype)
		# print(dataset.labels[index].dtype)

		# break

		image = dataset.images[index].tostring()
		label = dataset.labels[index].tostring()
		# print(dataset.labels[index])
		# break
		example = tf.train.Example(features=tf.train.Features(
				feature={
					# 'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[dataset.images.shape[1]])),
					# 'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[dataset.images.shape[2]])),
					# 'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[dataset.images.shape[3]])),
					'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
					'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))
					# 'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(dataset.labels[index])]))
				},
			)
		)
		writer.write(example.SerializeToString())

	writer.close()

					



