import os
import tensorflow as tf 
import model as modellib
import argparse
import sys
import traceback

class Evaluator(object):
    """Load the latest checkpoint and evalulate"""
    def __init__(self, model, saved_path, test_data, output):
        self.model=model
        self.saved_path=saved_path
        self.test_data=test_data
        self.output=output

    def evaluate(self):
        """Run evaluation, return loss and accuracy"""

        # create an evaluation graph
        with tf.Graph().as_default() as graph:
            self.tensors = self.model.build_eval_graph(self.test_data)
            self.summary = tf.summary.merge_all()
            self.saver = tf.train.Saver()

            global_init = tf.local_variables_initializer()
            local_init=tf.global_variables_initializer()

        sess1 = tf.Session(graph=graph)
        sess1.run(local_init)
        sess1.run(global_init)
                
        last_checkpoint = tf.train.latest_checkpoint(self.saved_path)        
        self.saver.restore(sess1, last_checkpoint)

        # for tensorboard
        logdir = os.path.join(self.output, 'test')
        writer = tf.summary.FileWriter(logdir, sess1.graph)

        # build for pipelines
        coord1 = tf.train.Coordinator()
        threads1 = tf.train.start_queue_runners(sess=sess1, coord=coord1)

        try:
            while not coord1.should_stop():
                metrics = sess1.run(self.tensors.metric_values)
        except tf.errors.OutOfRangeError:
            print("finish evaluation")
        finally:
            coord1.request_stop()
        
        coord1.join(threads1)

        print(metrics)
        writer.close()
        return metrics[0]

class Trainer(object):
    """Performs model training and optionally evaluation"""

    def __init__(self, model, train_data, output, test_data):
        self.args = None
        self.model = model
        self.train_data = train_data
        self.output = output
        self.evaluator = Evaluator(model, output, test_data, output)

    def run_training(self):
        with tf.Graph().as_default() as graph:
            tensors = self.model.build_train_graph(self.train_data)        
            init_global = tf.global_variables_initializer()
            init_local = tf.local_variables_initializer()
            self.saver = tf.train.Saver()
            merged = tf.summary.merge_all()

        sess = tf.Session(graph=graph)

        sess.run(init_global)
        sess.run(init_local)

        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        logdir = os.path.join(self.output, 'train')

        writer = tf.summary.FileWriter(logdir, sess.graph)
        
        checkpoint=os.path.join(self.output, 'trained')

        try:
            step = 0
            while not coord.should_stop():
                sess.run([tensors.train])
                step += 1
                if step%10 == 0:
                    self.saver.save(sess, checkpoint)
        except tf.errors.OutOfRangeError:
            print("finish training")
        finally:
            coord.request_stop()
            self.saver.save(sess, checkpoint)

        coord.join(threads)

        sess.close()
        writer.close()

    def evaluate(self):
        self.evaluator.evaluate()
        


def run(model):
    """run a training"""
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--train-data',
        type=str,
        action='append',
        required=True,
        help='The path to the training data files.')

    parser.add_argument(
        '--output',
        type=str,
        action='store',
        required=True,
        help='The path to which checkpoints and other outputs shoudl be saved')

    parser.add_argument(
        '--test-data',
        type=str,
        action='append',
        required=True)

    args=parser.parse_args()

    trainer = Trainer(model, 
        args.train_data,
        args.output, 
        args.test_data)

    trainer.run_training()

    # run test
    cnn1 =  modellib.create_model()

    eval = Evaluator(cnn1, args.output, args.test_data, args.output)
    print("accuracy", eval.evaluate())

def main(_):
    cnn =  modellib.create_model()
    run(cnn)

if __name__ == '__main__':
    tf.app.run()