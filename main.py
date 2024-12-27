import lib
from absl import app, flags
from ml_collections.config_flags import config_flags
import logging
import os
import random
import pickle
import torch 
from datasets import *
# import tensorflow as tf

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config', None, 'Training configuration.', lock_config=True
)
flags.DEFINE_string('workdir', None, 'Work directory.')
flags.DEFINE_enum('mode', None, ['train', 'eval', 'train_regressor', 'const_opt', 'cond_sample'],
                  'Running mode: train or eval')
flags.DEFINE_string('eval_folder', 'eval', 'The folder name for storing evaluation results')
flags.mark_flags_as_required(['workdir', 'config', 'mode'])


def main(argv):
    # Set random seed
    # run_lib.set_random_seed(FLAGS.config)
    lib.set_random_seed(FLAGS.config)
    if FLAGS.mode == 'train':
        # Create the working directory
        # tf.io.gfile.makedirs(FLAGS.workdir)
        if not os.path.exists(FLAGS.workdir):
            os.makedirs(FLAGS.workdir)
        # Set logger so that it outputs to both console and file
        # Make logging work for both disk and Google Cloud Storage
        gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'a')
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel('INFO')
        lib.sde_train(FLAGS.config, FLAGS.workdir)

                      
                     
    elif FLAGS.mode == 'eval':
        # Run the evaluation pipeline
        lib.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)


if __name__ == '__main__':
    app.run(main)

