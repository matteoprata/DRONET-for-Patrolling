
# Simple Keras Model

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from wandb.keras import WandbCallback

# Set an experiment name to group training and evaluation
experiment_name = wandb.util.generate_id()

# Start a run, tracking hyperparameters
wandb.init(
    project="intro-demo",
    group=experiment_name,
    config={
        "layer_1": 512,
        "activation_1": "relu",
        "dropout": 0.2,
        "layer_2": 10,
        "activation_2": "softmax",
        "optimizer": "sgd",
        "loss": "sparse_categorical_crossentropy",
        "metric": "accuracy",
        "epoch": 6,
        "batch_size": 32
    })

config = wandb.config