"""
Configuration for LSTM experiments
"""


class Config(object):
    INPUT_LENGTH = 128
    OUTPUT_LENGTH = 20
    HIDDEN_SIZE = 256
    ENCODER_HIDDEN_SIZE = 256
    DECODER_HIDDEN_SIZE = 256
    BATCH_SIZE = 100
    LEARNING_RATE = 3e-2
    MOMENTUM = 0.9
    PAD_TOKEN = "<PAD>"
    EOS_TOKEN = "<UNK>"
    EPOCHES = 100

    # How often in one training epoch to check
    # for performance on validation set
    VAL_CHECK_STEP = 1000


config = Config()
