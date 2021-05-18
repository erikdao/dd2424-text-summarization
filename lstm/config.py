"""
Configuration for LSTM experiments
"""

class Config(object):
    INPUT_LENGTH = 128
    OUTPUT_LENGTH = 20
    HIDDEN_SIZE = 128
    ENCODER_HIDDEN_SIZE = 128
    DECODER_HIDDEN_SIZE = 128
    BATCH_SIZE = 100
    LEARNING_RATE = 1e-3
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    EPOCHES = 20

config = Config()