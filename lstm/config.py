"""
Configuration for LSTM experiments
"""

class Config(object):
    INPUT_LENGTH = 256
    OUTPUT_LENGTH = 50
    HIDDEN_SIZE = 512
    ENCODER_HIDDEN_SIZE = 512
    DECODER_HIDDEN_SIZE = 512
    BATCH_SIZE = 100

config = Config()