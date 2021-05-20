"""
Configuration for LSTM experiments
"""


class Config(object):
    EMBEDDING_DIM = 50
    INPUT_LENGTH = 128
    OUTPUT_LENGTH = 32
    HIDDEN_SIZE = 256
    ENCODER_HIDDEN_SIZE = 256
    DECODER_HIDDEN_SIZE = 256
    BATCH_SIZE = 2
    LEARNING_RATE = 0.003
    MOMENTUM = 0.9
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    EPOCHES = 100

    # How often in one training epoch to check
    # for performance on validation set
    VAL_CHECK_STEP = 50

    # After how many training steps we print out
    # model's prediction -- for debugging
    PRINT_PREDICTION_STEP = 20


config = Config()
