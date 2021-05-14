"""
Logging configuration
"""
import logging

logging.getLogger('pytorch_lightning').setLevel(logging.DEBUG)
logger = logging.getLogger('pytorch_lightning')