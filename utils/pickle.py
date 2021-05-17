import pickle
import os

def pickle_save(filename, data):
    "Save data python object to a pickle file"
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load(filename):
    "Load python object from a pickle file"
    with open(filename, 'rb') as handle:
        return pickle.load(handle)
