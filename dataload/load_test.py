from utils.pickle import *

def main():
    input_dir = "./tokenized"
    mappings_pickle = input_dir + "/tokenized"
    mappings = pickle_load(mappings_pickle)
    print(mappings)
    word2index_pickle = input_dir + "/word2index"
    word2index = pickle_load(word2index_pickle)
    print(word2index)

if __name__ == '__main__':
    main()
