from utils.pickle import *
from dataload.dataloader import *

EMBEDDING_FILE = './glove.6B.50d.txt'

def main():
    input_dir_tok = "./tokenized-padded"
    input_dir_w2i = "./tokenized"
    print("Loading mappings...")
    mappings_pickle = input_dir_tok + "/tokenized-padded"
    mappings = pickle_load(mappings_pickle)
    print("Loading word2index...")
    word2index_pickle = input_dir_w2i + "/word2index"
    word2index = pickle_load(word2index_pickle)

    print("Loading train loader...")
    train_loader = create_dataloader_glove(
        mappings = mappings,
        word2index = word2index,
        embed_file = EMBEDDING_FILE,
        batch_size = 100,
        shuffle=True
    )
    for idx, data in enumerate(train_loader):
        input_vec = data['input']
        label_vec = data['label']
        print(input_vec, input_vec.shape)
        print(label_vec, label_vec.shape)
        if idx > 2:
            break

if __name__ == '__main__':
    main()
