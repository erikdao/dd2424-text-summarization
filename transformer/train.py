from utils.pickle import *
from dataload.dataloader import *

from utils.glove_embedding import *
from transformer.transformer_model import SummaryTransformer

EMBEDDING_FILE = '../preprocess/glove.6B.50d.txt'
EPOCHS = 4
HEADS = 2 # default = 8
N = 2 # default = 6
DIMFORWARD = 1024

def main():
    input_dir_tok = "./tokenized-padded"
    input_dir_w2i = "./tokenized"
    print("Loading mappings...")
    mappings_pickle = input_dir_tok + "/tokenized-padded"
    mappings = pickle_load(mappings_pickle)
    sequence_length = len(mappings['inputs'][0])
    print(f"seq length {sequence_length}")

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

    print("load embeddings...")
    word_emb_size, embeddings = load_glove_embeddings(EMBEDDING_FILE)

    print("Train model...")
    model = SummaryTransformer(
        vocab_size=len(word2index.keys()),
        d_model=word_emb_size,
        nhead=HEADS,
        num_encoder_layers=N,
        num_decoder_layers=N, 
        dim_feedforward=DIMFORWARD, 
        max_seq_length=sequence_length, 
        pos_dropout=0.1, 
        trans_dropout=0.1, 
        word2index=word2index,
        embeddings=embeddings
    )
    print("model init...")


    # in_emb = glove_embedding.forward(input_vec)
    for epoch in range(EPOCHS):
        for idx, data in enumerate(train_loader): #batches
            input_vec = data['input'] # indecies
            label_vec = data['label']

            model.forward(input_vec, label_vec)


if __name__ == '__main__':
    main()
