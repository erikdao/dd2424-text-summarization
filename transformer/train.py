from utils.pickle import *
from dataload.dataloader import *
import time

from utils.glove_embedding import *
from transformer.transformer_model import *

EMBEDDING_FILE = '../preprocess/glove.6B.50d.txt'
EPOCHS = 2
HEADS = 2 # default = 8
N = 2 # default = 6
DIMFORWARD = 512

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    print("load embeddings...")
    word_emb_size, embeddings = load_glove_embeddings(EMBEDDING_FILE)

    inputs = mappings['inputs']
    labels = mappings['labels']
    mappings_train = {'inputs': inputs[:4], 'labels': labels[:4]}
    mappings_val = {'inputs': inputs[4:6], 'labels': labels[4:6]}
    mappings_test = {'inputs': inputs[6:8], 'labels': labels[6:8]}

    print("Loading train loader...")
    train_loader = create_dataloader_glove(
        mappings = mappings_train,
        word2index = word2index,
        embeddings = embeddings,
        word_emb_size = word_emb_size,
        batch_size = 2,
        shuffle=True
    )
    print("Loading val loader...")
    val_loader = create_dataloader_glove(
        mappings = mappings_val,
        word2index = word2index,
        embeddings = embeddings,
        word_emb_size = word_emb_size,
        batch_size = 2,
        shuffle=True
    )
    print("Loading test loader...")
    test_loader = create_dataloader_glove(
        mappings = mappings_test,
        word2index = word2index,
        embeddings = embeddings,
        word_emb_size = word_emb_size,
        batch_size = 2,
        shuffle=True
    )

    print("Init. model...")
    transformer = SummaryTransformer(
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
    print("xavier init...")
    for p in transformer.transformer_encoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for p in transformer.transformer_decoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for p in transformer.generator.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    transformer = transformer.to(device)

    print("model init completed...")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    print("start training")
    for epoch in range(EPOCHS):
        transformer.train()
        losses = 0
        start_time = time.time()
        for idx, data in enumerate(train_loader): #batches
            src = data['input'].to(device) # indecies
            tgt = data['label'].to(device)
            
            #tgt_input = tgt[:-1, :]
            tgt_input = tgt
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device)
            logits = transformer(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

            optimizer.zero_grad()

            #tgt_out = tgt[1:,:]
            tgt_out = tgt
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()

            optimizer.step()
            losses += loss.item()

        avg_train_loss = losses / len(train_loader)
        ######### VAL #############
        transformer.eval()
        losses = 0
        for idx, data in enumerate(val_loader): 
            src = data['input'].to(device) # indexes
            tgt = data['label'].to(device)

            tgt_input = tgt[:-1, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device)

            logits = transformer(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
            tgt_out = tgt[1:,:]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()
        avg_val_loss = losses / len(val_loader)
        
        ### epoch finished        
        end_time = time.time()
        print((f"Epoch: {epoch}, Train loss: {avg_train_loss:.3f}, Val loss: {avg_val_loss:.3f}, "
          f"Epoch time = {(end_time - start_time):.3f}s"))




if __name__ == '__main__':
    main()
