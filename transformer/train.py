from utils.pickle import *
from dataload.dataloader import *
import time

from utils.glove_embedding import *
from transformer.transformer_model import *

EMBEDDING_FILE = '../preprocess/glove.6B.50d.txt'
SAVED_MODEL_FILE = '/Users/pacmac/Documents/GitHub/KTH_Projects/dd2424-text-summarization/transformer/transformer_model'
SAVE_EPOCHS = 5
EPOCHS = 6
HEADS = 5 # default = 8
N = 3 # default = 6
DIMFORWARD = 512

def load_checkpoint(model, optimizer, filename='transformer_model'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        train_loss = checkpoint['loss']['train_loss']
        val_loss = checkpoint['loss']['val_loss']
        load_flag = True
    else:
        print("=> no checkpoint found at '{}'".format(filename))
        load_flag = False
        train_loss = 0.
        val_loss = 0.
    return load_flag, model, optimizer, train_loss, val_loss

def save_checkpoint(transformer, optimizer, train_loss, val_loss):
    state = {'state_dict': transformer.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': {
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }
            }
    torch.save(state, SAVED_MODEL_FILE)

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
    mappings_train = {'inputs': inputs[:40], 'labels': labels[:20]}
    mappings_val = {'inputs': inputs[20:40], 'labels': labels[20:40]}
    mappings_test = {'inputs': inputs[40:60], 'labels': labels[40:60]}

    print("Loading train loader...")
    train_loader = create_dataloader_glove(
        mappings = mappings_train,
        word2index = word2index,
        embeddings = embeddings,
        word_emb_size = word_emb_size,
        batch_size = 10,
        shuffle=True
    )
    print("Loading val loader...")
    val_loader = create_dataloader_glove(
        mappings = mappings_val,
        word2index = word2index,
        embeddings = embeddings,
        word_emb_size = word_emb_size,
        batch_size = 10,
        shuffle=True
    )
    print("Loading test loader...")
    test_loader = create_dataloader_glove(
        mappings = mappings_test,
        word2index = word2index,
        embeddings = embeddings,
        word_emb_size = word_emb_size,
        batch_size = 10,
        shuffle=True
    )

    # https://pytorch.org/tutorials/beginner/translation_transformer.html
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
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.01, betas=(0.9, 0.98), eps=1e-9) # TODO
    load_flag, transformer, optimizer, train_loss, val_loss = load_checkpoint(transformer, optimizer, filename=SAVED_MODEL_FILE)
    if not load_flag:    
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
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    print("start training...")
    for epoch in range(EPOCHS):
        transformer.train()
        losses = 0
        start_time = time.time()
        for idx, data in enumerate(train_loader): #batches
            src = data['input'].to(device) # indecies
            tgt = data['label'].to(device)
            

            # print("input shape")
            # print(tgt.shape)
            # print()

            #tgt_input = tgt[:-1, :]
            tgt_input = tgt
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device)
            #print("src_mask")
            #print(src_mask)

            logits = transformer(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)
            # change batch and sequence dim back
            # print()
            # print("logits")
            # print(logits.shape)
            optimizer.zero_grad()

            #tgt_out = tgt[1:,:]
            tgt_out = tgt
            tgt_out = tgt_out.transpose(0, 1).contiguous()
            # print()
            # print("tgt_out")
            # print(tgt_out.shape)
            #loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss = loss_fn(logits.view(-1, logits.shape[-1]), tgt_out.view(-1))
            # print(f"loss = {loss}")
            #loss = loss_fn(logits, tgt_out)
            #print("back prop...")
            loss.backward()

            optimizer.step()
            losses += loss.item()
            #print(f"loss = {losses}\r", end="")

        avg_train_loss = losses / len(train_loader)
        ######### VAL #############
        transformer.eval()
        losses = 0
        #print("EVALUATING...")
        for idx, data in enumerate(val_loader): 
            src = data['input'].to(device) # indexes
            tgt = data['label'].to(device)

            tgt_input = tgt
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device)
            #print("src_mask")
            #print(src_mask)

            logits = transformer(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)
            # change batch and sequence dim back
            
            #tgt_out = tgt[1:,:]
            tgt_out = tgt
            tgt_out = tgt_out.transpose(0, 1).contiguous()
            loss = loss_fn(logits.view(-1, logits.shape[-1]), tgt_out.view(-1))
            losses += loss.item()
        avg_val_loss = losses / len(val_loader)
        
        ### epoch finished        
        end_time = time.time()
        print((f"Epoch: {epoch}, Train loss: {avg_train_loss:.3f}, Val loss: {avg_val_loss:.3f}, "
          f"Epoch time = {(end_time - start_time):.3f}s"))

        if epoch % SAVE_EPOCHS == 0:
            save_checkpoint(transformer, optimizer, avg_train_loss, avg_val_loss)
    # save final outcome
    save_checkpoint(transformer, optimizer, avg_train_loss, avg_val_loss)        

if __name__ == '__main__':
    np.random.seed(420)
    main()
