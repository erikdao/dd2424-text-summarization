from utils.pickle import *
from dataload.dataloader import *
import time
from tqdm import tqdm
import json

from utils.glove_embedding import *
from transformer.transformer_model import *

EMBEDDING_FILE = '/Users/pacmac/Documents/GitHub/KTH_Projects/dd2424-text-summarization/preprocess/glove.6B.50d.txt'
SAVED_MODEL_FILE = '/Users/pacmac/Documents/GitHub/KTH_Projects/dd2424-text-summarization/transformer/transformer_model'
SAVED_LOSS_LOG_FILE = '/Users/pacmac/Documents/GitHub/KTH_Projects/dd2424-text-summarization/transformer/loss_loggs.json'
SAVE_EPOCHS = 1
EPOCHS = 16
HEADS = 10 # default = 8 have to be dividable by d_model
N = 6 # default = 6
DIMFORWARD = 512
LEARN_RATE = 0.001
BATCH_SIZE = 50
TEST = False

def load_checkpoint(model, optimizer, filename='transformer_model'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        load_flag = True
    else:
        print("=> no checkpoint found at '{}'".format(filename))
        load_flag = False
    return load_flag, model, optimizer

def save_checkpoint(transformer, optimizer, train_loss_per_epoch, val_loss_per_epoch):
    state = {'state_dict': transformer.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

    if os.path.isfile(SAVED_LOSS_LOG_FILE):
        with open(SAVED_LOSS_LOG_FILE) as outfile:
            losses = json.load(SAVED_LOSS_LOG_FILE)
            tloss = losses['train_loss']
            vloss = losses['val_loss']
    else:
        tloss = []
        vloss = []
    loss_log = {
                'train_loss': tloss+train_loss_per_epoch,
                'val_loss': vloss+val_loss_per_epoch
                }
    with open(SAVED_LOSS_LOG_FILE, 'w') as outfile:
        json.dump(loss_log, outfile)
    torch.save(state, SAVED_MODEL_FILE)
    

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.use_deterministic_algorithms(False)
    torch.cuda.empty_cache()
    print("DEVICE: ", device)

    input_dir_tok = "./tokenized-padded"
    input_dir_w2i = "./tokenized"
    print("Loading mappings...")
    mappings_pickle = input_dir_tok + "/tokenized-padded"
    mappings = pickle_load(mappings_pickle)
    input_sequence_length = len(mappings['inputs'][0])
    label_sequence_length = len(mappings['labels'][0])
    print(f"seq length input {input_sequence_length}")
    print(f"seq length label {label_sequence_length}")

    inputs = mappings['inputs']
    labels = mappings['labels']

    print("Loading word2index...")
    word2index_pickle = input_dir_w2i + "/word2index"
    word2index = pickle_load(word2index_pickle)

    print("load embeddings...")
    glove_word_emb_size, embeddings = load_glove_embeddings(EMBEDDING_FILE)

    

    if not TEST:
        mappings_train = {'inputs': inputs[:327200], 'labels': labels[:327200]}
        mappings_val = {'inputs': inputs[327200:327200+40900], 'labels': labels[327200:327200+40900]}
        mappings_test = {'inputs': inputs[327200+40900:327200+2*40900], 'labels': labels[327200+40900:327200+2*40900]}
    else:
        mappings_train = {'inputs': inputs[:10], 'labels': labels[:10]}
        mappings_val = {'inputs': inputs[10:20], 'labels': labels[10:20]}
        mappings_test = {'inputs': inputs[30:40], 'labels': labels[30:40]}

    print("Loading train loader...")
    train_loader = create_dataloader_glove(
        mappings = mappings_train,
        word2index = word2index,
        embeddings = embeddings,
        word_emb_size = glove_word_emb_size,
        batch_size = BATCH_SIZE,
        shuffle=True
    )
    print("Loading val loader...")
    val_loader = create_dataloader_glove(
        mappings = mappings_val,
        word2index = word2index,
        embeddings = embeddings,
        word_emb_size = glove_word_emb_size,
        batch_size = BATCH_SIZE,
        shuffle=True
    )
    """
    print("Loading test loader...")
    test_loader = create_dataloader_glove(
        mappings = mappings_test,
        word2index = word2index,
        embeddings = embeddings,
        word_emb_size = glove_word_emb_size,
        batch_size = 10,
        shuffle=True
    )
    """

    # https://pytorch.org/tutorials/beginner/translation_transformer.html
    print("Init. model...")

    transformer = SummaryTransformer(
        vocab_size=len(word2index.keys()),
        word_emb_size=glove_word_emb_size,
        nhead=HEADS,
        num_encoder_layers=N,
        num_decoder_layers=N, 
        dim_feedforward=DIMFORWARD, 
        max_input_seq_length=input_sequence_length, 
        max_label_seq_length=label_sequence_length, 
        pos_dropout=0.1, 
        trans_dropout=0.1, 
        word2index=word2index,
        embeddings=embeddings
    ).to(device)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=LEARN_RATE, betas=(0.9, 0.98), eps=1e-9) # TODO tune params
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    
    load_flag, transformer, optimizer = load_checkpoint(transformer, optimizer, filename=SAVED_MODEL_FILE)
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
        print("init loss histories")
    train_loss_per_epoch = []
    val_loss_per_epoch = []

    transformer = transformer.to(device)

    print("model init completed...")
    loss_fn = nn.CrossEntropyLoss()
    

    print("start training...")
    for epoch in tqdm(range(EPOCHS)):
        transformer.train()
        losses = 0
        start_time = time.time()
        for idx, data in tqdm(enumerate(train_loader)): #batches
            src = data['input'].to(device) # indecies
            tgt = data['label'].to(device)
            

            # print("input shape")
            # print(tgt.shape)
            # print()

            #tgt_input = tgt[:-1, :]
            tgt_input = tgt
            src_attention_mask, tgt_attention_mask, src_key_padding_mask, tgt_key_padding_mask = create_mask(src, tgt_input, device)
            src_attention_mask = src_attention_mask.to(device)
            tgt_attention_mask = tgt_attention_mask.to(device)
            src_key_padding_mask = src_key_padding_mask.to(device)
            tgt_key_padding_mask = tgt_key_padding_mask.to(device)
            #print("src_mask")
            #print(src_mask)

            logits = transformer(src, tgt_input, src_attention_mask, tgt_attention_mask, src_key_padding_mask, tgt_key_padding_mask)
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
            save_checkpoint(transformer, optimizer, train_loss_per_epoch, val_loss_per_epoch)
    # save final outcome
    train_loss_per_epoch.append(avg_train_loss)
    val_loss_per_epoch.append(avg_val_loss)
    save_checkpoint(transformer, optimizer, train_loss_per_epoch, val_loss_per_epoch)        

if __name__ == '__main__':
    np.random.seed(420)
    main()
