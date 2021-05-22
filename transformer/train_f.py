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
LEARN_RATE = 0.0001
BATCH_SIZE = 50
TEST = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_IDX = 0

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

def save_checkpoint(transformer, optimizer, train_losses, val_losses, train_accs=None, val_accs=None):
    state = {'state_dict': transformer.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

    torch.save(state, SAVED_MODEL_FILE)
    loss_log = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs,
    }
    with open(SAVED_LOSS_LOG_FILE, 'w') as outfile:
        json.dump(loss_log, outfile)

# helper function
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_attention_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_attention_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_attention_mask, tgt_attention_mask, src_padding_mask, tgt_padding_mask

def create_src_masks(src):
    src_seq_len = src.shape[0]
    src_attention_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    return src_attention_mask, src_key_padding_mask

def create_tgt_masks(tgt):
    tgt_seq_len = tgt.shape[0]
    tgt_attention_mask = generate_square_subsequent_mask(tgt_seq_len)
    tgt_key_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return tgt_attention_mask, tgt_key_padding_mask

def train_epoch(model, train_iter, optimizer, loss_fn):
  model.train()
  losses = 0
  for idx, data in tqdm(enumerate(train_iter)):
    src = data['input'].transpose(0, 1).to(DEVICE)
    tgt = data['label'].transpose(0, 1).to(DEVICE)
    
    tgt_input = tgt[:-1, :]

    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

    logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)

    optimizer.zero_grad()

    tgt_out = tgt[1:,:]
    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
    loss.backward()

    optimizer.step()
    losses += loss.item()
  return losses / len(train_iter)

def evaluate(model, val_iter, loss_fn):
  model.eval()
  losses = 0
  for idx, data in (enumerate(val_iter)):
    src = data['input'].transpose(0, 1).to(DEVICE)
    tgt = data['label'].transpose(0, 1).to(DEVICE)

    tgt_input = tgt[:-1, :]

    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

    logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)
    tgt_out = tgt[1:,:]
    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
    losses += loss.item()
  return losses / len(val_iter)        

def main():
    torch.use_deterministic_algorithms(False)
    torch.cuda.empty_cache()
    print("DEVICE: ", DEVICE)

    input_dir_tok = "./tokenized-padded"
    input_dir_w2i = "./tokenized"
    print("Loading mappings...")
    mappings_pickle = input_dir_tok + "/tokenized-padded"
    mappings = pickle_load(mappings_pickle)
    input_sequence_length = len(mappings['inputs'][0])
    label_sequence_length = len(mappings['labels'][0])
    inputs = mappings['inputs']
    labels = mappings['labels']
    print("Loading word2index...")
    word2index_pickle = input_dir_w2i + "/word2index"
    word2index = pickle_load(word2index_pickle)
    print("load embeddings...")
    glove_word_emb_size, embeddings = load_glove_embeddings(EMBEDDING_FILE)
    # use all, or sub set of data
    if not TEST:
        mappings_train = {'inputs': inputs[:327200], 'labels': labels[:327200]}
        mappings_val = {'inputs': inputs[327200:327200+40900], 'labels': labels[327200:327200+40900]}
        mappings_test = {'inputs': inputs[327200+40900:327200+2*40900], 'labels': labels[327200+40900:327200+2*40900]}
    else:
        mappings_train = {'inputs': inputs[:100], 'labels': labels[:100]}
        mappings_val = {'inputs': inputs[105:110], 'labels': labels[105:110]}

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
    )
    transformer = transformer.to(DEVICE)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=LEARN_RATE, betas=(0.9, 0.98), eps=1e-9) # TODO tune params
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
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
    train_acc_per_epoch = []
    val_acc_per_epoch = []
    print("model init completed...")
    
    print("start training...")
    for epoch in tqdm(range(EPOCHS)):
        start_time = time.time()
        ######### TRAIN ###########
        avg_train_loss = train_epoch(transformer, train_loader, optimizer, loss_fn)
        train_loss_per_epoch.append(avg_train_loss)
        ######### VAL #############
        avg_val_loss = evaluate(transformer, val_loader, loss_fn)
        val_loss_per_epoch.append(avg_val_loss)
        end_time = time.time()
        ### epoch finished
        print((f"Epoch: {epoch}, Train loss: {avg_train_loss:.3f}, Val loss: {avg_val_loss:.3f}"
            f"Epoch time = {(end_time - start_time):.3f}s"))
        ### save epoch every X epochs
        if epoch % SAVE_EPOCHS == 0:
            save_checkpoint(
                transformer, optimizer, avg_train_loss, 
                avg_val_loss
            )
    
    save_checkpoint(
        transformer, optimizer, train_loss_per_epoch, 
        val_loss_per_epoch
    )

if __name__ == '__main__':
    np.random.seed(420)
    main()
