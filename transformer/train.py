from utils.pickle import *
from dataload.dataloader import *
import time
from tqdm import tqdm
import json
import pytorch_warmup as warmup

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
LEARN_RATE = 0.006
BATCH_SIZE = 60
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

def save_checkpoint(transformer, optimizer, train_losses, train_accs,val_losses,val_accs):
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

def accuracy(logits, tgt_input):
    """
    Computes batch accuracy of network output compared with target/label
    Accuracy is sum of correctly guessed words / (sum of sentences lengths - pad mask len)
    """
    # TODO: use CPU
    tgt_input = tgt_input.transpose(0, 1).contiguous() # (B,S)
    # output
    logits_trans = logits.transpose(0, 1).contiguous() # (B,S,V)
    logprobs = F.log_softmax(logits_trans, dim=1) # (B,S,V)
    #print("logprobs ", logprobs.shape)
    max_idxs = logprobs.argmax(dim=2) # (B,S)
    #print("max_idxs ",max_idxs.shape)
    #print("tgt_input ",tgt_input.shape)

    # compare outputs and target label
    equals = torch.eq(max_idxs, tgt_input).int() # (B,S-1)

    # pad away unnecessary comparisons
    pad_mask = (tgt_input != 0).int() # (B,S-1)
    assert equals.shape == pad_mask.shape
    equals_pad = equals * pad_mask # (B,S-1)

    lens_per_sentence = torch.count_nonzero(tgt_input,dim=1) # (B), <pad> has idx 0
    lens_per_batch = torch.sum(lens_per_sentence) # (1)
    equals_per_sentence = torch.sum(equals_pad, dim=1) # (B)
    equals_per_batch = torch.sum(equals_per_sentence) # (1)
    batch_accuracy = float(equals_per_batch / lens_per_batch)
    return batch_accuracy

def train_epoch(model, train_iter, optimizer, loss_fn, scheduler, warmup=False):
  model.train()
  losses = 0
  accs = 0
  for idx, data in tqdm(enumerate(train_iter)):
    # TODO: remooove hardcore
    if warmup and idx < 5000:
        new_lr = sched_exp(LEARN_RATE, idx+1, 5000)
        set_lr(optimizer, new_lr)

    elif warmup and idx >= 5000:
        warmup = False

    optimizer.zero_grad()

    src = data['input'].transpose(0, 1).to(DEVICE)
    tgt = data['label'].transpose(0, 1).to(DEVICE)
    
    tgt_input = tgt[:-1, :]

    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

    logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)

    tgt_out = tgt[1:,:]
    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
    loss.backward()

    optimizer.step()
    if not warmup and scheduler != None:
        scheduler.step()

    losses += loss.item()
    accs += accuracy(logits, tgt_input)
  return losses / len(train_iter), accs / len(train_iter), warmup

def evaluate(model, val_iter, loss_fn):
  model.eval()
  losses = 0
  accs = 0
  for idx, data in (enumerate(val_iter)):
    src = data['input'].transpose(0, 1).to(DEVICE)
    tgt = data['label'].transpose(0, 1).to(DEVICE)

    tgt_input = tgt[:-1, :]

    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

    logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)
    tgt_out = tgt[1:,:]
    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
    losses += loss.item()
    accs += accuracy(logits, tgt_input)
  return losses / len(val_iter), accs / len(val_iter) 

def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr

def sched_exp(end_lr, it, maxit):
    #return start * (end / start) ** pos
    return it * end_lr / maxit

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
        mappings_train = {'inputs': inputs[:BATCH_SIZE], 'labels': labels[:BATCH_SIZE]}
        mappings_val = {'inputs': inputs[BATCH_SIZE:2*BATCH_SIZE], 'labels': labels[BATCH_SIZE:2*BATCH_SIZE]}

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
    #optimizer = torch.optim.Adam(transformer.parameters(), lr=LEARN_RATE, betas=(0.9, 0.98), eps=1e-9) # TODO tune params

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=LEARN_RATE, weight_decay=0.5)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.9999)
    scheduler = None
    
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
    warmup = not TEST
    for epoch in tqdm(range(EPOCHS)):
        start_time = time.time()
        ######### TRAIN ###########
        avg_train_loss, avg_train_acc, warmup = train_epoch(
            transformer, 
            train_loader, 
            optimizer, 
            loss_fn,
            scheduler,
            warmup
        )
        train_loss_per_epoch.append(avg_train_loss)
        train_acc_per_epoch.append(avg_train_acc)
        ######### VAL #############
        avg_val_loss, avg_val_acc = evaluate(transformer, val_loader, loss_fn)
        val_loss_per_epoch.append(avg_val_loss)
        val_acc_per_epoch.append(avg_val_acc)
        end_time = time.time()
        ### epoch finished
        print((f"Epoch: {epoch}, Train loss: {avg_train_loss:.3f}, Train acc: {avg_train_acc:.3f}, Val loss: {avg_val_loss:.3f}, Val acc: {avg_val_acc:.3f}"
            f"Epoch time = {(end_time - start_time):.3f}s"))
        ### save epoch every X epochs
        if epoch % SAVE_EPOCHS == 0:
            save_checkpoint(
                transformer, optimizer, train_loss_per_epoch, train_acc_per_epoch,
                val_loss_per_epoch,val_acc_per_epoch
            )
    
    save_checkpoint(
        transformer, optimizer, train_loss_per_epoch, train_acc_per_epoch,
        val_loss_per_epoch,val_acc_per_epoch
    )

if __name__ == '__main__':
    np.random.seed(420)
    main()
