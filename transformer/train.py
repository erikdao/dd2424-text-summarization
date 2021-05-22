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
EPOCHS = 2
HEADS = 10 # default = 8 have to be dividable by d_model
N = 6 # default = 6
DIMFORWARD = 512
LEARN_RATE = 0.001
BATCH_SIZE = 50
TEST = True

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

def save_checkpoint(transformer, optimizer, train_losses, val_losses,train_accs,val_accs):
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
    
def accuracy(logits, tgt_input):
    """
    Computes batch accuracy of network output compared with target/label
    Accuracy is sum of correctly guessed words / (sum of sentences lengths - pad mask len) 
    """
    # TODO: use CPU
    logits_trans = logits.transpose(0, 1).contiguous() # (B,S,V)
    logprobs = F.log_softmax(logits_trans, dim=1) # (B,S,V)
    max_idxs = logprobs.argmax(dim=2) # (B,S)
    equals = torch.eq(max_idxs, tgt_input).int() # (B,S)
    pad_mask = (tgt_input != 0).int() # (B,S)
    assert equals.shape == pad_mask.shape
    equals_pad = equals * pad_mask # (B,S)
    lens_per_sentence = torch.count_nonzero(tgt_input,dim=1) # (B), <pad> has idx 0
    lens_per_batch = torch.sum(lens_per_sentence) # (1)
    equals_per_sentence = torch.sum(equals_pad, dim=1) # (B)
    equals_per_batch = torch.sum(equals_sums) # (1)
    batch_accuracy = float(equals_per_batch / lens_per_batch)
    return batch_accuracy


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
        mappings_train = {'inputs': inputs[:100], 'labels': labels[:100]}
        mappings_val = {'inputs': inputs[200:300], 'labels': labels[200:300]}

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
    train_acc_per_epoch = []
    val_acc_per_epoch = []

    transformer = transformer.to(device)
    print("model init completed...")
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    

    print("start training...")
    for epoch in tqdm(range(EPOCHS)):
        transformer.train()
        losses = 0
        accuracies = 0
        start_time = time.time()
        for idx, data in tqdm(enumerate(train_loader)): #batches
            src = data['input'].to(device) # indecies (B,S)
            tgt = data['label'].to(device) # (B,S)

            tgt_input = tgt
            src_attention_mask, tgt_attention_mask, src_key_padding_mask, tgt_key_padding_mask = create_mask(src, tgt_input, device)
            src_attention_mask = src_attention_mask.to(device)
            tgt_attention_mask = tgt_attention_mask.to(device)
            src_key_padding_mask = src_key_padding_mask.to(device)
            tgt_key_padding_mask = tgt_key_padding_mask.to(device)
            """ check masking
            print()
            print("src")
            print(src.size())
            print(src)
            print()
            print("tgt")
            print(tgt.size())
            print(tgt)
            print()
            print("src_key_padding_mask")
            print(src_key_padding_mask.size())
            print(src_key_padding_mask)
            print()
            print("tgt_key_padding_mask")
            print(tgt_key_padding_mask.size())
            print(tgt_key_padding_mask)
            print()
            print("src_attention_mask")
            print(src_attention_mask.size())
            print(src_attention_mask)
            print()
            print("tgt_attention_mask")
            print(tgt_attention_mask.size())
            print(tgt_attention_mask)
            print()
            """

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
            loss = loss_fn(logits.view(-1, logits.shape[-1]), tgt_out.view(-1)) # changing dimensions back
            # print(f"loss = {loss}")
            #loss = loss_fn(logits, tgt_out)
            #print("back prop...")
            loss.backward()

            optimizer.step()
            losses += loss.item()
            accuracies += accuracy(logits, tgt_input)
           

        # TRAIN LOSS AND ACC
        avg_train_loss = losses / len(train_loader)
        train_loss_per_epoch.append(avg_train_loss)
        train_acc = accuracies / len(train_loader) 
        train_acc_per_epoch.append(train_acc)

        ######### VAL #############
        transformer.eval()
        losses = 0
        accuracies = 0
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
            accuracies += accuracy(logits, tgt_input)

        # VAL LOSS AND ACC
        avg_val_loss = losses / len(val_loader)
        val_loss_per_epoch.append(avg_val_loss)
        val_acc = accuracies / len(val_loader) 
        val_acc_per_epoch.append(val_acc)
        
        ### epoch finished        
        end_time = time.time()
        print((f"Epoch: {epoch}, Train loss: {avg_train_loss:.3f}, Val loss: {avg_val_loss:.3f}, train_acc {train_acc}, val acc {val_acc}"
          f"Epoch time = {(end_time - start_time):.3f}s"))

        if epoch % SAVE_EPOCHS == 0:
            save_checkpoint(
                transformer, optimizer, avg_train_loss, 
                avg_val_loss, train_acc, val_acc
            )

    ### END TRAIN LOOP

    # save final outcome
    #train_loss_per_epoch.append(avg_train_loss)
    #val_loss_per_epoch.append(avg_val_loss)
    save_checkpoint(
        transformer, optimizer, train_loss_per_epoch, 
        val_loss_per_epoch, train_acc_per_epoch, val_acc_per_epoch
    )

if __name__ == '__main__':
    np.random.seed(420)
    main()
