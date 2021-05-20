from utils.pickle import *
from dataload.dataloader import *
import time
from torch.autograd import Variable

from utils.glove_embedding import *
from transformer.transformer_model import *

EMBEDDING_FILE = '../preprocess/glove.6B.50d.txt'
SAVED_MODEL_FILE = '/Users/pacmac/Documents/GitHub/KTH_Projects/dd2424-text-summarization/transformer/transformer_model'
SAVE_EPOCHS = 5
EPOCHS = 6
HEADS = 5 # default = 8
N = 3 # default = 6
DIMFORWARD = 512
BATCH_SIZE = 1

def load_checkpoint(model, optimizer, filename='transformer_model'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        train_loss = checkpoint['loss_epoch_history']['train_loss']
        val_loss = checkpoint['loss_epoch_history']['val_loss']
        load_flag = True
    else:
        print("=> no checkpoint found at '{}'".format(filename))
        load_flag = False
        train_loss = 0.
        val_loss = 0.
    return load_flag, model, optimizer, train_loss, val_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    
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
    index2word = {}
    for word in word2index.keys():
        idx = word2index.get(word)
        index2word[idx] = word

    print("load embeddings...")
    word_emb_size, embeddings = load_glove_embeddings(EMBEDDING_FILE)
    
    inputs = mappings['inputs']
    labels = mappings['labels']
    """
    mappings_train = {'inputs': inputs[:2*BATCH_SIZE], 'labels': labels[:2*BATCH_SIZE]}
    mappings_val = {'inputs': inputs[2*BATCH_SIZE:4*BATCH_SIZE], 'labels': labels[2*BATCH_SIZE:4*BATCH_SIZE]}
    """
    mappings_test = {'inputs': inputs[4*BATCH_SIZE:6*BATCH_SIZE], 'labels': labels[4*BATCH_SIZE:6*BATCH_SIZE]}

    """
    print("Loading train loader...")
    train_loader = create_dataloader_glove(
        mappings = mappings_train,
        word2index = word2index,
        embeddings = embeddings,
        word_emb_size = word_emb_size,
        batch_size = BATCH_SIZE,
        shuffle=True
    )
    print("Loading val loader...")
    val_loader = create_dataloader_glove(
        mappings = mappings_val,
        word2index = word2index,
        embeddings = embeddings,
        word_emb_size = word_emb_size,
        batch_size = BATCH_SIZE,
        shuffle=True
    )
    """
    print("Loading test loader...")
    test_loader = create_dataloader_glove(
        mappings = mappings_test,
        word2index = word2index,
        embeddings = embeddings,
        word_emb_size = word_emb_size,
        batch_size = BATCH_SIZE,
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
    ).to(device)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.01, betas=(0.9, 0.98), eps=1e-9) # TODO
    load_flag, transformer, optimizer, train_loss, val_loss = load_checkpoint(transformer, optimizer, filename=SAVED_MODEL_FILE)
    if not load_flag:
        print("\"Focking idiot!!!!\", Adam 19.05.2021")
        exit()
    transformer = transformer.to(device)
    print("model init completed...")
    
    print("start generate...")
    
    start_time = time.time()

    

    batch = next(iter(test_loader))
    

    src = batch['input'].to(device)
    tgt_real = batch['label']
    # get start of string
    sos_idx = word2index['<SOS>']
    #ys = torch.Tensor([[sos_idx] for b in range(BATCH_SIZE)])

    
    transformer.eval()
    
    
    trg_init_tok = sos_idx
    trg = torch.LongTensor([[trg_init_tok]  + [0 for p in range(src.shape[1] - 1)]]).to(device)


    # encoding once
    src_attention_mask, src_key_padding_mask = create_src_masks(src, device)
    memory = transformer.encode(src, src_key_padding_mask, device).to(device)
    
    translated_sentence = ""
    maxlen = 25
    for i in range(maxlen):
        print("iteration = " + str(i))

        #print(trg)
        #print(trg.shape)
        tgt_attention_mask, tgt_key_padding_mask = create_tgt_masks(trg, device)
        
        pred = transformer.decode(trg, memory, src_key_padding_mask, tgt_attention_mask, tgt_key_padding_mask).to(device)
        pred = pred.transpose(0, 1).contiguous()
        #print(pred.shape)
        #print(pred)
        #print()

        pred = F.log_softmax(pred, dim=1).to(device)
        #print(pred)

        pred_word_idx = int(pred[0,i+1].argmax())
        #print("max index")
        #print(pred_word_idx)
        add_word = index2word[pred_word_idx]
        #print("pred word" + add_word)
        translated_sentence += " " + add_word
        if add_word == "<EOS>":
            trg[0,i+1] = pred_word_idx
            break

        #print("concat")
        #print(trg.shape)
        #print(torch.LongTensor([[pred_word_idx]]).shape)
        trg[0,i+1] = pred_word_idx
        #trg[i] = torch.cat((trg, torch.LongTensor([[pred_word_idx]])))
        
    print("\nGENERATED: ")
    print(translated_sentence, "\n\n")
    print("REAL: ")
    print(" ".join( [index2word[i] for i in tgt_real[0].tolist()] ))
    


    end_time = time.time()
        
if __name__ == '__main__':
    np.random.seed(420)
    main()
