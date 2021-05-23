from utils.pickle import *
from dataload.dataloader import *
import time
from torch.autograd import Variable

from utils.glove_embedding import *
from transformer.transformer_model import *
import json

EMBEDDING_FILE = '../preprocess/glove.6B.50d.txt'
SAVED_MODEL_FILE = '/Users/pacmac/Documents/GitHub/KTH_Projects/dd2424-text-summarization/transformer/transformer_model'
HEADS = 10 # default = 8
N = 6 # default = 6
DIMFORWARD = 512
BATCH_SIZE = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_IDX = 0
EOS_WORD = "<EOS>"

def load_checkpoint(model, filename='transformer_model'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        load_flag = True
    else:
        print("=> no checkpoint found at '{}'".format(filename))
        load_flag = False
        model = None
    return load_flag, model

def generate(model, sentence_idx, word2index, index2word, max_label_seq_length=10):
    sentence_idx.to(DEVICE)
    sos_idx = word2index['<SOS>']
    trg = torch.LongTensor([[sos_idx]]).to(DEVICE)
    translated_sentence = ""

    for i in range(max_label_seq_length):
        #print(f"generate word {str(i)}")
        size = trg.size(0)
        np_mask = torch.triu(torch.ones(size, size)==1).transpose(0,1)
        np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
        np_mask = np_mask.to(DEVICE)
        pred = model(sentence_idx.transpose(0,1), trg, tgt_attention_mask = np_mask)
        pred_word_idx = int(pred.argmax(dim=2)[-1])
        #print(f"index predicted = {pred_word_idx}")
        add_word = index2word[pred_word_idx]
        if add_word==EOS_WORD:
            break
        trg = torch.cat((trg,torch.LongTensor([[pred_word_idx]]).to(DEVICE)))
        translated_sentence += " " + add_word
    return translated_sentence

def main():
    torch.cuda.empty_cache()
    
    input_dir_tok = "./tokenized-padded"
    input_dir_w2i = "./tokenized"
    print("Loading mappings...")
    mappings_pickle = input_dir_tok + "/tokenized-padded"
    mappings = pickle_load(mappings_pickle)
    max_input_seq_length = len(mappings['inputs'][0])
    max_label_seq_length = len(mappings['labels'][0])
    print(f"seq input length {max_input_seq_length }")
    print(f"seq label length {max_label_seq_length }")

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

    start = 327200+40900 
    end = len(inputs)
    choices = list(np.random.choice(np.arange(start,end), size=BATCH_SIZE, replace=False))
    inputs_choice = map(inputs.__getitem__, choices)
    labels_choice = map(labels.__getitem__, choices)
    #mappings_test = {'inputs': inputs[start:start+2*BATCH_SIZE], 'labels': labels[start:start +2*BATCH_SIZE]}
    mappings_test = {'inputs': inputs_choice, 'labels': labels_choice}

    #mappings_test = {'inputs': inputs[327200+40900:327200+2*40900], 'lab    els': labels[327200+40900:327200+2*40900]}
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
        word_emb_size=word_emb_size,
        nhead=HEADS,
        num_encoder_layers=N,
        num_decoder_layers=N, 
        dim_feedforward=DIMFORWARD, 
        max_input_seq_length=max_input_seq_length,
        max_label_seq_length=max_label_seq_length,
        pos_dropout=0.1, 
        trans_dropout=0.1, 
        word2index=word2index,
        embeddings=embeddings
    ).to(DEVICE)

    load_flag, transformer = load_checkpoint(transformer, filename=SAVED_MODEL_FILE)
    if not load_flag:
        print("\"Focking idiot!!!!\", Adam 19.05.2021")
        exit()
    transformer = transformer.to(DEVICE)
    print("model init completed...")
    


    print("start generate...")
    start_time = time.time()

    for idx, batch in enumerate(test_loader):
        print("---- BATCH {} ----".format(idx))
        src = batch['input'].to(DEVICE) # (B,S)
        tgt_real = batch['label'] # (B,S)
        transformer.eval()

        # decoding iteratively
        for i in range(BATCH_SIZE):
            src_i = src[i]
            #src_i_words = [ index2word[int(num)] for num in list(src_i)]
            #print("SRC I: ",src_i_words)
            src_i = torch.unsqueeze(src_i,0) # (1,S)
            summary = generate(transformer, src_i, word2index, index2word, max_label_seq_length=10)
            print("\nGENERATED: ")
            print(summary)
            print("REAL: ")
            print(" ".join( [index2word[int(i)] for i in tgt_real[i].tolist()] ))

    end_time = time.time()
        
if __name__ == '__main__':
    main()
