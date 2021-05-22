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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    start = 327200
    mappings_test = {'inputs': inputs[start+1*BATCH_SIZE:start+3*BATCH_SIZE], 'labels': labels[start + 1*BATCH_SIZE:start +3*BATCH_SIZE]}

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
    ).to(device)

    load_flag, transformer = load_checkpoint(transformer, filename=SAVED_MODEL_FILE)
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
    eos_idx = word2index['<EOS>']

    transformer.eval()
    trg = torch.LongTensor([[sos_idx]]).to(device)
    translated_sentence = ""

    # encoding once
    src_attention_mask, src_key_padding_mask = create_src_masks(src, device)
    memory = transformer.encode(src, src_key_padding_mask, device).to(device)
    # decoding iteratively
    for i in range(max_label_seq_length):
        print("iteration = " + str(i))
        if i < max_label_seq_length - 2:
            size = trg.size(0)
            tgt_attention_mask, tgt_key_padding_mask = create_tgt_masks(trg, device)
            
            pred = transformer.decode(trg, memory, src_key_padding_mask, tgt_attention_mask, tgt_key_padding_mask).to(device)
            
            #print(pred)
            pred = pred.transpose(0, 1).contiguous()
            print("transpose pred ", pred.shape)

            # TODO should we use a softmax
            #pred = F.log_softmax(pred, dim=1).to(device)
            #print(pred.shape)
            pred_word_idx = int(pred[:,i,:].argmax())
            #print("max index")
            print("PRED WORD IDX ",pred_word_idx)
            add_word = index2word[pred_word_idx]
            #print("pred word" + add_word)
            translated_sentence += " " + add_word
            if add_word == "<EOS>":
                trg = torch.cat((trg, torch.Tensor([[pred_word_idx]])), 1).to(device)
                break
            #print("idx = "+str(pred_word_idx))
            #print("old TRG")
            #print(trg.shape)
            #print(trg)
            trg = torch.cat((trg, torch.LongTensor([[pred_word_idx]]).to(device)), 1)
            #print("new TRG")
            #print(trg.shape)
            #print(trg)
        else:
            add_word = "<EOS>"
            translated_sentence += " " + add_word
            trg = torch.cat((trg, torch.Tensor([[pred_word_idx]])), 1)
            break

        
    print("\nGENERATED: ")
    print(translated_sentence)
    print("REAL: ")
    print(" ".join( [index2word[i] for i in tgt_real[0].tolist()] ))
    


    end_time = time.time()
        
if __name__ == '__main__':
    np.random.seed(420)
    main()
