from utils.pickle import *
from tqdm import tqdm

# average length of sentences are 80 words, some very large with thousands of words
# about 84 % less than 128 words
# will truncate longer sentences to MAX_SENTENCE_LEN words
MAX_INPUT_LEN = 64
MAX_OUTPUT_LEN = 10

def main():
    input_dir = "./tokenized"
    mappings_pickle = input_dir + "/tokenized_cleaned"
    print("Loading mappings...")
    mappings = pickle_load(mappings_pickle)

    n_points = len(mappings["inputs"])
    pad = '<PAD>'
    sos = '<SOS>'
    eos = '<EOS>'

    print("Padding sentences...")
    for i in tqdm(range(n_points)):
        in_sent = mappings["inputs"][i]
        lab_sent = mappings["labels"][i]
        
        if len(in_sent) > MAX_INPUT_LEN:
            in_sent = in_sent[:MAX_INPUT_LEN]
        if len(lab_sent) > MAX_OUTPUT_LEN-2:
            lab_sent = lab_sent[:MAX_OUTPUT_LEN-2]
        # insert sos and eos
        # in_sent = [(sos,2)] + in_sent + [(eos,3)]
        lab_sent = [(sos,2)] + lab_sent + [(eos,3)]

        while len(in_sent) < MAX_INPUT_LEN:
            in_sent.append((pad, 0))
        while len(lab_sent) < MAX_OUTPUT_LEN:
            lab_sent.append((pad, 0))
        
        mappings["inputs"][i] = in_sent
        mappings["labels"][i] = lab_sent

        assert len(in_sent) == MAX_INPUT_LEN
        assert len(lab_sent) == MAX_OUTPUT_LEN

    print("Saving padded mappings...")
    pickle_save("./tokenized-padded_cleaned", mappings) 

if __name__ == '__main__':
    main()
