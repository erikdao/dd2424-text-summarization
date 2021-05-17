from utils.pickle import *
from tqdm import tqdm

# average length of sentences are 80 words, some very large with thousands of words
# about 84 % less than 128 words
# will truncate longer sentences to MAX_SENTENCE_LEN words
MAX_SENTENCE_LEN = 128

def main():
    input_dir = "./tokenized"
    mappings_pickle = input_dir + "/tokenized"
    print("Loading mappings...")
    mappings = pickle_load(mappings_pickle)

    n_points = len(mappings["inputs"])
    pad = '<PAD>'

    print("Padding sentences...")
    for i in tqdm(range(n_points)):
        in_sent = mappings["inputs"][i]
        lab_sent = mappings["inputs"][i]
        while len(in_sent) < MAX_SENTENCE_LEN:
            in_sent.append((pad, 0))
        while len(lab_sent) < MAX_SENTENCE_LEN:
            lab_sent.append((pad, 0))
    if len(in_sent) > MAX_SENTENCE_LEN:
        in_sent = in_sent[:MAX_SENTENCE_LEN]
    if len(lab_sent) > MAX_SENTENCE_LEN:
        lab_sent = lab_sent[:MAX_SENTENCE_LEN]

    print("Saving padded mappings...")
    pickle_save("./tokenized-padded", mappings) 

if __name__ == '__main__':
    main()
