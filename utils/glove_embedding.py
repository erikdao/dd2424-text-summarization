from preprocess.glove import *

class GloveEmbedding():
    def __init__(self, embedding_file, w2i):
        self.w2i = w2i
        self.vocab_size = len(self.w2i.keys())
        self.word_emb_size, embeddings = load_glove_embeddings(embedding_file)

        self.word_emb = nn.Embedding(self.vocabulary_size, self.word_emb_size)
        self.word_emb.weight = nn.Parameter(torch.from_numpy(embeddings), requires_grad=False)