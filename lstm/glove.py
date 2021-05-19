import typing
import torch
import torchtext.vocab as vocab

from config import config


def extend_glove(glove):
    """
    Add <PAD> and <UNK> tokens to glove embedding
    """
    vocab_size = len(glove.stoi.keys())

    # Add the pad token
    pad_vectors = torch.zeros((1, glove.vectors.size()[1]))
    glove.vectors = torch.cat((glove.vectors, pad_vectors))
    glove.stoi["<PAD>"] = vocab_size
    glove.itos.extend("<PAD>")

    # Add the UNK token
    unk_vectors = torch.mean(glove.vectors, 0, keepdim=True)
    glove.vectors = torch.cat((glove.vectors, unk_vectors))
    glove.stoi["<UNK>"] = vocab_size + 1
    glove.itos.extend("<UNK>")

    new_vocab_size = len(glove.stoi.keys())
    assert new_vocab_size == vocab_size + 2

    return glove


# glove = extend_glove(glove)


def tokenize(
    sentence: typing.Optional[str], tokenizer: typing.Optional[typing.Any]
) -> typing.List[typing.Any]:
    sentence = sentence.strip().lower().replace("\n", "")
    punctuation = ["(", ")", ":", '"', " "]
    return [
        tok
        for tok in tokenizer(sentence)
        if (not tok.isspace() and tok not in punctuation)
    ]


def main():
    glove = vocab.GloVe(name="6B", dim=config.EMBEDDING_DIM)
    print(glove.vectors.size())
    glove = extend_glove(glove)
    print(config.PAD_TOKEN, glove.stoi["<PAD>"])
    print(config.UNK_TOKEN, glove.stoi["<UNK>"])
    print(glove.vectors.size())


if __name__ == "__main__":
    main()
