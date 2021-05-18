import typing
import torch
import torchtext.vocab as vocab
from torchtext.data import get_tokenizer

def tokenize(sentence: typing.Optional[str], tokenizer: typing.Optional[typing.Any]) -> typing.List[typing.Any]:
    sentence = sentence.strip().lower().replace("\n", "")
    punctuation = ['(', ')', ':', '"', ' ']
    return [tok for tok in tokenizer(sentence) if (not tok.isspace() and tok not in punctuation)]

def main():
    glove = vocab.GloVe(name='6B', dim=50)

    print('Loaded {} words'.format(len(glove.itos)))

    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    input_sentence = "these macaroons are better than the non vegan raw ones! i swear it! the blondes too   they taste like they are made with butter. truly a great product! buy people and you will be hooked  in a good way ,too good for words!"
    tokens = tokenize(input_sentence, tokenizer=tokenizer)
    print(tokens)

    for idx, tok in enumerate(tokens):
        index = glove.stoi[tok]
        vectors = glove.vectors[index]
        print(idx, tok, vectors)

if __name__ == '__main__':
    main()
