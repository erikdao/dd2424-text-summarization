"""
Main entrance for training LSTM for text summary generation
- The word embedding used is DistillBert
"""
import torch

from lstm.logger import logger
from lstm.model import LSTMGenerator
from lstm.dataloader import create_dataloader, tokenizer


def main():
    logger.info("Loading training data...")
    train_loader = create_dataloader(
        csv_file='data/amazon-product-reviews/Reviews.csv',
        batch_size=1,
        shuffle=True
    )

    logger.info("Initialize model...")
    model = LSTMGenerator(input_size=768)
    for datapoint in train_loader:
        input = datapoint['input']
        print('input', input.shape)
        label = datapoint['label']
        hidden = model.init_hidden()
        for idx in range(input.shape[-1]):
            output, hidden = model.forward(torch.tensor(input[0, 0, idx]), hidden)

            predicted_index = torch.argmax(output[0, :]).item()
            predicted_text = tokenizer.decode(predicted_index)
            print(idx, predicted_text)
        
        break

if __name__ == '__main__':
    main()
