"""
Main entrance for training LSTM for text summary generation
- The word embedding used is DistillBert
"""
from lstm.model import LSTMGenerator
from lstm.dataloader import ReviewDataset

def main():
    model = LSTMGenerator()
    print(model)

    dataset = ReviewDataset(csv_file='data/amazon-product-reviews/Reviews.csv')
    print(dataset[10])


if __name__ == '__main__':
    main()
