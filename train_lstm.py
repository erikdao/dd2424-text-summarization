"""
Main entrance for training LSTM for text summary generation
- The word embedding used is DistillBert
"""
from lstm.model import LSTMGenerator
from lstm.dataloader import create_dataloader

def main():
    model = LSTMGenerator()
    print(model)

    train_loader = create_dataloader(
        csv_file='data/amazon-product-reviews/Reviews.csv',
        shuffle=True
    )
    for idx, data in enumerate(train_loader):
        if idx == 2:
            input_vec = data['input']
            label_vec = data['label']
            print(input_vec, input_vec.shape)
            print(label_vec, label_vec.shape)
            break

if __name__ == '__main__':
    main()
