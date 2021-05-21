from utils.pickle import *
from dataload.dataloader import *
import time
import matplotlib.pyplot as plt
from utils.pickle import *

from utils.glove_embedding import *
from transformer.transformer_model import *
from torch.utils.tensorboard import SummaryWriter
import json

#SAVED_MODEL_FILE = '/Users/pacmac/Documents/GitHub/KTH_Projects/dd2424-text-summarization/transformer/transformer_model'
SAVED_LOSS_LOG_FILE = '/Users/pacmac/Documents/GitHub/KTH_Projects/dd2424-text-summarization/transformer/loss_loggs.json'


def load_losses(filename='loss_loggs.json'):
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        with open(filename) as json_file:
            data = json.load(json_file)
            train_losses = data['train_loss']
            val_losses = data['val_loss']
            load_flag = True
    else:
        print("=> no checkpoint found at '{}'".format(filename))
        load_flag = False
        train_loss = []
        val_loss = []
    return load_flag, train_loss, val_loss


def main():
    np.random.seed(420)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    load_flag, train_losses, val_losses = load_losses(filename=SAVED_LOSS_LOG_FILE)
    if not load_flag:    
        print("No losses...")
        exit()
    
    n_losses = len(train_losses):
    for i in range(n_losses):
        print("Train loss {}, Val loss {}".format(train_losses[i],val_losses[i]))
    #plt.plot(train_losses)  
    #plt.plot(val_losses)
    #plt.show()
    """
    writer = SummaryWriter()
    n_points = len(train_losses)

    for i in range(n_points):
        writer.add_scalar(f'losses/train',train_losses[i],i)
        writer.add_scalar(f'losses/val', val_losses[i], i)
    writer.flush()
    writer.close()
    """

if __name__ == '__main__':
    main()
