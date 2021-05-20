from utils.pickle import *
from dataload.dataloader import *
import time

from utils.glove_embedding import *
from transformer.transformer_model import *
from torch.utils.tensorboard import SummaryWriter

SAVED_MODEL_FILE = '/Users/pacmac/Documents/GitHub/KTH_Projects/dd2424-text-summarization/transformer/transformer_model'

def load_losses(filename='transformer_model'):
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        train_losses = checkpoint['loss_epoch_history']['train_loss']
        val_losses = checkpoint['loss_epoch_history']['val_loss']
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
   
    load_flag, train_losses, val_losses = load_losses(filename=SAVED_MODEL_FILE)
    if not load_flag:    
        print("No losses...")
        exit()
    
    writer = SummaryWriter()
    n_points = len(train_losses)
    for i in range(n_points):
        writer.add_scalar(f'losses/train',train_losses[i],i)
        writer.add_scalar(f'losses/val', val_losses[i], i)
    writer.flush()
    writer.close()

if __name__ == '__main__':
    main()
