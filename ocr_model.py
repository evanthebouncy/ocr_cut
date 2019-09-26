import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

from data import make_ocr_data_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    def to_torch(x, dtype, req = False):
        tor_type = torch.cuda.LongTensor if dtype == "int" else torch.cuda.FloatTensor
        x = Variable(torch.from_numpy(x).type(tor_type), requires_grad = req)
        return x
else:
    def to_torch(x, dtype, req = False):
        tor_type = torch.LongTensor if dtype == "int" else torch.FloatTensor
        x = Variable(torch.from_numpy(x).type(tor_type), requires_grad = req)
        return x

class Encoder(nn.Module):
    def __init__(self, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)
        self.fc1 = nn.Linear(400, 100)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.mean(x, dim=-1)
        x = x.reshape(-1,20*20)
        x = F.relu(self.fc1(x))
        return x

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, word_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.word_size = word_size
        # takes in word, takes in hidden, get new hidden
        self.step = nn.Linear(word_size+hidden_size, hidden_size)
        # take the new hidden, compute the out from it
        self.dec_out = nn.Linear(hidden_size, word_size)

    def forward(self, word, hidden):
        hid = F.sigmoid(self.step(torch.cat((word, hidden), dim=1)))
        out = F.softmax(self.dec_out(hid), dim=1)
        return out, hid

    def roll_out(self, hidden, num_rolls):
        cur_word = torch.zeros(hidden.size()[0], self.word_size).cuda()
        ret = []
        for i in range(num_rolls):
            cur_word, hidden = self(cur_word, hidden)
            ret.append(cur_word)
        return ret

class OCR:
    def __init__(self, encoder, decoder):
        self.enc = encoder
        self.dec = decoder

    def learn(self, img_batch, seq_batch):
        raise NotImplementedError

    def transcribe(self, img):
        x_enc = self.enc(img)
        seq_dec = self.dec.roll_out(x_enc, 13)
        ret = []
        for tstep in seq_dec:
            amax = np.argmax(tstep.detach().cpu().numpy(), axis=1)
            ret.append(amax)
        return ret


if __name__ == '__main__':
    X1, Y1 = make_ocr_data_batch(7, 2)
    
    enc = Encoder(100).cuda()
    dec = DecoderRNN(100, 10).cuda()
    ocr = OCR(enc, dec)

    seq_outputs = ocr.transcribe(to_torch(X1,"float").unsqueeze(1))
    print (seq_outputs)

