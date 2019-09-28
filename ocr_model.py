import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import tqdm

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

# return result mod 10
class OCR(nn.Module):
    def __init__(self):
        super(OCR, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)
        self.fc1 = nn.Linear(400, 400)
        self.pred = nn.Linear(400, 10)
        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # compute mean over the "x-axis" feature maps and collapse
        x = torch.mean(x, dim=-1)
        x = x.reshape(-1,20*20)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.pred(x), dim=1)
        return x
    
    def learn_once(self, X, Y):
        X = to_torch(X, "float").unsqueeze(1)
        Y = to_torch(Y, "int")
        # optimize 
        self.opt.zero_grad()
        output = self(X)
        loss = F.nll_loss(output, Y)
        loss.backward()
        self.opt.step()
        return loss

    def save(self, loc):
        torch.save(self.state_dict(), loc)

    def load(self, loc):
        self.load_state_dict(torch.load(loc))

def evaluate(ocr):
    for seq_l in range(1, 5):
        acc = 0
        for bb in range(100):
            X, Y = make_ocr_data_batch(seq_l)
            Y_pred = np.argmax(ocr(to_torch(X,"float").unsqueeze(1)).detach().cpu().numpy(), axis=1)
            acc += np.mean(Y_pred == Y)
        print (f"seq length {seq_l} accuracy {acc / 100}")


if __name__ == '__main__':
    ocr = OCR().cuda()
    for i in tqdm.tqdm(range(10000000)):
        seq_l = np.random.randint(1, 4)
        X, Y = make_ocr_data_batch(seq_l)
        loss = ocr.learn_once(X, Y)
        if i % 10000 == 0:
            ocr.save('ocr_model.mdl')
            evaluate(ocr)


