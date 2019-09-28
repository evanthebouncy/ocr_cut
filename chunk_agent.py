import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import tqdm

from data import make_ocr_data_batch
from ocr_model import to_torch, OCR

class ChunkEnvironment:

    def __init__(self, ocr, chunk_length):
        self.ocr = ocr
        self.chunk_length = chunk_length

    def chunk(self, x):
        # if the segment is already too short to cut in 1, we stop
        if x.shape[1] < self.chunk_length:
            return x, None
        # otherwise we break it off into chunk_length and rest
        else:
            return x[:,:self.chunk_length], x[:,self.chunk_length:]

    def reset(self, img, ocr_label):
        self.img, self.ocr_label = img, ocr_label
        self.cur, self.rest = self.chunk(img)
        self.chunks = []
        if self.rest is not None:
            assert self.img.shape[1] == self.cur.shape[1] + self.rest.shape[1]

    def observe(self):
        return self.cur

    # update the states and return whether we're done with all chunkings
    def step(self, chunked_len):
        assert chunked_len < self.cur.shape[1]
        chunk = self.cur[:, :chunked_len]
        self.chunks.append(chunk)

        remain = np.concatenate((self.cur[:, chunked_len:], self.rest), axis=1)
        self.cur, self.rest = self.chunk(remain)
        if self.rest is None:
            self.chunks.append(self.cur)
            return True
        else:
            return False

    def get_rollout(self, agent):
        done = False
        cur_state = self.observe()
        states, actions = [], []
        while not done:
            action = agent.act(cur_state)
            
            states.append(cur_state)
            actions.append(action)

            done = self.step(action)
            cur_state = self.observe()
        total_length = sum([chunk.shape[1] for chunk in self.chunks])
        assert total_length == self.img.shape[1]
        return self.chunks, np.array(states), np.array(actions)

    def visualise(self, chunks):
        from data import plot
        to_draw = []
        for c in chunks:
            to_draw.append(np.full((28,1),255))
            to_draw.append(c)
        chunks = np.concatenate(to_draw, axis=1)
        plot(chunks[:,1:], f"chunks")

    # assume the last chunk is blank
    def score(self, chunks):
        ocr_pred = 0
        for c in chunks:
            # if chunk is not empty
            if np.sum(c) > 200 and c.shape[1] > 15:
                chunk_transcribed = np.argmax(self.ocr(to_torch(c,"float").unsqueeze(0).unsqueeze(0)).detach().squeeze().cpu().numpy())
                ocr_pred += chunk_transcribed
        return 1.0 if ocr_pred % 10 == self.ocr_label else 0.0

class RandomAgent():
    def act(self, chunk):
        return np.random.randint(20,chunk.shape[1])

class ChunkAgent(nn.Module):

    def __init__(self, chunk_length):
        super(ChunkAgent, self).__init__()
        # how much prefix of image to consume at a time
        self.chunk_length = chunk_length

        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 10, 5)

        self.fc1 = nn.Linear(10*20*52,1000)
        self.fc2 = nn.Linear(1000, chunk_length)
        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 10*20*52)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

    # sample a stoicastic action
    def act(self, x):
        # reshape it to [batch, channel, Height, Width]
        x = to_torch(x, "float").unsqueeze(0).unsqueeze(0)
        # get the action probability
        action_logsoftmax = self(x)
        # add a small number so every action has at least a small fraction
        action_prob = torch.exp(action_logsoftmax).squeeze().detach().cpu().numpy() + 1e-3
        # mask out some actions to ensure we do not chunk too small
        # re-normalise the rest and sample from it
        action_prob[:20] = 0
        action_prob = action_prob / np.sum(action_prob)
        return np.random.choice(np.arange(self.chunk_length), p=action_prob)

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

if __name__ == '__main__':

    CHUNK_LENGTH = 60

    ocr = OCR().cuda()
    ocr.load('ocr_model_stable.mdl')
    environment = ChunkEnvironment(ocr, CHUNK_LENGTH)
    ragent = RandomAgent()
    chunk_agent = ChunkAgent(CHUNK_LENGTH).cuda()

    # train the chunk agent
    scores = [0]
    for i in range(100000):
        X, Y = make_ocr_data_batch(4)
        img, ocr_label = X[0], Y[0]
        environment.reset(img, ocr_label)
        if np.random.random() < 0.9:
            chunks, states, actions = environment.get_rollout(chunk_agent)
        else:
            chunks, states, actions = environment.get_rollout(ragent)
        score = environment.score(chunks)
        scores.append(score)
        if score == 1.0:
            chunk_agent.learn_once(states, actions)

        if i % 1000 == 0:
            print (f"last 100 score {np.sum(scores[-100:])}")
            environment.visualise(chunks)

