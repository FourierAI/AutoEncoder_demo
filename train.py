import os

import torch

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from data import *
from model import *
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
logger = SummaryWriter('./log')

BATCH_SIZE = 64
lr = 0.0001
EPOCH = 2

AE = AutoEncoder(1, 200)

data = PaintingDataset()
dataloader = DataLoader(data, shuffle=True, batch_size=BATCH_SIZE)

criterion = nn.MSELoss()
opt_AE = optim.Adam(AE.parameters(), lr=lr)


for epoch in range(EPOCH):
    for step, (paintings, condition) in enumerate(dataloader):

        g_paintings = AE(condition)
        loss = criterion(g_paintings, paintings)
        opt_AE.zero_grad()
        loss.backward()
        opt_AE.step()

        if step % 100 == 0:
            print(f'epoch: {epoch}, loss : {loss}')
            logger.add_scalar('loss', loss, epoch * len(dataloader)+step)

torch.save(AE, './ae.pth')