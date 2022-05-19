import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt

AE = torch.load('ae.pth')
BATCH_SIZE = 10

condition_quadratic = torch.ones(BATCH_SIZE)*1.09
condition_quadratic = condition_quadratic.view(-1, 1)
quadratic = AE(condition_quadratic)
Y = quadratic.data.numpy()[8]
plt.plot(Y)
plt.show()