import torch

from model import *
from utils import *


# Test
test_text = 'i love me'
tests = [[word2idx[n] for n in test_text.split()]]
test_batch = torch.LongTensor(tests).to(device)
# Predict
# model = model.eval()
model = torch.load(MODEL_DIR+'model_CnnText.pth')
model.to(device)
predict = model(test_batch).data.max(1, keepdim=True)[1]
if predict[0][0] == 0:
    print(test_text,"is Bad Mean...")
else:
    print(test_text,"is Good Mean!!")