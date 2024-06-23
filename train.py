from utils import *
from model import *
import torch.optim as optim
import torch.nn.functional as F
from dataprocess import *

model = TestCNN().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training
for epoch in range(5000):
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        if(epoch + 1) % 100 == 0:
            print('EPOCH', '%04d' % (epoch + 1) ,'loss=', '{:.6f}'.format(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
torch.save(model, MODEL_DIR + f'model_CnnText.pth')


"""
这是一段测试的文件
"""