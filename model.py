from utils import *
import torch.nn as nn


class TestCNN(nn.Module):
    def __init__(self):
        super(TestCNN, self).__init__()
        self.W = nn.Embedding(vocab_size, embedding_size)
        output_channel = 3
        self.conv = nn.Sequential(
            nn.Conv2d(1, output_channel, (2, embedding_size)),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )
        self.fc = nn.Linear(output_channel, num_classes)

    def forward(self, X):
        batch_size = X.shape[0]
        embedding_X = self.W(X)
        # 使用unsqueeze(1)函数使数据增加一个维度
        embedding_X = embedding_X.unsqueeze(1)
        conved = self.conv(embedding_X)
        flatten = conved.view(batch_size, -1)
        output = self.fc(flatten)
        return output


if __name__ == '__main__':
    print(TestCNN())
