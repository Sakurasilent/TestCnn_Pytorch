from utils import *
import torch.utils.data as Data

def make_data(sentences, labels):
    inputs = []
    for sen in sentences:
        inputs.append([word2idx[n] for n in sen.split()])

    target = []
    for out in labels:
        target.append(out)
    return inputs, target


input_batch, target_batch = make_data(sentences, labels)
input_batch, target_batch = torch.LongTensor(input_batch), torch.LongTensor(target_batch)

dataset = Data.TensorDataset(input_batch, target_batch)
loader = Data.DataLoader(dataset, batch_size, True)
if __name__ == '__main__':
    s, t = make_data(sentences, labels)
    print(s)
    print(t)
    print(dataset[:])
    print(loader)