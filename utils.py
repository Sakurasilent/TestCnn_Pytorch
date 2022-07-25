import torch

MODEL_DIR = 'output/model/'
# 学习率
LR = 1e-3
# 定义类型
dtype = torch.FloatTensor
# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 定义编造常 规数据
sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.
# 词向量维度
embedding_size = 2
# 最大句长
sequence_length = len(sentences[0])
# 文本类别
num_classes = len(set(labels))

batch_size = 3
word_list = ' '.join(sentences).split()

vocab = list(set(word_list))
word2idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)

if __name__ == '__main__':
    print(word_list)
    print(vocab)
    print(word2idx)
    print(vocab_size)
