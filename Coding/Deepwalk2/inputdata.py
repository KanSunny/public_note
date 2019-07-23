
# coding: utf-8

# In[1]:


import numpy
from collections import deque
from huffman import HuffmanTree

numpy.random.seed(12345)

class InputData:
    def __init__(self, input_file, min_count):
        self.input_file = input_file
        self.sentence_sum_length = 0 # 用于统计句子中出现单词的总数量
        self.sentence_count = 0 # 用于统计随机游走数量
        self.word_count = 0
        self.word2id = dict()
        self.id2word = dict()
        self.word_frequency = dict()# 词频：用于统计随机游走学列中单词出现的次数
        self.word_pair_catch = deque() # 是什么
        
        self.get_words(min_count)
        tree = HuffmanTree(self.word_frequency)
        self.huffman_positive, self.huffman_negative = tree.divide_pos_and_neg()
        
    def get_words(self, min_count):
        
        word_frequency = dict()
        input_file = open(self.input_file)
        for line in input_file:
            self.sentence_count += 1
            words = line.strip().split(' ')
            self.sentence_sum_length += len(words)
            for word in words:
                try:
                    word_frequency[word] += 1
                except:
                    word_frequency[word] = 1
        
        wid = 0
        
        for word, count in word_frequency.items():
            if count < min_count: # 当词频小于最小词频时讲单词出现次数归为0
                self.sentence_sum_length -= count# min_count:需要计算词向量的最小词频。这个值可以去掉一些很生僻的低频词，默认是5。如果是小语料，可以调低这个值。
                continue
            
            self.word2id[word] = wid
            self.id2word[wid] = word
            self.word_frequency[wid] = count# 词和id重复？
            wid += 1
            
        self.word_count = len(self.word2id)
        input_file.close()
        
    def get_batch_pairs(self, batch_size, window_size):
        input_file = open(self.input_file) # 貌似每次打开文件都只会读前几行？？？
        while len(self.word_pair_catch) < batch_size:
            sentence = input_file.readline()
            word_ids = []
            for word in sentence.strip().split(' '):
                try:
                    word_ids.append(self.word2id[word])
                except:
                    continue
            
            for i, center_wid in enumerate(word_ids):
                window_wids = word_ids[max(0, i-window_size) : i+window_size]
                for j, window_wid in enumerate(window_wids):
                    assert center_wid < self.word_count# 测试
                    assert window_wid < self.word_count
                    if i == j:
                        continue
                    self.word_pair_catch.append((center_wid, window_wid))
                    
        batch_pairs = []
        for _ in range(batch_size):
            batch_pairs.append(self.word_pair_catch.popleft())
        
        input_file.close()
        return batch_pairs
    
    def get_pairs_from_huffman(self, word_pairs): #把一个点对根据huffman变成一正一负（负采样）
        pos_word_pairs = []
        neg_word_pairs = []
        for index in range(len(word_pairs)):
            pair = word_pairs[index]
            pos_word_pairs += zip([pair[0]] * len(self.huffman_positive[pair[1]]),
                                 self.huffman_positive[pair[1]]) # zip将多个对象中对应的元素打包成原组
            neg_word_pairs += zip([pair[0]] * len(self.huffman_negative[pair[1]]),
                                 self.huffman_negative[pair[1]])
            
        return pos_word_pairs, neg_word_pairs# 是什么
    
    def evaluate_pair_count(self, window_size):
        return self.sentence_sum_length * (2 * window_size -1) - (self.sentence_count - 1) * (1 + window_size) * window_size
    # 窗口全满-窗口空缺

