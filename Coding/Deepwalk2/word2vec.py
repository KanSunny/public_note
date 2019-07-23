
# coding: utf-8

# In[ ]:


from inputdata import InputData
from skipgram import SkipGramModel
import torch.optim as optim

class Word2Vec:
    def __init__(self, input_file, output_file, emb_dimension=100, batch_size=100, window_size=5, iteration=5,
                initial_lr=0.025, min_count=5, context_size=2):
    # iteration: 进行几次walks，即几次number_paths*nodes 
    # min_count: ?
    # context_size: ?
    # batch_size: ?
        self.data = InputData(input_file, min_count)
        self.output_file = output_file
        self.emb_size = len(self.data.word2id)#？
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.window_size = window_size
        self.iteration = iteration
        self.initial_lr = initial_lr
        self.context_size = context_size
        
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension) # ?
        self.optimizer = optim.SGD(self.skip_gram_model.parameters(), lr=self.initial_lr) # 参数是什么如何传，如何优化
        
    def skip_gram_train(self):
        pair_count = self.data.evaluate_pair_count(self.window_size)# 输入窗口尺寸就可以得到点对数量（中心点和临近点）
        batch_count = self.iteration * pair_count / self.batch_size# 点对数量*迭代次数/batch_size 就是batch的数量
        for i in range(int(batch_count)):
            # 每次都一样 ？？？
            word_pairs = self.data.get_batch_pairs(self.batch_size, self.window_size)# 根据batch_size和window_size取出点对
            pos_pairs, neg_pairs = self.data.get_pairs_from_huffman(word_pairs)# huffman采样
            
            pos_center = [int(pair[0]) for pair in pos_pairs] # 是什么是huffman编码吗
            pos_window = [int(pair[1]) for pair in pos_pairs]
            neg_center = [int(pair[0]) for pair in neg_pairs]
            neg_window = [int(pair[1]) for pair in neg_pairs]
            
            self.optimizer.zero_grad()
            loss = self.skip_gram_model.forward(pos_center, pos_window, neg_center, neg_window)
            loss.backward()
            self.optimizer.step()
            
            if i * self.batch_size % 100000 == 0:
                lr = self.initial_lr * (1.0 - 1.0 * i / batch_count)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
        self.skip_gram_model.save_embedding(self.data.id2word, self.output_file)

