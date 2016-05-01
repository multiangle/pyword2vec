__author__ = 'multiangle'

import jieba
from collections import Counter
import File_Interface as FI

class WordCounter():
    def __init__(self, ori_text_list):
        self.ori_text_list = ori_text_list
        self.stop_word = self.Get_Stop_Words()
        self.cutted_text_list = []
        self.word_set = None

        self.Word_Count(self.ori_text_list)

    def Get_Stop_Words(self):
        ret = []
        ret = FI.load_pickle('./static/stop_words.pkl')
        return ret

    def Word_Count(self,text_list,cut_all=True):

        filtered_word_list = []

        for line in text_list:
            res = jieba.cut(line,cut_all=cut_all)
            res = list(res)
            self.Filter_Stop_Words(res)
            self.cutted_text_list.append(res)
            filtered_word_list += res

        self.word_set = Counter(filtered_word_list)

    def Filter_Stop_Words(self,word_list):
        for i in range(word_list.__len__())[::-1]:

            if word_list[i] in self.stop_word:
                word_list.pop(i)
                continue

            try:  # replace the number to 'number'
                int(word_list[i])
                word_list[i] = "/number"
                continue
            except:
                pass

# class MulCounter(Counter):
#     def __init__(self,element_list):
#         self.Counter.__init__(element_list)
#         self.


if __name__ == '__main__':
    # text = FI.load_pickle('./static/demo.pkl')
    # text =[ x['dealed_text']['left_content'][0] for x in text]
    # wc = WordCounter(text)
    wc = FI.load_pickle('./static/test.pkl')
    x = Counter(wc.word_set)
    print(x)