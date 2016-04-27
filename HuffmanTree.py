__author__ = 'multiangle'

import numpy

class HuffmanTreeNode():
    def __init__(self,value,possibility):
        # common part of leaf node and tree node
        self.possibility = possibility
        self.left = None
        self.right = None
        # value of leaf node  will be the word, and be
        # mid vector in tree node
        self.value = value

class HuffmanTree():
    def __init__(self, win_len=5, vec_len=15000, min_count=5, max_ratio=0.3):
        self.win_len = win_len      # the length of window. or the length of  n-gram model
        self.vec_len = vec_len      # the length of word vector

    def merge(self,node1,node2):
        top_pos = node1.possibility + node2.possibility
        top_node = HuffmanTreeNode([0]*self.vec_len, top_pos)
        






