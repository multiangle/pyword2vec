__author__ = 'multiangle'

import numpy as np

class HuffmanTreeNode():
    def __init__(self,value,possibility):
        # common part of leaf node and tree node
        self.possibility = possibility
        self.left = None
        self.right = None
        # value of leaf node  will be the word, and be
        # mid vector in tree node
        self.value = value
        self.Huffman = None # store the huffman code

class HuffmanTree():
    def __init__(self, word_dict, vec_len=15000):
        self.vec_len = vec_len      # the length of word vector
        self.root = None

        word_dict_list = list(word_dict.values())
        node_list = [HuffmanTreeNode(x['word'],x['possibility']) for x in word_dict_list]
        self.build_tree(node_list)
        self.generate_huffman_code(self.root, word_dict)

    def build_tree(self,node_list):
        node_list.sort(key=lambda x:x.possibility,reverse=True)
        for i in range(node_list.__len__()-1)[::-1]:
            top_node = self.merge(node_list[i],node_list[i+1])
            node_list.insert(i,top_node)
        self.root = node_list[0]

    def generate_huffman_code(self, node, word_dict):
        # use recursion in this edition
        if node.left==None and node.right==None :
            word = node.value
            code = node.Huffman
            word_dict[word]['Huffman'] = code
            return

        code = node.Huffman
        if code==None:
            code = ""
        node.left.Huffman = code + "1"
        node.right.Huffman = code + "0"
        self.generate_huffman_code(node.left, word_dict)
        self.generate_huffman_code(node.right, word_dict)

    def merge(self,node1,node2):
        top_pos = node1.possibility + node2.possibility
        top_node = HuffmanTreeNode(np.zeros(self.vec_len), top_pos)
        if node1.possibility >= node2.possibility :
            top_node.left = node1
            top_node.right = node2
        else:
            top_node.left = node2
            top_node.right = node1
        return top_node











