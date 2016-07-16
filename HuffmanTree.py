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
        self.value = value # the value of word
        self.Huffman = "" # store the huffman code

    def __str__(self):
        return 'HuffmanTreeNode object, value: {v}, possibility: {p}, Huffman: {h}'\
            .format(v=self.value,p=self.possibility,h=self.Huffman)

class HuffmanTree():
    def __init__(self, word_dict, vec_len=15000):
        self.vec_len = vec_len      # the length of word vector
        self.root = None

        word_dict_list = list(word_dict.values())
        node_list = [HuffmanTreeNode(x['word'],x['possibility']) for x in word_dict_list]
        self.build_tree(node_list)
        # self.build_CBT(node_list)
        self.generate_huffman_code(self.root, word_dict)

    def build_tree(self,node_list):
        # node_list.sort(key=lambda x:x.possibility,reverse=True)
        # for i in range(node_list.__len__()-1)[::-1]:
        #     top_node = self.merge(node_list[i],node_list[i+1])
        #     node_list.insert(i,top_node)
        # self.root = node_list[0]

        while node_list.__len__()>1:
            i1 = 0  # i1表示概率最小的节点
            i2 = 1  # i2 概率第二小的节点
            if node_list[i2].possibility < node_list[i1].possibility :
                [i1,i2] = [i2,i1]
            for i in range(2,node_list.__len__()): # 找到最小的两个节点
                if node_list[i].possibility<node_list[i2].possibility :
                    i2 = i
                    if node_list[i2].possibility < node_list[i1].possibility :
                        [i1,i2] = [i2,i1]
            top_node = self.merge(node_list[i1],node_list[i2])
            if i1<i2:
                node_list.pop(i2)
                node_list.pop(i1)
            elif i1>i2:
                node_list.pop(i1)
                node_list.pop(i2)
            else:
                raise RuntimeError('i1 should not be equal to i2')
            node_list.insert(0,top_node)
        self.root = node_list[0]

    def build_CBT(self,node_list): # build a complete binary tree
        node_list.sort(key=lambda  x:x.possibility,reverse=True)
        node_num = node_list.__len__()
        before_start = 0
        while node_num>1 :
            for i in range(node_num>>1):
                top_node = self.merge(node_list[before_start+i*2],node_list[before_start+i*2+1])
                node_list.append(top_node)
            if node_num%2==1:
                top_node = self.merge(node_list[before_start+i*2+2],node_list[-1])
                node_list[-1] = top_node
            before_start = before_start + node_num
            node_num = node_num>>1
        self.root = node_list[-1]

    def generate_huffman_code(self, node, word_dict):
        # # use recursion in this edition
        # if node.left==None and node.right==None :
        #     word = node.value
        #     code = node.Huffman
        #     print(word,code)
        #     word_dict[word]['Huffman'] = code
        #     return -1
        #
        # code = node.Huffman
        # if code==None:
        #     code = ""
        # node.left.Huffman = code + "1"
        # node.right.Huffman = code + "0"
        # self.generate_huffman_code(node.left, word_dict)
        # self.generate_huffman_code(node.right, word_dict)

        # use stack butnot recursion in this edition
        stack = [self.root]
        while (stack.__len__()>0):
            node = stack.pop()
            # go along left tree
            while node.left or node.right :
                code = node.Huffman
                node.left.Huffman = code + "1"
                node.right.Huffman = code + "0"
                stack.append(node.right)
                node = node.left
            word = node.value
            code = node.Huffman
            # print(word,'\t',code.__len__(),'\t',node.possibility)
            word_dict[word]['Huffman'] = code

    def merge(self,node1,node2):
        top_pos = node1.possibility + node2.possibility
        top_node = HuffmanTreeNode(np.zeros([1,self.vec_len]), top_pos)
        if node1.possibility >= node2.possibility :
            top_node.left = node1
            top_node.right = node2
        else:
            top_node.left = node2
            top_node.right = node1
        return top_node











