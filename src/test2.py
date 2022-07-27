from abc import ABC, abstractmethod
import pandas as pd






class Trainpip(ABC):

    @abstractmethod
    def read_data(self):
        pass

    @staticmethod
    @abstractmethod
    def read_data2():
        pass
    # @staticmethod
    def generate(self, value):
        return value

    def print_func(self):
        self.read_data()
        self.my_para()
        print(self.nn.shape, self.generate(88))

    @classmethod
    def print_func2(clf):
        aa = clf.read_data2()
        print('static', aa.shape)

    @abstractmethod
    def my_para(self):
        pass

class data_process(Trainpip):
    def read_data(self):
        self.nn = pd.read_csv('/users/hao/documents/MA_thesis/ncf-torch2/data/jobs/debug/test_pos_neg.csv')

    @staticmethod
    def read_data2():
        return pd.read_csv('/users/hao/documents/MA_thesis/ncf-torch2/data/jobs/debug/test_pos_neg.csv')
    def my_para(self):
        print('this is my own params', self.nn.shape)

aa = data_process()
aa.print_func2()