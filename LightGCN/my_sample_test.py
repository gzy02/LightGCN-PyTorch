import numpy as np
# from cppimport import imp_from_filepath
# path = "./my_sample.cpp"
# my_sample = imp_from_filepath(path)
import my_sample
my_dict = {1: {2, 3}, 2: {1, 0}}
fake_num = 1
weight = [6, 4, 2, 3]
my_sample.set_seed(2023)
res = my_sample.sample_weightBPR(my_dict, weight)
print(res)
my_sample.set_seed(2021)
res = my_sample.sample_weightBPR(my_dict, weight)
print(res)
