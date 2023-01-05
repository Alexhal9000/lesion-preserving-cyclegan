import glob
import csv
import os
import random
import math
import numpy as np


class PatientKeyInfo:

    def __init__(self, folder, keyfile, seed, train_size, combine_all=False, size_limit=-1, fold=0):

        names = glob.glob(folder)
        names = sorted(names)
        if size_limit > 0:
            names = names[0:size_limit]
        # print(names)
        assert len(names) > 0, "Could not find any training data"
        print("Number of samples: ", len(names))

        if keyfile != "none":


            with open(keyfile, newline='') as f:
                reader = csv.reader(f)
                data = list(reader)
                data = list(map(list, zip(*data)))
            # print(data)

            if train_size < 0:

                train_pre = []
                test_pre = []
                age_pre = []

                print(len(names))            

                for name in names:
                    name_extract = os.path.splitext(os.path.basename(name))[0]
                    separator_index = name_extract.index('_')
                    name_extract = name_extract[:separator_index]
                    # print(name_extract)
                    name_idx = data[0].index(name_extract)
                    if data[2][name_idx] == "train":
                        if combine_all:
                            train_pre.append(name)
                            test_pre.append(name)
                        else:
                            train_pre.append(name)
                        age_pre.append(int(data[1][name_idx]))

                    else:
                        if combine_all:
                            train_pre.append(name)
                            test_pre.append(name)
                        else:
                            test_pre.append(name)
                        age_pre.append(int(data[1][name_idx]))

                self.train = train_pre
                self.test = test_pre
                self.age = age_pre

            else:

                if train_size < 18:
                    train_pre = [None] * train_size
                    split_num = train_size
                else:
                    train_pre = [None] * 18
                    split_num = 18

                for n in range(split_num):
                    train_pre[n] = []

                test_pre = []

                for name in names:
                    name_extract = os.path.splitext(os.path.basename(name))[0]
                    separator_index = name_extract.index('_')
                    name_extract = name_extract[:separator_index]
                    # print(name_extract)
                    name_idx = data[0].index(name_extract)
                    if data[2][name_idx] == "train":
                        slot = math.floor(int(data[1][name_idx])/(18/split_num))
                        if slot > (split_num - 1):
                            slot = split_num - 1
                        train_pre[slot].append(name)
                    else:
                        test_pre.append(name)

                self.train = []
                if seed >= 0:
                    random.seed(seed)
                if train_size <= 18:
                    for i in range(train_size):
                        self.train.append(train_pre[i][random.randint(0, len(train_pre[i])-1)])

                self.test = test_pre

        else:

            if train_size < 0:

                train_pre = []
                val_pre = []
                test_pre = []

                print(len(names))   

                fold_range = int((len(names)*0.85)/5)
                kk = 0         

                for name in names:
                    if kk < int(len(names)*0.85):
                        if combine_all:
                            train_pre.append(name)
                            val_pre.append(name)
                            test_pre.append(name)
                        else:
                            if kk >= (fold_range * fold) and kk < (fold_range * (fold+1)):
                                val_pre.append(name)
                            else:
                                train_pre.append(name)

                    else:
                        if combine_all:
                            train_pre.append(name)
                            val_pre.append(name)
                            test_pre.append(name)
                        else:
                            test_pre.append(name)

                    kk = kk + 1


                self.train = train_pre
                self.test = test_pre
                self.val = val_pre

            else:

                if train_size < 18:
                    train_pre = [None] * train_size
                    split_num = train_size
                else:
                    train_pre = [None] * 18
                    split_num = 18

                for n in range(split_num):
                    train_pre[n] = []

                self.train = []
                self.test = []

                if seed>=0:
                    np.random.seed(seed)

                your_list = list(np.random.permutation(np.arange(0, int(len(names)/2) - 1))[:train_size])

                print(str(your_list))

                for n in your_list:
                    self.train.append(names[n])

                for n in range(int(len(names)/2), int(len(names))):
                    self.test.append(names[n])

                   