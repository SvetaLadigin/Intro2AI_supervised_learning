import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


class decisionTree(object):

    def __init__(self, data, M, pruning):
        self.data = data
        self.classification = None
        self.divider_feature = None
        self.divider_value = None
        self.big = None
        self.small = None
        self.pruning = pruning
        self.M = M

    def get_classification_tree(self):
        self.divider_feature, self.divider_value = self.get_divider_feature()
        if self.divider_feature is None:
            return
        self.data = self.data[np.argsort(self.data[:, self.divider_feature])]
        row_to_split_by = 0
        for row in self.data:
            if row[self.divider_feature] >= self.divider_value:
                break
            row_to_split_by += 1
        small_array = self.data[:row_to_split_by, :]
        big_array = self.data[row_to_split_by:, :]
        self.big = decisionTree(big_array, self.M, self.pruning)
        self.big.get_classification_tree()
        self.small = decisionTree(small_array, self.M, self.pruning)
        self.small.get_classification_tree()

    def get_id3_value(self, feature, value):
        big_healthy = 0
        big_sick = 0
        small_healthy = 0
        small_sick = 0
        info_gain = 0

        for row in self.data:
            if row[feature] >= value:
                if row[0] == 'B':
                    big_healthy += 1
                else:
                    big_sick += 1
            else:
                if row[0] == 'B':
                    small_healthy += 1
                else:
                    small_sick += 1

        small_subjects = small_sick + small_healthy
        big_subjects = big_sick + big_healthy

        if self.pruning == True:
            if small_subjects < self.M or big_subjects < self.M:
                return None
        else:
            if small_subjects == 0 or big_subjects == 0:
                return None

        sick_ratio = (small_sick + big_sick) / len(self.data)
        healthy_ratio = (small_healthy + big_healthy) / len(self.data)
        if sick_ratio != 0 and healthy_ratio != 0:
            sum_entropy = -(sick_ratio * np.log2(sick_ratio) + healthy_ratio * np.log2(healthy_ratio))
            info_gain += sum_entropy

        if small_sick != 0 and small_healthy != 0:
            small_sick_ratio = small_sick / small_subjects
            small_healthy_ratio = small_healthy / small_subjects
            small_entropy = -(small_sick_ratio * np.log2(small_sick_ratio) + small_healthy_ratio * np.log2(small_healthy_ratio))
            info_gain -= (small_subjects / len(self.data)) * small_entropy

        if big_sick != 0 and big_healthy != 0:
            big_sick_ratio = big_sick / big_subjects
            big_healthy_ratio = big_healthy / big_subjects
            big_entropy = -(big_sick_ratio * np.log2(big_sick_ratio) + big_healthy_ratio * np.log2(big_healthy_ratio))
            info_gain -= (big_subjects / len(self.data)) * big_entropy

        return info_gain

    def get_all_values_for_feature(self, feature):
        feature_values = self.data[:,feature]
        feature_values = list(dict.fromkeys(feature_values))
        feature_values.sort()
        result = []
        for j in range(1, len(feature_values)):
            result.append((feature_values[j] + feature_values[j - 1]) / 2)
        return result

    def get_divider_feature(self):
        if self.is_identical_and_classify() is True:
            return None, None
        feature = len(self.data[0]) - 1
        info_gain = {}
        while feature != 0:
            feature_values = self.get_all_values_for_feature(feature)
            for value in feature_values:
                info_gain_value = self.get_id3_value(feature, value)
                if info_gain_value is not None and info_gain_value not in info_gain:
                    info_gain[info_gain_value] = (feature, value)
            feature -= 1

        if len(info_gain.keys()) == 0:  # no feature found
            self.classify()
            return None, None

        max_info_gain = max(info_gain.keys())
        return info_gain[max_info_gain][0], info_gain[max_info_gain][1]

    def is_identical_and_classify(self):
        b_num = 0
        m_num = 0
        for row in self.data:
            if row[0] == 'M':
                m_num += 1
            else:
                b_num += 1
        if b_num == 0:
            self.classification = 'M'
            return True
        elif m_num == 0:
            self.classification = 'B'
            return True
        return False

    def classify(self):
        b_num = 0
        m_num = 0

        for row in self.data:
            if row[0] == 'M':
                m_num += 1
            else:
                b_num += 1

        if b_num > m_num:
            self.classification = 'B'
        else:
            self.classification = 'M'


class ID3Algo(object):

    def __init__(self, M=1, pruning=False):
        self.decision_tree = None
        self.pruning = pruning
        self.M = M

    def fit_predict(self, train, test):
        self.decision_tree = decisionTree(train, self.M, self.pruning)
        self.decision_tree.get_classification_tree()
        classification_list = []
        for row in test:
            current_decision_tree = self.decision_tree
            while current_decision_tree.divider_feature is not None:
                if row[current_decision_tree.divider_feature] < current_decision_tree.divider_value:
                    current_decision_tree = current_decision_tree.small
                else:
                    current_decision_tree = current_decision_tree.big

            if current_decision_tree.classification == 'M':
                classification_list.append(1)
            else:
                classification_list.append(0)
        return classification_list


def experiment(train_set):
    M_list = [1,2,3,4 ,5,6,7,8,9, 10,11,12,13,14, 15,16,17,18,19, 20, 25, 50]
    precisions_list = []
    for M in M_list:
        precision_sum = 0
        ID3_result = ID3Algo(M, True)
        kf = KFold(n_splits=5, shuffle=True, random_state=319649778)
        indexes = kf.split(train_set)
        for train_set_index, test_set_index in indexes:
            sub_test_list = []
            sub_train_list = []
            for i in range(len(train_set)):
                if i in train_set_index:
                    sub_train_list.append(train_set[i])
                else:
                    sub_test_list.append(train_set[i])
            test_sub_set = np.array(sub_test_list)
            train_sub_set = np.array(sub_train_list)
            numpy_array = ID3_result.fit_predict(train_sub_set, test_sub_set)
            right_counter = 0
            wrong_counter = 0
            for i in range(len(numpy_array)):
                if (numpy_array[i] == 1 and test_sub_set[i][0] == 'M') or \
                        (numpy_array[i] == 0 and test_sub_set[i][0] == 'B'):
                    right_counter += 1
                else:
                    wrong_counter += 1
            precision_sum += right_counter / (right_counter + wrong_counter)
        precision_avg = precision_sum / 5
        precisions_list.append(precision_avg)
    plt.plot(M_list, precisions_list, color='green', linestyle='solid', linewidth=1, marker='o', markerfacecolor='green', markersize=5)
    plt.xlabel('pruning values')
    plt.ylabel('precision')
    plt.show()


def main():
    train_set = pd.read_csv('train.csv', sep=',', header=None)
    train_set_ndarray = train_set.to_numpy()

    '''
    #  this note is for question 3 - the experiment part   
    '''
    experiment(train_set_ndarray)

if __name__ == '__main__':
    main()
