__author__ = 'Andrew'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compare_two_test_results(test_file_1, test_file_2):
    df_1 = pd.read_csv(test_file_1)
    df_2 = pd.read_csv(test_file_2)

    df_1_mismatch = []
    df_2_mismatch = []
    for num in range(10):
        df_1_mismatch.append((df_2[df_1.Label == num].Label != num).sum())
        df_2_mismatch.append((df_1[df_2.Label == num].Label != num).sum())

    df = pd.DataFrame({'K = 700, UFL data size = 3000': df_1_mismatch, 'K = 1000, UFL data size = 3000': df_2_mismatch})

    df.plot(kind='bar', color=('r', 'b'), alpha=0.6)
    plt.show()


def visualize_two_test_mismatch(test_file_1, test_file_2, path_test, n_row, n_col):
    df_1 = pd.read_csv(test_file_1)
    df_2 = pd.read_csv(test_file_2)

    mismatch_ind = []
    for ind in range(df_1.shape[0]):
        if df_1.Label[ind] != df_2.Label[ind]:
            mismatch_ind.append(ind)

    rand_indices = np.random.choice(mismatch_ind, size=n_row * n_col, replace=False)
    data = pd.read_csv(path_test).as_matrix()[rand_indices]

    plt.figure(figsize=(n_row, n_col))
    plt.suptitle('Files: %s, %s' % (test_file_1, test_file_2))
    for i, comp in enumerate(data):
        plt.subplot(n_row, n_col, i + 1)
        plt.title('%d, %d' % (df_1.Label[rand_indices[i]], df_2.Label[rand_indices[i]]))
        plt.imshow(comp.reshape((28, 28)), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.show()


def prediction_errors_bars(predict, y):
    bars = []
    for num in range(10):
        bars.append((predict[y == num] != num).sum())

    pd.Series(bars).plot(kind='bar', alpha=0.6)
    plt.show()


if __name__ == '__main__':
    # compare_two_test_results('mnist/results/test_labels1022.csv', 'mnist/results/test_labels2359.csv')

    # visualize_two_test_mismatch('mnist/results/test_labels1022.csv', 'mnist/results/test_labels2359.csv',
    #                             'mnist/data/test.csv', 5, 10)

    y = np.array([0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9])
    predict = np.array([1,0,0,1,2,3,3,2,4,4,5,5,6,6,7,7,8,9,8,9])
    prediction_errors_bars(predict, y)




