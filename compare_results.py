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

    plt.figure()
    plt.hist([df_1_mismatch, df_2_mismatch])
    plt.show()


compare_two_test_results('mnist/results/test_labels1022.csv', 'mnist/results/test_labels2359.csv')


