__author__ = 'Andrew'

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import svm, metrics, cross_validation, neighbors
import matplotlib.pyplot as plt
import kmeans_types


class ZCAWhitening(object):
    """ZCA whitening"""

    def __init__(self, data, eps):
        self.data = data
        self.eps = eps
        self.components = np.nan

    def fit(self):
        sigma = np.dot(self.data.T, self.data) / float(self.data.shape[0])
        U, S, V = np.linalg.svd(sigma)
        tmp = np.dot(U, np.diag(1 / np.sqrt(S + self.eps)))
        self.components = np.dot(tmp, U.T)

    def predict(self, X):
        return np.dot(X, self.components)


class KMeansUFLPipelineOCR(object):
    """K-means UFL OCR pipeline"""

    def __init__(self, kmeans_method, patch_size=8,
                 pooling_grid=2, eps_norm=10, eps_zca=0.1, step_size=1, encoder_type='simple', simple_encoder_alpha=0.5,
                 complex_encoder_b=0.5,
                 classifier=svm.SVC()):
        """
        encoder_type = ['simple', 'complex']
        """
        self.k_means_method = kmeans_method
        self.patch_size = patch_size
        self.pooling_grid = pooling_grid
        self.eps_norm = eps_norm
        self.eps_zca = eps_zca
        self.step_size = step_size
        self.encoder_type = encoder_type
        self.simple_encoder_alpha = simple_encoder_alpha
        self.complex_encoder_b = complex_encoder_b
        self.dictionary = np.zeros(0)
        self.ZCA = np.nan
        self.classifier = classifier

    def get_patch(self, image, i_coord, j_coord):
        return [image[i * 28 + j] for i in xrange(i_coord, i_coord + self.patch_size) for j in
                xrange(j_coord, j_coord + self.patch_size)]

    def get_image_width(self, image):
        return int(np.sqrt(image.shape[0]))

    def get_image_length(self, image):
        return int(np.sqrt(image.shape[0]))

    def get_patched_data(self, data):
        print ' Start patching data'
        return np.array([self.get_patch(img, i, j) for img in data for i in
                         xrange(self.get_image_width(img) - self.patch_size + 1) for j in
                         xrange(self.get_image_length(img) - self.patch_size + 1)])

    def normalize_data(self, data):
        return ((data.T - data.mean(1)) / (np.sqrt(data.var(1) + self.eps_norm))).T

    def learn_dictionary(self, data):
        kmeans = self.k_means_method
        if isinstance(kmeans, kmeans_types.KMeansSPSA):
            for data_point in data:
                kmeans.fit(data_point)
        else:
            kmeans.fit(data)
        self.dictionary = kmeans.cluster_centers_.T

    def get_patched_image(self, image):
        img_width = self.get_image_width(image)
        img_length = self.get_image_length(image)
        return np.array([self.get_patch(image, i, j) for i in
                         xrange(0, img_width - self.patch_size + 1, self.step_size) for j in
                         xrange(0, img_length - self.patch_size + 1, self.step_size)])

    def simple_encoder(self, x, alpha=0.5):
        return np.maximum(0, np.dot(self.dictionary.T, x) - alpha)

    def complex_encoder(self, x, b=0.5):
        return (1 + np.exp(-np.dot(self.dictionary.T, x) + b)) ** (-1)

    def features_representation(self, image):
        represent = []
        for patch in self.normalize_data(self.get_patched_image(image)):
            patch = self.ZCA.predict(patch)
            if self.encoder_type == 'simple':
                represent.append(self.simple_encoder(x=patch, alpha=self.simple_encoder_alpha))
            elif self.encoder_type == 'complex':
                represent.append(self.complex_encoder(x=patch, b=self.complex_encoder_b))
        represent = np.array(represent)
        return self.pooling(image, represent)

    def pooling(self, image, Y):
        img_width = self.get_image_width(image)
        img_length = self.get_image_length(image)
        i_intervals = []
        j_intervals = []

        i_grid = range(0, img_width - self.patch_size + 1, self.step_size)
        j_grid = range(0, img_length - self.patch_size + 1, self.step_size)
        bound_val_i = len(i_grid) / self.pooling_grid
        bound_val_j = len(j_grid) / self.pooling_grid

        for l in range(self.pooling_grid):
            i_intervals.append(i_grid[(l * bound_val_i):((l + 1) * bound_val_i)])
            j_intervals.append(j_grid[(l * bound_val_j):((l + 1) * bound_val_j)])

        Z = []
        for i_int in i_intervals:
            for j_int in j_intervals:
                pool = np.array([Y[i * 21 + j] for i in i_int for j in j_int])
                Z.extend(pool.mean(0))
        return np.array(Z)

    def get_data_representation(self, data):
        print ' Data features representation'
        data_repr = []
        for img in data:
            data_repr.append(self.features_representation(img))
        return np.array(data_repr)

    def unsupervised_features_learning(self, data):
        print 'Unsupervised features learning'

        patch_data = self.get_patched_data(data)
        print ' Patch dataset: count: %d, dim: %d' % patch_data.shape

        print ' Normalization'
        patch_data = self.normalize_data(patch_data)
        print ' ZCA whitening'
        self.ZCA = ZCAWhitening(patch_data, self.eps_zca)
        self.ZCA.fit()
        patch_data = self.ZCA.predict(patch_data)

        print ' Learning dictionary'
        self.learn_dictionary(patch_data)

    def fit(self, data, labels):
        print 'Training classifier'
        self.classifier.fit(self.get_data_representation(data), labels)

    def predict(self, data):
        print 'Prediction'
        return self.classifier.predict(self.get_data_representation(data))


def load_train_data(data_filename, is_random_part=False, part_size=1000):
    print 'Loading train data'
    df = pd.read_csv(data_filename)

    labels = df.label.as_matrix()
    data = df.ix[:, 1:].as_matrix()
    if is_random_part:
        indexes = np.random.choice(data.shape[0], size=part_size, replace=False)
        return data[indexes], labels[indexes]
    else:
        return data, labels


def load_test_data(path_test):
    print 'Loading test data'
    return pd.read_csv(path_test).as_matrix()


def write_labels_csv(path_labels, labels):
    print 'Writing labels'
    with open(path_labels, 'w') as f:
        f.write('ImageId,Label\n')
        for i in range(len(labels)):
            f.write(','.join([str(i + 1), str(labels[i])]) + '\n')


def describe_data(data):
    print 'Data count: %d, Data dim: %d' % data.shape


def plot_patches(data, n_row, n_col):
    plt.figure(figsize=(6, 6))
    for i, comp in enumerate(data):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.show()


if __name__ == '__main__':
    pipeline = KMeansUFLPipelineOCR(kmeans_method=kmeans_types.KMeansSPSA(n_clusters=25),
                                    classifier=svm.SVC())

    train_data, target = load_train_data('data/mnist/train.csv', is_random_part=True, part_size=1000)

    pipeline.unsupervised_features_learning(train_data)

    # train_data, target = load_train_data('data/mnist/train.csv')
    # pipeline.fit(train_data, target)

    # train_data = None
    # target = None

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_data, target, test_size=0.2,
                                                                         random_state=0)

    pipeline.fit(X_train, y_train)
    predicted = pipeline.predict(X_test)

    print("Classification report for classifier %s:\n%s\n"
          % (pipeline.classifier, metrics.classification_report(y_test, predicted)))

    plot_patches(pipeline.dictionary.T, 5, 5)

    # test_data = load_test_data('data/mnist/test.csv')
    # write_labels_csv('data/mnist/test_labels.csv', pipeline.predict(test_data))


