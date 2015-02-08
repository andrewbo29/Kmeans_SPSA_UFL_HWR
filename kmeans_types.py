__author__ = 'Andrew'

import numpy as np
import matplotlib.pyplot as plt


class KMeansClustering(object):
    """K-means clustering interface"""

    def __init__(self, n_clusters=8):
        self.n_clusters = n_clusters
        self.labels_ = np.zeros(0)
        self.cluster_centers_ = np.zeros(n_clusters)

    def cluster_decision(self, point):
        return np.argmin([((self.cluster_centers_[label] - point) ** 2).sum() for label in xrange(self.n_clusters)])

    def clusters_fill(self, data):
        self.labels_ = np.zeros(data.shape[0])
        for ind in xrange(data.shape[0]):
            label = self.cluster_decision(data[ind])
            self.labels_[ind] = label

    def fit_step(self, data):
        pass

    def fit(self, data):
        pass


class KMeansClassic(KMeansClustering):
    """Classic K-means implementation"""

    def __init__(self, n_clusters=8, max_iter=300, tol=0.0001, n_init=10, kmeans_pp=False):
        super(KMeansClassic, self).__init__(n_clusters=n_clusters)

        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.kmeans_pp = kmeans_pp

    def centers_init_rand(self, data):
        init_centers_ind = np.random.choice(data.shape[0], size=self.n_clusters, replace=False)
        self.cluster_centers_ = data[init_centers_ind]

    def centers_init_kmeans_pp(self, data):
        init_ind = np.random.choice(data.shape[0])
        centers_ind = [init_ind]
        init_centers = [data[init_ind]]
        for label in xrange(1, self.n_clusters):
            distances = []
            ind_set = []
            for i in xrange(data.shape[0]):
                if i not in centers_ind:
                    d = min([((data[i] - init_centers[prev_label]) ** 2).sum() for prev_label in xrange(label)])
                    distances.append(d)
                    ind_set.append(i)
            s = float(sum(distances))
            prob = [dst / s for dst in distances]
            mu_ind = np.random.choice(ind_set, replace=False, p=prob)
            centers_ind.append(mu_ind)
            init_centers.append(data[mu_ind])
        self.cluster_centers_ = np.array(init_centers)

    def centers_calc(self, data):
        for label in xrange(self.n_clusters):
            self.cluster_centers_[label] = data[self.labels_ == label].mean(0)

    def stop_criterion(self, old_centers):
        for label in xrange(self.n_clusters):
            if (np.abs(self.cluster_centers_[label] - old_centers[label]) > self.tol).any():
                return False
        return True

    def inter_cluster_dist(self, data):
        comm_sum = 0
        for label in xrange(self.n_clusters):
            comm_sum += sum(
                [((self.cluster_centers_[label] - point) ** 2).sum() for point in data[self.labels_ == label]])
        return comm_sum

    def fit_step(self, data):
        print '  Initialization classic K-means'
        if self.kmeans_pp:
            self.centers_init_kmeans_pp(data)
        else:
            self.centers_init_rand(data)
        self.clusters_fill(data)
        for label in xrange(self.n_clusters):
            if not data[self.labels_ == label].any():
                print 'Empty cluster'

        iter_num = 1
        while iter_num < self.max_iter:
            print '  My classic K-means iteration: %d' % iter_num
            old_centers = self.cluster_centers_
            self.centers_calc(data)
            self.clusters_fill(data)
            if self.stop_criterion(old_centers):
                break
            iter_num += 1

        return [self.cluster_centers_, self.inter_cluster_dist(data)]

    def fit(self, data):
        fit_steps = []
        for i in xrange(self.n_init):
            fit_steps.append(self.fit_step(data))
        best_ind = np.argmin([fit_steps[i][1] for i in xrange(self.n_init)])
        # print('Best k-means inter-clusters distance: %.2f' % fit_steps[best_ind][1])

        self.cluster_centers_ = fit_steps[best_ind][0]
        self.clusters_fill(data)


class KMeansSPSA(KMeansClustering):
    """SPSA K-means implementation"""

    def __init__(self, n_clusters, gamma=1./6, alpha=1./4, beta=15):
        super(KMeansSPSA, self).__init__(n_clusters=n_clusters)

        self.cluster_centers_ = []
        self.gamma = gamma
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.iteration_num = 1

    def fit(self, w):
        if self.iteration_num % 1000 == 0:
            print '  SPSA K-means iteration: %d' % self.iteration_num

        if self.iteration_num <= self.n_clusters:
            self.cluster_centers_.append(w)
        else:
            if self.iteration_num == self.n_clusters + 1:
                self.cluster_centers_ = np.array(self.cluster_centers_)
            self.fit_step(w)
        self.iteration_num += 1

    def y_vec(self, centers, w):
        return np.array([((w - centers[label]) ** 2).sum() for label in xrange(self.n_clusters)])

    def j_vec(self, w):
        y = self.y_vec(self.cluster_centers_, w)
        vec = np.zeros(self.n_clusters)
        vec[np.argmin(y)] = 1
        return vec

    def delta_fabric(self, d):
        return np.where(np.random.binomial(1, 0.5, size=d) == 0, -1, 1)

    def alpha_fabric(self):
        return self.alpha / (self.iteration_num ** self.gamma)

    def beta_fabric(self):
        return self.beta / (self.iteration_num ** (self.gamma / 4))

    def fit_step(self, w):
        delta_n_t = self.delta_fabric(w.shape[0])[np.newaxis]
        alpha_n = self.alpha_fabric()
        beta_n = self.beta_fabric()

        j_vec = self.j_vec(w)[np.newaxis].T
        j_vec_dot_delta_t = np.dot(j_vec, delta_n_t)

        y_plus = self.y_vec(self.cluster_centers_ + beta_n * j_vec_dot_delta_t, w)[np.newaxis]
        y_minus = self.y_vec(self.cluster_centers_ - beta_n * j_vec_dot_delta_t, w)[np.newaxis]
        self.cluster_centers_ -= j_vec_dot_delta_t * np.dot(alpha_n * (y_plus - y_minus) / (2. * beta_n), j_vec)


class KMeansSpherical(KMeansClustering):
    """Spherical K-means implementation"""

    def __init__(self, n_clusters=8, max_iter=300, tol=0.0001, damped_update=False, norm_dist_init=False):
        super(KMeansSpherical, self).__init__(n_clusters=n_clusters)

        self.max_iter = max_iter
        self.tol = tol
        self.damped_update = damped_update
        self.norm_dist_init = norm_dist_init
        self.S = np.zeros(0)

    def centers_init(self, data):
        if self.norm_dist_init:
            init_centers = []
            for _ in xrange(self.n_clusters):
                norm_dist_vec = np.random.multivariate_normal(np.zeros(data.shape[1]), np.eye(data.shape[1]))
                init_centers.append(norm_dist_vec / np.linalg.norm(norm_dist_vec))
            self.cluster_centers_ = np.array(init_centers).T
        else:
            init_centers_ind = np.random.choice(data.shape[0], size=self.n_clusters, replace=False)
            for i in xrange(len(init_centers_ind)):
                ind = init_centers_ind[i]
                if np.linalg.norm(data[ind]) == 0:
                    ind = np.random.choice(data.shape[0], size=1, replace=False)
                    while ind in init_centers_ind or np.linalg.norm(data[ind]) == 0:
                        ind = np.random.choice(data.shape[0], size=1, replace=False)
                init_centers_ind[i] = ind
            self.cluster_centers_ = data[init_centers_ind].T

    def stop_criterion(self, old_centers):
        for label in xrange(self.n_clusters):
            if (np.abs(self.cluster_centers_[:, label] - old_centers[:, label]) > self.tol).any():
                return False
        return True

    def fit(self, data):
        print '  Initialization spherical K-means'

        self.centers_init(data)
        self.S = np.zeros((self.n_clusters, data.shape[0]))

        iter_num = 1
        while iter_num <= self.max_iter:
            print '  Spherical K-means iteration: %d' % iter_num

            old_cluster_centers_ = self.cluster_centers_

            for i in xrange(data.shape[0]):
                j = np.nanargmax([np.abs(np.dot(self.cluster_centers_[:, l].T, data[i])) for l in xrange(self.n_clusters)])
                self.S[:, i] = 0
                self.S[j, i] = np.dot(self.cluster_centers_[:, j].T, data[i])

            if self.damped_update:
                self.cluster_centers_ = np.dot(data.T, self.S.T) + self.cluster_centers_
            else:
                self.cluster_centers_ = np.dot(data.T, self.S.T)

            for j in xrange(self.cluster_centers_.shape[1]):
                self.cluster_centers_[:, j] /= np.linalg.norm(self.cluster_centers_[:, j])

            if self.stop_criterion(old_cluster_centers_):
                break
            iter_num += 1
        self.cluster_centers_ = self.cluster_centers_.T

    def clusters_fill(self, data):
        self.labels_ = np.zeros(data.shape[0])
        for ind in xrange(data.shape[0]):
            label = np.argmax(self.S[:, ind])
            self.labels_[ind] = label


def plot_kmeans(data, kmeans):
    fig = plt.figure()

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.scatter(data[:, 0], data[:, 1], c='black')
    ax1.set_title('Input data')

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.scatter(data[:, 0], data[:, 1], c=kmeans.labels_)

    ax = fig.add_subplot(2, 1, 2)
    for label in xrange(kmeans.n_clusters):
        center_point = kmeans.cluster_centers_[label]
        ax.plot(center_point[0], center_point[1], 's')

    if isinstance(kmeans, KMeansClassic):
        ax.set_title('Classic K-means')
    elif isinstance(kmeans, KMeansSPSA):
        ax.set_title('SPSA K-means')
    elif isinstance(kmeans, KMeansSpherical):
        ax.set_title('Spherical K-means')
    plt.show()


if __name__ == '__main__':
    N = 500
    mix_prob = [0.5, 0.3, 0.2]
    clust_means = [[0, 0], [2, 2], [-2, 4]]
    clust_cov = [np.diag([1, 1]), [[1, -0.7], [-0.7, 1]], [[1, 0.7], [0.7, 1]]]
    data_set = []

    # kmeans = KMeansSPSA(n_clusters=3)
    for _ in xrange(N):
        mix_ind = np.random.choice(len(mix_prob), p=mix_prob)
        data_point = np.random.multivariate_normal(clust_means[mix_ind], clust_cov[mix_ind])
        data_set.append(data_point)
        # kmeans.fit(data_point)
    data_set = np.array(data_set)

    # kmeans.clusters_fill(data_set)

    # kmeans = KMeansClassic(n_clusters=3, kmeans_pp=False)
    # kmeans.fit(data_set)

    kmeans = KMeansSpherical(n_clusters=3, norm_dist_init=True)
    kmeans.fit(data_set)
    kmeans.clusters_fill(data_set)

    plot_kmeans(data_set, kmeans)