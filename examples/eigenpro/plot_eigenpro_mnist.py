"""
===============================================
Comparison of EigenPro and SVC on Fashion-MNIST
===============================================

Here we train a EigenPro Classifier and a Support
Vector Classifier (SVC) on subsets of MNIST of various sizes.
We halt the training of EigenPro after two epochs.
Experimental results on MNIST demonstrate more than 3 times
speedup of EigenPro over SVC in training time. EigenPro also
shows consistently lower classification error on test set.
"""
print(__doc__)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from time import time

from sklearn_extra.kernel_methods import EigenProClassifier
from sklearn.svm import SVC
from sklearn.datasets import fetch_openml

rng = np.random.RandomState(1)

#  Generate sample data from mnist
mnist = fetch_openml("Fashion-MNIST")
mnist.data = mnist.data / 255.0
print("Data has loaded")

p = rng.permutation(60000)
x_train = mnist.data[p]
y_train = np.int32(mnist.target[p])
x_test = mnist.data[60000:]
y_test = np.int32(mnist.target[60000:])

# Run tests comparing eig to svc
eig_fit_times = []
eig_pred_times = []
eig_err = []
svc_fit_times = []
svc_pred_times = []
svc_err = []

train_sizes = [500, 1000, 2000]

print("Train Sizes: " + str(train_sizes))

bandwidth = 5.0

# Fit models to data
for train_size in train_sizes:
    for name, estimator in [
        (
            "EigenPro",
            EigenProClassifier(
                n_epoch=2, bandwidth=bandwidth, random_state=rng
            ),
        ),
        (
            "SupportVector",
            SVC(
                C=5, gamma=1.0 / (2 * bandwidth * bandwidth), random_state=rng
            ),
        ),
    ]:
        stime = time()
        estimator.fit(x_train[:train_size], y_train[:train_size])
        fit_t = time() - stime

        stime = time()
        y_pred_test = estimator.predict(x_test)
        pred_t = time() - stime

        err = 100.0 * np.sum(y_pred_test != y_test) / len(y_test)
        if name == "EigenPro":
            eig_fit_times.append(fit_t)
            eig_pred_times.append(pred_t)
            eig_err.append(err)
        else:
            svc_fit_times.append(fit_t)
            svc_pred_times.append(pred_t)
            svc_err.append(err)
        print(
            "%s Classification with %i training samples in %0.2f seconds."
            % (name, train_size, fit_t + pred_t)
        )

# set up grid for figures
fig = plt.figure(num=None, figsize=(6, 4), dpi=160)
ax = plt.subplot2grid((2, 2), (0, 0), rowspan=2)

# Graph fit(train) time
train_size_labels = [str(s) for s in train_sizes]
ax.plot(train_sizes, svc_fit_times, "o--", color="g", label="SVC")
ax.plot(
    train_sizes, eig_fit_times, "o-", color="r", label="EigenPro Classifier"
)
ax.set_xscale("log")
ax.set_yscale("log", nonposy="clip")
ax.set_xlabel("train size")
ax.set_ylabel("time (seconds)")
ax.legend()
ax.set_title("Training Time")
ax.set_xticks(train_sizes)
ax.set_xticklabels(train_size_labels)
ax.set_xticks([], minor=True)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

# Graph prediction(test) time
ax = plt.subplot2grid((2, 2), (0, 1), rowspan=1)
ax.plot(train_sizes, eig_pred_times, "o-", color="r")
ax.plot(train_sizes, svc_pred_times, "o--", color="g")
ax.set_xscale("log")
ax.set_yscale("log", nonposy="clip")
ax.set_ylabel("time (seconds)")
ax.set_title("Prediction Time")
ax.set_xticks([])
ax.set_xticks([], minor=True)

# Graph training error
ax = plt.subplot2grid((2, 2), (1, 1), rowspan=1)
ax.plot(train_sizes, eig_err, "o-", color="r")
ax.plot(train_sizes, svc_err, "o-", color="g")
ax.set_xscale("log")
ax.set_xticks(train_sizes)
ax.set_xticklabels(train_size_labels)
ax.set_xticks([], minor=True)
ax.set_xlabel("train size")
ax.set_ylabel("Classification error %")
plt.tight_layout()
plt.show()
