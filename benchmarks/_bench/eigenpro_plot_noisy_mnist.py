import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from time import time

from sklearn.datasets import fetch_openml
from sklearn_extra.kernel_methods import EigenProClassifier
from sklearn.svm import SVC

rng = np.random.RandomState(1)

# Generate sample data from mnist
mnist = fetch_openml("mnist_784")
mnist.data = mnist.data / 255.0

p = rng.permutation(60000)
x_train = mnist.data[p][:60000]
y_train = np.int32(mnist.target[p][:60000])
x_test = mnist.data[60000:]
y_test = np.int32(mnist.target[60000:])

# randomize 20% of labels
p = rng.choice(len(y_train), np.int32(len(y_train) * 0.2), False)
y_train[p] = rng.choice(10, np.int32(len(y_train) * 0.2))
p = rng.choice(len(y_test), np.int32(len(y_test) * 0.2), False)
y_test[p] = rng.choice(10, np.int32(len(y_test) * 0.2))

# Run tests comparing fkc to svc
eig_fit_times = []
eig_pred_times = []
eig_err = []
svc_fit_times = []
svc_pred_times = []
svc_err = []

train_sizes = [500, 1000, 2000, 5000, 10000, 20000, 40000, 60000]

gamma = 0.02

# Fit models to data
for train_size in train_sizes:
    for name, estimator in [
        (
            "EigenPro",
            EigenProClassifier(n_epoch=2, gamma=gamma, random_state=rng),
        ),
        ("SupportVector", SVC(C=5, gamma=gamma)),
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
            "%s Classification with %i training samples in %0.2f seconds. "
            "Test error %.4f" % (name, train_size, fit_t + pred_t, err)
        )

# set up grid for figures
fig = plt.figure(num=None, figsize=(6, 4), dpi=160)
ax = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
train_size_labels = ["500", "1k", "2k", "5k", "10k", "20k", "40k", "60k"]

# Graph fit(train) time
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.plot(train_sizes, svc_fit_times, "o--", color="g", label="SVC")
ax.plot(train_sizes, eig_fit_times, "o-", color="r", label="EigenPro")
ax.set_xscale("log")
ax.set_yscale("log", nonposy="clip")
ax.set_xlabel("train size")
ax.set_ylabel("time (seconds)")
ax.legend()
ax.set_title("Train set")
ax.set_xticks(train_sizes)
ax.set_xticks([], minor=True)
ax.set_xticklabels(train_size_labels)

# Graph prediction(test) time
ax = plt.subplot2grid((2, 2), (0, 1), rowspan=1)
ax.plot(train_sizes, eig_pred_times, "o-", color="r")
ax.plot(train_sizes, svc_pred_times, "o--", color="g")
ax.set_xscale("log")
ax.set_yscale("log", nonposy="clip")
ax.set_ylabel("time (seconds)")
ax.set_title("Test set")
ax.set_xticks(train_sizes)
ax.set_xticks([], minor=True)
ax.set_xticklabels(train_size_labels)

# Graph training error
ax = plt.subplot2grid((2, 2), (1, 1), rowspan=1)
ax.plot(train_sizes, eig_err, "o-", color="r")
ax.plot(train_sizes, svc_err, "o-", color="g")
ax.set_xscale("log")
ax.set_xticks(train_sizes)
ax.set_xticklabels(train_size_labels)
ax.set_xticks([], minor=True)
ax.set_xlabel("train size")
ax.set_ylabel("classification error %")
plt.tight_layout()
plt.show()
