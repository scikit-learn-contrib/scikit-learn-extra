# Scikit-learn-extra Developer Documentation
**The official docs can be found [here](https://scikit-learn.org/stable/developers/index.html).**

### Scikit-learn-extra is a Python module that extends scikit-learn. It includes algorithms that do not satisfy scikit-learn [inclusion criteria](https://scikit-learn.org/stable/faq.html#what-are-the-inclusion-criteria-for-new-algorithms). For instance, this may be due to their novelty or lower citation number. 
----
# Installation Instructions:
### scikit-learn-extra requires:
- Python (>= 3.5)
- scikit-learn(>=0.21)
- Cython (>0.28)
### User Installation:
`pip install https://github.com/scikit-learn-contrib/scikit-learn-extra/archive/master.zip`

----

# User Guide:
### EigenPro for Regression and Classification
****
This is a very efficient implementation of kernel regressison/classification that uses an optimization method based on preconditioned stochastic gradient descent. Basically, it implements a "ridgeless" kernel regression. 
#### Examples:
More on EigenPro can be found [here](https://scikit-learn-extra.readthedocs.io/en/latest/modules/eigenpro.html)

### K-Medoids
****
K-Medoids is related to the KMeans algorithm. Rather than trying to minimize the within cluster sum-of-squares, K-Medoids tries to minimize the sum of distances between each point and the medoid of its cluster. K-Medoids can be more robust to noise and outliers in comparison to KMeans as well. 

#### Examples:
[A demo of K-Medoids clustering on the handwritten digits data (MNIST)](https://scikit-learn-extra.readthedocs.io/en/latest/auto_examples/plot_kmedoids_digits.html#sphx-glr-auto-examples-plot-kmedoids-digits-py) 
In this example, K-Medoids is applied to digits with various distance metrics. 

# scikit-learn-extra API:

### **Kernel Approximation**
****
` kernel_approximation.Fastfood([sigma, ...]) `

 Approximates feature map of an RBF kernel by Monte Carlo approximation of its Fourier transform.

### **EigenPro**
****
`kernel_methods.EigenProRegressor([...])`

 Regression using EigenPro iteration

`kernel_methods.EigenProClassifier([...])`

Classification using EigenPro iteration

### **Clustering**
****
`cluster.KMedoids([n_clusters, metric, init, ...])`

K-Medoids clustering