"""
@author Jacob
@time 2019/06/13


Preprocessing data

"""

"""
# 1. Standardization, or mean removal and variance scaling

Standardization of datasets is a common requirement for many machine leaning estimators implemented in sikit-learn;
they might behave badly if the individual features do not more or less look like standard normally distributed data:
Gaussian with zero mean and unit variance.

For instance, many elements used in the objective function of a learning algorithm (such as the RBF kernel of SVM or
the l1 and l2 regularizers of linear models) assume that all features are centered around zero and hva variance in 
the same order. If a feature has a variance that is orders of magnitude larger than others, it might dominate the 
objective function and make the estimator unable to learn from other features correctly as expected.
"""

from sklearn import preprocessing
import numpy as np

X_train = np.array([[1., -1, 2.],
                    [2., 0., 0.],
                    [0., 1., -1.]])
X_scaled = preprocessing.scale(X_train)

print(X_scaled)
# array([[ 0.  ..., -1.22...,  1.33...],
#        [ 1.22...,  0.  ..., -0.26...],
#        [-1.22...,  1.22..., -1.06...]])


# Scaled data has zero mean and unit variance:
print(X_scaled.mean(axis=0))
# [0, 0, 0]

print(X_scaled.std(axis=0))
# [1, 1, 1]

"""
# 1.1 Scaling features to a range

An alternative standardization is scaling features to lie between a given minimum and maximum value, often between zero 
and one, or so that the maximum absolute value of each feature is scaled to unit size. This can be achieved using 
MinMaxScaler or MaxAbsScaler, respectively.

The motivation to use this scaling include robustness to very small standard deviations of features and preserving zero
entries in sparse data.
"""

min_max_scaler = preprocessing.MinMaxScaler()

X_train_minmax = min_max_scaler.fit_transform(X_train)

print(X_train_minmax)
# array([[0.5       , 0.        , 1.        ],
#        [1.        , 0.5       , 0.33333333],
#        [0.        , 1.        , 0.        ]])


"""
The same instance of the transformer can then be applied to some new test data unseen during the fit call: the same 
scaling and shifting operations will be applied to be consistent with the transformation performed on the train data:
"""

X_test = np.array([[-3., -1., 4.]])

X_test_minmax = min_max_scaler.transform(X_test)

print(X_test_minmax)
# array([[-1.5       ,  0.        ,  1.66666667]])

"""
It it possible to introspect the scaler attributes to find about the exact nature of the transformation learned on the
training data:
"""

print(min_max_scaler.scale_)
# array([0.5       , 0.5       , 0.33...])

print(min_max_scaler.min_)
# array([0.        , 0.5       , 0.33...])


"""
MaxAbsScaler works in a very similar fashion, but scales in a way that the training data lies within the range [-1, 1] 
by dividing through the largest maximum value in each feature. It is meant for data that is already centered at zero or 
sparse data.
"""

max_abs_scaler = preprocessing.MaxAbsScaler()

X_train_maxabs = max_abs_scaler.fit_transform(X_train)

print(X_train_maxabs)
# array([[ 0.5, -1. ,  1. ],
#        [ 1. ,  0. ,  0. ],
#        [ 0. ,  1. , -0.5]])

X_test_maxabs = max_abs_scaler.transform(X_test)
print(X_test_maxabs)
# array([[-1.5, -1. ,  2. ]])

print(max_abs_scaler.scale_)
# array([2.,  1.,  2.])

"""
# 1.2 Scaling sparse data

...
"""

"""
# 2. Non-linear transformation

Two types of transformations are available: quantile transforms and power transforms. Both quantile and power transforms 
are based on monotonic transformations of the features and thus preserve the rank of the values along each feature.

Quantile transforms put all features into the same desired distribution based on the formula G-1(F(X)) where F is the 
cumulative distribution function of the feature and G-1 the quatile function of the desired output distribution G.
This formula is using the two following facts: (i) if X is a random variable with a continuous cumulative distribution
function F then F(X) is uniformly distributed on [0, 1]; (ii) if U is a random variable with uniform distribution on 
[0, 1] then G-1(U) has distribution G. By performing a rank transformation, a quantile transform smooths out unusual 
distributions and is less influenced by outliers than scaling methods. It does, however, distort correlations and 
distances within and across features.

Power transforms are a family of parametric transformations that aim to map data from any distribution to as close to a 
Gaussian distribution. 
"""

"""
# 2.1 Mapping to a Uniform distribution

QuantileTransformer and quantile_transform provide a non-parametric transformation to map the data to a uniform 
distribution with values between 0 and 1.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
X_train_trans = quantile_transformer.fit_transform(X_train)
X_test_trans = quantile_transformer.transform(X_test)

print(np.percentile(X_train[:, 0], [0, 25, 50, 75, 100]))
# array([ 4.3,  5.1,  5.8,  6.5,  7.9])

"""
This feature corresponds to the sepal length in cm. Once the quantile transformation applied, those landmarks approach 
closely the percentiles previously defined
"""

print(np.percentile(X_train_trans[:, 0], [0, 25, 50, 75, 100]))
# array([ 0.00... ,  0.24...,  0.49...,  0.73...,  0.99... ])

"""
# 2.2 Mapping to a Gaussian distribution

In many modeling scenarios, normality of the features in a dataset is desirable. Power transforms are a family of 
parametric, monotonic transformations that aim to map data from any distribution to as close to a Gaussian distribution
as possible in order to stabilize variance and minimize skewness.

PowerTransformer currently provides two such power transformations, the Yeo-Johnson transform and Box-Cox transform.

Box-Cox can only be applied to strictly positive data. In both methods, the transformation is parameterized by lambda, 
which is determined through maximum likelihood estimation. Here is an example of using Box-Cox to map samples drawn from
a lognormal distribution to a normal distribution:
"""

pt = preprocessing.PowerTransformer(method='box-cox', standardize=False)
X_lognormal = np.random.RandomState(616).lognormal(size=(3, 3))
print(X_lognormal)
# array([[1.28..., 1.18..., 0.84...],
#        [0.94..., 1.60..., 0.38...],
#        [1.35..., 0.21..., 1.09...]])

print(pt.fit_transform(X_lognormal))
# array([[ 0.49...,  0.17..., -0.15...],
#        [-0.05...,  0.58..., -0.57...],
#        [ 0.69..., -0.84...,  0.10...]])

"""
While the above example sets the `standardize` option to `False`, `PowerTransformer` will apply zero-mean, 
unit-variance normalization to the transformed output by default.

It is also possible to map data to a normal distribution using `QuantileTransformer` by setting 
`output_distribution='normal'`.
"""

quantile_transformer = preprocessing.QuantileTransformer(
    output_distribution='normal',
    random_state=0
)
X_trans = quantile_transformer.fit_transform(X)

print(quantile_transformer.quantiles_)
# array([[4.3, 2. , 1. , 0.1],
#        [4.4, 2.2, 1.1, 0.1],
#        [4.4, 2.2, 1.2, 0.1],
#        ...,
#        [7.7, 4.1, 6.7, 2.5],
#        [7.7, 4.2, 6.7, 2.5],
#        [7.9, 4.4, 6.9, 2.5]])

"""
# 3. Normalization

Normalization is the process of scaling individual samples to have unit norm. This process can be useful if you plan to
use a quadratic form such as the dot-product or any other kernel to quantify the similarity of any pair of samples.

This assumption is the base of the Vector Space Model often used in text classification and clustering contexts.

...
"""

"""
$ 4. Encoding categorical features

...
"""

"""
# 5. Discretization

Discretization (otherwise known as quantization or binning) provides a way to partition continuous features into 
discrete values. Certain datasets with continuous features may benefit from discretization, because discretization can 
transform the dataset of continuous attributes to one with only nominal attributes.

...
"""
