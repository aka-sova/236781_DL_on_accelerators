r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**
1. A test set enables us to test our model after we finished training it and it will give us an estimation how
our model will preform in real conditions.
2.Test set will be chosen from our dataset, percentage of the test set and how we choose it can be modified.
3.In training the test set will not be used, but It will be used to evaluated performance of the trained model. When we
preform cross-validation a test set is different every iteration and the trained model is tested on it. This methods should
increase performance of the model.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**

We always split our dataset into training,validation and test sets. Validation is the part were we fine tune our model 
parameters and prevent it from over fitting/under fitting and generalize it. It would be wrong to validate our model with a 
test set because it will overfit our model.


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
When k=1  0 classification errors occurs during training set, because we use the single observation to classify itself. 
Our predictions become less stable since only one point in the training set will predict the label thus,
generalization error increases for very low K values.

On the other hand as we increase the value of K, our predictions become more stable due to
majority voting, and thus, more likely to make more accurate predictions (up to an
optimal point). When we increase K above the optimal point, large number of data
points from the training set are used for prediction of the unseen data. This increases
generalization error and the accuracy reduces.


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**

If we train and select the model based on the same set, the train set prediction accuracy will
be high. Yet, for unseen data the accuracy will be low due to generalization error.
If we train on the entire set and select the model based on a sub-set of it, generalization
error is also relatively high as the model is selected based on a subset of the training set.
By using K-fold CV, we split the data into a number of folds. If we have N folds, then the first
step is to train the algorithm using (N−1) of the folds, and test the algorithm’s accuracy on
the single left-out fold. This is then repeated N times until each fold has been used as in
the test set.
This reduces generalization error by:
1. The train set and test set are distinct
2. Splitting the train/test randomly provides a high variance estimate since
changing which observations happen to be in the testing set can significantly
change testing accuracy; moreover, testing accuracy can change a lot
depending on a which observation happen to be in the testing set.
It can be concluded that the selection of test / train set is crucial and CV fold
reduces generalization error by optimizing this split.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**

The ideal pattern to see in a residual plot is a constant behavior of the error, centered around 0 and.
From the plots we can see that the fitness of the trained model improves for top-5-features improves after CV, error margin decrease
and the error points are concentrated around error value 0 in comparison to before we CV where they were more distributed.


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q3 = r"""
**Your answer:**

lambda was set using logspace instead of linspace because we want to search the best order of magnitude
that will produce the best parameters. 
Since lambda is used to penalize the parameters' magnitude, small changes in lambda will insure insignificant changes 
in the parameters' sizes. Therefore using values that are close is a waste of time while searching among different orders 
of magnitude is more efficient for this hyperparameter.


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
