r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Answer:**

1. The test set allows us to estimate the Expected loss (and not in-sample loss). We asses how
well the model is generalized to the whole set of inputs. A test set enables us to test our model after we finished training it and it will give us an estimation how
our model will preform in real conditions.

2. The test set is independent of the training set. It should represent the variety of
outputs which the model should learn to generalize upon. Test set will be chosen from our dataset, 
percentage of the test set and how we choose it can be modified.

3. The test set is used in the performance evaluation of the model.
On others:
- Training uses the training set only to update its parameters
- Cross-validation uses the training and validation set (splitting the original training set into 
training and validation subsets in different variations, e.g. k-fold)
- Choosing one model over another is done by looking at the performance on the validation dataset.
"""

part1_q2 = r"""
**Answer:**
Yes, we do need to split a part of the training set into the validation set. 
The validation set is used to display how well the model is learning to generalize, whether it's over/underfit.
We make a decisions on the network structure and the hyperparameter based on it
Test set is only used to assess the performance of the fully-trained model. This set is sometimes crafted manually,
to display the full spectrum of the input data that the model may encounter.
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Answer:**
Choosing the correct K value for the KNN algorithm hugely depends on the dataset that we
want to classify. 

When k=1  0 classification errors occurs during training set, because we use the single observation to classify itself. 
Our predictions become less stable since only one point in the training set will predict the label thus,
generalization error increases for very low K values.

Choosing small K value would lead to a big influence of the noise on the inference.


On the other hand as we increase the value of K, our predictions become more stable due to
majority voting, and thus, more likely to make more accurate predictions (up to an
optimal point). When we increase K above the optimal point, large number of data
points from the training set are used for prediction of the unseen data. This increases
generalization error and the accuracy reduces.

But with large K values we may miss some small classes which are represented with less amount of samples in the 
dataset, then some class with has similar features, but represented with a bigger amount of samples.





"""

part2_q2 = r"""
**Answer:**

1. Choosing a model with respect to the training set accuracy would be wrong. Training set accuracy
is valid for the in-sample Set, and doesn't reflect the accuracy on the Complete dataset (the Expected accuracy).

2. The difference of this method and K-fold CV is that the best model is chosen with regards to the test set, while
in the K-fold CV the best model is chosen with regard to the average validation error.
Using the method from the K-fold CV is better:
- eventually, all samples are used for both training and validation
- it gives a good estimation on the true error since it uses averaging over all 'folds'
- we check how well the model generalizes on the unseen data by eventually using all the dataset as the validation
dataset (by folds)

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Answer:**
This parameter is arbitrary, because it is included in the loss function. 
We have parameters w which are learnable and would be tuned to mininize the loss including this delta
using the parameters update.
"""

part3_q2 = r"""
**Answer:**
The linear model is actually learning how prevalent each feature (pixel) is
for each class. (digit). E.g. pixels which are mostly used for drawing the digit 2
will have their corresponding weights for this class more positive, and accordingly, pixels
which are less likely to be used for that digit will have their weights negative.
The vizualization shows the input image which maximixes the output class certainty.

The mistakes occur when some input received higher class classification (higher score) on the wrong class.
When a certain digit is written in an obscure way, resembling two or more digits, the classification errors occur,
and that is easily seen in the vizualization.

The interpretation is different from the KNN in different things:
1. KNN has no parameters (except of K...). It is based on the existing data. So it requires no training.
2. After the Logistic Regression model finished learning, the KNN is much slower at extracting the class of the input
data, since it has to compute the distances to each sample in the training set. This, it's computation time is
increasing with the training set size.
3. KNN can classify non-linearly separable datasets, while Logistic Regression only supports the solutions
which are linear 
4. Logistic Regression can output confidence on it's predictions


"""

part3_q3 = r"""
**Answer:**

1. Based on the graph, the learning rate seems to be good. We can see that the loss
has (almost) reached convergence over the epochs, so did the accuracy. 
- Low learning rate would mean that the convergence over the given number of epochs would not be reached
- High learning rate would show the loss descreasing really fast in the beginning, but then being unable
to converge, making loss graph 'jump' up and down. This means we update the weights too much with every 
epoch.

2. The model overfit is recognized when the training error keeps decreasing, while the test error keeps
increasing at certain point. Meaning we decrease the in-sample loss, but the expected loss (of the whole set X, not
just the given training samples) is increasing. 
The model underfit is observed when the loss still didn't reach convergence. Meaning that if some more epochs would 
occur, both training and test loss would decrease even more. The simple solution would be to simply give more epochs.

In our case, I see a slight underfit. I still see the validation loss decreasing, and the validation accuracy 
increasing, when the number of epochs (30) ends. We would train for some more epochs for better results.


"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Answer:**

The ideal pattern to see in a residual plot is a constant behavior of the error, centered around 0 and.
From the plots we can see that the fitness of the trained model improves for top-5-features improves after CV, error margin decrease
and the error points are concentrated around error value 0 in comparison to before we CV where they were more distributed.

"""

part4_q2 = r"""
**Answer:**
1.Our model is still a linear regression model because feature engineering allows us to reformulate non-linear 
connection/problems in our data problems to linear problems by adding a feature  to our feature vector,
thus expecting that our linear regression model to preform better.
2.we can not fit every non linear function of the original feature because it must be a proper inner product.
3.Adding a non linear feature increase the dimension of the decision boundary, thus the boundary is still a hyperplane
in a higher dimension, and it will enable a better separation of the data than we would have without the non linear feature


"""

part4_q3 = r"""
**Answer:**

lambda was set using logspace instead of linspace because we want to search the best order of magnitude
that will produce the best parameters. 
Since lambda is used to penalize the parameters' magnitude, small changes in lambda will insure insignificant changes 
in the parameters' sizes. Therefore using values that are close is a waste of time while searching among different orders 
of magnitude is more efficient for this hyperparameter.

"""

# ==============
