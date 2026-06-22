# Machine Learning

## Linear Models for Classification I

Manfred Jaeger\
Aalborg University\
ML F26 Machine Learning

______________________________________________________________________

# Course Logistics

## People and Time

**Teachers**

- Manfred Jaeger
- Thomas D. Nielsen
- Peter Dolog

**Times**

- Lectures and exercises: Mondays 8:15-12:00
- Self study: Wednesdays 12:30-16:15

**Places**

- Lectures: Seminar room
- Exercises and self study: Group rooms; work spaces

______________________________________________________________________

## Content

Course consists of 12 units:

1-3 (Manfred)

- Linear models for classification
- Support vector machines

4-7 (Thomas)

- Probabilistic graphical models
- Deep neural networks

8-11 (Peter)

- Learning with graph data
- Link prediction

12 (Manfred)

- Graph Kernels

______________________________________________________________________

## Self-study

**Self studies**

- Extended, applied exercises (implementation, experimentation)
- Best done in small groups (2-3 students)
- No hand-ins or deliverables
- Limited support available approximately Wed. 13:30-15:30

______________________________________________________________________

## Exam

**Oral or written exam**

- Oral or written: to be determined soon
- Either way: questions at the exam about
  - methods, theory, examples discussed in the lectures
  - applications, data investigated in the self studies (not specifics of Python code)

______________________________________________________________________

## Time allocation

Total expected workload: 150 hours

| Activity               | Hours  |
| ---------------------- | ------ |
| Lectures and exercises | 12 × 4 |
| Self studies           | 12 × 4 |
| Reading                | 12 × 2 |
| Exam preparation       | 25     |
| Other                  | 5      |

______________________________________________________________________

## Literature and Tools

**Literature**

- For the first 3 lectures: C. M. Bishop: *Pattern Recognition and Machine Learning*, Springer 2006

**Exercises**

- Theoretical exercises solvable with pencil and paper

**Self studies require**

- Python
- NumPy
- Scikit-learn
- Pandas
- Matplotlib
- Jupyter

______________________________________________________________________

# Classification and Regression

## Classification

Learning to predict:

- Is the sun shining tomorrow
- Predicting next player move
- Predicting customer product interest
- Diagnosing based on symptoms

Instances have

- feature values
- known or unknown class label

______________________________________________________________________

## Regression

Predicting continuous target values:

- Click-through rate
- Stock price
- Years a patient will survive

______________________________________________________________________

# K Nearest Neighbor Classifier

## Principle

Near neighbors tend to have the same label.

## Model

Dataset:

- Labeled training instances
- Each instance: $x\_i = (x\_{i,1}, \ldots, x\_{i,D})$ and label $y\_i$

Distance function: $d(x, x')$

Classification rule:

- Find the $K$ training instances closest to $x$
- Predict the most frequent class among them

______________________________________________________________________

## Decision Regions

Instance space: all possible input values

A classifier partitions instance space into decision regions:

- Each region corresponds to a class label

Examples shown:

- 1-nearest neighbor
- 5-nearest neighbor

______________________________________________________________________

# Perceptron and Naive Bayes

## The Perceptron

Neural network with

- Input layer
- No hidden layers
- One output neuron
- Sign activation

Function:

$$
O(x_1, \ldots, x_n) =
\begin{cases}
1 & \text{if } w_0 + w_1x_1 + \ldots + w_nx_n > 0 \\
-1 & \text{otherwise}
\end{cases}
$$

Decision regions separated by a linear hyperplane.

______________________________________________________________________

## XOR Example

The perceptron cannot learn the XOR classification.

______________________________________________________________________

## Naive Bayes Model

Features are independent given the class label.

Binary case: classify instance as $\oplus$ if

$$
P(\oplus \mid X_1, \ldots, X_n) \ge P(\ominus \mid X_1, \ldots, X_n)
$$

After transformations:\
Linear function in the $X\_i$ with coefficients defined by network parameters.

______________________________________________________________________

## Summary

Perceptron and Naive Bayes

- Limited
- Cannot learn XOR
- Still useful baseline models
- Integral components of more complex models

______________________________________________________________________

# Overfitting

## Example: Nearest Neighbor

1-NN:

- Train accuracy: 100 percent
- Test accuracy: approx 66 percent

5-NN:

- Train accuracy: approx 75 percent
- Test accuracy: approx 77 percent

______________________________________________________________________

## Overfitting Defined

Hypothesis $h$ overfits if

- It has smaller training error than some $h'$
- But larger true error under instance distribution

Source: Mitchell, p. 67

______________________________________________________________________

## Overfitting and Model Complexity

Hypothesis space structured by model parameter:

- NN: $k$
- Neural networks: number of hidden units
- BN: structure complexity

Small $k$ leads to complex decision regions.

______________________________________________________________________

# Why Linear Models

Benefits:

- Unlikely to overfit
- Easy to learn
- Well understood
- Core for more complex models

Different linear models have same capacity but differ in:

- objective functions
- learning methods
- results

Examples:

- Perceptron: minimize error function
- Naive Bayes: maximize likelihood

______________________________________________________________________

# Linear Functions

## Definition

$$
y(x_1, \ldots, x_D) = w_0 + w_1x_1 + \ldots + w_Dx_D
$$

## Vector notation

$$
y(x) = w_0 + \mathbf{w} \cdot \mathbf{x}
$$

Decision regions:

- $R\_1 = {x \mid y(x) \ge 0}$
- $R\_2 = {x \mid y(x) < 0}$

______________________________________________________________________

## Geometry

$w$ determines orientation of decision boundary

Distance from origin:

$$
\frac{w_0}{\|w\|}
$$

![Geometry hyp](./images/geometry_hyp.png)

______________________________________________________________________

## Multiple Classes

Use either

- one against all
- one against one

![Multiple class](./images/multiple_classes.png)

______________________________________________________________________

## Discriminant Functions

Construct linear function $y\_k$ for each class $k$\
Classify to class with maximal $y\_k(x)$

______________________________________________________________________

## Matrix Representation

For $K$ discriminant functions:

$$
\mathbf{y}(x) = \mathbf{w}_0 + W^T x
$$

Shorter:

$$
\mathbf{y}(x) = \tilde{W}^T \tilde{x}
$$

______________________________________________________________________

## Classification by Least Squares Regression

Target vector $t\_n$ is one-hot encoding of class label.

Minimize sum of squares error:

$$
E_D(\tilde{W}) = \frac{1}{2} \sum_{n=1}^N \|\tilde{W}^T \tilde{x}_n - t_n\|^2
$$

Matrix form:

$$
E_D(\tilde{W}) = \frac{1}{2} \text{Tr}[(\tilde{X}\tilde{W} - T)^T(\tilde{X}\tilde{W} - T)]
$$

______________________________________________________________________

## Computing $\tilde{W}$

The error function in pedestrian notation:

$$
E_D(\tilde{W}) = \frac{1}{2} \sum_n \sum_k \left( \sum_d w_{kd} \, \tilde{x}_{nd} - t_{nk} \right)^2
$$

Derivative with respect to each weight parameter $w\_{kd}$:

$$
\sum_n \left( \sum_{d'} w_{kd'} \, \tilde{x}_{nd'} - t_{nk} \right) \tilde{x}_{nd}
$$

Setting derivative to zero:

$$
\sum_n \tilde{x}_{nd} \sum_{d'} \tilde{x}_{nd'} w_{kd'} = \sum_n \tilde{x}_{nd} t_{nk}
$$

From derivative conditions:

$$
\tilde{X}^T\tilde{X}\tilde{W} = \tilde{X}^T T
$$

Solution:

$$
\tilde{W} = (\tilde{X}^T\tilde{X})^{-1}\tilde{X}^T T
$$

______________________________________________________________________

## Problems: Outliers

Least squares may fail even for linearly separable data.\
It does not directly minimize classification error.

![outliers](./images/outliers.png)

in the right image the oulier create a high error rate even if they are classified correctly so this creates the slightly bent purple line to minimize the error.

______________________________________________________________________

## Example

Three data points with example matrices $X$, $T$, $W\_1$, $W\_2$ and corresponding outputs.

$W\_1$ achieves 100 percent accuracy\
$W\_2$ preferred by sum of squares criterion.

______________________________________________________________________
