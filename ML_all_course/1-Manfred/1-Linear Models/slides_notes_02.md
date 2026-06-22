## Linear Models for Classification II

Manfred Jaeger  
Aalborg University  
ML F26 Linear Models 2

***

# Probabilistic Models

## Probabilistic Classifiers

Classify data instance **x** to belong to the class k, for which probability

$$
P(Y = k \mid \mathbf{x})
$$

is maximal.    

### Generative Approach

*   Define a joint distribution for the features

    $$
    \mathbf{X} = (X_1,\ldots,X_D)
    $$

and the class variable $Y$ with values $1,\ldots,K$.

*   Usually factored as

$$
    P(Y,\mathbf{X}) = P(Y) P(\mathbf{X} \mid Y).

$$

*   Compute

$$
    P(Y = k \mid \mathbf{X} = \mathbf{x})
    = \frac{P(\mathbf{X} = \mathbf{x} \mid Y = k)\,P(Y = k)}{P(\mathbf{X} = \mathbf{x})}.
$$

Examples: Naive Bayes, LDA.

### Discriminative Approach

Directly learn the conditional distribution

$$
P(Y \mid \mathbf{X}).
$$

Examples: Logistic Regression, Neural Networks(it returns the probability distribution of the classes).    

***

## Generative vs. Discriminative

Operations supported:

| Operation                                  | Generative | Discriminative |
| ------------------------------------------ | ---------- | -------------- |
| Predicting $Y$ given $\mathbf{X}$      | yes        | yes            |
| Predicting any other variable $X_i$     | yes(compute arbitrary conditional probabilities $P(X_i\|...)$)        | no             |
| Generating data points $(\mathbf{x}, y)$ | yes        | no             |

Additional notes:

*   Operations supported in principle by generative models are not always computationally efficient.
*   Discriminative models can be more efficient to learn and more interpretable.
*   Notation: upper case $X$ for variables, lower case $x$ for specific values. Example: $X =$ Temperature, $x = 23.5C$.    

***

## Likelihood

Main criterion for learning probabilistic models is maximizing likelihood.

### Generative

$$
\prod_{n=1}^N P(\mathbf{x}_n, y_n)
$$

### Discriminative

$$
\prod_{n=1}^N P(y_n \mid \mathbf{x}_n)
$$

Always required: a parametric model for the distributions.
specific funcitonal forms 

$$
P(Y \mid \mathbf{w}), P(\mathbf{X} \mid Y, \mathbf{w}), P(Y \mid \mathbf{w, X})
$$

Example: multinomial model for $P(Y \mid \mathbf{w})$.    
**multinomial**: $\mathbf{w} = (P(Y = 1), P(Y = 2), ... , P(Y= K - 1))$

***

# Naive(assumption of independence) Bayes

A probabilistic generative model. $Y$ is the father of al the $X_i$

$$
X_1, X_2, X_3, X_4, X_5 \leftarrow Y
$$

### Components

*   $P(Y \mid \mathbf{w})$: conditional probability table for node $Y$.
*   $P(X_i \mid Y, \mathbf{w})$:
    *   If $X_i$ categorical: probability tables defined by $\mathbf{w}$.
    *   If $X_i$ continuous: often Gaussian

        $$
        P(X_i \mid Y, \mathbf{w}) = \mathcal{N}(\mu, \sigma^2).
        $$

        Equivalent to a simple Gaussian mixture model.    

***

# Gaussian Mixtures

### General Gaussian mixture model

For D dimensional feature space, without naive Bayes independence:

$$
P(Y = k)
$$

is multinomial.

$$
P(\mathbf{X} = \mathbf{x} \mid Y = k)
= \frac{1}{(2\pi)^{D/2}\,|\Sigma_k|^{1/2}}
\exp\left(-\frac12(\mathbf{x}-\mu_k)^T \Sigma_k^{-1}(\mathbf{x}-\mu_k)\right)
$$

Defined by mean vectors $\mu_k$ and covariance matrices $\Sigma_k$.    

Example

$Y = Region ∈ \{Nordjylland, . . . , Hovedstaden\} \space (K = 5)$

$X = (X_1, X_2)$ with $X_1$: annual income, $X_2$: annual housing expenditure (mortgage, rent).

$μ_k = (μ_{1,k} , μ_{2,k} )$: average income and housing costs in region $k$.

$$
Σk =
\begin{matrix}
σ1,1,k & σ1,2,k\\
σ2,1,k & σ2,2,k
\end{matrix}
$$

* Presumably: X1 and X2 are positively correlated, i.e., σ1,2,k (= σ2,1,k ) > 0.

***

## Example (D = 2, diagonal covariances)

$$
P(Y = 1) = 0.3,\quad P(Y = 2)=0.6,\quad P(Y = 3)=0.1
$$

$$
\mu_1 = (2,3),\ \mu_2 = (4,5),\ \mu_3 = (7,0)
$$

Covariances:

$$
\Sigma_1 = 
\begin{pmatrix}
1 & 0 \\ 
0 & 0.25
\end{pmatrix},\quad
\Sigma_2 = 
\begin{pmatrix}
4 & 0 \\ 
0 & 2.25
\end{pmatrix},\quad
\Sigma_3 = 
\begin{pmatrix}
0.25 & 0 \\ 
0 & 1
\end{pmatrix}
$$

Decision boundaries shown on slide.    


## Linear Discriminant Analysis (LDA)

Special case: all covariance matrices are identical.  
Decision boundaries are linear.   

LDA: special case where all covariance matrices are identical. 

> [!DANGER] \
> Warning: not to be confused with
> Fisher’s linear discriminant – Bishop Sec. 4.2.1 describes LDA, but does not use this name!

***

# Learning Gaussian Mixtures

![Learning Gaussian Mixtures ex](./images/learning_gaussian_mixture.png)

Given data $(\mathbf{x}_n, y_n)$:

$$
P(Y = k) = \frac{N_k}{N}
$$

$$
\mu_k = \frac{1}{N_k} \sum_{n:y_n=k} \mathbf{x}_n
$$

Unrestricted covariance:

$$
\Sigma_k[i,j] = \frac{1}{N_k} \sum_{n:y_n=k} (x_n[i] - \mu_k[i])(x_n[j] - \mu_k[j])
$$

LDA shared covariance:

$$
\Sigma[i,j] = \frac{1}{N} \sum_k \sum_{n:y_n=k} (x_n[i] - \mu_k[i])(x_n[j] - \mu_k[j])
$$


***

# Logistic Regression

Model: discriminative with continuous inputs, two classes $Y \in \{0,1\}$.

$$
P(Y = 1 \mid \mathbf{X}=\mathbf{x},\mathbf{w})
= \sigma(\mathbf{w}\cdot\mathbf{x})
$$

![sigmoid](./images/sigmoid.png)

$$
\sigma(x) = \frac{e^x}{1+e^x} = \frac{1}{1+e^{-x}}
$$

Linear decision boundary from condition

$$
P(Y=1\mid x) \ge 0.5 \iff \mathbf{w}\cdot\mathbf{x} \ge 0.
$$

Not a generative model.    

***

## Logistic Regression: Extensions

### Categorical Features

Encoded via one‑hot encoding.

### More than 2 Classes

$$
\frac{P(Y=1\mid x,w)}{P(Y=0\mid x,w)}
= e^{\mathbf{w}\cdot\mathbf{x}}
$$

or 

$$
\log (\frac{P(Y=1\mid x,w)}{P(Y=0\mid x,w)})
= \mathbf{w}\cdot\mathbf{x}
$$

* The lienar funciton $\mathbf{w}\cdot\mathbf{x}$ defines the **log-odds** of class 1 against class 0.

For multi-class: $Y\in\{1,\ldots,K\}$:

*   Pick reference class $K$.
*   Learn $K-1$ models with parameters $\mathbf{w}_k$.
*   Odds:

    $$
    \text{odds}(k,K,x) = 
    \frac{P(Y = k \mid x, w_k)}{P(Y = K \mid x, w_k)}
    $$

* Classify as $k$ for maximal odds if odds greater than 1, else classify as $K$.    

$K$ discriminant functions: $\text{odds}(k, K , x) (k = 1, . . . , K − 1)$ and $\text{odds}(K , K , x) ≡ 1$

***

# Learning Logistic Regression

Likelihood:

$$
\prod_{n: y_n=1} P(Y=1\mid \mathbf{x}_n,w)
\prod_{n:y_n=0} (1 - P(Y=1\mid \mathbf{x}_n,w))
$$

Log likelihood:

$$
\sum_{n=1}^N 
\left[
y_n \log P(Y=1\mid x_n,w)
+ (1-y_n)\log(1-P(Y=1\mid x_n,w))
\right]
$$

> Set derivative to zero; find root of derivative using numerical methods, often Newton-Raphson
> not easy task so optimize using numerical methods such as Newton Raphson.    

***

# Comparison

![comparison](./images/comparison.png)

***

# Maximum Margin Hyperplanes

![max_margin](./images/max_margin.png)

*   Margin: minimum distance of any datapoint to the hyperplane.
*   **Maximum‑margin hyperplane** has the largest possible margin.
    * minimum distance of point from green class to hyperplane = minimum distance of point from blue class to hyperplane
    * there are datapoint in both classes whose distance to the hyperplane equals the margin. These are the **support vectors** of the hyperplane
*   Datapoints achieving equality are support vectors.    

## Distance to Hyperplane

![geomtric hyperplane](./images/geometry_hyp.png)

For hyperplane

$$
y(\mathbf{x}) = w_0 + \mathbf{w}\cdot\mathbf{x}
$$

distance:

$$
\frac{|y(\mathbf{x})|}{\|\mathbf{w}\|}
$$

For labels $Y\in\{-1,1\}$:

$$
\text{distance} = \frac{y_n\,y(\mathbf{x}_n)}{\|\mathbf{w}\|}
$$

With class encoding $Y ∈ {−1, 1}$:

distance of datapoint $x_n$ with label $y_n$ to hyperplane $y = 0$ (if datapoint lies on correct side of the hyperplane, i.e., $y_n = 1$ and $y(x_n) > 0$, or $y_n = −1$ and $y(x_n) < 0$): 

$$
\frac{y_n y(\mathbf{x})}{\|\mathbf{w}\|}
$$

(“negative distance” for points that lie on the wrong side of the hyperplane

***

# Max Margin Objective

> Assume data linearly separable. 

To find the hyperplane defined by $w, b$ as
Maximize:

$$
arg\max_{w,b} \{\frac{1}{\|w\|}
\min_n \left[y_n (w\cdot x_n + b)\right]\}
$$

**simplify** \
The decision boundary (and distance of datapoints to decision boundary) is not changed by **scaling**($→$) w , b with a common factor κ:

$$
w 7 → κw , b 7 → κb
$$

can calibrate $w , b$ so that for support vectors $x_n$:

$$
y_n (w\cdot x_n + b) = 1,
$$

and for all datapoints:

$$
y_n (w\cdot x_n + b) \ge 1.
$$


***

# SVM Learning

![simplify](./images/simplification.png)

Replacing maximization of $1/ ∥ w ∥$ by equivalent minimization of $∥ w ∥^2 /2$ leads to the formulation: find


$$
\min_{w,b} \frac12 \|w\|^2
\quad
\text{subject to the constraint } y_n (w\cdot x_n + b) \ge 1.
$$

### Lagrangian

The SVM learning problem is solved using the method of Lagrange multipliers.

* the solution identifies the support vectors: $x_i$ that lie on the margin (= those $x_i$ whose constraint is active at the optimal solution)

* the vector $w$ is a linear combination of support vectors:

$$
w = \sum_{i:λ_i >0}
λ_i y_i x_i
$$
* a test instance $z$ is classified as

$$
sign(w · z + b) = sign( \sum_{i:λ_i >0} λ_i y_i x_i \cdot z + b)
$$

* In the Lagrange optimization process: to determine the support vectors $x_i$ , the $λ_i$ and $b$: the only operations required on data items is to compute dot products $x_i \cdot x_j$.
* for classification: only need to compute dot products $x_i \cdot z$




<!-- Constraint gradients satisfy: -->
<!---->
<!-- $$ -->
<!-- \nabla f = \sum_i -\lambda_i \nabla c_i, -->
<!-- \quad \lambda_i \ge 0. -->
<!-- $$ -->
<!---->
<!-- Lagrange function: -->
<!---->
<!-- $$ -->
<!-- f + \sum_i \lambda_i c_i. -->
<!-- $$ -->


***

# SVM Solution Properties

*   Support vectors are those datapoints with $$\lambda_i > 0$$.
*   Weight vector:

    $$
    w = \sum_{i:\lambda_i>0} \lambda_i y_i x_i.
    $$

*   Classification rule:

    $$
    \text{sign}(w\cdot z + b)
    = \text{sign}\left(\sum_{i:\lambda_i>0} \lambda_i y_i x_i\cdot z + b\right).
    $$

*   Only dot products $x_i\cdot x_j$ and $x_i\cdot z$ are required.    

***

# Data Transformations

A mapping

$$
\phi: \mathbb{R}^D \to \mathbb{R}^{D'}
$$

transforms input $x$ into

$$
\phi(x) = (\phi_1(x),\ldots,\phi_{D'}(x)).
$$

Often $D' > D$ and $\phi_i$ are nonlinear.    

***

# Classifying in Feature Space

Example mapping: shown in slide (feature space transformation).  
Linear models applied in feature space correspond to nonlinear boundaries in input space.    


