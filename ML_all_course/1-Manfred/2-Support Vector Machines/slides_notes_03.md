# SVMs

Manfred Jaeger  
Aalborg University  
ML F26 SVM 1 / 26 
***

# Data Transformations

ML F26 SVM 2 / 26    
## Feature Space

Any mapping $\phi : \mathbb{R}^D \to \mathbb{R}^{D'}$ defines a data transformation that transforms the original data instance **x** into a transformed data instance

$$
\phi(\mathbf{x}) = (\phi_1(\mathbf{x}), \ldots, \phi_{D'}(\mathbf{x})).
$$

*   The components $\phi\_i(\mathbf{x})$ are also called **features** or **basis functions**, and $\mathbb{R}^{D'}$ the **feature space** of $\phi$.
*   Often: $D' > D$, and $\phi\_i$ typically non‑linear.  
    ML F26 SVM 2 / 26 


***

# Classifying in Feature Space

Original data (left), transformed data in feature space (right), decision boundaries (black lines).  

![4.12](./images/fig4.12.png)

Here:

$$
\phi_1(x_1, x_2) = \exp\left(-((x_1 + 1)^2 + (x_2 + 1)^2)\right)
$$

$$
\phi_2(x_1, x_2) = \exp(-(x_1^2 + x_2^2)).
$$

*   All our linear models can be applied to the transformed data
*   The linear decision boundary in feature space then corresponds to a non‑linear decision boundary in the original space  
    ML F26 SVM 3 / 26 
***

# Nonlinear SVM

ML F26 SVM 4 / 26 
## SVM in Feature Space

SVM learning after a non‑linear transformation:  

![svm learn](./images/svm_learning.png)

Problem: suitable feature spaces may need to be very high-dimensional. That makes computations with transformed vectors $\phi(\mathbf{x})$ very expensive.  
ML F26 SVM 4 / 26    
***

# Remember ...

*   In the Lagrange optimization process: to determine the support vectors $\mathbf{x}\_i$, the $\lambda\_i$ and $b$: the only operations required on data items is to compute dot products $\mathbf{x}\_i \cdot \mathbf{x}\_j$.
*   For classification: only need to compute dot products $\mathbf{x}\_i \cdot \mathbf{z}$.  
    ML F26 SVM 5 / 26 
***

# From Dot Products to Kernels

Example mapping:

$$
\phi : (x_1, x_2) \mapsto (x_1^2, x_2^2, \sqrt{2}x_1, \sqrt{2}x_2, \sqrt{2}x_1x_2, 1)
$$

Then:

$$
\phi(\mathbf{x}) \cdot \phi(\mathbf{x}')
= x_1^2 x_1'^2 + x_2^2 x_2'^2 + 2 x_1 x_1' + 2 x_2 x_2' + 2 x_1 x_1' x_2 x_2' + 1
= (\mathbf{x} \cdot \mathbf{x}' + 1)^2
$$

Thus

$$
K(\mathbf{x}, \mathbf{x}') = (\mathbf{x} \cdot \mathbf{x}' + 1)^2
$$

is a **kernel function**.  

> Kernel functions can be interpreted as measures of similarity.  

ML F26 SVM 6 / 26 
***

# From Kernels to Dot Products I

Can we interpret $K (x, z)$ as a dot product $ϕ(x) · ϕ(z)$ in some feature space?

**Mercer's Theorem:**  
A kernel $K(\mathbf{x}, \mathbf{x}')$ is of the form $\phi(\mathbf{x}) \cdot \phi(\mathbf{x}')$ if and only if for all functions $g(\mathbf{x})$ with

$$
\int g(\mathbf{x})^2 d\mathbf{x} < \infty:
$$


$$
\iint K(\mathbf{x}, \mathbf{x}') g(\mathbf{x}) g(\mathbf{x}')\, d\mathbf{x} d\mathbf{x}' \ge 0.
$$

The kernel is then called **positive semi-definite**.  
ML F26 SVM 7 / 26 
***

# From Kernels to Dot Products II

Kernel matrix for datapoints $\mathbf{x}\_1, \ldots, \mathbf{x}\_n$:

$$
\begin{pmatrix}
K(\mathbf{x}_1, \mathbf{x}_1) & \cdots & K(\mathbf{x}_1, \mathbf{x}_n) \\
\vdots & \ddots & \vdots \\
K(\mathbf{x}_n, \mathbf{x}_1) & \cdots & K(\mathbf{x}_n, \mathbf{x}_n)
\end{pmatrix}
$$

is the (symmetric) Kernel matrix (also called Gram matrix) for $x_1,...,x_n$

Matrix Version of Mercer's Theorem:  
A symmetric function $K(\mathbf{x}, \mathbf{z})$ is a kernel if and only if the kernel matrix is positive semi-definite for all finite sets of points.  
ML F26 SVM 8 / 26 
***

# Positive Semi‑Definite Matrices

*   A vector $\mathbf{x}$ is an eigenvector if $M\mathbf{x} = \lambda \mathbf{x}$.
*   A matrix is **positive semidefinite** if all eigenvalues are non‑negative.
*   Quadratic form:

$$
x \rightarrow\mathbf{x}^T M \mathbf{x} = \sum_{i,j} x_i x_j M_{ij} \in\math{R}.
$$

Positive semi-definite if $\mathbf{x}^T M \mathbf{x} \ge 0$ for all $\mathbf{x}$.  
ML F26 SVM 9 / 26 
***

# Feature Space and Kernels

![cloud](./images/cloud_image.png)

When we have a positive semi-definite kernel $K(\mathbf{x}, \mathbf{x}')$ we need not know the actual feature mapping $\phi$ in order to perform computations involving dot products $\phi(\mathbf{x}) \cdot \phi(\mathbf{x}')$.  
ML F26 SVM 10 / 26 
***

# The Kernel Trick

Strategy to construct kernelized SVM classifiers:

*   **Given**: a classification problem
*   **Define** or **select** a similarity function 
    $$K(\mathbf{x}, \mathbf{z})$$
*   **Verify** that $K(\mathbf{x}, \mathbf{z})$ is positive semi-definite
*   **Learn** an SVM using the values $K(\mathbf{x}\_i, \mathbf{x}\_j)$ in place of dot products $\mathbf{x}\_i \cdot \mathbf{x}\_j$

The learning algorithm requires as input only the **kernel matrix** for the training datapoints.  
ML F26 SVM 11 / 26 
***

# Constructing Kernels

Basic Kernels:

*   $(\mathbf{x} \cdot \mathbf{z} + 1)^p$ Polynomial kernel
*   $\exp(-|\mathbf{x} - \mathbf{z}| / 2\sigma^2)$ Gaussian kernel
*   $\tanh(\kappa, \mathbf{x} \cdot \mathbf{z} - \delta)$ Hyperbolic tangent kernel

$p, \sigma, \kappa, delta$ are all hyperparameters.

Kernel Building Rules:  
If $K(\mathbf{x}, \mathbf{x}')$ is positive semi-definite, then so is:

*   $q(K(\mathbf{x}, \mathbf{x}'))$, where $q$ is a polynomial with nonnegative coefficients
*   $\exp(K(\mathbf{x}, \mathbf{x}'))$
*   $K(\mathbf{x}, \mathbf{x}') K'(\mathbf{x},\mathbf{x}')$
* ...
*   $\frac{K(\mathbf{x},\mathbf{x}')}{\sqrt{K(\mathbf{x},\mathbf{x}) K(\mathbf{x}',\mathbf{x}')}}$ (the normalization of $K()$)

ML F26 SVM 12 / 26 
***

# Non linearly separable data

Implicit assumption so far: data linearly separable, maybe after kernel transformation. In reality: unlikely to be true.

Example: scikit‑learn SVC(kernel='linear') also handles non-separable data.  

![nonlin_sep_data](./images/nonlin_sep_data.png)

ML F26 SVM 13 / 26 
***

# Relaxing the Objective

Slack variables $\zeta\_n$ $(n \in \{1,...,N\})$:

Original SVM:

$$
\min \frac12 \|\mathbf{w}\|^2
$$

subject to

$$
y_n(\mathbf{w}\cdot \mathbf{x}_n + b) \ge 1.
$$

Relaxed SVM:

$$
\min \frac12 \|\mathbf{w}\|^2 + C \sum_{n=1}^N \zeta_n
$$

subject to

$$
y_n(\mathbf{w}\cdot \mathbf{x}_n + b) \ge 1 - \zeta_n,\quad \zeta_n \ge 0.
$$

* the $ζ_n$ are constrained to be non-negative
* the hyper-parameter $C$ controls how much the loss increases when we relax the constraints with $ζ_n > 0$.
* for linearly separable data: small values of $C$ can lead to solutions that do not separate the data

The hyperparameter $C$ controls the penalty for constraint violations.  
For linearly separable data: small $C$ may lead to solutions that do not separate the data.  
ML F26 SVM 14 / 26    
***

# Kernels for Non‑Standard Data

ML F26 SVM 15 / 26 
***

## Non‑Numeric Data

Define a kernel function $K(\text{shape}\_i, \text{shape}\_j)$ and verify positive semidefiniteness of the kernel matrix.  

![non-numeric-data](./images/non_numeric_data.png)

(values here are hypothetical values computed by some imaginary kernel).

Can now learn SVM from Train kernel matrix, and classify test case(s) using the Train×Test
kernel matrix

ML F26 SVM 16 / 26 
***

# Text Data

Example movie reviews.  
Classification task: positive or negative sentiment.  
ML F26 SVM 17 / 26 
***

# From Text to Vectors

Term frequency vector $tf(t)$:

*   Vocabulary size $n$
*   $tf(t)[i] =$ number of occurrences of term $i$ in text $t$
*   This rappresentation is called Bag‑of‑words model
*   Since most components of tf(t) are zero, should use a sparse representation.

ML F26 SVM 18 / 26    
***

# Similarity of TF Vectors

![tf_vec](./images/tf_vectors.png)

If $t_1$ and $t_2$ do not have any terms in common, then their $tf$ vectors are orthogonal ($cosine(θ)=0$).

If $t_1$ and $t_2$ contain exactly the same terms with the same relative freqencies, then $tf(t_1) = r · tf(t_2)$ for some constant $r$ , and $cosine(θ)=1$. (E.g.: $t_1$ is the concatenation of two copies of $t_2$, and $r = 2$).

Cosine similarity:

$$
\text{cos-sim}(t_1,t_2) = \text{cosine}(\theta) = 
\frac{tf(t_1)\cdot tf(t_2)}
{\|tf(t_1)\|\cdot \|tf(t_2)\|}.
$$

ML F26 SVM 19 / 26 
***

# Cosine Similarity as a Kernel

kernel \
cos-sim is a positive semi-definite kernel: normalization of plain dot product (because $∥ x ∥= √x · x$).
No kernel trick here: we work with explicitly constructed feature vectors $tf(t)$.

Information retrieval \
cos-sim is extensively used in information retrieval as a measure of similarity between documents:

* t1: a short query text
* t2: a candidate document that may be returned for query t1. 

***

# Strings and Graphs

ML F26 SVM 21 / 26 
***

# String Data
A data instance has the form

$$(s_n, c_n)$$

with

* $s_n ∈ Σ^∗$ for some alphabet $Σ$
* $c_n ∈ {0, 1}$ (class label)

Basic string features:  
For $u \in \Sigma^*$:

*   $\phi\_u(s) :=$ number of substring occurrences
*   $\phi\_u^+(s) :=$ number of subsequence occurrences(how many time u appears in s in the same order but may not be consecuntive)

Example: $s =$ statistics  
$u = \{ti, tis, actics\}$


   i tis atics
ϕu (s) 2 1 0
ϕ+u (s) 5 7 3

ML F26 SVM 22 / 26 
***

# p‑Spectrum Kernel

Defined by feature vector $\{\phi\_u | u\in\Sigma^p\}$.  

![pspec](./images/pspectrum_kernel.png)

Example: 2‑spectrum kernel matrix for {bar, bat, car, cat}.  
ML F26 SVM 23 / 26 
***

# All‑Subsequences Kernel

Feature vector ${\phi\_u^+}\_{u\in\Sigma^\*}$.  
Cannot compute whole feature vector.  
Can compute kernel using dynamic programming:  
Time $O(|s||t|)$.  
ML F26 SVM 24 / 26 
***

# Computing All‑Subsequence Kernels

![all sub](./images/all_sub.png)


Dynamic Programming Approach
* $s = s_1 . . . s_n, t = t_1 . . . t_m$
* For $i ⊆ {1, . . . , n}$: $s[i]$: subsequence defined by $i$
* $K (s, t) = \#(i, j) : s[i] = t[j]$

Recursion:

$$
K(\epsilon,t)=1
$$

$$
K(s[1:i], t[1:j]) = K(s[1:i-1],t[1:j]) + \sum_{k\le j: t_k=s_i} K(s[1:i-1], t[1:k-1]).
$$

Straightforward implementation: $O(n m^2)$  
Optimized: $O(nm)$.  
ML F26 SVM 25 / 26 
***

# Pros and Cons

SVM + Kernel Functions:

*   Powerful method for classification
*   Successful applications in bio‑informatics
*   Wide variety of data types  
    - Binary classifier only (multiclass via multiple binary classifiers)  
    - Complexity quadratic in number of instances(you need the kernel matrix that is quadratic to the data)  
    - Finding the right kernel may require engineering

ML F26 SVM 26 / 26 

