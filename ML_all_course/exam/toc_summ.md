Manfred’s topics:

* [Linear Models](./1-Manfred/1-Linear%20Models/)

    Linear models are a family of classification and regression methods that assume the decision boundary is a linear (or affine) function of the input features. They are simple, interpretable, and often serve as a baseline for more complex approaches. The topic covers both discriminative models (least squares, logistic regression) and generative models (LDA), along with fundamental challenges like overfitting.
   <img width="500" height="450" alt="image" src="https://github.com/user-attachments/assets/43ca09ea-9822-476a-9256-60abc4dfde5c" />


    * [Decision Regions](./1-Manfred/1-Linear%20Models/ml-F26-01.pdf#page=16): Given that the instance/input space is the space of all poissible values for the input features, Decision regions describe how the instance space is divided into areas assigned to different classes by a classifier. In linear models, these regions are separated by straight lines or hyperplanes.
      <img width="750" height="384" alt="image" src="https://github.com/user-attachments/assets/16f0ab99-186a-42a2-aea2-8011fe482e17" />


    * [Overfitting](./1-Manfred/1-Linear%20Models/ml-F26-01.pdf#page=24):A situation a model can be in, in which there exists an alternative state for the model in which the train error is higher, but the test/validation error is lower.
    * Knn classifier is the perfect example of this, in which k=1, the train error will be zero, but the test/validation error will not.
    * Overfitting happens when a model learns noise and details from the training data instead of the general pattern. As a result, it performs well on training data but poorly on unseen examples.
    * It relates to the bias/variance tradeoff, where bias is the inherent error a model has with respect to a given set of data, and variance is how many different representations that modeling method can generate depending on different subsets of that dataset

    <img width="813" height="534" alt="image" src="https://github.com/user-attachments/assets/f04ac1be-98fb-42a4-9798-7fadb93321b2" />


    * [Least squares regression](./1-Manfred/1-Linear%20Models/ml-F26-01.pdf#page=24) (corresponding to sklearn LinearRegression in self study 1) Least Squares Regression is a linear modeling technique that tries to fit a decision boundary to a dataset by minimising the sum of squared errors
    * The dataset must be linearly separable, and be numerically encoded using one-hot encoding
    * SSE does not directly measure classification error, only the distance to the boundary. What it does is measure the quadratic error between a given data point and a on-hot encoded target vector
    * Very sensitive to outliers, due to the quadratic error term
    * In multi-class classification, some decision regions can become very small or are simply ignored. This is called masking.
    * The minimisation problem can be transformed from an open-form expression to a closed-formed one
    * Least squares classification treats classification as regression. The class labels are one-hot encoded, and the model learns weights by minimizing the squared difference between predicted score vectors and target vectors. However, squared error is not the same as classification error, so least squares can be sensitive to outliers.

    <img width="906" height="427" alt="image" src="https://github.com/user-attachments/assets/da24a585-588e-411f-b590-c36f32c610dc" />
    <img width="1507" height="708" alt="image" src="https://github.com/user-attachments/assets/95d6d80e-e88d-43ef-ad6b-bff6578e0984" />
    <img width="1047" height="271" alt="image" src="https://github.com/user-attachments/assets/7241786b-b069-44de-83a2-652884fb33fd" />


    * [Linear discriminant analysis](./1-Manfred/1-Linear%20Models/ml-F26-02.pdf#page=13):
    * LDA is a generative probabilistic model, approximating the joint probability distribution of the classes
    * LDA is a specific case of Gaussian Mixture Model, in which all of the co-variance matrices are the same
    * LDA is different because it is *generative*. It models `P(Y)` and `P(X|Y)`, assuming each class has a Gaussian distribution. In LDA, all classes share the same covariance matrix, which leads to linear decision boundaries. The model estimates class priors, class means, and a shared covariance matrix from the data.
    *
        1. Estimate class priors  
           Compute the prior probabilities:  
           `P(Y = k)`  
           as the proportion of training points in each class.
        
        2. Estimate class means  
           For each class `k`, compute the mean vector `μ_k` using all data points belonging to that class.
        
        3. Estimate a shared covariance matrix  
           Compute a single covariance matrix `Σ` using all data.
        
        4. Compute class-conditional densities  
           Use Gaussian distributions:  
           `P(X | Y = k)`  
           with mean `μ_k` and shared covariance `Σ`.
        
        5. Compute posterior probabilities, Apply Bayes’ rule:  
           `P(Y = k | X = x) = (P(Y = k) * P(X | Y = k)) / P(X)`
        
        6. Classify the input  
           Assign `x` to the class with the highest posterior probability.

    <img width="905" height="456" alt="image" src="https://github.com/user-attachments/assets/548b8fce-cb84-4bdf-a5d5-c4a31d017495" />


    * [Logistic Regression](./1-Manfred/1-Linear%20Models/ml-F26-02.pdf#page=15):
    * Logistic regression is a discriminative classification model that directly models the conditional probability P(Y∣X) rather than the joint distribution. It assumes that the probability of a class is given by a sigmoid function applied to a linear combination of the input features, i.e. P(Y=1∣X=x)=σ(w⋅x). The model learns a linear decision boundary, where classification is based on whether this probability exceeds 0.5.
    1. Define the model  
       Assume `P(Y = 1 | X = x) = σ(w · x)`, where `σ` is the sigmoid function.
    
    2. Choose a likelihood function  
       Use the likelihood of the observed labels given the inputs (based on Bernoulli probabilities).
    
    3. Maximize the log-likelihood  
       Convert to log-likelihood and optimize it to find the best parameters `w`.
    
    4. Optimize parameters  
       Solve using iterative numerical methods (e.g., gradient-based methods or Newton–Raphson).
    
    5. Compute probabilities for new inputs  
       Plug new `x` into `σ(w · x)` to get class probabilities.
    
    6. Classify the input  
       Assign class 1 if `σ(w · x) ≥ 0` (probability ≥ 0.5), otherwise class 0, giving a linear decision boundary.

    * . Unlike least squares, logistic regression optimizes likelihood or cross-entropy, which is more appropriate for classification, but it requires iterative optimization.
    <img width="621" height="414" alt="image" src="https://github.com/user-attachments/assets/a43c1e6c-2540-490d-b415-b41e0311cc37" />

    

3. [Support Vector Machines](./1-Manfred/2-Support%20Vector%20Machines/) A Support Vector Machine (SVM) is a supervised machine learning algorithm used mainly for classification. It works by finding the best boundary (called a hyperplane) that separates data points from different classes while maximizing the margin between them and the closest points (called support vectors). For data that is not linearly separable, SVM can map the data into a higher-dimensional feature space and still find a separating boundary, often using kernel functions to efficiently compute similarities without explicitly performing the transformation.

    * [Maximum margin hyperplanes](./1-Manfred/1-Linear%20Models/ml-F26-02.pdf#page=21): A maximum margin hyperplane separates classes while maximizing the distance to the nearest training points. This larger margin often improves the model’s ability to generalize to new data.

    * [Feature transformations](./1-Manfred/2-Support%20Vector%20Machines/ml-F26-03.pdf#page=2) and [kernel functions](./1-Manfred/2-Support%20Vector%20Machines/ml-F26-03.pdf#page=2): Feature transformations map the original input data into a new (ofter higher-dimensional) space allowing linear models to capture non-linear patterns.
    * In this transformed feature space, a simple linear decision boundary can correspond to a complex, non-linear boundary in the original data. Kernel functions provide an efficient way to compute inner products ϕ(xi) ⋅ ϕ(xj) without explicitly performing the transformation, enabling algorithms like Support Vector Machines to operate in high-dimensional spaces implicitly and efficiently.
    * This matters because SVM learning and classification only require dot products between data points. So, if we replace ordinary dot products x_i ⋅ x_j ​ with kernel values K(x_i,x_j), the model behaves as if it were working in a richer feature space, without ever explicitly constructing that space.
    * [The kernel trick](./1-Manfred/2-Support%20Vector%20Machines/ml-F26-03.pdf#page=17): The kernel trick lets algorithms operate in a high-dimensional feature space by only evaluating pairwise kernel similarities. This makes nonlinear classification computationally feasible.
    * The kernel trick is a shortcut used in algorithms like SVMs to handle non-linear classification efficiently. Instead of explicitly transforming data into a higher-dimensional feature space using `phi ϕ(x)`, we use a kernel function `K(xi,xj)` that behaves as if it directly computed the dot product in the transformed space.
    * A caveat is that the kernel function must be positive semi-definite, which can be understood as one that approximates similarities that are consistent and do not produce impossible geometric relationships
      <img width="780" height="357" alt="image" src="https://github.com/user-attachments/assets/dd24cd78-f25a-4d65-993d-b017e3556764" />



    * [String kernels](./1-Manfred/2-Support%20Vector%20Machines/ml-F26-03.pdf#page=25): String kernels measure similarity between text or sequence data by comparing shared substrings or patterns. They are useful in domains such as text classification and bioinformatics where inputs are symbolic sequences.
    * Types of kernels for non-standard data covered in the slides:
        * **Non-numeric data kernels** — Kernel functions defined directly on non-numeric objects (e.g., shapes) by designing a similarity function
        * **Bag-of-words / Term Frequency kernel** — Text is represented by term frequency (TF) vectors; cosine similarity between TF vectors
        * **p‑Spectrum Kernel** — String kernel using all substrings of length `p`
        * **All‑Subsequences Kernel**

4. [Graph kernels](./1-Manfred/3-Graph%20kernels/) Graph kernels are similarity functions defined on graphs that allow them to be used in machine learning algorithms such as support vector machines.

    * [Convolution kernels](./1-Manfred/3-Graph%20kernels/ml-F26-12.pdf#page=10): Convolution kernels compare structured objects by decomposing them into smaller parts and summing similarities across those parts. In graphs, this can mean comparing nodes, edges, paths, or substructures.

    * [Subgraph isomorphisms](./1-Manfred/3-Graph%20kernels/ml-F26-12.pdf#page=17) and [graphlet kernels](./1-Manfred/3-Graph%20kernels/ml-F26-12.pdf#page=27): Subgraph isomorphism checks whether a smaller graph pattern appears inside a larger graph. Graphlet kernels use counts of small subgraphs to represent graphs and compare them based on their local structural motifs.

    * [Random walk kernels](./1-Manfred/3-Graph%20kernels/ml-F26-12.pdf#page=36): Random walk kernels compare graphs by counting matching walks in both graphs. They capture structural similarity through sequences of connected nodes, although they can be computationally expensive.

    * [Weisfeiler Lehman kernel](./1-Manfred/3-Graph%20kernels/ml-F26-12.pdf#page=36): The Weisfeiler-Lehman kernel compares graphs by iteratively relabeling nodes based on the labels of their neighbors. This captures increasingly rich neighborhood structure and makes graph comparison efficient and powerful.

Thomas' topics:

1. [Learning and neural networks](./2-Thomas/1-Learning%20and%20Neaural%20networks/) Learning is the process of iteratively adjusting the parameters of a model using data to improve its performance on a well defined task.

    * [Loss functions](./2-Thomas/1-Learning%20and%20Neaural%20networks/ML-2026-1.pdf#page=12): A loss function measures how far a neural network’s prediction is from the correct target value. During training, the network uses this error to decide how its weights should be adjusted so that future predictions become more accurate. Common loss functions include **squared error**, often used for regression problems, and **cross-entropy**, often used for classification when outputs are interpreted as probabilities. The goal of learning is to find the set of weights that minimizes the overall loss across the training examples, helping the model gradually improve its performance.

    * [Back propagation and the chain rule](./2-Thomas/1-Learning%20and%20Neaural%20networks/ML-2026-1.pdf#page=28): Backpropagation is the main algorithm used to train neural networks by calculating how much each weight contributed to the final error. After a forward pass produces an output and a loss, backpropagation works backward through the network, using the chain rule to compute derivatives for each operation in the computational graph. These derivatives tell the model how to adjust its weights to reduce the loss during training, making learning more efficient even when the network has hidden layers where target values are not directly available.
    The chain rule is a calculus rule used to find the derivative of a function made from other functions. If one function depends on another, such as f(g(x)), the derivative is found by multiplying the derivative of the outer function by the derivative of the inner function

    * [Computational graphs](./2-Thomas/1-Learning%20and%20Neaural%20networks/ML-2026-1.pdf#page=30): A computational graph represents a mathematical expression as a network of operations and variables. It makes forward computation and automatic differentiation easier to organize and implement.

    * [Gradient descent](./2-Thomas/1-Learning%20and%20Neaural%20networks/ML-2026-1.pdf#page=16) and [stochastic gradient descent](./2-Thomas/1-Learning%20and%20Neaural%20networks/ML-2026-1.pdf#page=60): Gradient descent is an iterative optimization algorithm that minimizes a loss function `L(w)` by updating parameters opposite the gradient: `w' = w - η ∇_w L(w)`. The **learning rate** `η` controls step size, and `∇_w L` is the gradient (direction of steepest ascent, so we move opposite to descend). Too large a learning rate can cause divergence; too small makes convergence slow.
    * **Batch variants:** Full-batch GD computes the gradient over the entire training set (accurate but expensive). **Stochastic gradient descent (SGD)** uses a single random training example per update, introducing noise that can help escape local minima but makes convergence less stable. **Mini-batch SGD** strikes a balance by computing the gradient over a small random subset (e.g., 32–256 examples), combining efficiency with stable convergence.

2. [Probabilistic graphical models](./2-Thomas/2-Probabilistic%20graphical%20models/):
Probabilistic graphical models are a family of methods that use acyclic graph to represent probabilistic relationships between variables, where nodes denote variables and edges encode dependencies. This representation allows complex joint distributions to be expressed compactly and enables efficient inference by exploiting conditional independence.
    
    * [Maximum likelihood estimation](./2-Thomas/2-Probabilistic%20graphical%20models/ML-2026-2.pdf#page=20): Maximum likelihood learning (MLE) estimates model parameters by choosing the values that make the observed data most probable: `θ̂_MLE = argmax_θ log p(D | θ)`. The **likelihood function** `L(θ) = P(D | θ)` measures how probable the observed data is given the parameters. The **log-likelihood** is used instead (products become sums, numerically more stable). For complete data with simple multinomial models, the MLE reduces to counting observed frequencies. In Gaussian models, MLE yields the sample mean and (biased) sample variance.

    * [The EM algorithm](./2-Thomas/2-Probabilistic%20graphical%20models/ML-2026-2.pdf#page=43): The Expectation-Maximization algorithm handles maximum likelihood learning when data contains **hidden or missing variables**, making direct likelihood optimization intractable. It iterates two steps until convergence:

        - **E-step (Expectation):** Compute the **expected sufficient statistics** given current parameters `θ^t`. For each variable `X_i`, compute the expected count table by averaging over the posterior of the hidden variables:

          `E[N(X_i, pa(X_i) | D)] = Σ_{d in D} P(X_i, pa(X_i) | d, θ^t)`

          This is a **soft completion** — each missing value is fractionally assigned to all possible states weighted by their posterior probability (e.g., `P(Pr=yes | evidence) = 0.7`, `P(Pr=no | evidence) = 0.3`), rather than a single hard assignment.

        - **M-step (Maximization):** Update parameters using the expected counts as if they were observed counts (standard MLE, but with fractional counts):

          `θ̂^{t+1}_{ijk} = E[N(X_i = k, pa(X_i) = j | D)] / Σ_k E[N(X_i = k, pa(X_i) = j | D)]`

          This re-estimates each conditional probability table entry as: expected count of `(state, parent_config)` divided by expected count of `parent_config` across all states.

        Each iteration is guaranteed to **increase** (or not decrease) the observed-data log-likelihood, converging to a local optimum.

    * [Bayesian learning](./2-Thomas/2-Probabilistic%20graphical%20models/ML-2026-2.pdf#page=56): Bayesian learning treats model parameters as **random variables** and computes a posterior distribution over them using Bayes' rule: `p(θ | D) = p(θ) p(D | θ) / p(D)`. The **prior** `p(θ)` encodes beliefs before seeing data, the **likelihood** `p(D | θ)` measures how well the data supports each parameter value, and the **posterior** `p(θ | D)` is the updated belief after observing data.
        - **MLE vs MAP vs Bayesian:** MLE picks the single best `θ` maximizing likelihood; MAP adds a prior (`argmax log p(D|θ) + log p(θ)`); full Bayesian keeps the entire posterior distribution.
        - **Conjugacy:** When prior and posterior belong to the same family (e.g., Beta-Bernoulli: `Beta(a+N_1, b+N_0)`), inference reduces to simple parameter updates.

4. [Variational inference in probabilistic models](./2-Thomas/3-Variational%20inference%20in%20probabilistic%20models/)

    * [Variational inference basics](./2-Thomas/3-Variational%20inference%20in%20probabilistic%20models/ML-2026-4.pdf#page=6) (objective function, Evidence lower bound, mean field assumption): In Bayesian models we often want a posterior `p(z|x)` (where `x` is observed data, `z` are latent variables), but computing it exactly requires an intractable integral `p(x) = integral p(x,z) dz`. Variational inference avoids this by **turning inference into optimization**:
        - Choose a simpler family of distributions `q_lambda(z)` (the **variational family**) parametrized by `lambda`.
        - Find the member of that family closest to the true posterior by minimizing the **KL divergence** `KL(q || p)`. However, KL still depends on the unknown true posterior, so instead we maximize a different objective.
        - **ELBO (Evidence Lower Bound):** `ELBO(q) = E_q[log p(z,x)] - E_q[log q(z)]`. Maximizing the ELBO is equivalent to minimizing `KL(q || p)`. The log evidence decomposes as `log p(x) = ELBO(q) + KL(q || p)`, and since KL is non-negative, ELBO is a lower bound on `log p(x)`.
        - **Mean-field assumption:** Assume the latent variables are independent in the approximation: `q(z) = product_i q_i(z_i)`. For example, if `z = (z1, z2, z3)`, we use `q(z1,z2,z3) = q1(z1) q2(z2) q3(z3)`. This makes inference scalable, but may miss dependencies between variables.

    * [Black-box variational inference](./2-Thomas/3-Variational%20inference%20in%20probabilistic%20models/ML-2026-4.pdf#page=22): BBVI makes variational inference a **generic algorithm** that works for many models without manual derivations. It uses the **score-function gradient estimator** to estimate gradients of the ELBO via sampling, and then performs gradient ascent: `lambda' = lambda + eta * gradient ELBO`. The user does **not** need to derive model-specific update equations — the system computes gradient estimates from samples of `q(z)` and the joint `p(z,x)`. This is the key that makes variational inference practical for complex models.

    * [Variational inference and probabilistic programming](./2-Thomas/3-Variational%20inference%20in%20probabilistic%20models/ML-2026-4.pdf#page=36): Probabilistic programming languages (like Pyro, Stan, PyMC) let users **specify the model** (the joint distribution `p(z,x)`) directly in code, and then **automatically apply inference** methods like BBVI to approximate the posterior. The user declares variables, defines priors and likelihoods, and the system handles the optimization automatically. This separates model design from inference engineering, making advanced Bayesian analysis accessible without having to derive and implement variational updates by hand.

Peter's topics:

1. [Link prediction with node embeddings](./3-Peter/2-Link%20Prediction%20with%20Node%20Embedding%20Models/) Link prediction is the task of predicting missing or future edges in a graph, and It is based on the notion that **nodes which are similar should be connected**

    * [Encoder/Decoder View](./3-Peter/2-Link%20Prediction%20with%20Node%20Embedding%20Models/L9_linkprediction_withnodeembeddings.pdf#page=7): The encoder/decoder framework separates representation learning from prediction. The **encoder** `ENC(v) = Z[v]` maps each node to a low-dimensional embedding vector `z_v` in `R^d`. The **decoder** `DEC(z_u, z_v)` takes two embeddings and scores how likely a link exists between them — usually using a dot product `DEC(z_u, z_v) = z_u^T z_v` (cosine similarity). The **loss function** measures how well the decoded scores reconstruct the true adjacency, typically: `L = sum_{(u,v) in D} ||DEC(z_u, z_v) - S[u,v]||^2_2` where `S[u,v]` is the true similarity (e.g., 1 if edge exists, 0 otherwise). Minimizing this reconstruction loss forces embeddings to capture graph structure.

    * [Matrix Factorization](./3-Peter/2-Link%20Prediction%20with%20Node%20Embedding%20Models/L9_linkprediction_withnodeembeddings.pdf#page=17): Matrix factorization methods like **SVD** decompose the similarity matrix into lower-dimensional factors: `R = U * Sigma * V^T`. By keeping only the top `K` singular values (e.g., 20-100), we get a low-rank approximation that captures the most important structural patterns while filtering noise. The decoder is the dot product `z_u^T z_v`, and learning is done via SGD: for each observed pair `(u,v)`, update `A[u,k]` and `B[k,v]` to minimize `(R[u,v] - sum_k A[u,k]*B[k,v])^2`. A regularization term `lambda(||A||^2 + ||B||^2)` prevents overfitting. The result is that unobserved entries get filled in — that's the "magic" of matrix factorization for link prediction.

    * [Random Walk Based Methods](./3-Peter/2-Link%20Prediction%20with%20Node%20Embedding%20Models/L9_linkprediction_withnodeembeddings.pdf#page=42): Instead of using the adjacency matrix directly, random walk methods model the probability that two nodes co-occur on short random walks: `z_u^T z_v ∝ P_R(v|u)` (the probability of reaching `v` from `u` via a random walk). The loss uses **negative sampling** to avoid expensive computation over all node pairs: `L = sum_{(u,v) in D} -log(sigma(z_u^T z_v)) - gamma * sum_{i=1}^k log(sigma(-z_u^T z_k))`. Here `k` negative samples `z_k` are drawn from a noise distribution, and `sigma` is the sigmoid function. This makes training scalable even for large graphs.

2. [Graph Neural Networks](./3-Peter/3-Linkprediction%20with%20GNN/) Graph Neural Networks (GNNs) are neural network models that operate directly on graph data. They overcome the limitations of shallow node embeddings by using **shared parameters across nodes** and incorporating **node features** into the encoding process. This makes GNNs more expressive and able to generalize to unseen nodes.

    * [Message Passing Networks](./3-Peter/3-Linkprediction%20with%20GNN/L10_linkprediction_withGNN.pdf#page=33): GNNs replace shallow embedding lookups with neural encoders that use **both graph structure and node features**. The basic message passing layer: `h_u^{(k+1)} = sigma(W_self * h_u^{(k)} + W_neigh * sum_{v in N(u)} h_v^{(k)} + b)`. Each node updates its representation by combining its own previous state (`h_u`) with an **aggregated message** from neighbors (e.g., sum of neighbor features), followed by a linear transformation and non-linearity `sigma` (ReLU/tanh). After `K` layers, node embeddings contain information from the `K`-hop neighborhood. This overcomes shallow embedding problems: shared parameters across nodes, support for node features, and ability to generalize to unseen nodes.

    * [Generalizations with normalization](./3-Peter/3-Linkprediction%20with%20GNN/L10_linkprediction_withGNN.pdf#page=36), [set aggregation](./3-Peter/3-Linkprediction%20with%20GNN/L10_linkprediction_withGNN.pdf#page=39), and [attention](./3-Peter/3-Linkprediction%20with%20GNN/L10_linkprediction_withGNN.pdf#page=44) (including multi head attention): These extensions improve how neighborhood information is combined in GNNs:
        - **Normalization:** Plain summation makes high-degree nodes dominate. Using **mean** (`m = 1/|N(u)| * sum h_v`) or **symmetric normalization** stabilizes training by reducing sensitivity to node degree. Normalization is most helpful when node features matter more than graph structure.
        - **Set aggregation (pooling):** Beyond sum/mean, we can use **max pooling** over neighbors or more sophisticated methods like LSTM-based attention that processes neighbors in sequence, learning which neighbors contribute most to the representation.
        - **Attention:** Each neighbor gets a learned importance weight: `m_{N(u)} = sum_{v in N(u)} a_{u,v} * h_v`, where `a_{u,v}` is computed from the node features (e.g., via a learned vector `a` and softmax over neighbors). **Multi-head attention** runs several attention mechanisms in parallel and concatenates the results (`m = [head1 || head2 || ... || headK]`), letting the model capture different types of relationships simultaneously.

    * [Generalized Updates](./3-Peter/3-Linkprediction%20with%20GNN/L10_linkprediction_withGNN.pdf#page=46): Beyond the simple sum-and-update pattern, generalized updates can use **gating mechanisms** (like GRU/LSTM cells) to control how much of the previous node state is kept versus overwritten by the neighborhood message. This helps with **over-smoothing** (where many GNN layers make all node representations converge to the same value). Other approaches include jumping connections (skip connections from earlier layers) to preserve local information.

3. [Node and Graph Classification with shallow node embeddings and GNNs](./3-Peter/4-Node%20and%20Graph%20Classification/)

    * [Iterative Classification](./3-Peter/4-Node%20and%20Graph%20Classification/L11_node_and_graph_classification.pdf#page=37) (explaining on selected GNN or Node Embedding Method): Iterative classification exploits **homophily** (similar nodes tend to be connected) and **co-citation** (similar nodes link to the same things). It works in rounds:
        1. Train a base classifier on labeled nodes using their features.
        2. For each unlabeled node, aggregate features/labels from its neighbors (e.g., count of neighbor labels per class, or mean of neighbor feature vectors).
        3. Update predictions using both the node's own features AND the aggregated neighborhood information.
        4. Repeat until convergence — each iteration uses the latest predicted labels as input for the next round.
        This captures relational dependencies: a node's label depends not just on its own attributes but on what its neighbors are predicted to be.

    * [Node aggregations for graph embeddings](./3-Peter/4-Node%20and%20Graph%20Classification/L11_node_and_graph_classification.pdf#page=15): To classify whole graphs (not just nodes), we need one fixed-size vector per graph. This is done by **pooling** node embeddings:
        - **Sum or Mean:** `h_G = sum_{u in V} h_u` or `h_G = 1/|V| * sum_{u in V} h_u`. Simple and often sufficient for small graphs.
        - **LSTM + Attention:** Process nodes sequentially with an LSTM, computing attention weights per node at each step, then combine them: `h_G = [h_1 || h_2 || ... || h_T]` (concatenation of T attention-guided reads).
        - **Graph Coarsening:** Apply clustering repeatedly, each time aggregating clusters into super-nodes, then take the final pooled representation.
        The resulting graph embedding can be fed into any standard classifier (SVM, MLP).

    * [Random Walk based classifications](./3-Peter/4-Node%20and%20Graph%20Classification/L11_node_and_graph_classification.pdf#page=41): **Label propagation via random walks** uses the idea that a walk starting at an unlabeled node will eventually reach labeled nodes, and the probability of landing on each label determines the classification. The transition matrix `P` is split into labeled-labeled (`P_ll`), labeled-unlabeled (`P_lu`), unlabeled-labeled (`P_ul`), and unlabeled-unlabeled (`P_uu`) blocks. The stationary distribution for unlabeled nodes is: `P_infinity = (I - P_uu)^(-1) * P_ul`. This gives the probability that a random walk from each unlabeled node ends at each labeled class — and we classify by picking the most probable label. It's a simple, transductive method that works well when the graph structure strongly correlates with labels.
