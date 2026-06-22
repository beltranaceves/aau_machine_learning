Manfred’s topics:

1. [Linear Models](./1-Manfred/1-Linear%20Models/)

    * [Decision Regions](./1-Manfred/1-Linear%20Models/ml-F26-01.pdf#page=16): Decision regions describe how the input (or instance) space is divided into areas assigned to different classes by a classifier. In linear models, these regions are separated by straight lines or hyperplanes.

    * [Overfitting](./1-Manfred/1-Linear%20Models/ml-F26-01.pdf#page=24): Overfitting happens when a model learns noise and details from the training data instead of the general pattern. As a result, it performs well on training data but poorly on unseen examples.

    * [Least squares regression](./1-Manfred/1-Linear%20Models/ml-F26-01.pdf#page=24) (corresponding to sklearn LinearRegression in self study 1): Least squares classification treats classification as regression. The class labels are one-hot encoded, and the model learns weights by minimizing the squared difference between predicted score vectors and target vectors. However, squared error is not the same as classification error, so least squares can be sensitive to outliers.

    * [Linear discriminant analysis](./1-Manfred/1-Linear%20Models/ml-F26-02.pdf#page=13): LDA is different because it is *generative*. It models $P(Y)$ and $P(X∣Y)$, assuming each class has a Gaussian distribution. In LDA, all classes share the same covariance matrix, which leads to linear decision boundaries. The model estimates class priors, class means, and a shared covariance matrix from the data.

    * [Logistic Regression](./1-Manfred/1-Linear%20Models/ml-F26-02.pdf#page=15): Logistic regression is discriminative. It directly models $P(Y∣X)$. For binary classification, it uses the sigmoid function $P(Y=1∣x)=\sigma(w\cdot x)$. If this probability is at least 0.5, the point is classified as class 1. This again gives a linear decision boundary. Unlike least squares, logistic regression optimizes likelihood or cross-entropy, which is more appropriate for classification, but it requires iterative optimization.

2. [Support Vector Machines](./1-Manfred/2-Support%20Vector%20Machines/)

    * [Maximum margin hyperplanes](./1-Manfred/1-Linear%20Models/ml-F26-02.pdf#page=21): A maximum margin hyperplane separates classes while maximizing the distance to the nearest training points. This larger margin often improves the model’s ability to generalize to new data.

    * [Feature transformations](./1-Manfred/2-Support%20Vector%20Machines/ml-F26-03.pdf#page=2) and [kernel functions](./1-Manfred/2-Support%20Vector%20Machines/ml-F26-03.pdf#page=2): Feature transformations map data into a new space where patterns may become easier to separate. Kernel functions allow this to be done implicitly, without computing the transformed coordinates directly.

    * [The kernel trick](./1-Manfred/2-Support%20Vector%20Machines/ml-F26-03.pdf#page=17): The kernel trick lets algorithms operate in a high-dimensional feature space by only evaluating pairwise kernel similarities. This makes nonlinear classification computationally feasible.

    * [String kernels](./1-Manfred/2-Support%20Vector%20Machines/ml-F26-03.pdf#page=25): String kernels measure similarity between text or sequence data by comparing shared substrings or patterns. They are useful in domains such as text classification and bioinformatics where inputs are symbolic sequences.

3. [Graph kernels](./1-Manfred/3-Graph%20kernels/)

    * [Convolution kernels](./1-Manfred/3-Graph%20kernels/ml-F26-12.pdf#page=10): Convolution kernels compare structured objects by decomposing them into smaller parts and summing similarities across those parts. In graphs, this can mean comparing nodes, edges, paths, or substructures.

    * [Subgraph isomorphisms](./1-Manfred/3-Graph%20kernels/ml-F26-12.pdf#page=17) and [graphlet kernels](./1-Manfred/3-Graph%20kernels/ml-F26-12.pdf#page=27): Subgraph isomorphism checks whether a smaller graph pattern appears inside a larger graph. Graphlet kernels use counts of small subgraphs to represent graphs and compare them based on their local structural motifs.

    * [Random walk kernels](./1-Manfred/3-Graph%20kernels/ml-F26-12.pdf#page=36): Random walk kernels compare graphs by counting matching walks in both graphs. They capture structural similarity through sequences of connected nodes, although they can be computationally expensive.

    * [Weisfeiler Lehman kernel](./1-Manfred/3-Graph%20kernels/ml-F26-12.pdf#page=36): The Weisfeiler-Lehman kernel compares graphs by iteratively relabeling nodes based on the labels of their neighbors. This captures increasingly rich neighborhood structure and makes graph comparison efficient and powerful.

Thomas' topics:

1. [Learning and neural networks](./2-Thomas/1-Learning%20and%20Neaural%20networks/)

    * [Loss functions](./2-Thomas/1-Learning%20and%20Neaural%20networks/ML-2026-1.pdf#page=12): A loss function measures how far a model’s predictions are from the true targets. It provides the objective that learning algorithms try to minimize during training.

    * [Back propagation and the chain rule](./2-Thomas/1-Learning%20and%20Neaural%20networks/ML-2026-1.pdf#page=28): Backpropagation computes gradients in a neural network by applying the chain rule from calculus layer by layer. These gradients tell us how to update the parameters to reduce the loss.

    * [Computational graphs](./2-Thomas/1-Learning%20and%20Neaural%20networks/ML-2026-1.pdf#page=30): A computational graph represents a mathematical expression as a network of operations and variables. It makes forward computation and automatic differentiation easier to organize and implement.

    * [Gradient decent](./2-Thomas/1-Learning%20and%20Neaural%20networks/ML-2026-1.pdf#page=16) and [stochastic gradient descent](./2-Thomas/1-Learning%20and%20Neaural%20networks/ML-2026-1.pdf#page=60): Gradient descent updates parameters by moving them in the direction that most decreases the loss. Stochastic gradient descent uses one or a few training examples at a time, making learning faster and more scalable.

2. [Probabilistic graphical models](./2-Thomas/2-Probabilistic%20graphical%20models/)

    * [Maximum likelihood learning](./2-Thomas/2-Probabilistic%20graphical%20models/ML-2026-2.pdf#page=20): Maximum likelihood learning estimates model parameters by choosing the values that make the observed data most probable. It is a fundamental principle for fitting probabilistic models.

    * [The EM algorithm](./2-Thomas/2-Probabilistic%20graphical%20models/ML-2026-2.pdf#page=43): The Expectation-Maximization algorithm is used when models involve hidden or missing variables. It alternates between estimating latent variables and updating parameters until convergence.

    * [Bayesian learning](./2-Thomas/2-Probabilistic%20graphical%20models/ML-2026-2.pdf#page=56): Bayesian learning treats model parameters as random variables and updates beliefs about them using observed data. This combines prior knowledge with evidence and naturally captures uncertainty.

3. [Variational inference in probabilistic models](./2-Thomas/3-Variational%20inference%20in%20probabilistic%20models/)

    * [Variational inference basics](./2-Thomas/3-Variational%20inference%20in%20probabilistic%20models/ML-2026-4.pdf#page=6) (objective function, Evidence lower bound, mean field assumption): Variational inference approximates difficult posterior distributions with a simpler family of distributions. It does this by optimizing the ELBO, often under assumptions such as mean field independence between variables.

    * [Black-box variational inference](./2-Thomas/3-Variational%20inference%20in%20probabilistic%20models/ML-2026-4.pdf#page=22): Black-box variational inference uses generic gradient estimators so inference can be applied without deriving model-specific update rules. This makes variational methods easier to use in complex probabilistic models.

    * [Variational inference and probabilistic programming](./2-Thomas/3-Variational%20inference%20in%20probabilistic%20models/ML-2026-4.pdf#page=36): Probabilistic programming allows users to define probabilistic models in code, and variational inference can then be used automatically to approximate the posterior. This makes advanced inference more flexible and accessible.

Peter's topics:

1. [Link prediction with node embeddings](./3-Peter/2-Link%20Prediction%20with%20Node%20Embedding%20Models/)

    * [Encoder/Decoder View](./3-Peter/2-Link%20Prediction%20with%20Node%20Embedding%20Models/L9_linkprediction_withnodeembeddings.pdf#page=7): In the encoder/decoder view, the encoder maps nodes to low-dimensional embeddings and the decoder uses those embeddings to predict whether a link exists. This framework separates representation learning from the actual prediction mechanism.

    * [Matrix Factorization](./3-Peter/2-Link%20Prediction%20with%20Node%20Embedding%20Models/L9_linkprediction_withnodeembeddings.pdf#page=17): Matrix factorization learns node embeddings by decomposing an adjacency or similarity matrix into lower-dimensional factors. The learned vectors capture latent structure that can be used to reconstruct or predict edges.

    * [Random Walk Based Methods](./3-Peter/2-Link%20Prediction%20with%20Node%20Embedding%20Models/L9_linkprediction_withnodeembeddings.pdf#page=42): Random walk based methods generate node sequences by traversing the graph and use co-occurrence information to learn embeddings. Nodes that appear in similar walk contexts end up with similar vector representations.

2. [Graph Neural Networks](./3-Peter/3-Linkprediction%20with%20GNN/)

    * [Message Passing Networks](./3-Peter/3-Linkprediction%20with%20GNN/L10_linkprediction_withGNN.pdf#page=33): Message passing networks update each node by aggregating information from its neighbors. Repeating this over several layers allows nodes to build representations that reflect larger graph neighborhoods.

    * [Generalizations with normalization](./3-Peter/3-Linkprediction%20with%20GNN/L10_linkprediction_withGNN.pdf#page=36), [set aggregation](./3-Peter/3-Linkprediction%20with%20GNN/L10_linkprediction_withGNN.pdf#page=39), and [attention](./3-Peter/3-Linkprediction%20with%20GNN/L10_linkprediction_withGNN.pdf#page=44) (including multi head attention): These extensions improve how neighborhood information is combined in GNNs. Normalization stabilizes updates, set aggregation respects permutation invariance, and attention learns which neighbors are most important, possibly using multiple attention heads.

    * [Generalized Updates](./3-Peter/3-Linkprediction%20with%20GNN/L10_linkprediction_withGNN.pdf#page=46): Generalized updates refer to flexible node update rules that go beyond simple averaging of neighbor features. They allow richer transformations, gating, or learned combinations that improve expressiveness.

3. [Node and Graph Classification with shallow node embeddings and GNNs](./3-Peter/4-Node%20and%20Graph%20Classification/)

    * [Iterative Classification](./3-Peter/4-Node%20and%20Graph%20Classification/L11_node_and_graph_classification.pdf#page=37) (explaining on selected GNN or Node Embedding Method): Iterative classification repeatedly predicts node labels while using the current predicted labels of neighboring nodes as additional features. This captures relational dependencies and can improve classification in graph-structured data.

    * [Node aggregations for graph embeddings](./3-Peter/4-Node%20and%20Graph%20Classification/L11_node_and_graph_classification.pdf#page=15): Graph embeddings can be built by aggregating node representations into one fixed-size vector for the whole graph. Common aggregation methods include sum, mean, and max pooling.

    * [Random Walk based classifications](./3-Peter/4-Node%20and%20Graph%20Classification/L11_node_and_graph_classification.pdf#page=41): Random walk based classification methods use traversal patterns in a graph to extract structural context for nodes or graphs. These patterns can then be transformed into features or embeddings and used for classification tasks.