## Intro to Link Prediction Based on Structural Similarity

## Use cases
- Node Clalssification
- Link Prediction
- Network/Graph Clalssification


## Graphs
G = (V, E)

- Directed
- Undirected
- Weighted
- Labeled
- Signed

### Degre of a graph
-For directed graph:
    - In-degree of node 𝑣_𝑖 = number of edges pointing into the node. d_in_i
    - Out-degree of node 𝑣_𝑖 = number of edges pointing away from the node. d_out_i
    - Degree = In-degree + Out-degree. 𝑑_𝑖
For undirected graph:
    - Undirected edge = two opposite directed edges (aka. bi-directional or reciprocal edge)
    - In-degree = Out-degree = number of edges connected to node
    - Degree = 2 times the number of edges
### Degre Distribution
p_d = n_d/n, where
    n_d: number of nodes with degree d
    n: number of nodes
### Power-Law Degree Distribution
𝑝_𝑑 = 𝛽_𝑑^(−𝛼)
log 𝑝_𝑑 = log 𝛽 − 𝛼 . log(𝑑)

𝛼: the power-law exponent and its value is typically in the range of [2, 3]
𝛽: power-law intercep

### Graph representation
- Adjacency matrix
    - Dense
- Adjacency list
    - Sparse
    - Multivalued
- Edge list
    - Less useful for our use cases

## Page Ranking

## Link prediction

- Feature/property approach
- Structural approach
    - Main point for the lectures

### Structural similarity 
- Local measures, useful but non-optimal, cheap and vulnerable to spamming.
    - Vertex simm
    - Jaccard simm
    - Cosine simm


