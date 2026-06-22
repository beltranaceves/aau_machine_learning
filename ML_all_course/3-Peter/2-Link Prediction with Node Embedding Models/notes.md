> [!Problem]
> Computing pairwise similarities and complete random walks can be computationally difficult for large graphs and all nodes
> Spamming is a problem
> Simple changes can affect the results
> Coverage especially for local measures poor

The general idea is to encode the original network in to a space(embbeded space). This is the learning part. Then for prediction we are going to decode the encoding space. This apporace is general.

## Intuition
Consider a two-dimensional space given by a drawing canvas
Draw a graph according to selected drawing algorithm

Observe and record coordinates of nodes in that space

> Use that as an embeddings

e.g.
```python
[[-0.08608312 0.35674596]
[-0.10669304 0.20477623]
[-0.06612635 0.03965846]
[-0.2406135 0.2481614 ]
[ 0.10655595 0.60888028]
[-0.0755657 0.76935902]
[-0.01136205 0.73449742]
[-0.25088574 0.15759357]
[ 0.0472217 -0.07626688]
[ 0.38513712 -0.10531281]
[ 0.10468862 0.73088794]
[-0.27860864 0.66405851]]
```

### ENCODER
Generally: $ENC: 𝒱 → ℝ^𝑑$

Specifically in node embeddings: ENC(v) = 𝐙[𝑣]
Where 𝐙 ∈ ℝ^{|𝒱|×𝑑}

### DECODER

Generally: DEC: ℝ^d × ℝ^𝑑 → ℝ+
• Specifically in reconstruction of the relationships:
DEC(ENC(𝑢) , ENC(𝑣)) = DEC(z_u , z_v ) ≈ 𝐒[𝑢, 𝑣]

### LEARNING
We want to **minimize** the reconstruction loss, since we want to reconstruct the links

as usual the loss function is the sum is the sum of the loss between the decode and the estimated similarity.

To define: loss function, similarity, DEC function

## Matrix Factorization / Inner Product Methods / Dimmensionality Reduction

Basic idea:

Trade more complex offline model building for faster online prediction generation

Singular Value Decomposition(SVD) for dimensionality reduction of rating matrices(works also in non square matrices):

- Captures important factors/aspects and their weights in the data
- Factors can be genre, actors but also non-understandable ones
- Assumption that k dimensions capture the signals and filter out noise (K = 20 to 100)

## Formally

decoder:
is a dotproduct

loss:
Usually the loss is a square loss

similarity:
if we use matrix factorization we can apply matrix factorization approach such as SVD

## Intuition behind the dot product(LOOK AT SLIDE 16)

## Matrix factorization

Informally, the SVD theorem (Golub and Kahan 1965) states that a given
matrix 𝑀 can be decomposed into a product of three matrices as follows

𝑅 = 𝑈 × Σ × 𝑉^T

• where 𝑈 and 𝑉 are called left and right singular vectors and the values of the diagonal of Σ are called the singular values

We can approximate the full matrix 𝑹 by observing only the most important features – those with the largest singular values

example slide 18

we do not do to a 3 matrix multiplication we will like just to use 2 page 24

we just take some vectors and update the matrices

## Better alg page 30 
we do sub sampling reduces precision but increases speed
 
 