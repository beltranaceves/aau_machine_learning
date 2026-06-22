## Link Prediction with Graph Neural Networks

## outline
### Problems with previous methods

L8: • Computing pairwise similarities and complete random walks can be 
computationally difficult for large graphs and all nodes• Spamming is a problem• Simple changes can affect the results• Coverage especially for local measures poor• ...

L9:• There are no shared parameters between nodes• No node features used in encoding•  They know only nodes which have been there during training
Machine Learning

> Replacing “shallow encoders” by more general encodersUsing both, graph structure and graph attributes/featuresUsing Neural Network

Encoding => Learning
Decoding => Prediction

For many prediction tasks incl. link prediction s

the illustration on slide 8 is aggregating the local (only one iteration) neighborhood this is our embedding

formalization slide 14

> The K used of the embedding space must be not too large otherwise the nodes will be all 1 and you can connect everything with everything

Options for Node Features (Input)
•Features from a dataset describing a node
•Node statistics – node degree, node prestige, ...
•Identity features – one hot encoding of node features –
makes the model transudative though – you can consider this 
as a vector with dimensions for all nodes with 1 where the 
node identity is

**Still following the same idea of embeddings. But now also with node features and aggregations**

### Basic GNN
#### Self Loops

Generalized Neighborhood Aggregation
• To address stability and sensitivity to node degrees
• Going beyond Sum as an aggregation
• Dealing with different importance of different neighbors

Generalized Update Methods
• To deal with over–smoothing and neighborhood influence


Normalize or Not To Normalize?
This question is application specific
Normalization is most helpful when node features are far more 
important than graph structural information

Normalization makes differences between node degrees smaller

Normalization can be also useful when there is a wide range of node degrees


### Message PassingGNN based on Simple 
### Message Passing
### Generalized Aggregation
### Generalized Update
### Conclusions
