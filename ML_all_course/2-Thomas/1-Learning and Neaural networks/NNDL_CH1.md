# neural network semantics
Can be seen as a composite function layer by layer.

The dimesionality 

of the inputs is $x$ a colum vector of dim 4x1

the bias is a vec col 5x1 

of the weight matrix is 4x5 

the second bas is 3x1 

the second weight is 5x3 

if the output of your nn is a distribution I need to use the log-loss functions. Having little values translates in to large negative values 

cross-entropy is used in case of categorical values.

# Perceptron

What is a neural network? To get started, I'll explain a type of artificial neuron called a perceptron. Perceptrons were developed in the 1950s and 1960s by the scientist Frank Rosenblatt, inspired by earlier work by Warren McCulloch and Walter Pitts. Today, it's more common to use other models of artificial neurons - in this book, and in much modern work on neural networks, the main neuron model used is one called the sigmoid neuron. We'll get to sigmoid neurons shortly. But to understand why sigmoid neurons are defined the way they are, it's worth taking the time to first understand perceptrons.

> [!IMPORTANT]
> A way you can think about the perceptron is that it's a device that makes decisions by weighing up evidence.

By varying the weights and the threshold, we can get different models of decision-making. Dropping the threshold means you're more willing to accept the output.

It should seem plausible that a complex network of perceptrons could make quite subtle decisions: 

![mlp](./images/multi_l_perceptron.png)

In this network, the first column of perceptrons - what we'll call the first layer of perceptrons - is making three very simple decisions, by weighing the input evidence. What about the perceptrons in the second layer? Each of those perceptrons is making a decision by weighing up the results from the first layer of decision-making. In this way a perceptron in the second layer can make a decision at a more complex and more abstract level than perceptrons in the first layer. And even more complex decisions can be made by the perceptron in the third layer. In this way, a many-layer network of perceptrons can engage in sophisticated decision making.

# Sigmoid neurons
Sigmoid neurons are similar to perceptrons, but modified so that small changes in their weights and bias cause only a small change in their output. That's the crucial fact which will allow a network of sigmoid neurons to learn.

Just like a perceptron, the sigmoid neuron has inputs, $x1,x2,…$. But instead of being just $0$ or $1$, these inputs can also take on any values between $0$ and $1$. So, for instance, $0.638…$ is a valid input for a sigmoid neuron. Also just like a perceptron, the sigmoid neuron has weights for each input, $w1,w2,…,$ and an overall bias, $b$. But the output is not $0$ or $1$. Instead, it's $σ(w⋅x+b)$, where $σ$ is called the sigmoid function 

(Incidentally, $σ$ is sometimes called the logistic function, and this new class of neurons called logistic neurons. It's useful to remember this terminology, since these terms are used by many people working with neural nets. However, we'll stick with the sigmoid terminology.), 

and is defined by:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

To put it all a little more explicitly, the output of a sigmoid neuron with inputs $x1,x2,…$, weights $w1,w2,…$, and bias $b$ is

$$
\frac{1}{1+exp(−∑_j w_jx_j−b)}
$$

the exact form of $σ$ isn't so important - what really matters is the shape of the function when plotted. Here's the shape:

![sigmoid_fun](./images/sigmoid_fun.png)

This shape is a smoothed out version of a step function:

![step_fun](./images/step_fun.png)

> [!IMPORTANT]
> If $σ$ had in fact been a step function, then the sigmoid neuron would be a perceptron
