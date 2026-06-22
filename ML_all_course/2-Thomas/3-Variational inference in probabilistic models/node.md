# Important to remember slides for Variational Inference [EXAM IMPORTANT]
n 5: given a binomial distribution 

p is always the real distribution(whe do not know the explicit formula) q is an aproximation.

x is data 
z rappresents the unobserved thing that I'm interested in(for bayesianNN would be the weights)
lambda are the parameters of the model

q_lambda(z) is a class of aproximations "candidates"(might use more than one)

we are going to use a KL convergence that provides 0 if its an exact match. What might be the best fit?

How do we minimize KL?
if I do not know z and x how do I know what I'm doing?

n 6
KL is and expectation of the log between two distribuitions. 

> Is **not symmetric** 

when the distributions are equal the fraction goes to 1 and the log to 0.

n 7

after applying the properties of log we move out log p(x) because does not depend on z

to have the joint probability we reduce the p(z|x)p(x)

log p(x) it the log marginal likelihood we already saw

we rewrite everything in function on x so we do not need to care about x.

green and pink are inversely proportional

we prefer to maximize GREEN because we just need to know p(z,x) instead of needing p(z|x)(that should be integreated later)

n 8 

n 9
the bayesian as a negative correlation the line should go down if we increase our bias

for the variational solution we have a posterior distribution well but we fail to catch the correlation because the mean field assumption that considers a independece

n 16 
in the differentiation vanishes the fraction and the exponential leaving to differentiation only for the argument for exp

for the expectation we just need to do sampling and compute the mean over the plugged in samples

