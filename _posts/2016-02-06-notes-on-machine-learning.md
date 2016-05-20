---
layout: post
title: "Notes on Machine Learning"
tags:
    - python
    - notebook
---

{% include _toc.html %}

--- 
These notes are a result of my preparation for a midterm exam in [Pascal Poupart](https://cs.uwaterloo.ca/~ppoupart)'s [course on Machine Learning](https://cs.uwaterloo.ca/~ppoupart/teaching/cs485-winter16/) at University of Waterloo in the winter 2016 term. The contents are therefore based on the corresponding presentation slides available online. 
 
## Introduction

Definition by Tom Mitchell (1988): "A computer program is said to learn from **experience E** with respect to some class of **tasks T** and **performance measure P** if its performance for tasks in T, as measured by P, improves with experience E."

Inductive Learning: given a training set of examples of the form $(x, f(X))$, return a function $h$ (**hypothesis**) that approximates $f$ (**true underlying function**).

The quality measure for hypotheses is **generalization**. A good hypothesis will generalize well, i.e., predict unseen examples correctly. **Ockham's razor** suggests to prefer the simplest hypothesis consistent with the input data.

Two different types of inductive learning:

* **Classification**: range/output space of $f$ is categorical (or discrete)
* **Regression**: range/output space of $f$ is continuous

**Hypothesis space $H$**: set of all possible $h$

**Consistency**: $h$ is consistent if it agrees with $f$ on all examples; it is not always possible to find a consistent $h$ (e.g., due to an insufficient $H$ or noisy input data)

**Realizability**: a learning problem is realizable if and only if $f$ is in $H$

In general, we will observe a tradeoff between **expressiveness** (i.e., the size of $H$) and the **complexity** of finding a good $h$.

**Overfitting**: given a hypothesis space $H$, a hypothesis $h \in H$ is said to overfit the training data if there exists some alternative hypothesis $h' \in H$ such that $h$ has smaller error than $h'$ over the training examples, but $h'$ has smaller error than $h$ over the entire distribution of instances.

**$k$-fold cross-validation:** run $k$ experiments, each time putting aside $\frac{1}{k}$ of the data to test on and, finally, compute the average accuracy of the experiments 
 
## Decision Trees

Decision trees (a.k.a. **C**lassification **a**nd **R**egression **T**rees [CART]) represent disjunctions (OR) of conjunctions (AND) of constraints of attribute values.

Decision trees have the following **structure**:

* **nodes**: attributes
* **edges**: attribute values
* **leaves**: classes / regression values

**Quality measure** for decision trees:

* small size
* high consistency

**Greedy induction** of a decision tree:

* depth-first search like construction
* a good attribute splits the examples into subsets that are ideally all from the same class or, in other words, that minimizes the residual error

**Residual error for classification**:

||Error Frequency|Gini Index|Entropy|
|---|---|---|---|
|**Definition**|$Q_\tau=num(\tau)-max_k num(k)$|$Q_\tau=\sum_kp_\tau(k)(1-p_\tau(k))$|$Q_t=-\sum_kp_\tau(k) \ ld \ p_\tau(k)$|
|**Explanation**|Number of examples in leaf $\tau$ minus maximum number of examples in leaf $\tau$ that belong to any class $k$|Expected misclassification when choosing the class according to $p_\tau(k)$|Expected # of bits to encode the class of an instance chosen at random according to $p_\tau(k)$|

**Variables:**

* $\tau$: index for leaf $\tau$
* $k$: index for class $k$
* $num(\tau)$: number of examples in leaf $\tau$
* $num(k)$: number of examples in leaf $\tau$ belonging to class $k$
* $p_\tau(k) = \frac{num(k)}{num(\tau)}$

**Residual error for regression**:

* Let $t_n = f(x_n)$ be the target for the $n^{th}$ example.
* Let $y_t$ be the value returned by leaf $\tau$.
* Let $R_\tau$ be the set of examples in leaf $\tau$.
* Euclidean error for leaf $\tau$: $Q_\tau=\sum_{n \in R_\tau}(t_n-y_n)^2$

**Choosing the best attribute** for the next decision layer in the tree:

* In leaf $\tau$, choose attribute $A$ that reduces the residual error the most when expanded:
* $A^*=argmax_AQ_\tau-\sum_{a \in A}p_\tau(A=a)Q_{\tau a}$, where
  * $p_\tau(A=a)=\frac{num(A=a)}{num(\tau)}$ denotes the proportion of examples with value $a$ (in attribute $A$) inside node $\tau$
  * $\tau a$ indexes the node reached by following the edge for attribute value $a$, starting from node $\tau$
  
**Techniques to avoid overfitting:**

* Stop learning when the curve for testing accuracy, plotted against the tree size, has reached its global peak; in practice, this is sometimes sometimes hard to achieve because the curve might exhibit high fluctuation
* Pruning of statistically irrelevant nodes in a bottom-up fashion
  * Remove nodes that improve testing accuracy by less than some threshold
  * Regularization: add a penalty term that reflects the tree complexity (e.g., $\|T\|=$ #leaves in the tree) and remove leaves with a negative "regularized" error reduction: $Q_\tau-\sum_{a \in A}p_\tau(A=a)Q_{\tau a}-\lambda\|T\|$ 
 
## K-Nearest Neighbors

In the limit, single-attribute thresholding to construct decision trees for attributes with continuous inputs will lead to a full tree with one input example per leaf. Decision boundaries will always be axis-aligned. A better approach without this restriction is $K$-Nearest Neighbors.

**Approach:**

* Let $knn(x)$ be the $K$ nearest neighbors of $x$ according to distance $d$
* Label $y_x=mode(\{y_x' \mid x' \in knn(x)\})$, i.e., the most frequent label among the $K$ nearest neighbors
* $K$ controls the degree of smoothing and can be optimized using $k$-fold cross-validation (if $K$ is too small this will lead to overfitting; if $K$ is too high this will lead to underfitting)

**Comparison of complexity** between decision trees and $K$-Nearest Neighbors with respect to:

* $N$: size of training set
* $M$: number of attributes

||Training &nbsp;  &nbsp;  &nbsp; |Testing &nbsp;  &nbsp;  &nbsp; |
|---|---|---|
|**Decision Tree**|$O(M^2N)$|$O(M)$|
|**$K$-Nearest Neighbors**|$O(MN)$|$O(MN)$|

A good visualization of the decision boundaries for the two-dimensional 1-Nearest Neighbor case assuming Euclidean distance is the Voronoi diagram. 
 
## Linear Regression

**Data**: $\{(x_1, t_1),(x_2,t_2),...,(x_N,t_N)\}$ where $x_n \in \mathbb{R}^D$ and $t_n \in \mathbb{R}$

**Problem**: find linear hypothesis $h$ that maps $x$ to $t$; in other words, try to find a weight vector $w \in \mathbb{R}^{D+1}$ so that the error for $h(x,w)=w^T\overline{x}$ with $\overline{x}=\begin{pmatrix} 1 \\ x \end{pmatrix}$ over all $x$ in the dataset is minimal

**Solution**: $ Aw=b $ with:

* $A=\sum_{n=1}^N\overline{x}_n\overline{x}_n^T$ (invertible if the training instances span $\mathbb{R}^{D+1}$)
* $b=\sum_{n=1}^Nt_n\overline{x}_n$

**Tikhonov Regularization**: a technique, applied to avoid a form of overfitting where small changes to the input data lead to big changes in the learned weight vector: $(\lambda I+A)w=b$. The greater $\lambda$ the smaller the magnitude of $w$ will be. 
 
## Statistical Learning

**Idea**: learning simply reduces the uncertainty in our knowledge of the world

**Marginal Probability Distribution**: specification of a probability for each event in our sample space; all probabilities must sum up to $1$

**Joint Probability Distribution**: specification of probabilities for all combinations of events: $Pr(A=a
\wedge B=b)$ for all $a$, $b$

**Marginalization** (sumout rule): $Pr(A=a)=\sum_bPr(A=a
\wedge B=b)$

**Conditional Probability**: fraction of worlds in which $B$ is true that also have $A$ true: $Pr(A \mid B)=\frac{Pr(A \wedge B)}{Pr(B)}$; also $Pr(A \wedge B)=Pr(A \mid B) \ Pr(B)$

**Bayes' Rule**: $Pr(B \mid A)=\frac{Pr(A \mid B) \ Pr(B)}{Pr(A)}$

**Bayesian Inference**: $P(H \mid e)=kP(e \mid H) \ P(H)$

* $H$: hypothesis
* $e$: evidence
* $P(H)$: **prior** probability of $H$
* $P(e \mid H)=\prod_nP(e_n \mid H)$: **likelihood** of observing $e$, given $H$
* $P(H \mid e)$: **posterior** probability of $H$, given $e$
* $k$: normalizing factor, applied so that all posteriors sum up to $1$

**Bayesian Prediction**: $P(X \mid e)=\sum_iP(X \mid h_i) \ P(h_i \mid e)$

**Properties of Bayesian Learning**: optimal and not prone to overfitting, but potentially intractable if the hypothesis space is large

**Approximations of Bayesian Learning**:

* **Maximum a Posteriori (MAP)**:
  * making predictions, based on the most probable hypothesis $h_{MAP}=argmax_{h_i}P(h_i \mid e)$
  * less accurate than Bayesian prediction, but both converge in accuracy as data increases
  * controlled overfitting (prior can be used to penalize complex hypotheses)
  * MAP for linear regression leads to regularized least square problem
* **Maximum Likelihood (ML)**:
  * MAP with uniform prior: $h_{ML}=argmax_{h_i}P(e \mid h_i)$
  * less accurate than Bayesian and MAP prediction, but all three converge in accuracy as data increases
  * prone to overfitting
  * ML for linear regression leads to non-regularized least square problem
  
**Bias-Variance Decomposition** for linear regression: $expected \ loss = (bias)^2 + variance + noise$

* bias increases as regularization parameter $\lambda$ increases
* variance decreases as regularization parameter $\lambda$ increases
* noise is constant 
 
## Mixture of Gaussians

**Purpose**: linear classification technique

**Assumptions**:

* $Pr(C=c_k)=\pi_k$: prior probability of class $k$
* $Pr(x \mid C=c_k) \propto e^{-0.5(x-\mu_k)^T\Sigma^{-1}(x-\mu_k)}$: likelihood of data point $x$, given class $k$, is a Gaussian distribution with the same covariance matrix $\Sigma$ for all classes
* $Pr(C=c_k \mid x)=kPr(x \mid C=c_k)Pr(C=c_k)$: posterior probability of class $k$, given data point $x$

**For two classes $A$ and $B$** (using sigmoid):

$Pr(C=A)=\sigma(w^Tx+w_0)$ where

* $\sigma(a) = \frac{1}{1+e^{-a}}$ (sigmoid function)
* $w=\Sigma^{-1}(\mu_A-\mu_B)$
* $w_0=-0.5\mu_A^T\Sigma^{-1}\mu_A+0.5\mu_B^T\Sigma^{-1}\mu_B+ln\frac{\pi_A}{\pi_B}$
* $\pi_k:$ fraction of training examples that belong to class $k$ (via maximum likelihood)
* $\mu_k:$ empirical mean of all training examples that belong to class $k$ (via maximum likelihood)
* $\Sigma=\frac{S_A+S_B}{N}$: normalized sum of covariance matrices (via maximum likelihood)
* $S_k=\sum_{n \in c_k}(x_n-\mu_k)(x_n-\mu_k)^T$

**For multiple classes** (using softmax):

$Pr(C=c_k \mid x)=\frac{e^{w_k^T\overline{x}}}{\sum_je^{w_j^T\overline{x}}}$

**Prediction**: best class $k^*=argmax_kPr(c_k \mid x)$ 
 
## Logistic Regression

**Purpose**: linear classification technique (can be viewed as a regression where the goal is to estimate a posterior probability which is continuous)

**Assumption**: $Pr(x \mid C=c_k)$ are members of the exponential family: $Pr(x \mid \Theta_k)=exp(\Theta_k^TT(x)-A(\Theta_k)+B(x))$

**Derivation**: $Pr(C=c_k \mid x)=\sigma(w^T\overline{x})$: the posterior probability of class $k$ is a sigmoid logistic linear in $x$ (or softmax linear in $x$ for more than two classes)

**Idea**: learning $Pr(C=c_k \mid x)$ directly by maximum likelihood

**Implementation for binary classification**:

* $y \in \{0,1\}$
* $w^*=argmax_w\prod_n\sigma(w^T\overline{x})^{y_n}(1-\sigma(w^T\overline{x}))^{1-y_n}$
* $ \ \ \ \ \ =argmin_w-\sum_nln(\sigma(w^T\overline{x}))+(1-y_n)ln(1-\sigma(w^T\overline{x}))$
* Derivative $\nabla L(w)=\sum_n(\sigma(w^T\overline{x})-y_n)\overline{x}_n$
* Solve derivative iteratively for $0$ using Newton's method: $w_{i+1}=w_i-H^{-1}\nabla L(w)$ where
  * $H=\overline{X}R\overline{X}^T$ is the Hessian matrix, $X$ being the training data matrix where each column is represents one input vector
  * $R$ is a diagonal matrix of size $N*N$ with entries of $\sigma_n(1-\sigma_n)$
  * $\sigma_n=\sigma(w_i^T\overline{x}_n)$ 
 
## Generalized Linear Models

**Purpose**: non-linear classification/regression

**Idea**: map inputs to a different space using a set of basis functions and do linear classification/regression in that space

**Common basis functions**:

* Polynomial: $\theta_j(x)=x^j$
* Gaussian: $\theta_j(x)=e^{-\frac{(x-\mu_j)^2}{2s^2}}$
* Sigmoid: $\theta_j(x)=\sigma(\frac{(x-\mu_j)^2}{s})$
* Fourier, Wavelets etc. 
 
## Artificial Neural Networks

**Purpose**: non-linear classification/regression

**Idea**: network of units similar to neurons in a human brain

**Implementation**: numerical output of unit $j$, $h(a_j)$ where

* $a_j=\sum_{i}W_{ji}x_i+w_0=W_j\overline{x}$
* $x_i$ is the output of unit $i$
* $W_{ji}$ denotes the strength of the link from unit $i$ to unit $j$
* $h(x)$ is the activation function (e.g., threshold, sigmoid, Gaussian, hyperbolic tangent, identity)

**Structures**:

* feed-forward network (directed acyclic graph)
* recurrent network (directed cyclic graph)

**Perceptron**: single-layer feed-forward network

* **Threshold Perceptron Learning**:
  * done separately for each unit $j$
  * for each $(x,y)$ pair, correct weight $W_{ji}$ if incorrect output is produced:
    * if output produced is $0$ instead of $1$: $W_{ji}=W_{ji}+x_i$
    * if output produced is $1$ instead of $0$: $W_{ji}=W_{ji}-x_i$
  * convergence if and only if the dataset is linearly separable
* **Sigmoid Perceptron Learning**:
  * same hypothesis space as logistic regression

**Multi-layer neural networks**: flexible non-linear models by learning non-linear basis functions

* examples of 2-layer feed-forward networks:
  * $h_1$ non-linear and $h_2$ **sigmoid**: non-linear **classification**
  * $h_1$ non-linear and $h_2$ **identity**: non-linear **regression**

**Back Propagation**:

* Purpose: learning by iteratively adjusting network's weights to minimze output error
* Two phases:
  * Forward phase: compute output $z_j$ for each unit $j$
  * Backward phase: compute delta $\delta_j$ at each unit $j$:
    * if $j$ is an output unit: $\delta_j=h'(a_j)(y_j-z_j)$
    * if $j$ is a hidden unit: $\delta_j=h'(a_j)\sum_kw_{kj}\delta_k$ (recursion) where all units $k$ are the ones that receive input from unit $j$
    * update weights: $w_{ji} \leftarrow w_{ji} - \alpha \ \delta_j z_i$ 
 
## Kernel Methods

**Key idea**: use a large (possibly infinite) set of fixed, non-linear functions; normally, the computational complexity depends on the number of basis functions used, but by a "dual trick", complexity depends on the amount of data.

**Kernel function**: $k(x, x') = \phi(x)^T \phi(x')$

**Gram matrix**: $K = \Phi^T \Phi$

**Prediction for kernelized linear regression**: $y_n = k(\cdot, x_n)^T (K + \lambda I)^{-1}y$ where $k(\cdot, x_n) = k(x, x_n) \ \forall \ x$

**Constructing kernels $K$**:

* Valid kernels must be positive semi-definite (= all eigenvalues must be $\geq 0$); in other words, the $K$ must factor into a product of a transposed matrix by itself ($K = \Phi^T \Phi$)
* There are well-defined rules that can be applied to combine kernels with each other to create new kernels, preserving the property of positive semi-definiteness
* Kernels can be defined with respect to other things than vector such as sets, strings or graphs.

**Common kernels ($k(x, x') =$ ...)**:

* **Polynomial**: $(x^T x')^M$; feature space: all degree $M$ products of entries in $x$
* **General Polynomial**: $(x^T x' + c)^M$ with $c > 0$; feature space: all products of **up to** $M$ products of entries in $x$
* **Gaussian**: $\exp({-\frac{\mid x - x' \mid^2}{2 \sigma^2}})$; feature space: infinite! 
 
## Gaussian Processes

**Key idea**: approximating the true function $f(x)$ by an infinite-dimensional Gaussian distribution over functions $P(f)$

**Distribution over functions**: $f(x) \sim GP(m(x), k(x, x')) \ \forall x, x'$ where

* $m(x) = E(f(x))$ is the mean (= zero because the expectation of a zero-centered Gaussian is zero)
* $k(x, x') = E((f(x) - m(x)) (f(x') - m(x'))) = \frac{\phi(x)^T \phi(x')}{\alpha}$ is the kernel covariance function

**Gaussian Process Regression**: corresponds to kernelized Bayesian linear regression with a function view instead of a weight space view, posteriors over $f$ instead of $w$ and a complexity, cubic in the number of training points instead of cubic in the number of basis functions. Prediction:

* $P(y' \mid x', X, y) = N(\bar{f}(x'), k'(x', x'))$ where
* $\bar{f}(\cdot) = k(\cdot, X) \ A \ y$
* $k'(\cdot, \cdot) = k(\cdot, \cdot) - k(\cdot, X) \ A \ k(X, \cdot)$
* $A = (K + \sigma^2 I)^{-1}$ (the inversion of this step is cubic in the number of training points 
 
## Support Vector Machines

**Key idea**: find the decision hyperplane that maximizes the distances to the closest data point, resulting in a unique and globally optimal max-margin separator that can be found in polynomial time.

**Optimization problem**:

* $\max_w \frac{1}{\mid w \mid} \min_n y_n w^T \phi(x_n)$
* Alternatively, set the minimum distance to 1 and minimize $\mid w \mid$: $\min_w \frac{1}{2}{\mid w \mid}^2$ such that $y_n w^T \phi(x_n) \geq 1 \ \forall \ n$ (only the points where the distance is 1 are necessary to define the active constraints and are called **support vectors**)
* This optimization can be reformulated as a kernelized dual problem, given by: $\max_a \sum_n a_n - \frac{1}{2} \sum_n \sum_{n'} a_n a_{n'} y_n y_{n'} k(x_n, x_{n'})$ such that $\sum_n a_n y_n = 0$ and $a_n \geq 0 \ \forall \ n$ (many $a_n$ will be zero, they will be non-zero only for the support vectors)

**Prediction**:

* Primal problem: $y_* = sign(w^T \phi(x_*))$
* Dual problem: $y_* = sign(\sum_n a_n y_n k(x_n, x_*))$

**Soft margin**: for data that is not linearly separable, slack variables may be introduced into the optimzation problem to handle minor misclassifications: $\min_w C \sum_n \xi_n + \frac{1}{2}{\mid w \mid}^2$ such that $y_n w^T \phi(x_n) \geq 1 - \xi_n$ and $\xi_n \geq 0 \ \forall \ n$; slack variable $\xi_n$ will be $> 0$ for misclassified examples and $C$ controls the trade-off between the slack variable penalty and the margin; $C$ can also be interpreted as a regularization parameter; when $C \rightarrow \infty$ we arrive again at the hard margin problem; **support vectors** are all points that are in the margin or misclassified;

**Multiclass SMVs**:

* **One-Against-All**: for $K$ classes, train $K$ SMVs to distinguish each class from the rest; drawback: there will be regions that are either claimed by no class or by multiple classes
* **Pairwise Comparison**: train $O(K^2)$ SMVs to compare each pair of classes; drawbacks: computationally expensive and it is not obvious how the best class should be picked
* **Continuous Ranking**: single SVM that returns a continuous value to rank all classes (most popular approach today); idea: instead of computing the sign of a linear separator, compare the values of linear functions for each class $k$; classification: $y_* = \arg\max_k w_k^T \phi(x_*)$ 
 
## Hidden Markov Models

**Key idea**: make use of sequential correlations between classes

**Assumptions**:

* **Stationary process**: transition and emission distributions are identical at each time step: $P(x_t \mid y_t) = P(x_{t+1} \mid y_{t+1})$ and $P(y_t \mid y_{t-1}) = P(y_{t+1} \mid y_t) \ \forall \ t$
* **Markovian process**: next state is independent of previous states given the current state: $P(y_{t+1} \mid y_{t}, y_{t-1}, ..., y_{1}) = P(y_{t+1} \mid y_{t}) \ \forall \ t$

**Parametetrization**:

* **Initial state distribution**: $\pi = P(y_1)$
* **Transition distribution**: $\theta = P(y_{t} \mid y_{t - 1})$
* **Emission distribution**: $\phi = P(x_t \mid y_t)$

**Inference**:

* **Monitoring**: $P(y_t \mid x_{1..t})$; forward algorithm has linear complexity in $t$
* **Prediction**: $P(y_{t+k} \mid x_{1..t})$; forward algorithm has linear compelxity in $t + k$
* **Hindsight**: $P(y_{k} \mid x_{1..t})$ where $k < t$; forward-backward algorithm has linear complexity in $t$
* **Most likely explanation**: $\arg\max_{y_1,...y_t}P(y_{1..t} \mid x_{1..t})$; Viterbi algorithm has linear complexity in $t$

**Maximum likelihood objectives**:

* **Supervised**: $\pi',\theta',\phi' = \arg\max_{\pi,\theta,\phi} P(y_{1..T},x_{1..T} \mid \pi,\theta,\phi)$
* **Unsupervised**: $\pi',\theta',\phi' = \arg\max_{\pi,\theta,\phi} P(x_{1..T} \mid \pi,\theta,\phi)$ 
 
## Deep Neural Networks

**Definition**: neural network with many hidden layers, providing a high level of expressivity (as the number of layers is increased, the number of units needed may decrease exponentially; example: parity function)

**Problems**:

* **Gradient vanishing**: deep neural networks of sigmoid and hyperbolic units often suffer from vanishing gradients (because the derivative of these activation functions is always $< 1$); for this problem, two popular solutions exist:

  * **Pre-training** of weights
  * Other types of activation functions:

    * **Rectified Linear Units**:
      * Hard version: $g(x) = \max(0, x)$
      * Soft version ("Softplus"): $g(x) = \log(1 + e^x)$
    * **Maxout Units**: operate on vector-typed weights and output the maximum value of all the linear combinations of different subsets of input weights
    
* **Overfitting**: high expressivity increases the risk for overfitting; for this problem, three popular solutions exist:

  * **Regularization**
  * **Data augmentation** (to ensure that the number of parameters in the deep neural net is lower than the amount of data
  * **Dropout**: random "drop" some (input and hidden) units from the network when training; during prediction, multiply the output of each unit by one minus its dropout probability; dropout can be seen as a form of ensemble learning for neural networks 
 
## Convolutional Neural Networks

**Key idea**: learn convolutional features in sequential, spatial or tensor data (e.g., pixels from an input image) in a flexible way. An equivariant representation of the learned model ensures (partial) translation invariance and handling of variable-length inputs.

**Convolution**: in neural networks, a convolution denotes the linear combination of a subset of units based on a specific pattern of weights. Convolutions are often combined with activation functions to produce a feature value.

**Pooling**: commutative mathematical operation that combines several units (e.g., max, sum, product, Euclidean norm etc.)

**Convolutional Neural Network (CNN)**: any ANN that includes an alternation of convolution and pooling layers where some of the convolution weights are shared. CNNs are also trained using backpropagation, but gradients of shared weights are combined into a single gradient. 
 
## Recurrent and Recursive Neural Networks

**Key idea**: facilitate processing of variable-length data by instantiating recurrent or recursive patterns in the network.

**Recurrent Neural Network**: outputs can be fed back to the network as inputs, creating a recurrent structure that can be unrolled to handle variable-length data. Recurrent neural networks are trained by backpropagation on the unrolled network, combining gradients of shared weights into a single gradient. The **Encoder-Decoder** model is a model mapping input sequences to output sequences (e.g., for machine translation, question answering, dialog).

**Recursive Neural Network**: generalize recurrent neural networks from chains to trees. Parse trees or dependency graphs are used as structures for recursive neural networks. The **Long-Short-Term-Memory** (LSTM) model is a special gated structure to control memorization and forgetting in recursive neural networks, facilitating long-term memory. 
 
## Ensemble Learning

**Key idea**: combine several imperfect hypotheses into a better hypothesis

**Assumptions**:

* each hypothesis $h_i$ makes error with probability $p$
* the hypotheses are independent

**Probabilities of making errors** using majority vote of $n$ hypotheses under these assumptions*:

* $k \leq n$ hypotheses make an error: $p_e(k) = \binom{n}{k} p^k (1-p)^{n-k}$
* majority makes an error: $\sum_{k=\frac{1}{2}n}^n p_e(k)$

**Weighted majority**: as the above two assumptions are rarely true, hypotheses can be weighted lower if:

* they are correlated
* they have a lower classification performance

**Boosting**: technique to "boost" a weak learner by training multiple parametrizations of the same base model using weighted variations of the training set. One popular implementation is **AdaBoost**.

* **Boosting algorithm**:

  * set all instance weights $w_x$ to 1
  * repeat until sufficient number of hypotheses:
    * $h_i \leftarrow$ learn(dataset, weights)
    * increase $w_x$ of misclassified instances $x$
  * ensemble hypothesis is the weighted majority of all $h_i$ with weights $z_i$ proportional to the accuracy of $h_i$

* **Requirement**: a weak learner (i.e., a model producing hypotheses at least as good as random), e.g., rules of thumb, decision stumps (= decision trees of one node), perceptrons, naive bayes models

**Bagging** (= **b**ootstrap **agg**regation): technique to improve the accuracy of a weak learner by sampling a subset of both the samples and features to obtain independent classifiers; prediction is then done by a simple majority vote (without weighting). **Random forests** are bags of decision trees.

* **Bagging algorithm**:

  * repeat until sufficient number $K$ of hypotheses:
    * bootstrap sampling: $D_k \leftarrow$ sample data subset
    * random projection: $F_k \leftarrow$ sample feature subset
    * training: $h_k \leftarrow$ learn($D_k$, $F_k$)
  * ensemble hypothesis is the (non-weighted) majority of all $h_k$ for classification or the average output for regression


||Boosting &nbsp;  &nbsp;  &nbsp; |Bagging &nbsp;  &nbsp;  &nbsp; |
|---|---|---|
|**Sampling**|reweight instances in dataset|sample subset of data and features|
|**Classifiers obtained**|complementary|independent|
|**Prediction**|weighted majority vote|majority vote (no weighting)| 
 
## Stream Learning

**Key idea**: instead of training one hypothesis on a fixed dataset, continuously update a preliminary hypothesis as new data points arrive.

**Challenges**:

* since the data is streaming, it cannot all be stored $\rightarrow$ old data cannot necessarily be revisited
* time to process new incoming data must be constant and less than the arrival time for the next batch of data

**Solutions**:

* **Bayesian Learning**: lends itself naturally to stream learning because it stores information about training data in a product of likelihood distributions which can be updated incrementally every time a new data point comes in
* **Stochastic Gradient Descent** (SGD): for optimization-based learners like least square regression, logistic regression, maximum likelihood, support vector machines and neural networks, gradient descent can be implemented as an incremental approach where the model parameters are updated based on every incoming data point $(x_n, y_n)$ in isolation (as opposed to all existing data points at once), weighting the corresponding loss gradient by a data point specific learning rate $\alpha_n$.

  * **Robins-Monro conditions**: convergence is only guaranteed if the learning rate staisfies the Robins-Monro conditions of $\sum_{\alpha_n=1}^\infty \alpha_n = \infty$ and $\sum_{\alpha_n=1}^\infty \alpha_n^2 < \infty$; an example of such an update rule for learning rate $\alpha_n$ that satisfies these conditions is: $\alpha_n = \frac{1}{(\tau + n)^k}$ where $\tau \geq 0$ and $k \in (0.5, 1]$
  
  * **AdaGrad**: form of SDG that is often used in backpropagation. In AdaGrad, the model parameters $\theta$ are updated as follows: $\theta_m^{(n+1)} \leftarrow \theta_m^{(n)} - \frac{\alpha}{\tau + \sqrt{s_m^{(n)}}} \frac{\delta Loss(x_n; y_n; \theta^{(n)})}{\delta \theta_m^{(n)}}$ where $s_m^{(n)} \leftarrow s_m^{(n - 1)} + (\frac{\delta Loss(x_n; y_n; \theta^{(n)})}{\delta \theta_m^{(n)}})^2$ 
