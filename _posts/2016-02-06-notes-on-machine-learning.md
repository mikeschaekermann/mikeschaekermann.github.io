---
layout: post
title: "Notes on Machine Learning"
tags:
    - python
    - notebook
---

{% include _toc.html %}

--- 
These notes are a result of my preparation for a midterm exam in [Pascal Poupart](https://cs.uwaterloo.ca/~ppoupart)'s [course on Machine Learning](https://cs.uwaterloo.ca/~ppoupart/teaching/cs485-winter16/) at University of Waterloo in the winter 2016 term. This post is currently a **work in progress (as of February 6, 2016)**. 
 
## Introduction

Definition by Tom Mitchell (1988): "A computer program is said to learn from **experience E** with respect to some class of **tasks T** and **performance measure P** if its performance for tasks in T, as measured by P, improves with experience E."

Inductive Learning: given a training set of examples of the form $(x, f(X))$, return a function $h$ (**hypothesis**) that approximates $f$ (**true underlying function**).

The quality measure for hypotheses is **generalization**. A good hypothesis will generalize well, i.e., predict unseen examples correctly. **Ockham's razor** demands to prefer the simplest hypothesis consistent with the input data.

Two different types of inductive learning:

* **Classification**: range/output space of $f$ is categorical (or discrete)
* **Regression**: range/output space of $f$ is continuous

**Hypothesis space $H$**: set of all possible $h$

**Consistency**: $h$ is consistent if it agrees with $f$ on all examples; it is not always possible to find a consistent $h$ (e.g., due to an insufficient $H$ or noisy input data)

**Realizability**: a learning problem is realizable if and only if $f$ is in $H$

In general, we will observe a tradeoff between **expressiveness** (i.e., the size of $H$) and the **complexity** of finding a good $h$.

**Overfitting**: given a hypothesis space $H$, a hypothesis $h \in H$ is said to overfit the training data if there exists some alternative hypothesis $h' \in H$ such that $h$ has smaller error than $h'$ over the training examples, but $h'$ has smaller error than $h$ over the entire distribution of instances.

**$k$-fold cross-validation:** you run $k$ experiments, each time putting aside $\frac{1}{k}$ of the data to test on and, finally, compute the average accuracy of the experiments 
 
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
|**Definition**|$Q_\tau=\\#\tau-max_k\\#k$|$Q_\tau=\sum_kp_\tau(k)(1-p_\tau(k))$|$Q_t=-\sum_kp_\tau(k) \ ld \ p_\tau(k)$|
|**Explanation**|Number of examples in leaf $\tau$ minus maximum number of examples in leaf $\tau$ that belong to any class $k$|Expected misclassification when choosing the class according to $p_\tau(k)$|Expected # of bits to encode the class of an instance chosen at random according to $p_\tau(k)$|

**Variables:**

* $\tau$: index for leaf $\tau$
* $k$: index for class $k$
* $\\#\tau$: number of examples in leaf $\tau$
* $\\#k$: number of examples in leaf $\tau$ belonging to class $k$
* $p_\tau(k) = \frac{\\#k}{\\#\tau}$

**Residual error for regression**:

* Let $t_n = f(x_n)$ be the target for the $n^th$ example.
* Let $y_t$ be the value returned by leaf $\tau$.
* Let $R_\tau$ be the set of examples in leaf $\tau$.
* Euclidean error for leaf $\tau$: $Q_\tau=\sum_{n \in R_\tau}(t_n-y_n)^2$

**Choosing the best attribute** for the next decision layer in the tree:

* In leaf $\tau$, choose attribute $A$ that reduces the residual error the most when expanded:
* $A^*=argmax_AQ_\tau-\sum_{a \in A}p_\tau(A=a)Q_{\tau a}$, where
  * $p_\tau(A=a)=\frac{\\#(A=a)}{\\#\tau}$ denotes the proportion of examples with value $a$ (in attribute $A$) inside node $\tau$
  * $\tau a$ indexes the node reached by following the edge for attribute value $a$, starting from node $\tau$
  
**Techniques to avoid overfitting:**

* Stop learning when the curve for testing accuracy, plotted against the tree size, goes down again; this is in practice sometimes hard to achieve because the curve might exhibit high fluctuation
* Pruning of statistically irrelevant nodes in a bottom-up fashion
  * Remove nodes that improve testing accuracy by less than some threshold
  * Regularization: add a penalty term that reflects the tree complexity (e.g., $\|T\|=$ #leaves in the tree) and remove leaves with a negative "regularized" error reduction: $Q_\tau-\sum_{a \in A}p_\tau(A=a)Q_{\tau a}-\lambda\|T\|$ 
 
## k-Nearest Neighbors 
 
## Linear Regression 
 
## Statistical Learning 
 
## Bayesian Learning 
 
## Mixture of Gaussians 
 
## Logistic Regression 
 
## Artificial Neural Networks 
