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
 
## K-Nearest Neighbors

In the limit, single-attribute thresholding to construct decision trees for attributes with continouous inputs will lead to a full tree with one example per tree. Decision boundaries will always axis-aligned. A better approach without this restriction is $K$-Nearest Neighbors.

**Approach:**

* Let $knn(x)$ be the $K$ nearest neighbors of $x$ according to distance $d$
* Label $y_x=mode(\{y_x' \ \| \ x' \in knn(x)\})$, i.e., the most frequent label among the $K$ nearest neighbors
* $K$ controls the degree of smoothing and can be optimized using $k$-fold cross-validation (if $K$ is too small this will lead to overfitting; if $K$ is too high this will lead to underfitting)

**Comparison of complexity** between decision trees and $K$-Nearest Neighbors with respect to:

* $N$: size of training set
* $M$: number of attributes

||Training &nbsp;  &nbsp;  &nbsp; |Testing &nbsp;  &nbsp;  &nbsp; |
|---|---|---|
|**Decision Tree**|$O(M^2N)$|$O(M)$|
|**$K$-Nearest Neighbors**|$O(MN)$|$O(MN)$|

A good visualization of the decision boundaries for the two-dimensional 1-Nearest Neighbor case assuming Euclidean distance is the Voronoi diagram: 

**In [5]:**

{% highlight python linenos %}
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all([v >= 0 for v in vertices]):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

# make up data points
np.random.seed(1234)
points = np.random.rand(15, 2)

# compute Voronoi tesselation
vor = Voronoi(points)

# plot
regions, vertices = voronoi_finite_polygons_2d(vor)

# colorize
for region in regions:
    polygon = vertices[region]
    plt.fill(*zip(*polygon), alpha=0.4)

plt.plot(points[:,0], points[:,1], 'ko')
plt.axis('equal')
plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)
{% endhighlight %}




    (-0.086231550409317764, 0.98264119063611655)



 
![png]({{ site.url }}/images/2016-02-06-notes-on-machine-learning_4_1.png) 

 
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

**Approximatinos of Bayesian Learning**:

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

* bias increases with regularization parameter $\lambda$
* variance decreases with regularization parameter $\lambda$
* noise is constant 
 
## Mixture of Gaussians 
 
## Logistic Regression 
 
## Artificial Neural Networks 
