---
layout: post
title: "Notes on Artificial Intelligence"
tags:
    - python
    - notebook
---

{% include _toc.html %}

--- 
These notes are a result of my preparation for a midterm exam in [Kate Larson](https://cs.uwaterloo.ca/~klarson/)'s introductory [course on Artificial Intelligence](https://cs.uwaterloo.ca/~klarson/teaching/W16-486/) at University of Waterloo in the winter 2016 term. **This post is currently a work in progress (as of February 8th, 2016).** 
 
## Introduction

**Definition dimensions**:

||Reasoning|Behavior|
|---|---|---|
|**Human-like**|Thinking like a human|Acting like a human|
|**Rational**|Thinking rationally|Acting rationally (course focuses on this field)|

**Rational agents**:

* Agent: entity that perceives and acts
* Rationality: acting optimally towards a specific goal in a given environment
* Task environment: performance measure, environment, sensors and actuators

**Properties of the Task Environment**:

* Fully observable vs Partially observable
* Deterministic vs Stochastic
* Episodic vs Dynamic
* Discrete vs Continous
* Single agent vs Multi agent 
 
## Search

**Task environment**:

* Fully observable
* Deterministic
* Episodic
* Discrete
* Single Agent 
 
## Uninformed Search

**Techniques**:

||Depth First|Breadh First|Iterative Deepening|Uniform Cost|
|---|---|---|---|---|
|**acronym**|DFS|BFS|IDS|UCS|
|**dequeuing method**|LIFO|FIFO|alternating LIFO and FIFO|minimal backward cost of path|
|**complete?**|no|yes|yes|if all $\epsilon > 0$ and $C^*<\infty$|
|**optimal?**|no|for ident. cost|for ident. cost|yes|
|**time complexity**|$O(b^m)$|$O(b^d)$|$O(b^d)$|$O(b^{C^*/\epsilon})$|
|**space complexity**|$O(bm)$|$O(b^d)$|$O(bd)$|$O(b^{C^*/\epsilon})$|

**Variables**:

* $b$: branching factor of search tree
* $m$: maximum depth of search tree
* $d$: depth of shallowest goal node
* $C^*$: cheapest solution cost
* $\epsilon$: minimum edge cost

**Helper technique**:

If the state space graph is cyclic the search tree will be infinite. In this case, a “closed list” may be used to keep track of nodes which have already been expanded in order to avoid infinite traversals of cyclic structures 
 
## Informed Search

**Heuristic function**: function $h(n)$ that estimates the cost of reaching a goal from a given state (requirement: $h(n_{goal})=0$)

**Admissibility**: a heuristic is admissible if $0 \leq h(n) \leq h^*(n)$ where $h^*(n)$ is the true shortest path from node $n$ to one of the goal states

**Consistency**: a heuristic is consistent if $h(n) \leq cost(n,n')+h(n')$

**Backward cost**: function $g(n)$ that tells how expensive it was to reach node $n$ from the start node

**Estimate of cost of entire path**: function $f(n)=g(n)+h(n)$

**Greedy Best First Search**: expand the most promising node according to the heuristic only; only complete when used with a closed-list

**Techniques**:

||Greedy Best First|A*|Iterative Deepening A*|Simplified Memory-bounded A*|
|---|---|---|---|---|
|**acronym**|GBFS|A*|IDA*|SMA*|
|**dequeuing method**|minimal $h(n)$|minimal $f(n)$|minimal $f(n)$ with f-limit|see A*|
|**complete?**|no|yes|see A*|if memory needed for path to shallowest goal node $\leq$ memory size|
|**optimal?**|no|only for graph-search and a consistent $h(n)$|see A*|see above|
|**time complexity**|$O(b^m)$|$O(b^m)$|see A*|see A*|
|**space complexity**|$O(b^m)$|$O(b^m)$|less than A*|will drop nodes from memory if it runs out of memory| 
 
## Constraint Satisfaction Problems

A special subset of search problems where:

* **States** are defined by (continuous or discrete) **variables** $X_i$ with values from (finite or infinite) **domains** $D_i$
* **Goal test** is a set of (unary, binary, higher-order or soft) **constraints** specifying allowable combinations of values for subsets of variables
* **Commutativity** is in place, i.e., the order of actions taken does not effect the outcome; variables can be assigned in any order

**Formal abstraction** as a search problem:

* **States**: partial assignments of values to variables
* **Initial State**: empty assignment $\{\}$
* **Successor Function**: assign a value to an unassigned variable
* **Goal Test**: the current assignment is complete and satisfies all constraints

**Backtracking**:

* Select unassigned variable $X$ and try out first valid assignment $x_i$
* If a valid assignment is found move to next variable
* If no valid assignment is found back up and try a different assignment for $X$

**Improvements** to backtracking using:

* **Ordering**:
  * **Most Constrained Variable**: choose the variable which has the fewest legal moves
  * **Most Constraining Variable**: choose variable with most constraints on unassigned variables (tie-breaker for most constrained variable)
  * **Least Constraining Value**: given a variable, choose the value that rules out the fewest values in unassigned variables
* **Filtering**:
  * **Forward Checking**: keep track of remaining legal values for unassigned variables and terminate search if any variable has no legal values
  * **Arc Consistency**: given two domains $D_1$ and $D_2$, an arc is consistent if, for all $x$ in $D_1$, there is a $y$ in $D_2$ such that $x$ and $y$ are consistent
* **Structure**:
  * **Independent Subproblems**: break down constraint graph into connected components and solve them separately; can reduce time complexity from $O(d^n)$ to $O(d^c n/c)$ where $d$ is the domain size, $n$ is the total number of variables and $c$ is the average number of variables per component
  * **Tree Structures**: perform topological sort; back to front: make consistent from children to parents; front to back: assign values consistent with parent; time complexity is $O(nd^2)$
  * **Cutsets**: choose a subset $S$ of variables such that the constraint graph becomes a tree when $S$ is removed ($S$ is the cycle subset); for each possible valid assignment to the variables of $S$: remove from the domains of remaining variables all values that are inconsistent with $S$; if the remaining CSP has a solution, return it; time complexity is $O(d^c(n-c)d^2)$ where $c$ is the size of the cutset
  * **Tree Decomposition**: decompose graph into subproblems that constitute a tree structure; solve each subproblem independently; solve constraints connecting the subproblems using the tree-based algorithm; time complexity is $O(nd^w)$ where $w$ is the size of the largest subproblem 
 
## Local Search 
 
## Planning 
 
## Adversarial Search 
 
## Decision Making 
 
## Markov Decision Processes 
 
## Reinforcement Learning 
