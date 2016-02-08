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

**Admissibility**: a heuristic is admissible if $0 \leq h(n) \leq h'(n)$ where $h'(n)$ is the true shortest path from node $n$ to one of the goal states

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

For many problems, the search path is unimportant. Instead, oftentimes it is simply important to find a viable/good comnbinatorial solution without knowing the path to get there.

**Iterative Improvement**

* **Approach**:
  * Start at some random point
  * Generate all possible points to move to (i.e., the moveset)
  * If the set is empty, restart
  * If the set is not empty, choose point from it and move to it
* **Methods**:
  * **Hill Climbing / Gradient Descent**
    * **Idea**: always take a step in the direction that improves the current solution value the most
    * **Pros:**
      * straightforward implementation
      * low memory consumption
    * **Cons**:
      * not complete
      * not optimal (can get stuck in local optima/plateaus)
    * **Modifications**:
      * allow sideway moves to escape plateaus
      * random restarts to escape local optima
      * random selection of next move, but only take the step if it improves the solution
      * allow bad moves to escape local optima (see simulated annealing)
  * **Simulated Annealing**
    * **Idea**:
      * choose random move from moveset
      * if it improves the solution make the move
      * if not (bad move) take it anyways with probability $p$
      * $p=\frac{V(S_i)-V(S)}{T}$ (Boltzmann distribution)
      * $T$ is a temperature parameter which will decrease over time:
        * exploration phase when $T$ is high (random walk)
        * exploitation phase when $T$ is low (randomized hill climbing)
    * **Properties**:
      * optimal if $T$ decreases slowly enough

**Genetic Algorithms**

* **Idea**: simluation of natural evolutionary processes to approach a global optimum
* **Requirements**:
  * **Encoding representation** of individuals (normally a bitstring)
  * **Fitness function** to evaluate the quality of an individual
  * **Operations**:
    * **Selection**: selection of candidates for reproduction may be...
      * fitness-proportionate (can lead to overcrowding)
      * tournament-based (select two individuals at random and, with constant probability, choose the fitter one)
      * rank-based
      * softmax-based
    * **Crossover**
    * **Mutation** (normally done with a low probability)
* **Algorithm**:
  * Initialize population randomly
  * Compute fitness for each individual
  * $N$ times do:
    * Select two parents
    * Crossover the parents to create new child
    * With low probability, mutate child
    * Add child to population
  * Return "fittest" individual in population 
 
## Planning

**Purpose**: construct a sequence of actions for performing some task / reaching some goal

**Stanford Research Institute Problem Solver (STRIPS)** language:

* **Domain**: set of typed, concrete objects (no variables allowed)
* **States**: conjunctions of first-order predicates over objects (no variables allowed)
* **Goals**: conjunctions of **positive** ground literals (no negative ground literals allowed)
* **Closed-World Assumption**: any conditions not mentioned in a state are assumed to be false (see Frame Problem)
* **Actions**: tuples of preconditions (conjunction of function-free positive literals) and effects (description of how the state changes when the action is executed, sometimes defined as **delete- and add-lists**)

**Planning as Search**: planning is a specific type of search in which the search space is reduced significantly by the use of a highly structured and restriced planning language (e.g., Planning Domain Definition Language **PDDL**, a generalization of STRIPS):

* Progression Planning (Forward Planning): classical search which can strongly benefit from good heuristics
* Regression Planning (Backward Planning): **start from goal state** and find a sequence of **consistent** (i.e., must not undo any desired state), **relevant** (i.e., must achieve one of the conjuncts of the goal) actions

**Frame Problem**: when the consequences of an action are described the frame problem poses the question what has happened to components of the world that were not mentioned in this description

**Sussman's Anomaly**: stack-based regression planning might not work if a problem is decomposed into sub-problems that are interdependent

**Planning Graphs**: a form of representation of a planning problem

* **Levels**:
  * $S_0$ has a node for each literal that holds in the initial state
  * $A_0$ has a node for each action that could be taken in $S_0$
  * $S_i$ contains all literals that could hold given the actions taken in level $A_{i-1}$
  * $A_i$ contains all actions whose preconditions could hold in $S_i$
* **Persistence Actions (no-op)**: literal will persist until an action negates it
* **Mutual Exclusion (Mutex) links**: record conflicts between actions or states that cannot occur together for one of the following reasons:
  * Inconsistent Effects (actions)
  * Interference (actions)
  * Competing Needs (actions)
  * Inconsistent Support (states)
* **Heuristics**:
  * Level-cost: for a single goal literal, the level in which it appears first
  * Max-level: $argmax_i levelcost(g_i)$
  * Sum-level: $\sum_i levelcost(g_i)$ (may be inadmissible!)
  * Set-level: for multiple goal literals, the level where all appear and are not mutex (dominates max-level)
* **GraphPlan**:
  * Forward construction of the planning graph (in polynomial time)
  * Solution extraction (backward search through the graph, may be intractable because PSPACE-complete) 
 
## Adversarial Search

**Task environment**: multi-agent

**Types of Games**:

||Perfect Information|Imperfect Information|
|---|---|---|
|**Deterministic**|Chess|Other Card Games|
|**Stochastic**|Rolling the Dice|Poker|

**Zero-sum Perfect Information Games**:

* **Agents**:
  * MAX: aims to maximize the utility of the terminal node (i.e., win the game)
  * MIN: aims to minimize the utility of the terminal node (i.e., make MAX lose the game)
* **Goal**: finding an optimal strategy for MAX (i.e., a strategy that leads to outcomes at least as good for MAX as any other strategy, given that MIN is playing optimally)
* **Minimax**: a search algorithm to extract the optimal strategy
  * Complete if tree is finite
  * Time complexity: $O(b^m)$
  * Space complexity: $O(bm)$ (DFS)
* **Alpha-Beta Pruning**: elimination of large parts of the minimax search tree
  * $\alpha$: value of best choice (highest value) we have found so far on path for MAX
  * $\beta$: value of best choice (lowest value) we have found so far on path for MIN
  * Prune branches that are worse than $\alpha$ or $\beta$ for MAX and MIN respectively
* **Evaluation Functions**: compute expected utility for non-terminal states (and actual utility for terminal states) to allow for real-time decisions instead of going down the search tree for part of the search space

**Stochastic Games**:

* **Agents**:
  * MIN and MAX like above
  * CHANCE
* **Expectiminimax**:
  * CHANCE will compute the expected value
  * MIN will compute the minimum
  * MAX will compute the maximum 
 
## Decision Making

A decision problem under uncertainty is $<D,S,U,P>$ where:

* $D$ is a set of decisions
* $S$ is a set of states
* $U$ is a function that maps a real utility value to every state $\in S$ (unique up to a positive affine transformation)
* $P$ is a probability distribution which will tell how likely it is that decision $d$ will lead to state $s$

**Expected Utility**: $EU(d)=\sum_{s \in S}P_d(s)U(s)$

**Solution**: any $d' \in D$ such that $EU(d') \geq EU(d)$ for all $d \in D$

**Policies**: for a sequence of actions, a policy assigns an action decision to each state; policies may be obtained by bottom-up analysis of decision trees, incorporating a NATURE agent, representing probability distributions of outcomes for actions, taken in states 
 
## Markov Decision Processes 
 
## Reinforcement Learning 
