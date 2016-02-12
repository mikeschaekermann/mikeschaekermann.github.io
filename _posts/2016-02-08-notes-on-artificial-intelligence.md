---
layout: post
title: "Notes on Artificial Intelligence"
tags:
    - python
    - notebook
---

{% include _toc.html %}

--- 
These notes are a result of my preparation for a midterm exam in [Kate Larson](https://cs.uwaterloo.ca/~klarson/)'s introductory [course on Artificial Intelligence](https://cs.uwaterloo.ca/~klarson/teaching/W16-486/) at University of Waterloo in the winter 2016 term. The contents are therefore based on the corresponding presentation slides available online. 
 
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
* Discrete vs Continuous
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
  * **Arc Consistency**: given two domains $D_1$ and $D_2$, an arc from $D_1$ to $D_2$ is consistent if, for all $x$ in $D_1$, there is a $y$ in $D_2$ such that $x$ and $y$ are consistent
* **Structure**:
  * **Independent Subproblems**: break down constraint graph into connected components and solve them separately; can reduce time complexity from $O(d^n)$ to $O(d^c n/c)$ where $d$ is the domain size, $n$ is the total number of variables and $c$ is the average number of variables per component
  * **Tree Structures**: perform topological sort; back to front: make mutually consistent between children and parents; front to back: assign values consistent with parent; time complexity is $O(nd^2)$
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
      * $p=e^\frac{V(S_i)-V(S)}{T}$ (Boltzmann distribution)
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
  * Set-level: for multiple goal literals, the first level where all appear and are not mutex (dominates max-level)
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

**Markov Chain**:

* A set of probability distributions of the next state given the current state (may be represented as a **transition probability matrix**)
* **History Independence (Markov Property)**: the probability of reaching state $s_{t+1}$ from state $s_t$ does not depend on how the agent got to the current state $s_t$
* **Discounted sum of future rewards** $U'(s)$ of state $s$: is the sum of the reward for state $s$ and of all future rewards that can be reached from state $s$ where the utility of each future state $x$ which is $n$ steps away will be discounted by a factor of $\gamma^n$, $y$ being a constant discount factor with $0 < \gamma < 1$:
  * $U'(s_i)=r_i+\gamma\sum_{j=1}^nP_{ij}U'(s_j)$
  * $U=(I-\gamma P)^{-1}R$ ($P$ being the transition probability matrix and $R$ being the rewards vector)
  * This system may be solved directly by **matrix inversion** or, if this is too costly, approximated by **Value Iteration**:
    * Compute $U^n(s)$ values for each state $s$ and step length $n$ (starting with $n=1$)
    * Use dynamic programming by computing $U^n(s)$ by the previously computed and stored values of $U^{n-1}(s)$

**Markov Decision Process (MDP)**: similar to a Markov Chain, but incorporating the notion of actions. In every state $s_i$, the agent may decide to take an action $a_k$ which may lead to state $s_j$ with probability $P(s_j \mid s_i,a_k)$

* **Expected discounted sum of future rewards** assuming the optimal policy and a step length of $t$, starting from state $s_i$, $V^t(s_i)$:
  * $V^{t+1}(s_i)=max_k r_i+\gamma\sum_{j=1}^nP_{ij}^kV^t(s_j)$
  * $V^*(s_i)$ is $V^t(s_i)$ with $t=\infty$
* **Policy Optimization**: for every MDP, there is an optimal policy (i.e., a mapping from state to action) such that for every possible start state, there is no better option than to follow the policy; it can be found in polynomial time (in the number of states) by:
  * **Value Iteration**: iteratively compute $V^*(s_i)$ for all $s_i$ and select the best action $k$ according to $argmax_k r_i+\gamma\sum_{j=1}^nP_{ij}^kV^t(s_j)$
  * **Policy Iteration**:
    * **Policy Evaluation**: given policy $\pi$, compute $V_i^\pi$ for all states $s_i$
    * **Policy Improvement**: calculate a new policy $\pi_{i+1}$ using 1-step lookahead
    * Repeat both steps until $V^\pi(s_i)$ converges
    
**Partially Observable MDP (POMDP)**: in a POMDP, the agent does not know for sure what state it is in; therefore, it also stores a set of observations $O=\{o_1,...,o_k\}$, an observation model $P(o_t \mid s_t)$ and a belief state $b$ which is a probability distribution over all possible states; $b(s)$ is the probability assigned to state $s$; here, a policy is a mapping from a belief state to an action; generally, finding an approximately optimal policy is PSPACE-hard 
 
## Reinforcement Learning

**Task Environment**:

* Fully observable
* Stochastic
* Dynamic
* Discrete
* Single Agent

**Characteristics**:

* the agent learns a policy to act with the aim maximize the resulting reinforcement signals (numerical reward)
* the reinforcement signals may be delayed (credit assignment problem)
* the goal is to find the optimal policy, but we start without knowing the underlying Markov Decision Process (MDP), i.e., the rewards and transition probabilities are not known
* formally, we can describe this as the following problem: learn policy $\pi:S \mapsto A$ that maximizes $E[r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+...]$ from any starting state $\in S$

**Forms of Reinforcement Learning**:

||Passive (evaluate a given policy)|Active (learning to act optimally)|
|---|---|---|
|**Model-based**|Adaptive Dynamic Programming (ADP): evaluate a given policy, based on observations after running the policy a number of times|$\{\}$|
|**Model-free**|Temporal Difference: use observed transitions to adjust values of observed states so that they satisfy Bellman equations|Q-Learning|

**Reward Shaping**: consider delays in rewards and add additional rewards for "making progress" using domain knowledge about important steps for reaching the final reward; this bears the risk of the agent optimizing for the pseudo rewards 
