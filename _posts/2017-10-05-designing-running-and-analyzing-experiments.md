---
layout: post
title: "Notes on Designing, Running and Analyzing Experiments"
tags:
    - python
    - notebook
---

{% include _toc.html %}

These notes are a result of taking the online course [Designing, Running and Analyzing Experiments](https://www.coursera.org/learn/designexperiments) taught by Scott Klemmer and Jacob Wobbrock. The contents are therefore based on the corresponding presentations available online.

## Basic Experiment Design Concepts

**Participants**:

* **Sampling**:
  * Probability Sampling (uses random approaches)
  * Non-probability Sampling (purposive, convenience, snowball)

* **Criteria**:
  * Inclusion Criteria
  * Exclusion Criteria

**Apparatus**:

* **Environment**:
  * Lab Study
  * Online Study
  * Remote Study

* **Data Capturing**:
  * Log Files
  * Video Recording
  * Personal Observations

**Procedure**:

* **Trials**:
  * Number
  * Duration

* **Temporal Effects**:
  * Fatigueness
  * Learning

* **Tasks**:
  * Open Exploration
  * Task

**Design & Analysis**:

* Formal Design Characteristics
* Appropriate Statistical Analysis

## Tests of Proportions

**Types of Tests**:

* **Exact**: computes exact p-value
* **Aymptotic**: approximates p-value


**Reporting p-values (referred to as {P VALUE REPORT} below)**:

* **If statistically significant**:
  * $p < .05$
  * $p < .01$
  * $p < .001$
  * $p < .0001$
* **If on the edge of significance (i.e., $.05 < p < .1$)**:
  * Report as a "trend"
* **If not statistically significant**:
  * *n.s.* (do not treat statistically non-significant differences as there being no difference, but rather as there being no detectable difference based on the observed data)

**One-sample Test of Proportions**:

* **Pearson $\chi^2$ Test** (asymptotic test):
  * R call: `chisq.test(table)`
  * R output: `X-squared = {TEST STATISTIC}, df = {DEGREES OF FREEDOM}, p-value = {P VALUE}`
  * Report as: $\chi^2(\text{\{DEGREES OF FREEDOM\}}, N=\text{\{SAMPLE SIZE\}}) = \text{\{TEST STATISTIC\}}, \text{\{P VALUE REPORT\}}$

* **Binomial Test** (exact test):
  * R call: `binom.test(table)`
  * R output: `number of successes = {NUM. SUCCESSES}, number of trials = {SAMPLE SIZE}, p-value = {P VALUE}`
  * Report as: We had a binomial test with {SAMPLE SIZE} data points, {P VALUE REPORT}

* **Multinomial Test** (exact test for more than two response categories, should be followed by a series of post-hoc binomial tests and a Bonferroni adjustment of p-values to determine significance for individual response categories):
  * R call: `library(XNomial); xmulti(table, c(1/3, 1/3, 1/3), statName="Prob")` (this notation is correct for three response categories; adjust the second argument for more than three response categories)
  * R output: `P value (Prob) = {P VALUE}`
  * Report as: We had a multinomial test with {SAMPLE SIZE} data points, {P VALUE REPORT}


**N-sample Test of Proportions**:

* **N-sample Pearson $\chi^2$ Test** (asymptotic test, see above)
* **G Test** (asymptotic test):
  * R call: `library(RVAideMemoire); G.test(table)`
  * R output: `G = {TEST STATISTIC}, df = {DEGREES OF FREEDOM}, p-value = {P VALUE}`
* **Fisher's Test** (exact test):
  * R call: `fisher.test(table)`
  * R output: `p-value = {P VALUE}`


## The T-Test

**Variable Types**:

* **Independent Variables**: The variables the experimenter manipulates, also called the *treatments*, or *factors* (with different levels, i.e., the specific values a factor can take on)
  * **Between-Subjects Factor**: Each participant experiences only one level of a factor
  * **Within-Subjects Factor**: Each participant experiences more than one level of a factor (partial within-subjects factors expose participants to more than one, but not all levels of the factor)
* **Dependent Variables**: The variables that are potentially influenced by the independent variables
* **Notation in R**: $Y \sim X + \epsilon$ ($Y$: dependent variable; $X$: independent variable; $\epsilon$: random measurement error)

**Design Types**:

* **Balanced vs. Unbalanced**: depending on whether there are about the same number of participants in every condition
* **N-Measure**: indicating how many data points we measure from each participant

**Independent-samples t-test / Two-samples t-test** (parametric form of ANOVA, appropriate for between-subjects factors with two levels)

* R call: `t.test(Y ~ X, data=dataframe, var.equal=TRUE)` (use `var.equal=FALSE` for the Welch t-test for unequal variances, e.g., when the homoscedasticity assumption for ANOVAs is violated, see below)
* R output: `t = {TEST STATISTIC}, df = {DEGREES OF FREEDOM}, p-value = {P VALUE}`
* Report as: $t(\text{\{DEGREES OF FREEDOM\}}) = \text{\{TEST STATISTIC\}}, \text{\{P VALUE REPORT\}}$


## Validity in Design and Analysis

**Experimental Control**:

* **Goal**: ensuring that systematic differences in observed responses can be attributed to systematic changes in manipulated factors
* **Trade-off between experimental control and ecological validity**
* **Confounds**: non-random effects that introduce uncontrolled variation in the experiment; strategies to mitigate the effects of confounds:
  1. **Manipulate it** (by turning confounds into independent variables that are manipulated systematically)
  2. **Control for it** (keep the confound constant or evenly spread out across all participants)
  3. **Measure it** (record it to control for it in the subsequent analysis)
  4. **Consider it a *Hidden Effect* otherwise**

**Types of Analyses**

* **Parametric**: does make assumptions about the distribution of the response variable to gain statistical power
* **Non-Parametric**: does not make assumptions about the distribution of the response variable lacking statistical power (typically operate on ranks)

**Data Distributions**:

* **Continuous**:
  * **Normal / Gaussian**: $\mu$ (mean); $\delta^2$ (variance); applies to most response variables
  * **Log Normal**: $\mu$ (mean); $\delta^2$ (variance); e.g., task time
  * **Gamma**: $k$ (shape); $\Theta$ (scale); e.g., waiting times in lines
  * **Exponential**: $\lambda$ (rate); e.g., people's wealth; special case of the Gamma distribution when shape $k=1$; $\lambda = \Theta^{-1}$
* **Discrete**:
  * **Poisson**: $\lambda$; e.g., counts of rare events
  * **Binomial / Multinomial**: $n$ (number of trials); $p$ (probabilities of success for individual outcomes, only one scalar for the binomial distribution, and a vector for the multinomial distribution); for categorical response variables

**3 Assumptions of Analysis of Variance (ANOVA, parametric test)**:

* **Independence**: each participant is sampled independently from other participants (violated in snowball sampling); also, each measure on a given participant is independent from measures on other subjects
* **Normality**: the residuals (i.e., the differences between the observed response variable and the statistical model's predictions) are normally distributed (i.e., follow the Gaussian bell curve); use Shapiro-Wilk normality test on the residuals (`shapiro.test(residuals(model))` --> `W = {TEST STATISTIC}, p-value = {P VALUE}`, must *not* be significant in order to comply with the normality assumption) and visualize with a QQ-plot (`qqnorm(residuals(model)); qqline(residuals(model))` --> all points should be close to the diagonal line); if this assumption is violated, try to transform the data in a way that the data, and thus likely also the residuals, are normally distributed; for example, test for log-normality of the data using the Kolmogorov-Smirnov test (`library(MASS); fit = fitdistr(data, "lognormal")$estimate;` `ks.test(data, "plnorm", meanlog=fit[1], sdlog=fit[2], exact=TRUE)` --> `D = {TEST STATISTIC}, p-value = {P VALUE}`, must *not* be significant in order to assume a log-normal distribution); if the data is follows a log-normal distribution, apply a log transform to it before performing the ANOVA
* **Homoscedasticity / Homogeneity of Variance**: the variance among groups being compared is similar; use Levene's test (`leveneTest(Y ~ X, data=data, center=mean)` --> `Df {DEGREES OF FREEDOM} F value {TEST STATISTIC} Pr(>F) {P VALUE}`, must *not* be significant in order to comply with the homoscedasticity assumption) and Brown-Forsythe test (`leveneTest(Y ~ X, data=data, center=median)` --> `Df {DEGREES OF FREEDOM} F value {TEST STATISTIC} Pr(>F) {P VALUE}`, must *not* be significant in order to comply with the homoscedasticity assumption; preferred as it uses the median and is more robust to outliers) to test for this assumption; if this assumption is violated (even after a potential log transform of the data), use the Welch t-test for unequal variances

**Mann-Whitney U test** (non-parametric form of ANOVA, appropriate for between-subjects factors with two levels, i.e., the non-parametric equivalent of the independent-samples t-test):

* R call: `library(coin); wilcox_test(Y ~ X, data=data, distribution="exact")`
* R output: `Z = {TEST STATISTIC}, p-value = {P VALUE}`
* Report as: $Z = \text{\{TEST STATISTIC\}}, \text{\{P VALUE REPORT\}}$