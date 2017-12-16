---
layout: post
title: "Notes on Designing, Running and Analyzing Experiments"
tags:
    - python
    - notebook
---

<style>
  table {
    border: 1px solid #000000;
    text-align: center;
  }

  td {
    border-right: 1px solid #000000;
  }

  tbody tr {
    border-top: 1px solid #dddddd;
  }

  thead tr {
    border-bottom: 1px solid #000000;
  }
</style>

{% include _toc.html %}

These notes are a result of taking the online course [Designing, Running and Analyzing Experiments](https://www.coursera.org/learn/designexperiments) taught by [Scott Klemmer](https://d.ucsd.edu/srk/) and [Jacob Wobbrock](https://faculty.washington.edu/wobbrock/). The contents are therefore based on the corresponding presentations available online.

## Tests Cheatsheet

**Tests of Proportions**:

| Samples | Response Categories | Asymptotic Tests         | Exact Tests           |
|---------|---------------------|--------------------------|-----------------------|
| 1       | 2                   | One-sample $\chi^2$ test | Binomial test         |
| 1       | > 2                 | One-sample $\chi^2$ test | Multinomial test      |
| > 1     | >= 2                | N-sample $\chi^2$ test   | G-test; Fisher's test |


**Analyses of Variance**:

| Factors | Levels | (B)etween or (W)ithin | Parametric Tests                                             | Non-Parametric Tests                                                 |
|---------|--------|-----------------------|--------------------------------------------------------------|----------------------------------------------------------------------|
| 1       | 2      | B                     | Independent-samples T-test                                   | Mann-Whitney U Test                                                  |
| 1       | > 2    | B                     | One-way ANOVA                                                | Kruskal-Wallis Test                                                  |
| 1       | 2      | W                     | Paired-samples t-test                                        | Wilcoxon signed-rank test                                            |
| 1       | > 2    | W                     | One-way repeated measures ANOVA                              | Friedman test                                                        |
| > 1     | >= 2   | B                     | Factorial ANOVA; Linear Models (LM)                          | Aligned Rank Transform (ART); Generalized Linear Models (GLM)        |
| > 1     | >= 2   | W                     | Factorial repeated measures ANOVA; Linear Mixed Models (LMM) | Aligned Rank Transform (ART); Generalized Linear Mixed Models (GLMM) |


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
    * **Pros**: Avoids **carryover effects** (see below)
    * **Cons**: More participants needed; higher subject-dependent variance in the response variables
  * **Within-Subjects Factor**: Each participant experiences more than one level of a factor (partial within-subjects factors expose participants to more than one, but not all levels of the factor); also called **Repeated Measures** factor:
    * **Pros**: Less participants needed; lower subject-dependent variance in the response variables
    * **Cons**: Is prone to **carryover effects** (like fatigue, practice effects, boredom, skill transfer etc.); carryover effects can be accounted for by controlling (e.g., randomizing or rotating) and logging the order in which individual participants are exposed to the different levels of a factor
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


## One-Factor Between-Subjects Experiments

**One-Way ANOVA** (parametric; for experiments with a single between-subjects factor of more than two levels):

* **F-test** (overall / omnibus test):

  * R call: `m = aov(Y ~ X, data=data); anova(m)`
  * R output: `Df: {NUM. DOF} Sum Sq: ... Mean Sq: ... F value: {TEST STATISTIC} Pr(>F): {P VALUE} Residuals: {DENOM. DOF}`
  * Report as: $F(\text{\{NUM. DOF\}}, \text{\{DENOM. DOF\}}) = \text{\{TEST STATISTIC\}}, \text{\{P VALUE REPORT\}}$
  
* **Post-hoc pairwise comparisons** (using independent samples t-tests):
  
  * R call: `library(multcomp); summary(glht(m, mcp(IDE="Tukey")), test=adjusted(type="holm"))`

**Non-parametric Equivalent**:

* **Kruskal-Wallis Test** (overall / omnibus test):

  * R call: `library(coin); kruskal_test(Y ~ X, data=data, distribution="asymptotic")`

* **Post-hoc pairwise comparisons** (using either multiple Mann-Whitney U tests [see above] or one combined test by Cover and Iman [see R call below]):

  * R call: `library(PMCMR); posthoc.kruskal.conover.test(Y ~ X, data=data, p.adjust.method="holm")`


## One-Factor Within-Subjects Experiments

**Counterbalancing Repeated Measures Factors** (how to assign order of presentation of factor levels to avoid carryover effects):

* **Full Counterbalancing**: every possible order is represented equally in the study; preferred method if the participant sample is large enough to represent each order equally often; the number of possible orders is the factorial of the number of factor levels; the participant should be a multiple of the number of possible orders;

* **Latin Square**: each factor level appears in each order position equally often; this is done by rotating a fixed order of factor levels; the participant sample should be a multiple of the number of factor levels;
  * 1, 2, 3, 4, 5
  * 2, 3, 4, 5, 1
  * 3, 4, 5, 1, 2
  * 4, 5, 1, 2, 3
  * 5, 1, 2, 3, 4

* **Balanced Latin Square**: first row (1, 2, n, 3, n-1, 4, n-2, ...); subsequent (n-1) rows increment the values from each preceding row and wrap around $(n_p + 1 \mod n)$; if n, i.e., the number of factor levels, is odd, repeat the block in reverse order); below is an example for an odd number of factor levels:
  * **Block 1** (forward order):
    * 1, 2, 5, 3, 4
    * 2, 3, 1, 4, 5
    * 3, 4, 2, 5, 1
    * 4, 5, 3, 1, 2
    * 4, 1, 4, 2, 3
  * **Block 2** (reverse order; only needed if n is odd):
    * 4, 3, 5, 2, 1
    * 5, 4, 1, 3, 2
    * 1, 5, 2, 4, 3
    * 2, 1, 3, 5, 4
    * 3, 2, 4, 1, 4


**Paired t-test** (parametric form of ANOVA, appropriate for within-subjects factors with two levels)

* R call: `t.test(Y ~ X, data=dataframe, paired=TRUE, var.equal=TRUE)` (use `var.equal=FALSE` for the Welch t-test for unequal variances, e.g., when the homoscedasticity assumption for ANOVAs is violated)
* R output: `t = {TEST STATISTIC}, df = {DEGREES OF FREEDOM}, p-value = {P VALUE}`
* Report as: $t(\text{\{DEGREES OF FREEDOM\}}) = \text{\{TEST STATISTIC\}}, \text{\{P VALUE REPORT\}}$


**Wilcoxon Signed-Rank Test** (nonparametric equivalent of paired t-test):

* R call: `library(coin); wilcoxsign_test(Y ~ X | Subject, data=dataframe, distribution="exact")`
* R output: `Z = {TEST STATISTIC}, p-value = {P VALUE}`
* Report as: $Z = \text{\{TEST STATISTIC\}}, \text{\{P VALUE REPORT\}}$


**One-way Repeated Measures ANOVA** (parametric form of ANOVA, appropriate for within-subjects factors with more than two levels)

* R call: `library(ez); m = ezANOVA(dv=Y, within=X, wid=Subject, data=dataframe); m$Mauchly; m$ANOVA;`
`pos = match(m$``Sphericity Corrections``$Effect, m$ANOVA$Effect)`
`m$Sphericity$GGe.DFn = m$Sphericity$GGe * m$ANOVA$DFn[pos] # Greenhouse-Geisser`
`m$Sphericity$GGe.DFd = m$Sphericity$GGe * m$ANOVA$DFd[pos]`
`m$Sphericity$HFe.DFn = m$Sphericity$HFe * m$ANOVA$DFn[pos] # Huynh-Feldt`
`m$Sphericity$HFe.DFd = m$Sphericity$HFe * m$ANOVA$DFd[pos]`
`m$Sphericity`
* Followed by post-hoc pairwise comparisons using paired t-tests


**Friedman's Test** (non-parametric equivalent of one-way repeated measures ANOVA):

* R call: `library(coin); friedman_test(Y ~ X | Subject, data=dataframe, distribution="asymptotic")`
* R output: `chi-squared = {TEST STATISTIC}, df = {DEGREES OF FREEDOM}, p-value = {P VALUE}`
* Report as: $\chi^2(\text{\{DEGREES OF FREEDOM\}}) = \text{\{TEST STATISTIC\}}, \text{\{P VALUE REPORT\}}$
* Followed by post-hoc pairwise comparisons using Wilcoxon signed-rank tests


## Multi-Factor Experiments

**NxM mixed / within-subjects / between-subjects factorial designs**:

* **N**: number of levels in the first factor
* **M**: number of levels in the second factor (there can be more than two factors)
* **mixed**: means that some factors are within-subjects factors and some factors are between-subjects factors; if the factorial design contains any within-subjects factors, always use Repeated Measures ANOVA, not regular ANOVA
* **within-subjects**: means that all factors are within-subjects factors
* **between-subjects**: means that all factors are between-subjects factors

**Effects**:

* **Main effect**: means that changing levels within one factor leads to significant differences in the dependent variable
* **Interaction effect**: means that changing levels in one factor differentially affects outcomes in the dependent variable for different levels of another factor

**Mixed Factorial ANOVA**:

* **R code**: `library(ez); ezANOVA(dv={DEPENDENT VARIABLE}, between={BETWEEN-SUBJECTS FACTOR}, within={WITHIN-SUBJECTS FACTOR}, wid={SUBJECT COLUMN NAME}, data=data)`

**Aligned Rank Transform (ART) Procedure** (non-parametric equivalent to Mixed Factorial ANOVA):

* **R code**: `library(ARTool); m = art({DEPENDENT VARIABLE} ~ {BETWEEN-SUBJECTS FACTOR} * {WITHIN-SUBJECTS FACTOR} + (1|{SUBJECT COLUMN NAME}), data=data); anova(m)`

