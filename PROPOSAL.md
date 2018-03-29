# Project proposal 

### Team

- Melanie Tosik (mt3685), `tosik@nyu.edu`
- Antonio Mallia, `me@antoniomallia.it`
- Kedar Gangopadhyay, `kedarg@nyu.edu`

## Problem description

The Fake News Challenge Stage 1 (the "FNC-1") was a competition in 2017 to advance the application of AI to help solve the fake news problem.  Fifty competing teams developed machine learning systems for fake news detection. Earlier this year, the datasets were released to the public. Our team's objective is to utilize the datasets and attempt to beat the official
baseline by using traditional NLP features and classifier.

FNC-1 concerns the first step in detecting fake news-- stance detection. The idea is that automating a stance detection process could serve as the crucial first step in a complementary AI-assisted fact-checking pipeline. Stance detection commonly involves estimating the relative perspective (i.e., stance) of two pieces of text relative to a topic. However, the FNC-1 _specifically_ extends the stance detection work of Ferreira & Vlachos (see Related work). In FNC-1, the task is to estimate the stance of a body text from a news article relative to the headline and categorize it into one of four categories: agree, disagree, discusses, or unrelated.

<p align="center"><img src="https://github.com/amallia/FakeNewsChallenge/blob/master/report/images/fnc-eval.png" ></p>


The top scorers of FNC-1 implemented deep learning methods for stance detection by analyzing and optimizing neural-based methods. This was not surprising as this family of techniques gained popularity in 2017, around the time of the competition. Top scorers fared very well with "unrelated" categories, but performed poorly with respect to accuracy on "agree" and "disagree" stances (see Dataset). Our team's objective is to explore the possibility of improving accuracies in the "agree" and "disagree" stances. To achieve this, our plan is to build a fairly simple system using tradition NLP features and classifier (see Methodology). The team's success will be measured against the official baseline accuracy of 79.53% weighted accuracy (see Evaluation plan).

## Dataset
The distribution of the output labels is overwhelmingly "unrelated" to the headline. The distribution of the stances are outlined below:

|   rows  |   unrelated |   discuss |     agree |   disagree |
|--------:|------------:|----------:|----------:|-----------:|
|  49,972 |    0.73131  |  0.17828  | 0.0736012 |  0.0168094 |

Our team's objective is two-fold: 1) determine whether the heading/content is related/unrelated (syntax/semantics); and 2) as mentioned previously, determine whether the heading/content discusses/agree/disagrees (sentiment). Moreover, we will test whether traditional NLP features can improve upon the low accuracies in the "agree" and "disagree" stances of previous participants.

_Sources_:
1) http://www.fakenewschallenge.org/
2) https://github.com/FakeNewsChallenge/fnc-1

## Methodology

Feature engineering, including:

- possibly use semantic role labeling to generate triples of (agent (subject), verb, patient (object))
- dependency relations/syntactic features
- vector features
- TF-IDF
- LSA/SVD
- ??? 

- Python/spaCy/gensim

## Related works
As part of our initial background research, we relied upon two previous related works:

1)  William Ferreira, Andreas Vlachos. Emergent: a novel data-set for stance classification. http://aclweb.org/anthology/N/N16/N16-1138.pdf, June 2016.

2)  Benjamin Riedel, Isabelle Augenstein, Georgios P. Spithourakis, Sebastian Riedel. A simple but tough-to-beat baseline for the Fake News Challenge stance detection task. https://arxiv.org/abs/1707.03264, July 2017.

## Evaluation plan

### Baseline

FNC-1 provides a [baseline model](https://github.com/FakeNewsChallenge/fnc-1-baseline) consisting of hand-engineered features, including n-gram co-occurrence counts between the headline and article and indicator features for polarity and refutation, to train a gradient-boosting classifier. The baseline accuracy is 79.53%.

The baseline implementation performs well separating related stances from the rest, but rather poorly when differentiating between agree, disagree, and discuss. Our goal is to improve upon this area of weakness while still achieving the baseline.
