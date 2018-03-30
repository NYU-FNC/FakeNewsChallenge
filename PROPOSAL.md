# Project proposal 

### Team

- Melanie Tosik (mt3685), `tosik@nyu.edu`
- Antonio Mallia, `me@antoniomallia.it`
- Kedar Gangopadhyay, `kedarg@nyu.edu`

## Problem description

The Fake News Challenge[1] Stage 1 (the "FNC-1") was a competition in 2017 to advance the application of AI to help solve the fake news problem.  Fifty competing teams developed machine learning systems for fake news detection. Earlier this year, the datasets were released to the public. Our team's objective is to utilize the datasets and attempt to beat the official
baseline by using traditional NLP features and classifier.

FNC-1 concerns the first step in detecting fake news-- stance detection. The idea is that automating a stance detection process could serve as the crucial first step in a complementary AI-assisted fact-checking pipeline. Stance detection commonly involves estimating the relative perspective (i.e., stance) of two pieces of text relative to a topic. However, the FNC-1 _specifically_ extends the stance detection work of Ferreira & Vlachos (see Related work). In FNC-1, the task is to estimate the stance of a body text from a news article relative to the headline and categorize it into one of four categories: agree, disagree, discusses, or unrelated.

<p align="center"><img src="https://github.com/amallia/FakeNewsChallenge/blob/master/report/images/fnc-eval.png" ></p>

The top scorers of FNC-1 implemented deep learning methods for stance detection by analyzing and optimizing neural-based methods. This was not surprising as this family of techniques gained popularity in 2017, around the time of the competition. Top scorers fared very well with "unrelated" categories, but performed poorly with respect to accuracy on "agree" and "disagree" stances (see Dataset). Our team's objective is to explore the possibility of improving accuracies in the "agree" and "disagree" stances. To achieve this, our plan is to build a fairly simple system using tradition NLP features and classifier (see Methodology). The team's success will be measured against the official baseline accuracy of 79.53% weighted accuracy (see Evaluation plan).

## Dataset
The dataset [2] consist of 49,972 labeled article headline and body pairs, which are derived from the Emergent Dataset [3].

A sample of the data looks like this:
```csv
Headline,Body ID,Stance
Police find mass graves with at least '15 bodies' near Mexico town where 43 students disappeared after police clash,712,unrelated
```

The distribution of the output labels is overwhelmingly "unrelated" to the headline and is outlined below:

|   rows  |   unrelated |   discuss |     agree |   disagree |
|--------:|------------:|----------:|----------:|-----------:|
|  49,972 |    0.73131  |  0.17828  | 0.0736012 |  0.0168094 |

## Methodology

Our team's objective is two-fold: 1) determine whether the heading/content is related/unrelated (syntax/semantics); and 2) as mentioned previously, determine whether the heading/content discusses/agrees/disagrees (sentiment). Moreover, we will test whether traditional NLP features can improve upon the low accuracies in the "agree" and "disagree" stances of previous participants.

The related/unrelated classification task is expected to be much easier and less relevant for detecting fake news, so it is given less weight in the evaluation metric. The Stance Detection task (classify as discusses/agrees/disagrees) is more difficult and relevant to fake news detection, thereby given much more weight in the evaluation metric.

On a preliminary basis, we intend to attack the problem by at least considering the following features:
- overlapping words between the headline and body
- LSA/SVD
- TF-IDF, cosine similarity between the headline and body
- n-gram counts
- semantic role labeling to generate triples of (agent/subject, verb, patient/object)
- dependency relations/syntactic features
- vector features

### Baseline

FNC-1 provides a [baseline model](https://github.com/FakeNewsChallenge/fnc-1-baseline) consisting of hand-engineered features, including n-gram co-occurrence counts between the headline and article and indicator features for polarity and refutation, to train a gradient-boosting classifier. Based on this mode, the baseline accuracy was 79.53%.

The baseline implementation performs well separating related stances from the rest, but rather poorly when differentiating between agree, disagree, and discuss. Our goal is to improve upon this area of weakness while still achieving the baseline accuracy.

## References

[1] Fake News Challenge. http://www.fakenewschallenge.org/

[2] FNC-1 Dataset. https://github.com/FakeNewsChallenge/fnc-1

[3] William Ferreira and Andreas Vlachos. Emergent: a novel data-set for stance classiﬁcation. In Proceedings of NAACL: Human Language Technologies. Association for Computational Linguistics, 2016.
