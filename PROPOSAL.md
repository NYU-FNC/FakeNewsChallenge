# Project proposal 

### Team

- Kedar Gangopadhyay, `kedarg@nyu.edu`
- Antonio Mallia, `me@antoniomallia.it`
- Melanie Tosik (mt3685), `tosik@nyu.edu`

## Problem description

- work on Fake News Challenge
- definition of fake news and specifically stance detection
- previous work all neural-based method
- data distribution overwhelmingly "unrelated", only 0.17828 percent of data "discusses" (0.0736012 "agree"/0.0168094 "disagree")
- previous methods low accuracy on "agree" stances
- see if we can beat official baseline by using traditional NLP features/classifier

## Dataset

Problem is 2-fold:

1. determine is heading/content are related/unrelated (syntax/semantics)
2. if related, determine whether or not discusses/agree/disagrees (sentiment)

http://www.fakenewschallenge.org/
https://github.com/FakeNewsChallenge/fnc-1

## Methodology

Feature engineering, including:

- possibly use semantic role labeling to generate triples of (agent (subject), verb, patient (object))
- dependency relations/syntactic features
- vector features
- TF-IDF
- LSA/SVD
- ??? 

- Python/spaCy/gensim

## Related work

- Benjamin Riedel, Isabelle Augenstein, Georgios P. Spithourakis, Sebastian Riedel. A simple but tough-to-beat baseline for the Fake News Challenge stance detection task. https://arxiv.org/abs/1707.03264.

## Evaluation plan


