# IRT and DKT implementation

This library contains implementations of IRT models and 
[Deep Knowledge Tracing (DKT)](http://papers.nips.cc/paper/5654-deep-knowledge-tracing.pdf) that reproduces the results reported in "Back to the Basics: Bayesian extensions of IRT outperform neural networks for proficiency estimation" (Wilson, Karklin, Han, Ekanadham EDM2016).

# Implemented models 

## IRT 


Bayesian versions of one and two parameter Item Response Theory models.  The likelihood is given by the ogive item response function, and priors on student and item parameters are standard normal distributions.

## Hierarchical IRT

Implementation of an IRT model that extends the model above with a Gaussian hyper-prior on item difficulties.

## DKT

Recurrent neural network implemented using Theano.

# Requirements (see `requirements.in`)
- python
- theano
- numpu
- scipy
- ipython
- pandas
- igraph

# Usage
```
    Usage: rnn_prof [OPTIONS] COMMAND [ARGS]...

      Collection of scripts for evaluating RNN proficiency models

    Options:
      -h, --help  Show this message and exit.

    Commands:
      irt    Run IRT to get item parameters and compute...
      naive  Just report the percent correct across all...
      rnn    RNN based proficiency estimation :param str...
```


# To reproduce results in the EDM2016 paper:

1. construct the 20/80 split data sets (20% for model parameter selection, e.g.,
prior parameters, RNN layer sizes; 80% for train/test).  Detailed instructions:
  a. TBD

2. execute the following commands:

## IRT
    rnn_prof irt assistments skill_builder_data_80.txt --onepo \
    --drop-duplicates --no-remove-skill-nans --num-folds 5 \
    --item-id-col problem_id --concept-id-col single 

    rnn_prof irt kddcup bridge_to_algebra_2006_2007_train_80.txt \
    --onepo --drop-duplicates --no-remove-skill-nans --num-folds 5 \
    --item-id-col 'Step Name' --concept-id-col single

## HIRT
    rnn_prof irt assistments skill_builder_data_80.txt --onepo \
    --drop-duplicates --no-remove-skill-nans --num-folds 5 \
    --item-precision 4.0 --template-precision 2.0 \
    --template-id-col template_id --item-id-col problem_id \
    --concept-id-col single

    rnn_prof irt kddcup bridge_to_algebra_2006_2007_train_80.txt  --onepo \
    --drop-duplicates --no-remove-skill-nans --num-folds 5 \
    --item-precision 2.0 --template-precision 4.0 -m 5000 \
    --template-id-col template_id --item-id-col problem_id \
    --concept-id-col single

## DKT
    rnn_prof rnn assistments skill_builder_data_80.txt  \
    --no-remove-skill-nans --drop-duplicates --num-folds 5 \
    --item-id-col problem_id --num-iters 50 --dropout-prob 0.25 \
    --first-learning-rate 5.0  --compress-dim 50 --hidden-dim 100 

    rnn_prof rnn kddcup bridge_to_algebra_2006_2007_train_80.txt  \
    --no-remove-skill-nans --drop-duplicates --num-folds 5 --item-id-col KC \
    --num-iters 50 --dropout-prob 0.25 --first-learning-rate 5.0 \
    --compress-dim 50 --hidden-dim 100 



