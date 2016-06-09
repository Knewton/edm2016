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
- numpy
- scipy
- ipython
- pandas
- igraph

# Data

The ASSISTments data set may be found [here](https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data/skill-builder-data-2009-2010). Note that the authors of the data set have since removed several duplicates from the original data set which we used. However, as we explain in the paper, our preprocessing steps involved removing these duplicates as well. Thus, while we used the original data set, both the original and the corrected versions should duplicate our results.

The KDD Cup data set may be found [here](https://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp). We used the Bridge to Algebra 2006-2007 data set, and specifically the training data set.

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
prior parameters, RNN layer sizes; 80% for train/test) using `data/split_data.py`, 
`python split_data.py bridge_to_algebra_2006_2007_train.txt "Anon Student Id" "\t"`, 
`python split_data.py skill_builder_data.csv user_id ","`

2. execute the following commands:

#### IRT
    rnn_prof irt assistments skill_builder_data_big.txt --onepo \
    --drop-duplicates --no-remove-skill-nans --num-folds 5 \
    --item-id-col problem_id --concept-id-col single 

    rnn_prof irt kddcup bridge_to_algebra_2006_2007_train_big.txt \
    --onepo --drop-duplicates --no-remove-skill-nans --num-folds 5 \
    --item-id-col 'Step Name' --concept-id-col single

#### HIRT
    rnn_prof irt assistments skill_builder_data_big.txt --onepo \
    --drop-duplicates --no-remove-skill-nans --num-folds 5 \
    --item-precision 4.0 --template-precision 2.0 \
    --template-id-col template_id --item-id-col problem_id \
    --concept-id-col single

    rnn_prof irt kddcup bridge_to_algebra_2006_2007_train_big.txt  --onepo \
    --drop-duplicates --no-remove-skill-nans --num-folds 5 \
    --item-precision 2.0 --template-precision 4.0 -m 5000 \
    --template-id-col template_id --item-id-col problem_id \
    --concept-id-col single

#### DKT
    rnn_prof rnn assistments skill_builder_data_big.txt  \
    --no-remove-skill-nans --drop-duplicates --num-folds 5 \
    --item-id-col problem_id --num-iters 50 --dropout-prob 0.25 \
    --first-learning-rate 5.0  --compress-dim 50 --hidden-dim 100 

    rnn_prof rnn kddcup bridge_to_algebra_2006_2007_train_big.txt  \
    --no-remove-skill-nans --drop-duplicates --num-folds 5 --item-id-col KC \
    --num-iters 50 --dropout-prob 0.25 --first-learning-rate 5.0 \
    --compress-dim 50 --hidden-dim 100 



