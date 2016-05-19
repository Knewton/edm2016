from __future__ import division

import click
import numpy as np

from . import run_irt
from . import run_rnn
from .cliutils import (CommonOptionGroup, ensure_directory_callback, logging_callback,
                       valid_which_fold, require_value_callback)
from .data.assistments import SKILL_ID_KEY, PROBLEM_ID_KEY, TEMPLATE_ID_KEY
from .data.constants import USER_IDX_KEY, SINGLE
from .data.kddcup import KC_NAME_STARTS_WITH, PROBLEM_NAME, STEP_NAME
from .data.splitting_utils import split_data
from .data.wrapper import load_data, DataOpts


# Setup common options
common_options = CommonOptionGroup()

# System options
common_options.add('--log-level', '-l', type=click.Choice(['warn', 'info', 'debug']),
                   default='info', help="Set the logging level", extra_callback=logging_callback)
common_options.add('--seed', '-r', type=int, default=0,
                   help="Random number seed for data splitting and model initialization")

# Data options
common_options.add('--remove-skill-nans/--no-remove-skill-nans', is_flag=True, default=False,
                   help="Remove interactions from the data set whose skill_id column is NaN. "
                        "This will occur whether or not the item_id_col is skill_id")
common_options.add('--item-id-col', type=str, nargs=1,
                   help="(Required) Which column should be used for identifying items from the "
                        "dataset. Depends on source as to which names are valid.",
                   extra_callback=require_value_callback((SKILL_ID_KEY, PROBLEM_ID_KEY,
                                                          TEMPLATE_ID_KEY, SINGLE,
                                                          KC_NAME_STARTS_WITH, PROBLEM_NAME,
                                                          STEP_NAME)))
common_options.add('--drop-duplicates/--no-drop-duplicates', default=True,
                   help="Remove duplicate interactions: only the first row is retained for "
                        "duplicate row indices in Assistments")
common_options.add('--max-inter', '-m', type=int, default=0, help="Maximum interactions per user",
                   extra_callback=lambda ctx, param, value: value or None)
common_options.add('--min-inter', type=int, default=2,
                   help="Minimum number of interactions required after filtering to retain a user",
                   extra_callback=lambda ctx, param, value: value or None)
common_options.add('--proportion-students-retained', type=float, default=1.0,
                   help="Proportion of user ids to retain in data set (for testing sensitivity "
                        "to number of data points). Default is 1.0, i.e., all data retained.")

# Learning options
common_options.add('--num-folds', '-f', type=int, nargs=1, default=5,
                   help="Number of folds for testing.", is_eager=True)
common_options.add('--which-fold', type=int, nargs=1, default=None, extra_callback=valid_which_fold,
                   help="If you want to parallelize folds, run several processes with this "
                        "option set to run a single fold. Folds are numbered 1 to --num-folds.")

# Reporting options
common_options.add('--output', '-o', default='rnn_result',
                   help="Where to store the pickled output of training",
                   extra_callback=ensure_directory_callback)


@click.group(context_settings={'help_option_names': ['-h', '--help']})
def cli():
    """ Collection of scripts for evaluating RNN proficiency models """
    pass


@cli.command('rnn')
@click.argument('source')
@click.argument('data_file')
@click.option('--compress-dim', '-d', type=int, nargs=1, default=100,
              help="The dimension to which to compress the input. If -1, will do no compression")
@click.option('--hidden-dim', '-h', type=int, nargs=1, default=100,
              help="The number of hidden units in the RNN.")
@click.option('--output-compress-dim', '-od', type=int, nargs=1, default=None,
              help="The dimension to which we should compress the output vector. "
                   "If not passed, no compression will occur.")
@click.option('--test-spacing', '-t', type=int, nargs=1, default=10,
              help="How many iterations before running the tests?")
@click.option('--recurrent/--no-recurrent', default=True,
              help="Whether to use a recurrent architecture")
@click.option('--use-correct/--no-use-correct', default=True,
              help="If True, record correct and incorrect responses as different input dimensions")
@click.option('--num-iters', '-n', type=int, default=50,
              help="How many iterations of training to perform on the RNN")
@click.option('--dropout-prob', '-p', type=float, default=0.0,
              help="The probability of a node being dropped during training. Default is 0.0 "
                   "(i.e., no dropout)")
@click.option('--use-hints/--no-use-hints', default=False,
              help="Should we add a one-hot dimension to represent whether a student used a hint?")
@click.option('--first-learning-rate', nargs=1, default=30.0, type=float,
              help="The initial learning rate. Will decay at rate `decay_rate`. Default is 30.0.")
@click.option('--decay-rate', nargs=1, default=0.99, type=float,
              help="The rate at which the learning rate decays. Default is 0.99.")
@common_options
def rnn(common, source, data_file, compress_dim, hidden_dim, output_compress_dim, test_spacing,
        recurrent, use_correct, num_iters, dropout_prob, use_hints, first_learning_rate,
        decay_rate):
    """ RNN based proficiency estimation.
    SOURCE specifies the student data source, and should be 'assistments' or 'kddcup'.
    DATA_FILE is the filename for the interactions data.
    """
    data_opts = DataOpts(num_folds=common.num_folds, item_id_col=common.item_id_col,
                         concept_id_col=None, template_id_col=None, use_correct=use_correct,
                         remove_skill_nans=common.remove_skill_nans, seed=common.seed,
                         use_hints=use_hints,
                         drop_duplicates=common.drop_duplicates,
                         max_interactions_per_user=common.max_inter,
                         min_interactions_per_user=common.min_inter,
                         proportion_students_retained=common.proportion_students_retained)

    data, _, item_ids, _, _ = load_data(data_file, source, data_opts)
    num_questions = len(item_ids)
    data_folds = split_data(data, num_folds=common.num_folds, seed=common.seed)
    run_rnn.run(data_folds, common.num_folds, num_questions, num_iters, output=common.output,
                compress_dim=compress_dim, hidden_dim=hidden_dim, test_spacing=test_spacing,
                recurrent=recurrent, data_opts=data_opts, dropout_prob=dropout_prob,
                output_compress_dim=output_compress_dim,
                first_learning_rate=first_learning_rate, decay_rate=decay_rate,
                which_fold=common.which_fold)


@cli.command('irt')
@click.argument('source')
@click.argument('data_file')
@click.option('--twopo/--onepo', default=False, help="Use a 2PO model (default is False)")
@click.option('--concept-id-col', type=str, nargs=1,
              help="(Required) Which column should be used for identifying "
                   "concepts from the dataset. Depends on source as to which names are valid. "
                   "If ``single``, use single dummy concept.",
              callback=require_value_callback((SKILL_ID_KEY, PROBLEM_ID_KEY, SINGLE,
                                               TEMPLATE_ID_KEY, KC_NAME_STARTS_WITH)))
@click.option('--template-id-col', type=str, default=None, nargs=1,
              help="If using templates, this option is used to specify the column in the dataset "
                   "you are using to represent the template id")
@click.option('--template-precision', default=None, type=float, nargs=1,
              help="Use template_id in IRT learning. Item means will be distributed around a "
                   "template mean. The precision of that distribution is the argument of this "
                   "parameter.")
@click.option('--item-precision', default=None, type=float, nargs=1,
              help="If using a non-templated model, this is the precision of the Gaussian "
                   "prior around item difficulties. If using a templated model, it is the "
                   "precision of the template hyperprior's mean. Default is 1.0.")
@common_options
def irt(common, source, data_file, twopo, concept_id_col, template_precision,
        template_id_col, item_precision):
    """ Run IRT to get item parameters and compute online metrics on a held-out set of students
    SOURCE specifies the student data source, and should be 'assistments' or 'kddcup'.
    DATA_FILE is the filename for the interactions data.
    """

    if (template_precision is None) != (template_id_col is None):
        raise ValueError("template_precision and template_id_col must both be set or both be None")
    data_opts = DataOpts(num_folds=common.num_folds, item_id_col=common.item_id_col,
                         concept_id_col=concept_id_col, template_id_col=template_id_col,
                         remove_skill_nans=common.remove_skill_nans,
                         seed=common.seed, use_correct=True,
                         use_hints=False, drop_duplicates=common.drop_duplicates,
                         max_interactions_per_user=common.max_inter,
                         min_interactions_per_user=common.min_inter,
                         proportion_students_retained=common.proportion_students_retained)

    data, _, _, _, _ = load_data(data_file, source, data_opts)
    data_folds = split_data(data, num_folds=common.num_folds, seed=common.seed)
    run_irt.irt(data_folds, common.num_folds, output=common.output, data_opts=data_opts,
                is_two_po=twopo,
                template_precision=template_precision,
                single_concept=concept_id_col is None,
                which_fold=common.which_fold,
                item_precision=item_precision)


@cli.command('naive')
@click.argument('source')
@click.argument('data_file')
@common_options
def naive(common, source, data_file):
    """ Just report the percent correct across all events.
    SOURCE specifies the student data source, and should be 'assistments' or 'kddcup'.
    DATA_FILE is the filename for the interactions data.
    """
    data_opts = DataOpts(num_folds=common.num_folds, item_id_col=common.item_id_col,
                         concept_id_col=None, template_id_col=None, use_correct=True,
                         remove_skill_nans=common.remove_skill_nans, seed=common.seed,
                         use_hints=True,
                         drop_duplicates=common.drop_duplicates,
                         max_interactions_per_user=common.max_inter,
                         min_interactions_per_user=common.min_inter,
                         proportion_students_retained=common.proportion_students_retained)
    data, _, _, _, _ = load_data(data_file, source, data_opts)
    print "Percentage correct in data set is {}".format(data.correct.mean())

    agged = data.groupby(USER_IDX_KEY).correct.agg([np.sum, len]).reset_index()
    mask = agged['sum'] <= agged['len'] // 2
    agged.loc[mask, 'sum'] = agged.loc['len', mask] - agged.loc['sum', mask]
    print "Percent correct for naive classifier is {}".format(agged['sum'].sum() /
                                                              agged['len'].sum())


def main():
    cli()
