"""
Utility functions for the cli to unclutter the actual definition.
"""
import logging
import os

import click


LOGGER_FORMAT_STRING = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


class State(object):
    """ An object that holds the values of common options """
    pass


class CommonOptionGroup(object):
    """
    Generate common options and decorate the command with the resulting object.
    Use thus::

        >>> common_options = CommonOptionGroup()
        >>> common_options.add('--some-option', '-s', type=int, default=10, nargs=1,
        ...     help="Here's an interesting option.",
        ...     extra_callback=lambda ctx, param, value: value + 10)
        >>>
        >>> @click.argument('anotherarg')
        ... @common_options
        ... def some_command(common_state, anotherarg):
        ...     print common_state.some_option, anotherarg
    """
    def __init__(self):
        self.options = []
        self.pass_state = click.make_pass_decorator(State, ensure=True)

    def add(self, *args, **kwargs):
        """
        To a common option group, add a new common option. All args and kwargs are
        passed to a click.option.

        WARNING: If you pass a `callback` or `expose_value` option, it will be
        overwritten. If you want a specific callback, use the option `extra_callback`.
        The value returned by `extra_callback` will replace the passed value.
        """
        def decorator(f):
            """ The actual decorator that will decorate the command """
            def callback(ctx, param, value):
                """
                The click callback that will be executed. Saves the `value` in the `param`
                attribute of a common `State` object after executing any `extra_callback`s passed,
                which may, in particular, modify the passed value.
                """
                if 'extra_callback' in kwargs:
                    # Execute the extra_callback if passed
                    value = kwargs['extra_callback'](ctx, param, value)
                # Get the singleton State object
                state = ctx.ensure_object(State)

                # Set the param attribute to value
                setattr(state, param.name, value)
                # Return the value
                return value

            # Setup the callback for the click decorator
            kwargs['callback'] = callback

            # Don't expose the value to the command function; it will be stored in `State`
            kwargs['expose_value'] = False

            # Decorate the command
            return click.option(*args,
                                **{k: v for k, v in kwargs.iteritems() if k != 'extra_callback'})(f)
        self.options.append(decorator)

    def __call__(self, f):
        """ Decorate the command with all the common options added """
        for option in self.options:
            f = option(f)
        return self.pass_state(f)


def ensure_directory_callback(ctx, param, value):
    """
    This callback ensures that the dirname of the passed value is created. If not,
    it creates it.

    :param ctx: The current click context
    :param param: The parameter name as determined by click
    :param str value: The directory whose existence we're ensuring
    :return: value
    :rtype: str
    """
    if not value:
        return value

    dirname = os.path.dirname(value)
    if not dirname:
        return value

    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    return value


def logging_callback(ctx, param, value):
    """
    A callback that sets the level of the root logger to the passed level.

    :param ctx: The current click context
    :param param: The parameter name as determined by click
    :param value: The desired logger level. Can be any type that
        logging.getLogger().setLevel(...) accepts
    :return: value
    """
    root_logger = logging.getLogger()
    formatter = logging.Formatter(fmt=LOGGER_FORMAT_STRING)
    handler = logging.StreamHandler()
    handler.formatter = formatter
    root_logger.addHandler(handler)
    root_logger.setLevel(value.upper())
    return value


def valid_which_fold(ctx, param, value):
    """
    A click callback that checks if the --which-fold argument is between
    1 and --num-folds. If not, raises a click.BadParameter
    """
    if value is not None:
        if not (1 <= value <= ctx.obj.num_folds):
            raise click.BadParameter(("--which-fold ({which_fold}) must be between 1 and "
                                     "--num-folds ({num_folds}) inclusive").format(
                                     which_fold=value, num_folds=ctx.obj.num_folds))
    return value


def require_value_callback(valid_options=None):
    """ Raise exception if input is not supplied.
    :param tuple|None valid_options: valid options for the parameter. If None, anything goes
    """
    def callback(ctx, param, value):
        if value is None:
            raise click.BadParameter("you must supply " + param.name)
        elif valid_options is not None and value not in valid_options:
            raise click.BadParameter("value must be one of %s" % str(valid_options))
        return value
    return callback
