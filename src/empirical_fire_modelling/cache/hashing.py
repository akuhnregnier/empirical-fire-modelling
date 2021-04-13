# -*- coding: utf-8 -*-
from collections.abc import Sequence, Set

import joblib
from wildfires.cache.hashing import Hasher

from .. import variable


def get_variable_parent_name(var):
    """Get the name of the parent if it exists or return the variable name otherwise."""
    if hasattr(var, "parent") and var.parent is not None:
        return var.parent.name
    else:
        return var.name


class OffsetVariableHasher(Hasher):
    @staticmethod
    def test_argument(arg):
        return isinstance(arg, variable.OffsetVariable)

    @staticmethod
    def hash(arg):
        # Ignore the Variable 'rank' and 'units' attributes for the hash.
        return joblib.hashing.hash(
            (arg.name, arg.shift, arg.comp_shift, get_variable_parent_name(arg))
        )


class VariableHasher(Hasher):
    @staticmethod
    def test_argument(arg):
        return isinstance(arg, variable.Variable)

    @staticmethod
    def hash(arg):
        return joblib.hashing.hash((arg.name, arg.shift, get_variable_parent_name(arg)))


_variable_hasher = VariableHasher()
_offset_variable_hasher = OffsetVariableHasher()


def get_variable_hash(var):
    if _offset_variable_hasher.test_argument(var):
        return _offset_variable_hasher.hash(var)
    if not _variable_hasher.test_argument(var):
        raise ValueError("A Variable needs to be given.")
    return _variable_hasher.hash(var)


class VariableSequenceHasher(Hasher):
    @staticmethod
    def test_argument(arg):
        # Hash of a sequence of Variable.
        return (
            isinstance(arg, Sequence)
            and all(isinstance(v, variable.Variable) for v in arg)
        ) or (
            isinstance(arg, Set) and all(isinstance(v, variable.Variable) for v in arg)
        )

    @staticmethod
    def hash(arg):
        # Hash of a sequence of Variable.

        arg_hash = None

        if isinstance(arg, Sequence):
            if all(isinstance(v, variable.Variable) for v in arg):
                arg_hash = joblib.hashing.hash(tuple(get_variable_hash(v) for v in arg))
        elif isinstance(arg, Set):
            if all(isinstance(v, variable.Variable) for v in arg):
                arg_hash = joblib.hashing.hash(set(get_variable_hash(v) for v in arg))

        if arg_hash is None:
            raise ValueError("Expected a collection of variables.")

        return arg_hash
