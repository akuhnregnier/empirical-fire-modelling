# -*- coding: utf-8 -*-
from operator import methodcaller

import pytest

from empirical_fire_modelling import variable


@pytest.fixture
def dummy_variables():
    variables = []
    for rank, name, units in [(0, "foo", "1"), (1, "bar", "%")]:
        for shift in [0, 1, 3, 6, 9]:
            variables.append(
                variable.VariableFactory(rank=rank, name=name, units=units)[shift]
            )
    return tuple(variables)


def test_transform_offset(dummy_variables):
    transformed = tuple(map(methodcaller("transform_offset"), dummy_variables))
    assert transformed == dummy_variables


def test_get_matching(dummy_variables):
    assert (
        variable.get_matching(dummy_variables, single=False, name="foo")
        == variable.get_matching(dummy_variables, single=False, units="1")
        == variable.get_matching(dummy_variables, single=False, name="foo", units="1")
        == variable.get_matching(dummy_variables, single=False, units="1", name="foo")
        == dummy_variables[:5]
    )


def test_get_matching_single_exc(dummy_variables):
    with pytest.raises(RuntimeError):
        variable.get_matching(dummy_variables, name="foo")


def test_get_matching_strict_exc(dummy_variables):
    assert variable.get_matching(dummy_variables, strict=False, name="foobar") == ()
    with pytest.raises(RuntimeError):
        variable.get_matching(dummy_variables, name="foobar")
