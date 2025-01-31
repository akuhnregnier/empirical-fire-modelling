# -*- coding: utf-8 -*-
import numpy as np
import pytest

from empirical_fire_modelling.data.cv import apply_structure


@pytest.mark.parametrize(
    "array, structure, expected",
    [
        (
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.bool_,
            ),
            np.array([[1]], dtype=np.bool_),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.bool_,
            ),
        ),
        (
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.bool_,
            ),
            np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.bool_),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.bool_,
            ),
        ),
        (
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.bool_,
            ),
            np.array(
                [
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                ],
                dtype=np.bool_,
            ),
            np.array(
                [
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                ],
                dtype=np.bool_,
            ),
        ),
        (
            np.array(
                [
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.bool_,
            ),
            np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.bool_),
            np.array(
                [
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.bool_,
            ),
        ),
        (
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                ],
                dtype=np.bool_,
            ),
            np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.bool_),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                ],
                dtype=np.bool_,
            ),
        ),
        (
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.bool_,
            ),
            np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.bool_),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 1],
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.bool_,
            ),
        ),
        (
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.bool_,
            ),
            np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.bool_),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1],
                    [1, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.bool_,
            ),
        ),
        (
            np.array(
                [
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.bool_,
            ),
            np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.bool_),
            np.array(
                [
                    [1, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.bool_,
            ),
        ),
        (
            np.array(
                [
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.bool_,
            ),
            np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.bool_),
            np.array(
                [
                    [1, 1, 0, 0, 1],
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.bool_,
            ),
        ),
        (
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                ],
                dtype=np.bool_,
            ),
            np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.bool_),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 1],
                ],
                dtype=np.bool_,
            ),
        ),
        (
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1],
                ],
                dtype=np.bool_,
            ),
            np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.bool_),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1],
                    [1, 0, 0, 1, 1],
                ],
                dtype=np.bool_,
            ),
        ),
        (
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.bool_,
            ),
            np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.bool_),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1],
                    [1, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1],
                ],
                dtype=np.bool_,
            ),
        ),
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.bool_,
            ),
            np.array(
                [
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 0],
                    [0, 0, 1, 1, 0],
                ],
                dtype=np.bool_,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.bool_,
            ),
        ),
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.bool_,
            ),
            np.array(
                [
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 0],
                    [0, 0, 1, 1, 0],
                ],
                dtype=np.bool_,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.bool_,
            ),
        ),
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                ],
                dtype=np.bool_,
            ),
            np.array(
                [
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 0],
                    [0, 0, 1, 1, 0],
                ],
                dtype=np.bool_,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1, 1, 1, 1],
                ],
                dtype=np.bool_,
            ),
        ),
    ],
)
def test_apply_structure(array, structure, expected):
    assert np.all(apply_structure(array, structure) == expected)
