import numpy as np
import pytest

from numpy_tasks import sum_prod
from grader_contracts.numpy_tasks import MatrixVectorBatchInput


def test_sum_prod_basic():
    matrices = np.array([[[1, 2], [3, 4]], [[0, 1], [1, 0]]])
    vectors = np.array([[[1], [1]], [[5], [6]]])
    result = sum_prod(MatrixVectorBatchInput(matrices, vectors))
    np.testing.assert_array_equal(result, [[9], [12]])


def test_sum_prod_single_pair():
    matrices = np.array([np.eye(3)])
    vectors = np.array([[[1], [2], [3]]])
    result = sum_prod(MatrixVectorBatchInput(matrices, vectors))
    np.testing.assert_allclose(result, [[1], [2], [3]])


def test_sum_prod_size_one():
    # матрицы 1x1 (n = 1)
    matrices = np.array([[[2]], [[3]]])
    vectors = np.array([[[4]], [[5]]])
    result = sum_prod(MatrixVectorBatchInput(matrices, vectors))
    np.testing.assert_array_equal(result, [[23]])  


def test_sum_prod_zero_matrices():
    matrices = np.zeros((4, 3, 3))
    vectors = np.ones((4, 3, 1))
    result = sum_prod(MatrixVectorBatchInput(matrices, vectors))
    np.testing.assert_array_equal(result, np.zeros((3, 1)))


def test_sum_prod_result_shape():
    matrices = np.ones((5, 4, 4))
    vectors = np.ones((5, 4, 1))
    result = sum_prod(MatrixVectorBatchInput(matrices, vectors))
    assert result.shape == (4, 1)
    np.testing.assert_array_equal(result, np.full((4, 1), 20)) 
