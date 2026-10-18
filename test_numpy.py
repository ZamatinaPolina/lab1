import numpy as np
import pytest


from numpy_tasks import sum_prod, binarize
from grader_contracts.numpy_tasks import MatrixVectorBatchInput, BinarizeInput


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

#binarize
def test_binarize_basic():
    matrix = np.array([[1, 5], [3, 7]])
    result = binarize(BinarizeInput(matrix, 4))
    np.testing.assert_array_equal(result, [[0, 1], [0, 1]])


def test_binarize_strictly_greater():
    # числа, равные порогу, становятся 0
    matrix = np.array([[2, 3], [4, 5]])
    result = binarize(BinarizeInput(matrix, 3))
    np.testing.assert_array_equal(result, [[0, 0], [1, 1]])


def test_binarize_default_threshold():
    # порог по умолчанию 0.5, его не передаём
    matrix = np.array([[0.2, 0.5], [0.7, 0.51]])
    result = binarize(BinarizeInput(matrix))
    np.testing.assert_array_equal(result, [[0, 0], [1, 1]])


def test_binarize_negative_values():
    matrix = np.array([[-3, -1], [0, 2]])
    result = binarize(BinarizeInput(matrix, -1))
    np.testing.assert_array_equal(result, [[0, 0], [1, 1]])


def test_binarize_all_above_and_all_below():
    matrix = np.full((2, 3), 10)
    np.testing.assert_array_equal(binarize(BinarizeInput(matrix, 0)), np.ones((2, 3)))
    np.testing.assert_array_equal(binarize(BinarizeInput(matrix, 100)), np.zeros((2, 3)))


def test_binarize_single_element():
    np.testing.assert_array_equal(binarize(BinarizeInput(np.array([[5]]), 4)), [[1]])
    np.testing.assert_array_equal(binarize(BinarizeInput(np.array([[5]]), 5)), [[0]])


def test_binarize_does_not_change_original():
    matrix = np.array([[1, 5], [3, 7]])
    copy = matrix.copy()
    result = binarize(BinarizeInput(matrix, 4))
    np.testing.assert_array_equal(matrix, copy)  
    assert result is not matrix                    


def test_binarize_keeps_shape():
    matrix = np.arange(6).reshape(2, 3)      
    assert binarize(BinarizeInput(matrix, 2)).shape == (2, 3)
