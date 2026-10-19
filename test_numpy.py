import numpy as np
import pytest


from numpy_tasks import sum_prod, binarize, unique_rows, unique_columns, matrix_statistics, chess
from grader_contracts.numpy_tasks import MatrixVectorBatchInput, BinarizeInput, MatrixInput, RandomMatrixInput, ChessInput


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
    matrix = np.array([[2, 3], [4, 5]])
    result = binarize(BinarizeInput(matrix, 3))
    np.testing.assert_array_equal(result, [[0, 0], [1, 1]])


def test_binarize_default_threshold():
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

#unique_rows unique_columns

def test_unique_basic():
    matrix = np.array([[7, 3, 7], [5, 5, 1]])
    assert unique_rows(MatrixInput(matrix)) == [[3, 7], [1, 5]]
    assert unique_columns(MatrixInput(matrix)) == [[5, 7], [3, 5], [1, 7]]


def test_unique_single_element():
    matrix = np.array([[-2]])
    assert unique_rows(MatrixInput(matrix)) == [[-2]]
    assert unique_columns(MatrixInput(matrix)) == [[-2]]


def test_unique_single_row():
    matrix = np.array([[9, 4, 9, 4, 6]])
    assert unique_rows(MatrixInput(matrix)) == [[4, 6, 9]]                 
    assert unique_columns(MatrixInput(matrix)) == [[9], [4], [9], [4], [6]]  


def test_unique_single_column():
    matrix = np.array([[8], [8], [2], [5]])
    assert unique_rows(MatrixInput(matrix)) == [[8], [8], [2], [5]]
    assert unique_columns(MatrixInput(matrix)) == [[2, 5, 8]]


def test_unique_all_same():
    matrix = np.array([[1, 1], [1, 1]])
    assert unique_rows(MatrixInput(matrix)) == [[1], [1]]
    assert unique_columns(MatrixInput(matrix)) == [[1], [1]]


def test_unique_all_different():
    matrix = np.array([[1, 2], [3, 4]])
    assert unique_rows(MatrixInput(matrix)) == [[1, 2], [3, 4]]
    assert unique_columns(MatrixInput(matrix)) == [[1, 3], [2, 4]]


def test_unique_floats_and_negative():
    matrix = np.array([[0.5, 0.5], [-1.0, 2.0]])
    assert unique_rows(MatrixInput(matrix)) == [[0.5], [-1.0, 2.0]]
    assert unique_columns(MatrixInput(matrix)) == [[-1.0, 0.5], [0.5, 2.0]]


#matrix_statistics
def test_matrix_statistics_shapes():
    result = matrix_statistics(RandomMatrixInput(rows=3, columns=4, seed=1))
    assert result.matrix.shape == (3, 4)
    assert result.row_means.shape == (3,)        
    assert result.column_means.shape == (4,)     
    assert result.row_variances.shape == (3,)
    assert result.column_variances.shape == (4,)


def test_matrix_statistics_first_row_and_column():
    result = matrix_statistics(RandomMatrixInput(rows=3, columns=4, seed=1))

    first_row = result.matrix[0]                 # первая строка таблицы
    manual_row_mean = sum(first_row) / 4         # в строке 4 числа: сложили и поделили на 4
    assert result.row_means[0] == pytest.approx(manual_row_mean)

    first_column = result.matrix[:, 0]           # первый столбец таблицы
    manual_column_mean = sum(first_column) / 3   # в столбце 3 числа: сложили и поделили на 3
    assert result.column_means[0] == pytest.approx(manual_column_mean)


def test_matrix_statistics_same_seed_same_result():
    first = matrix_statistics(RandomMatrixInput(rows=3, columns=3, seed=7))
    second = matrix_statistics(RandomMatrixInput(rows=3, columns=3, seed=7))
    np.testing.assert_array_equal(first.matrix, second.matrix)


def test_matrix_statistics_different_seed_different_result():
    first = matrix_statistics(RandomMatrixInput(rows=3, columns=3, seed=7))
    second = matrix_statistics(RandomMatrixInput(rows=3, columns=3, seed=8))
    assert not np.array_equal(first.matrix, second.matrix)


def test_matrix_statistics_zero_std():
    result = matrix_statistics(RandomMatrixInput(rows=2, columns=3, mean=5.0, std=0.0))
    np.testing.assert_array_equal(result.matrix, np.full((2, 3), 5.0))
    np.testing.assert_array_equal(result.row_variances, [0.0, 0.0])
    np.testing.assert_array_equal(result.column_variances, [0.0, 0.0, 0.0])


def test_matrix_statistics_single_element():
    result = matrix_statistics(RandomMatrixInput(rows=1, columns=1, mean=2.0, std=1.0, seed=3))
    assert result.matrix.shape == (1, 1)
    assert result.row_means[0] == pytest.approx(result.matrix[0, 0])
    assert result.row_variances[0] == 0.0    
    assert result.column_variances[0] == 0.0


def test_matrix_statistics_single_row():
    result = matrix_statistics(RandomMatrixInput(rows=1, columns=5, seed=4))
    assert result.row_means.shape == (1,)
    assert result.column_means.shape == (5,)
    np.testing.assert_array_equal(result.column_variances, np.zeros(5))   # в каждом столбце одно число


def test_matrix_statistics_follows_mean_and_std():
    result = matrix_statistics(RandomMatrixInput(rows=200, columns=200, mean=5.0, std=2.0, seed=0))
    assert result.matrix.mean() == pytest.approx(5.0, abs=0.1)
    assert result.matrix.std() == pytest.approx(2.0, abs=0.1)


def test_matrix_statistics_default_parameters():
    result = matrix_statistics(RandomMatrixInput(rows=2, columns=2))
    assert result.matrix.shape == (2, 2)

#chess
def test_chess_example_from_task():
    result = chess(ChessInput(rows=3, columns=4, first=0, second=1))
    np.testing.assert_array_equal(result, [[0, 1, 0, 1],
                                           [1, 0, 1, 0],
                                           [0, 1, 0, 1]])


def test_chess_top_left_is_first():
    result = chess(ChessInput(rows=2, columns=2, first=7, second=3))
    np.testing.assert_array_equal(result, [[7, 3], [3, 7]])


def test_chess_single_cell():
    result = chess(ChessInput(rows=1, columns=1, first=5, second=9))
    np.testing.assert_array_equal(result, [[5]])


def test_chess_single_row():
    result = chess(ChessInput(rows=1, columns=5, first=1, second=2))
    np.testing.assert_array_equal(result, [[1, 2, 1, 2, 1]])


def test_chess_single_column():
    result = chess(ChessInput(rows=4, columns=1, first=1, second=2))
    np.testing.assert_array_equal(result, [[1], [2], [1], [2]])


def test_chess_odd_square():
    result = chess(ChessInput(rows=3, columns=3, first=1, second=2))
    np.testing.assert_array_equal(result, [[1, 2, 1],
                                           [2, 1, 2],
                                           [1, 2, 1]])


def test_chess_shape():
    result = chess(ChessInput(rows=5, columns=2, first=0, second=1))
    assert result.shape == (5, 2)


def test_chess_equal_values():
    result = chess(ChessInput(rows=2, columns=3, first=4, second=4))
    np.testing.assert_array_equal(result, np.full((2, 3), 4))


def test_chess_float_and_negative_values():
    result = chess(ChessInput(rows=2, columns=2, first=0.5, second=-1.5))
    np.testing.assert_allclose(result, [[0.5, -1.5], [-1.5, 0.5]])

#draw_rectangle и draw_ellipse
