"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    total = np.zeros_like(vectors[0], dtype=float)  
    for matrix, vector in zip(matrices, vectors):  
        total = total + matrix @ vector           
    return total


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    matrix = np.asarray(matrix)
    return (matrix > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [np.unique(row).tolist() for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [np.unique(column).tolist() for column in matrix.T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)                       
    matrix = rng.normal(loc=mean, scale=std, size=(rows, columns))  

    return MatrixStatistics(
        matrix=matrix,
        row_means=matrix.mean(axis=1),            # среднее по каждой строке
        column_means=matrix.mean(axis=0),         # среднее по каждому столбцу
        row_variances=matrix.var(axis=1),         # дисперсия для каждой строки
        column_variances=matrix.var(axis=0),      # дисперсия для каждого столбца
    )

def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    row_index, column_index = np.indices((rows, columns))
    return np.where((row_index + column_index) % 2 == 0, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    
    image = np.full((image_height, image_width, 3), background_color, dtype=np.uint8)

    top = max((image_height - height) // 2, 0)
    left = max((image_width - width) // 2, 0)
    bottom = min(top + height, image_height)
    right = min(left + width, image_width)

    image[top:bottom, left:right] = shape_color
    return image

def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    image = np.full((image_height, image_width, 3), background_color, dtype=np.uint8)

    y, x = np.indices((image_height, image_width))
    x0 = (image_width - 1) / 2
    y0 = (image_height - 1) / 2

    inside = ((x - x0) / semi_axis_x) ** 2 + ((y - y0) / semi_axis_y) ** 2 <= 1
    image[inside] = shape_color
    return image




def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    raise NotImplementedError  # TODO


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    raise NotImplementedError  # TODO
