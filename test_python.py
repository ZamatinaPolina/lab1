from types import SimpleNamespace

import pytest

from python_basics import (
    count_vowels,
    has_unique_characters,
    count_one_bits,
    multiplicative_persistence,
    prime_factorization,
    pyramid,
    is_balanced_number,
    mse
)

def text(value):
    return SimpleNamespace(value=value)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("", 0),            # пустая строка
        ("bcdfg", 0),       # гласных нет
        ("aeiou", 5),       # все гласные
        ("AEIOU", 5),       # заглавные
        ("Hello World", 3), # предложение
        ("aaaa", 4),        # повторы
        ("12!5:@6$%", 0),   # цифры и спецсимволы
        ("a e i o u", 5),   # гласные с пробелами
    ],
)
def test_count_vowels(value, expected):
    assert count_vowels(text(value)) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("", True),       # пустая строка
        ("a", True),      # один символ
        ("abc", True),    # нет повторений
        ("aba", False),   # повтор
        ("aa", False),    # только повторы
        ("aA", True),     # регистр различается
    ],
)
def test_has_unique_characters(value, expected):
    assert has_unique_characters(text(value)) is expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (1, 1),        # 1
        (2, 1),        # 10
        (3, 2),        # 11
        (7, 3),        # 111
        (8, 1),        # одна единица 100...
        (255, 8),      # 11111111
        (1023, 10),    # 1111111111
        (2**40, 1),    # 2*x одна единица
    ],
)
def test_count_one_bits(value, expected):
    assert count_one_bits(text(value)) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (4, 0),         # однозначное
        (10, 1),        # 1*0 = 0
        (25, 2),        # 2*5=10, 1*0=0
        (39, 3),         
        (77, 4),        # 49, 36, 18, 8
        (999, 4),       
        (10**6, 1),     #  1*0*0*0*0*0... = 0
    ],
)
def test_multiplicative_persistence(value, expected):
    assert multiplicative_persistence(text(value)) == expected


def vectors(predicted, expected):
    return SimpleNamespace(predicted=predicted, expected=expected)


def test_mse_basic():
    assert mse(vectors([1, 2, 3], [1, 2, 3])) == 0           # векторы совпадают
    assert mse(vectors([1, 2, 3], [2, 3, 4])) == 1           # разность равна 1
    assert mse(vectors([0, 0], [3, 4])) == pytest.approx(12.5)  # (9 + 16) / 2


def test_mse_single_and_negative():
    assert mse(vectors([5], [2])) == 9                        # один элемент
    assert mse(vectors([-1, -2], [1, 2])) == pytest.approx(10.0)  # отрицательные числа


def test_mse_invalid():
    with pytest.raises(ValueError):
        mse(vectors([], []))           # пустые векторы
    with pytest.raises(ValueError):
        mse(vectors([1, 2], [1]))      # разная длина


@pytest.mark.parametrize(
    "value, expected",
    [
        (1, "(1)"),                      
        (2, "(2)"),                      
        (4, "(2**2)"),                  
        (12, "(2**2)(3)"),               
        (97, "(97)"),                    
        (86240, "(2**5)(5)(7**2)(11)"),  
        (2 * 3 * 5 * 7, "(2)(3)(5)(7)"), 
        (2**10, "(2**10)"),              
        (99991, "(99991)"),   
    ],
)
def test_prime_factorization(value, expected):
    assert prime_factorization(text(value)) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (1, 1),    
        (5, 2),    
        (14, 3),    
        (55, 5),
        (91, 6),
    ],
)
def test_pyramid_possible(value, expected):
    assert pyramid(text(value)) == expected


@pytest.mark.parametrize("value", [2, 6, 13, 15, 100])
def test_pyramid_impossible(value):
    assert pyramid(text(value)) == "It is impossible"


@pytest.mark.parametrize(
    "value, expected",
    [  
        (8, True),
        (11, True),     
        (12, True),     
        (123, False),    
        (121, True),      
        (1234006, True),   
        (123456, False),   
        (1221, True),     
        (1331, True),
        (1241, True),      
        (4321, False),     
    ],
)
def test_is_balanced_number(value, expected):
    assert is_balanced_number(text(value)) is expected
