from types import SimpleNamespace

import pytest

from python_basics import (
    count_vowels,
    has_unique_characters,
    count_one_bits,
    multiplicative_persistence,
    prime_factorization,
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
