from types import SimpleNamespace

import pytest

from python_basics import count_vowels
from python_basics import has_unique_characters
from python_basics import count_one_bits

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
    assert count_one_bits(SimpleNamespace(value=value)) == expected
