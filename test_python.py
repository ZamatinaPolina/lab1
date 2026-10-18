from types import SimpleNamespace

import pytest

from python_basics import count_vowels
from python_basics import has_unique_characters


def text(value):
    return SimpleNamespace(value=value)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("", 0),            # пустая строка
        ("bcdfg", 0),       # гласных нет
        ("aeiou", 5),       # все гласные
        ("AEIOU", 5),       # заглавные
        ("Hello World", 3), # обычный пример
        ("aaaa", 4),        # повторы считаются
        ("12!5:@6$%", 0),      # цифры и спецсимволы
        ("a e i o u", 5),       # гласные с пробелами
    ],
)
def test_count_vowels(value, expected):
    assert count_vowels(text(value)) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("", True),       # пустая строка
        ("a", True),      # один символ
        ("abc", True),    # все разные
        ("aba", False),   # повтор
        ("aa", False),    # только повторы
        ("aA", True),     # регистр различается
    ],
)
def test_has_unique_characters(value, expected):
    assert has_unique_characters(text(value)) is expected
