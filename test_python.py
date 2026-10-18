from types import SimpleNamespace

import pytest

from python_basics import count_vowels


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
