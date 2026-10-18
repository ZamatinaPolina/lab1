"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:  
    text = data.value
    return sum(1 for char in text.lower() if char in "aeiou")


def has_unique_characters(data: TextInput) -> bool: 
    text = data.value
    return len(set(text)) == len(text)


def count_one_bits(data: PositiveIntegerInput) -> int: 
    number = data.value
    return bin(number).count("1")


def multiplicative_persistence(data: PositiveIntegerInput) -> int: 
    number = data.value
    count = 0
    while number >= 10:
        product = 1
        for digit in str(number):
            product *= int(digit)
        number = product
        count += 1
    return count


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    if len(predicted) != len(expected):
        raise ValueError("Векторы должны быть одинаковой длины")
    if len(predicted) == 0:
        raise ValueError("Векторы не должны быть пустыми")
    total = sum((p - e) ** 2 for p, e in zip(predicted, expected))
    return total / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    if number == 1:
        return "(1)"
    parts = []
    divisor = 2
    while divisor * divisor <= number:
        power = 0
        while number % divisor == 0:
            number //= divisor
            power += 1
        if power == 1:
            parts.append(f"({divisor})")
        elif power > 1:
            parts.append(f"({divisor}**{power})")
        divisor += 1
    if number > 1:
        parts.append(f"({number})")
    return "".join(parts)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    total = 0
    level = 0
    while total < cube_count:
        level += 1
        total += level * level
    if total == cube_count:
        return level
    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    raise NotImplementedError  # TODO
