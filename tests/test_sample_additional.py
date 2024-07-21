import pytest
from embdata.sample import Sample

def test_flatten_recursive():
    sample = Sample(
        a=1,
        b={
            "c": 2,
            "d": [3, 4]
        },
        e=Sample(
            f=5,
            g={
                "h": 6,
                "i": 7
            }
        )
    )
    flattened = Sample.flatten_recursive(sample.dump())
    expected = [
        ('a', 1),
        ('b.c', 2),
        ('b.d.0', 3),
        ('b.d.1', 4),
        ('e.f', 5),
        ('e.g.h', 6),
        ('e.g.i', 7)
    ]
    assert flattened == expected, f"Expected {expected}, but got {flattened}"

def test_flatten_recursive_with_ignore():
    sample = Sample(
        a=1,
        b={
            "c": 2,
            "d": [3, 4]
        },
        e=Sample(
            f=5,
            g={
                "h": 6,
                "i": 7
            }
        )
    )
    flattened = Sample.flatten_recursive(sample.dump(), ignore={"b"})
    expected = [
        ('a', 1),
        ('e.f', 5),
        ('e.g.h', 6),
        ('e.g.i', 7)
    ]
    assert flattened == expected, f"Expected {expected}, but got {flattened}"

def test_group_values():
    flattened = [
        ('a', 1),
        ('b.c', 2),
        ('b.d.0', 3),
        ('b.d.1', 4),
        ('e.f', 5),
        ('e.g.h', 6),
        ('e.g.i', 7)
    ]
    grouped = Sample.group_values(flattened, ["a", "b.c", "e.g.h"])
    expected = {
        "a": [1],
        "b.c": [2],
        "e.g.h": [6]
    }
    assert grouped == expected, f"Expected {expected}, but got {grouped}"

def test_group_values_with_wildcard():
    flattened = [
        ('a', 1),
        ('b.c', 2),
        ('b.d.0', 3),
        ('b.d.1', 4),
        ('e.f', 5),
        ('e.g.h', 6),
        ('e.g.i', 7)
    ]
    grouped = Sample.group_values(flattened, ["a", "b.*", "e.g.h"])
    expected = {
        "a": [1],
        "b.*": [2, 3, 4],
        "e.g.h": [6]
    }
    assert grouped == expected, f"Expected {expected}, but got {grouped}"

def test_group_values_with_multiple_matches():
    flattened = [
        ('a', 1),
        ('b.c', 2),
        ('b.d', 3),
        ('b.e', 4),
        ('c.d', 5),
        ('c.e', 6)
    ]
    grouped = Sample.group_values(flattened, ["a", "b.*", "c.*"])
    expected = {
        "a": [1],
        "b.*": [2, 3, 4],
        "c.*": [5, 6]
    }
    assert grouped == expected, f"Expected {expected}, but got {grouped}"

if __name__ == "__main__":
    pytest.main()
