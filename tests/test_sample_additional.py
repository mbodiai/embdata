import pytest
import numpy as np
import torch
from embdata.sample import Sample

# def test_flatten_recursive():
#     sample = Sample(
#         a=1,
#         b={
#             "c": 2,
#             "d": [3, 4]
#         },
#         e=Sample(
#             f=5,
#             g={
#                 "h": 6,
#                 "i": 7
#             }
#         )
#     )
#     flattened = Sample.flatten_recursive(sample.dump())
#     expected = [
#         ('a', 1),
#         ('b.c', 2),
#         ('b.d.0', 3),
#         ('b.d.1', 4),
#         ('e.f', 5),
#         ('e.g.h', 6),
#         ('e.g.i', 7)
#     ]
#     assert flattened == expected, f"Expected {expected}, but got {flattened}"

# def test_flatten_recursive_with_ignore():
#     sample = Sample(
#         a=1,
#         b={
#             "c": 2,
#             "d": [3, 4]
#         },
#         e=Sample(
#             f=5,
#             g={
#                 "h": 6,
#                 "i": 7
#             }
#         )
#     )
#     flattened = Sample.flatten_recursive(sample.dump(), ignore={"b"})
#     expected = [
#         ('a', 1),
#         ('e.f', 5),
#         ('e.g.h', 6),
#         ('e.g.i', 7)
#     ]
#     assert flattened == expected, f"Expected {expected}, but got {flattened}"

# def test_group_values():
#     flattened = [
#         ('a', 1),
#         ('b.c', 2),
#         ('b.d.0', 3),
#         ('b.d.1', 4),
#         ('e.f', 5),
#         ('e.g.h', 6),
#         ('e.g.i', 7)
#     ]
#     grouped = Sample.group_values(flattened, ["a", "b.c", "e.g.h"])
#     expected = {
#         "a": [1],
#         "b.c": [2],
#         "e.g.h": [6]
#     }
#     assert grouped == expected, f"Expected {expected}, but got {grouped}"

# def test_group_values_with_wildcard():
#     flattened = [
#         ('a', 1),
#         ('b.c', 2),
#         ('b.d.0', 3),
#         ('b.d.1', 4),
#         ('e.f', 5),
#         ('e.g.h', 6),
#         ('e.g.i', 7)
#     ]
#     grouped = Sample.group_values(flattened, ["a", "b.*", "e.g.h"])
#     expected = {
#         "a": [1],
#         "b.*": [2, 3, 4],
#         "e.g.h": [6]
#     }
#     assert grouped == expected, f"Expected {expected}, but got {grouped}"

# def test_group_values_with_multiple_matches():
#     flattened = [
#         ('a', 1),
#         ('b.c', 2),
#         ('b.d', 3),
#         ('b.e', 4),
#         ('c.d', 5),
#         ('c.e', 6)
#     ]
#     grouped = Sample.group_values(flattened, ["a", "b.*", "c.*"])
#     expected = {
#         "a": [1],
#         "b.*": [2, 3, 4],
#         "c.*": [5, 6]
#     }
#     assert grouped == expected, f"Expected {expected}, but got {grouped}"

# def test_flatten_recursive_with_numpy_and_torch():
#     sample = Sample(
#         a=1,
#         b=np.array([2, 3]),
#         c=torch.tensor([4, 5]),
#         d=Sample(
#             e=6,
#             f=np.array([7, 8]),
#             g=torch.tensor([9, 10])
#         )
#     )
#     flattened = Sample.flatten_recursive(sample.dump())
#     expected = [
#         ('a', 1),
#         ('b', np.array([2, 3])),
#         ('c', torch.tensor([4, 5])),
#         ('d.e', 6),
#         ('d.f', np.array([7, 8])),
#         ('d.g', torch.tensor([9, 10]))
#     ]
#     assert len(flattened) == len(expected), f"Expected length {len(expected)}, but got {len(flattened)}"
#     for (key1, val1), (key2, val2) in zip(flattened, expected):
#         assert key1 == key2, f"Expected key {key2}, but got {key1}"
#         if isinstance(val1, (np.ndarray, torch.Tensor)):
#             assert np.array_equal(val1, val2), f"Expected {val2}, but got {val1}"
#         else:
#             assert val1 == val2, f"Expected {val2}, but got {val1}"

# def test_group_values_with_nested_structure():
#     flattened = [
#         ('a', 1),
#         ('b.c', 2),
#         ('b.d.0', 3),
#         ('b.d.1', 4),
#         ('e.f', 5),
#         ('e.g.h', 6),
#         ('e.g.i', 7),
#         ('x.y.z', 8)
#     ]
#     grouped = Sample.group_values(flattened, ["a", "b.*", "e.g.*", "x.*"])
#     expected = {
#         "a": [1],
#         "b.*": [2, 3, 4],
#         "e.g.*": [6, 7],
#         "x.*": [8]
#     }
#     assert grouped == expected, f"Expected {expected}, but got {grouped}"

# def test_flatten_recursive_with_list_of_samples():
#     sample = Sample(
#         a=1,
#         b=[
#             Sample(c=2, d=3),
#             Sample(c=4, d=5)
#         ],
#         e=Sample(
#             f=6,
#             g=[
#                 Sample(h=7, i=8),
#                 Sample(h=9, i=10)
#             ]
#         )
#     )
#     flattened = Sample.flatten_recursive(sample.dump())
#     expected = [
#         ('a', 1),
#         ('b.0.c', 2),
#         ('b.0.d', 3),
#         ('b.1.c', 4),
#         ('b.1.d', 5),
#         ('e.f', 6),
#         ('e.g.0.h', 7),
#         ('e.g.0.i', 8),
#         ('e.g.1.h', 9),
#         ('e.g.1.i', 10)
#     ]
#     assert flattened == expected, f"Expected {expected}, but got {flattened}"

# def test_match_wildcard():
#     # Test cases from group_values_with_wildcard
#     assert Sample.match_wildcard("a", "a") == True
#     assert Sample.match_wildcard("b.c", "b.*") == False
#     assert Sample.match_wildcard("b.d.0", "b.*") == False
#     assert Sample.match_wildcard("b.d.1", "d") == True
#     assert Sample.match_wildcard("b.d.0", "d") == True
#     assert Sample.match_wildcard("e.g.h", "e.g.h") == True

#     # Test cases from group_values_with_nested_structure
#     assert Sample.match_wildcard("b.c", "b.*") == False
#     assert Sample.match_wildcard("e.g.h", "e.g.*") == False
#     assert Sample.match_wildcard("e.g.i", "e.g.*") == False

#     # Test cases from group_values_flatten_merge_dicts
#     assert Sample.match_wildcard("b.0.d.0", "b.*.d.*") == True
#     assert Sample.match_wildcard("b.1.d.0", "b.*.d.*") == True
#     assert Sample.match_wildcard("b.2.d.0", "b.*.d.*") == True
#     assert Sample.match_wildcard("b.0.e.g.0", "b.*.e.g.*") == True
#     assert Sample.match_wildcard("b.1.e.g.0", "b.*.e.g.*") == True
#     assert Sample.match_wildcard("b.2.e.g.0", "b.*.e.g.*") == True

#     # Test cases from group_values_nested_dicts_and_lists
#     assert Sample.match_wildcard("b.0.c", "b.*.c") == True
#     assert Sample.match_wildcard("b.1.c", "b.*.c") == True
#     assert Sample.match_wildcard("b.0.d.0", "b.*.d.*") == True
#     assert Sample.match_wildcard("b.1.d.0", "b.*.d.*") == True
#     assert Sample.match_wildcard("b.1.d.0", "b.*.d.*") == True
#     assert Sample.match_wildcard("b.2.d.0", "b.*.d.*") == True
#     assert Sample.match_wildcard("b.0.e.g.0", "b.*.e.g.*") == True
#     assert Sample.match_wildcard("b.1.e.g.0", "b.*.e.g.*") == True
#     assert Sample.match_wildcard("b.2.e.g.0", "b.*.e.g.*") == True

#     # Additional test cases for clarity
#     assert Sample.match_wildcard("b.0.c", "c") == True
#     assert Sample.match_wildcard("b.1.c", "c") == True
#     assert Sample.match_wildcard("b.0.d.0", "d") == True
#     assert Sample.match_wildcard("b.1.d.0", "d") == True


# def test_group_values_with_wildcard():
#     flattened = [
#         ('a', 1),
#         ('b.c', 2),
#         ('b.d.0', 3),
#         ('b.d.1', 4),
#         ('e.f', 5),
#         ('e.g.h', 6),
#         ('e.g.i', 7)
#     ]
#     grouped = Sample.group_values(flattened, ["a", "b.d", "e.g.h"])
#     expected = {
#         "a": [[1]],
#         "b.d": [[3, 4]],
#         "e.g.h": [[6]]
#     }
#     assert grouped == expected, f"Expected {expected}, but got {grouped}"

# def test_group_values_with_exact_match():
#     flattened = [
#         ('a.b.c', 1),
#         ('a.b.d', 2),
#         ('b.c.d', 3),
#         ('c.d.e', 4)
#     ]
#     grouped = Sample.group_values(flattened, ["a.b.c", "b.c.d", "c.d.e"])
#     expected = {
#         "a.b.c": [1],
#         "b.c.d": [3],
#         "c.d.e": [4]
#     }
#     assert grouped == expected, f"Expected {expected}, but got {grouped}"

# def test_process_groups():
#     grouped_values = {
#         "a": [1, 2, 3],
#         "b": [4, 5, 6],
#         "c": [7, 8, 9]
#     }
#     result = Sample.process_groups(grouped_values)
#     expected = [
#         [1, 4, 7],
#         [2, 5, 8],
#         [3, 6, 9]
#     ]
#     assert result == expected, f"Expected {expected}, but got {result}"

# def test_process_groups_empty():
#     grouped_values = {}
#     result = Sample.process_groups(grouped_values)
#     expected = []
#     assert result == expected, f"Expected {expected}, but got {result}"

# def test_process_groups_single_item():
#     grouped_values = {
#         "a": [1],
#         "b": [2],
#         "c": [3]
#     }
#     result = Sample.process_groups(grouped_values)
#     expected = [[1, 2, 3]]
#     assert result == expected, f"Expected {expected}, but got {result}"

# def test_process_groups_unequal_lengths():
#     grouped_values = {
#         "a": [1, 2, 3],
#         "b": [4, 5],
#         "c": [6, 7, 8, 9]
#     }
#     result = Sample.process_groups(grouped_values)
#     expected = [[1, 4, 6], [2, 5, 7], [3, None, 8], [None, None, 9]]
#     assert result == expected, f"Expected {expected}, but got {result}"

# def test_group_values_flatten_merge_dicts():
#     sample = Sample(
#         a=1,
#         b=[
#             {"c": 2, "d": [3, 4], "e": {"f": 5, "g": [6, 7]}},
#             {"c": 5, "d": [6, 7], "e": {"f": 8, "g": [9, 10]}},
#             {"c": 11, "d": [12, 13], "e": {"f": 14, "g": [15, 16]}},
#         ],
#         e=Sample(f=8, g=[{"h": 9, "i": 10}, {"h": 11, "i": 12}]),
#     )
#     flattened = Sample.flatten_recursive(sample.dump())
#     print(f"Flattened: {flattened}")
#     grouped = Sample.group_values(flattened, ["b.*.d", "b.*.e.g"])
#     print(f"Grouped: {grouped}")
#     expected = {
#         "b.*.d": [[3, 4], [6, 7], [12, 13]],
#         "b.*.e.g": [[6, 7], [9, 10], [15, 16]]
#     }
#     assert grouped == expected, f"Expected {expected}, but got {grouped}"

# def test_group_values_nested_dicts_and_lists():
#     sample = Sample(
#         a=1, b=[{"c": 2, "d": [3, 4]}, {"c": 5, "d": [6, 7]}], e=Sample(f=8, g=[{"h": 9, "i": 10}, {"h": 11, "i": 12}])
#     )
#     flattened = Sample.flatten_recursive(sample.dump())
#     print(f"Flattened: {flattened}")
#     grouped = Sample.group_values(flattened, ["c", "d"])
#     print(f"Grouped: {grouped}")
#     expected = {
#         "c": [[2], [5]],
#         "d": [[3, 4], [6, 7]]
#     }
#     assert grouped == expected, f"Expected {expected}, but got {grouped}"

# def test_group_by_simple():
#     sample = Sample(a=1, b={"c": 2, "d": [3, 4]}, e=Sample(f=5, g={"h": 6, "i": 7}))
#     flattened = Sample.flatten_recursive(sample.dump())
#     print(f"Flattened: {flattened}")
#     grouped = Sample.group_values(flattened, ["a", "b.c", "e.g.h"])
#     expected = {
#         "a": [[1]],
#         "b.c": [[2]],
#         "e.g.h": [[6]]
#     }
#     assert grouped == expected, f"Expected {expected}, but got {grouped}"

# def test_flatten_with_to_and_process_groups():
#     sample = Sample(a=1, b={"c": 2, "d": [3, 4]}, e=Sample(f=5, g={"h": 6, "i": 7}))
#     flattened = Sample.flatten_recursive(sample.dump())
#     grouped = Sample.group_values(flattened, ["a", "b.c", "e.g.h"])
#     result = Sample.process_groups(grouped)
#     expected = [[1, 2, 6]]
#     assert result == expected, f"Expected {expected}, but got {result}"

def test_flatten_merge_dicts():
    sample = Sample(
        a=1,
        b=[
            {"c": 2, "d": [3, 4], "e": {"f": 5, "g": [6, 7]}},
            {"c": 5, "d": [6, 7], "e": {"f": 8, "g": [9, 10]}},
            {"c": 11, "d": [12, 13], "e": {"f": 14, "g": [15, 16]}},
        ],
        e=Sample(f=8, g=[{"h": 9, "i": 10}, {"h": 11, "i": 12}]),
    )

    flattened = sample.flatten(to=["d", "g"], output_type="dict")
    expected = [{"d": [3, 4], "g": [6, 7]}, {"d": [6, 7], "g": [9, 10]}, {"d": [12, 13], "g": [15, 16]}]
    assert flattened == expected, f"Expected {expected}, but got {flattened}"

    flattened = sample.flatten(to=["d", "g"], output_type="list")
    expected = [[3, 4, 6, 7], [6, 7, 9, 10], [12, 13, 15, 16]]
    assert flattened == expected, f"Expected {expected}, but got {flattened}"

# def test_sample_with_nested_dicts_and_lists():
#     sample = Sample(
#         a=1, b=[{"c": 2, "d": [3, 4]}, {"c": 5, "d": [6, 7]}], e=Sample(f=8, g=[{"h": 9, "i": 10}, {"h": 11, "i": 12}])
#     )
#     flattened = sample.flatten()
#     expected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
#     assert flattened == expected, f"Expected {expected}, but got {flattened}"

#     flattened = sample.flatten(to=["c", "d"])
#     expected = [[2, 3, 4], [5, 6, 7]]

#     assert flattened == expected, f"Expected {expected}, but got {flattened}"

#     flattened = sample.flatten(to={"c", "d"}, output_type="np")
#     expected = np.array([[2, 3, 4], [5, 6, 7]])

#     flattened_dict = sample.flatten(output_type="dict")
#     expected_dict = {
#         "a": 1,
#         "b.0.c": 2,
#         "b.0.d.0": 3,
#         "b.0.d.1": 4,
#         "b.1.c": 5,
#         "b.1.d.0": 6,
#         "b.1.d.1": 7,
#         "e.f": 8,
#         "e.g.0.h": 9,
#         "e.g.0.i": 10,
#         "e.g.1.h": 11,
#         "e.g.1.i": 12,
#     }
#     assert flattened_dict == expected_dict, f"Expected {expected_dict}, but got {flattened_dict}"

#     unflattened_sample = Sample.unflatten(flattened, sample.schema())
#     assert unflattened_sample == sample, f"Expected {sample}, but got {unflattened_sample}"

#     unflattened_sample_dict = Sample.unflatten(flattened_dict, sample.schema())
#     assert unflattened_sample_dict == sample, f"Expected {sample}, but got {unflattened_sample_dict}"



if __name__ == "__main__":
    pytest.main()
