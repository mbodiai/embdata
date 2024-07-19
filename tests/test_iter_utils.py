from itertools import zip_longest
import pytest
import numpy as np
from embdata.sample import Sample

# def test_accumulate_with_to():
#     accumulator = {}
#     to = ["a", "b"]
#     full_keys = {"a": "a", "b": "b"}
    
#     accumulator = accumulate("a", 1, accumulator, to, full_keys, "allow")
#     accumulator = accumulate("b", 2, accumulator, to, full_keys, "allow")
    
#     assert accumulator == {"a": [[1]], "b": [[2]]}

# def test_accumulate_without_to():
#     accumulator = []
    
#     accumulator = accumulate("a", 1, accumulator, None, {}, "allow")
#     accumulator = accumulate("b", 2, accumulator, None, {}, "allow")
    
#     assert accumulator == [1, 2]

# def test_accumulate_non_numerical():
#     accumulator = []
    
#     accumulator = accumulate("a", 1, accumulator, None, {}, "exclude")
#     accumulator = accumulate("b", "string", accumulator, None, {}, "exclude")
    
#     assert accumulator == [1]

# def test_process_accumulated_with_to():
#     accumulated = {"a": [[1], [3]], "b": [[2], [4]]}
#     to = ["a", "b"]
    
#     result = process_accumulated(accumulated, to, "list")
#     assert result == [[1, 2], [3, 4]]
    
#     result = process_accumulated(accumulated, to, "dict")
#     assert result == [{"a": [1], "b": [2]}, {"a": [3], "b": [4]}]

# def test_process_accumulated_without_to():
#     accumulated = [1, 2, 3, 4]
    
#     result = process_accumulated(accumulated, None, "list")
#     assert result == [1, 2, 3, 4]

def test_flatten_recursive_nested_dicts_and_lists():
    obj = Sample(
        a=1,
        b=[
            {'c': 2, 'd': [3, 4], 'e': {'f': 5, 'g': [6, 7]}},
            {'c': 8, 'd': [9, 10], 'e': {'f': 11, 'g': [12, 13]}}
        ],
        h={'i': 14, 'j': [15, 16]}
    )
    result = obj.flatten()
    expected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    assert result == expected, f"Expected {expected}, but got {result}"


# def test_accum_with_to():
#     obj = Sample(
#         a=1,
#         b=[
#             {'c': 2, 'd': [3, 4]},
#             {'c': 5, 'd': [6, 7]}
#         ],
#         e={'f': 8, 'g': [{'h': 9, 'i': 10}, {'h': 11, 'i': 12}]}
#     )
#     result = obj.flatten(to=["b.*.d", "b.*.c"])
#     expected = [[2, [3, 4]], [5, [6, 7]]]
#     assert result == expected, f"Expected {expected}, but got {result}"

def test_flatten_with_to():
    obj = Sample(
        a=1,
        b=[
            {'c': 2, 'd': [3, 4]},
            {'c': 5, 'd': [6, 7]}
        ],
        e={'f': 8, 'g': [{'h': 9, 'i': 10}, {'h': 11, 'i': 12}]}
    )
    result = obj.flatten(to=["b.*.d", "b.*.c"])
    expected = [[2, [3, 4]], [5, [6, 7]]]
    assert result == expected, f"Expected {expected}, but got {result}"

# def test_flatten_with_numpy_array():
#     obj = Sample(
#         a=1,
#         b=np.array([2, 3, 4]),
#         c={'d': np.array([5, 6, 7])}
#     )
#     result = obj.flatten()
#     expected = [1, 2, 3, 4, 5, 6, 7]
#     assert result == expected, f"Expected {expected}, but got {result}"

# def test_flatten_with_to_and_output_type():
#     obj = Sample(a=1, b={"c": 2, "d": [3, 4]}, e=Sample(f=5, g={"h": 6, "i": 7}))
#     result = obj.flatten(to=["a", "b.c", "e.g.h"])
#     expected = [[1, 2, 6]]
#     assert result == expected, f"Expected {expected}, but got {result}"

#     result_dict = obj.flatten(to=["a", "b.c", "e.g.h"], output_type="dict")
#     expected_dict = [{"a": 1, "b.c": 2, "e.g.h": 6}]
#     assert result_dict == expected_dict, f"Expected {expected_dict}, but got {result_dict}"

# def test_flatten_merge_dicts():
#     sample = Sample(
#         a=1,
#         b=[
#             {"c": 2, "d": [3, 4], "e": {"f": 5, "g": [6, 7]}},
#             {"c": 5, "d": [6, 7], "e": {"f": 8, "g": [9, 10]}},
#             {"c": 11, "d": [12, 13], "e": {"f": 14, "g": [15, 16]}},
#         ],
#         e=Sample(f=8, g=[{"h": 9, "i": 10}, {"h": 11, "i": 12}]),
#     )

#     flattened = sample.flatten(to=["b.*.d", "b.*.e.g"], output_type="dict")
#     expected = [{"d": [3, 4], "g": [6, 7]}, {"d": [6, 7], "g": [9, 10]}, {"d": [12, 13], "g": [15, 16]}]
#     assert flattened == expected, f"Expected {expected}, but got {flattened}"

#     flattened = sample.flatten(to=["b.*.d", "b.*.e.g"], output_type="list")
#     expected = [[3, 4, 6, 7], [6, 7, 9, 10], [12, 13, 15, 16]]
#     assert flattened == expected, f"Expected {expected}, but got {flattened}"

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

# def test_flatten_with_to():
#     sample = Sample(a=1, b={"c": 2, "d": [3, 4]}, e=Sample(f=5, g={"h": 6, "i": 7}))
#     flattened = sample.flatten(to=["a", "b.c", "e.g.h"])
#     expected = [[1, 2, 6]]
#     assert flattened == expected, f"Expected {expected}, but got {flattened}"

if __name__ == "__main__":
    pytest.main()
