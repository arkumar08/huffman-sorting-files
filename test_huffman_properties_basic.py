from __future__ import annotations

from random import shuffle

import pytest
from hypothesis import given, assume, settings
from hypothesis.strategies import binary, integers, dictionaries, text

from compress import *

settings.register_profile("norand", settings(derandomize=True, max_examples=200))
settings.load_profile("norand")


# === Test Byte Utilities ===
# Technically, these utilities are given to you in the starter code, so these
# first 3 tests below are just intended as a sanity check to make sure that you
# did not modify these methods and are therefore using them incorrectly.
# You will not be submitting utils.py anyway, so these first three tests are
# solely for your own benefit, as a sanity check.

@given(integers(0, 255))
def test_byte_to_bits(b: int) -> None:
    """ Test that byte_to_bits produces binary strings of length 8."""
    assert set(byte_to_bits(b)).issubset({"0", "1"})
    assert len(byte_to_bits(b)) == 8


@given(text(["0", "1"], min_size=0, max_size=8))
def test_bits_to_byte(s: str) -> None:
    """ Test that bits_to_byte produces a byte."""
    b = bits_to_byte(s)
    assert isinstance(b, int)
    assert 0 <= b <= 255


@given(integers(0, 255), integers(0, 7))
def test_get_bit(byte: int, bit_pos: int) -> None:
    """ Test that get_bit(byte, bit) produces  bit values."""
    b = get_bit(byte, bit_pos)
    assert isinstance(b, int)
    assert 0 <= b <= 1


# === Test the compression code ===

@given(binary(min_size=0, max_size=1000))
def test_build_frequency_dict(byte_list: bytes) -> None:
    """ Test that build_frequency_dict returns dictionary whose values sum up
    to the number of bytes consumed.
    """
    # creates a copy of byte_list, just in case your implementation of
    # build_frequency_dict modifies the byte_list
    b, d = byte_list, build_frequency_dict(byte_list)
    assert isinstance(d, dict)
    assert sum(d.values()) == len(b)


@given(dictionaries(integers(min_value=0, max_value=255), integers(min_value=1, max_value=1000), dict_class=dict,
                    min_size=2, max_size=256))
def test_build_huffman_tree(d: dict[int, int]) -> None:
    """ Test that build_huffman_tree returns a non-leaf HuffmanTree."""
    t = build_huffman_tree(d)
    assert isinstance(t, HuffmanTree)
    assert not t.is_leaf()


@given(dictionaries(integers(min_value=0, max_value=255), integers(min_value=1, max_value=1000), dict_class=dict,
                    min_size=2, max_size=256))
def test_get_codes(d: dict[int, int]) -> None:
    """ Test that the sum of len(code) * freq_dict[code] is optimal, so it
    must be invariant under permutation of the dictionary.
    Note: This also tests build_huffman_tree indirectly.
    """
    t = build_huffman_tree(d)
    c1 = get_codes(t)
    d2 = list(d.items())
    shuffle(d2)
    d2 = dict(d2)
    t2 = build_huffman_tree(d2)
    c2 = get_codes(t2)
    assert sum([d[k] * len(c1[k]) for k in d]) == \
           sum([d2[k] * len(c2[k]) for k in d2])


@given(dictionaries(integers(min_value=0, max_value=255), integers(min_value=1, max_value=1000), dict_class=dict,
                    min_size=2, max_size=256))
def test_number_nodes(d: dict[int, int]) -> None:
    """ If the root is an interior node, it must be numbered two less than the
    number of symbols, since a complete tree has one fewer interior nodes than
    it has leaves, and we are numbering from 0.
    Note: this also tests build_huffman_tree indirectly.
    """
    t = build_huffman_tree(d)
    assume(not t.is_leaf())
    count = len(d)
    number_nodes(t)
    assert count == t.number + 2


@given(dictionaries(integers(min_value=0, max_value=255), integers(min_value=1, max_value=1000), dict_class=dict,
                    min_size=2, max_size=256))
def test_avg_length(d: dict[int, int]) -> None:
    """ Test that avg_length returns a float in the interval [0, 8], if the max
    number of symbols is 256.
    """
    t = build_huffman_tree(d)
    f = avg_length(t, d)
    assert isinstance(f, float)
    assert 0 <= f <= 8.0


@given(binary(min_size=2, max_size=1000))
def test_compress_bytes(b: bytes) -> None:
    """ Test that compress_bytes returns a bytes object that is no longer
    than the input bytes. Also, the size of the compressed object should be
    invariant under permuting the input.
    Note: this also indirectly tests build_frequency_dict, build_huffman_tree,
    and get_codes.
    """
    d = build_frequency_dict(b)
    t = build_huffman_tree(d)
    c = get_codes(t)
    compressed = compress_bytes(b, c)
    assert isinstance(compressed, bytes)
    assert len(compressed) <= len(b)
    lst = list(b)
    shuffle(lst)
    b = bytes(lst)
    d = build_frequency_dict(b)
    t = build_huffman_tree(d)
    c = get_codes(t)
    compressed2 = compress_bytes(b, c)
    assert len(compressed2) == len(compressed)


@given(binary(min_size=2, max_size=1000))
def test_tree_to_bytes(b: bytes) -> None:
    """ Test that tree_to_bytes generates a bytes representation of a postorder
    traversal of a tree's internal nodes.
    Since each internal node requires 4 bytes to represent, and there are
    1 fewer internal nodes than distinct symbols, the length of the bytes
    produced should be 4 times the length of the frequency dictionary, minus 4.
    Note: also indirectly tests build_frequency_dict, build_huffman_tree, and
    number_nodes.
    """
    d = build_frequency_dict(b)
    assume(len(d) > 1)
    t = build_huffman_tree(d)
    number_nodes(t)
    output_bytes = tree_to_bytes(t)
    dictionary_length = len(d)
    leaf_count = dictionary_length
    assert (4 * (leaf_count - 1)) == len(output_bytes)


# === Test a roundtrip conversion

@given(binary(min_size=1, max_size=1000))
def test_round_trip_compress_bytes(b: bytes) -> None:
    """ Test that applying compress_bytes and then decompress_bytes
    will produce the original text.
    """
    text = b
    freq = build_frequency_dict(text)
    assume(len(freq) > 1)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    compressed = compress_bytes(text, codes)
    decompressed = decompress_bytes(tree, compressed, len(text))
    assert text == decompressed



# MY TEST CASES

#THIS IS WORKING
def testing_build_frequency_dict():
    # Test case 1: Normal case with some repeated and unique bytes
    text = bytes([65, 66, 67, 66])
    expected = {65: 1, 66: 2, 67: 1}
    assert build_frequency_dict(text) == expected, "Test Case 1 Failed"

    # Test case 2: Empty input, should return an empty dictionary
    text = bytes([])
    expected = {}
    assert build_frequency_dict(text) == expected, "Test Case 2 Failed"

    # Test case 3: All characters are the same, should return a dictionary with one entry
    text = bytes([65, 65, 65])
    expected = {65: 3}
    assert build_frequency_dict(text) == expected, "Test Case 3 Failed"

    # Test case 4: No repeated characters, every byte has frequency 1
    text = bytes([65, 66, 67, 68, 69])
    expected = {65: 1, 66: 1, 67: 1, 68: 1, 69: 1}
    assert build_frequency_dict(text) == expected, "Test Case 4 Failed"

    # Test case 5: Large input with mixed repetitions
    text = bytes([65, 65, 66, 66, 66, 67, 67, 67, 67])
    expected = {65: 2, 66: 3, 67: 4}
    assert build_frequency_dict(text) == expected, "Test Case 5 Failed"

    # Test case 6: Single character byte
    text = bytes([255])
    expected = {255: 1}
    assert build_frequency_dict(text) == expected, "Test Case 6 Failed"

    # Test case 7: Empty string with whitespace (should handle any byte input)
    text = bytes([32, 32, 32, 65])
    expected = {32: 3, 65: 1}
    assert build_frequency_dict(text) == expected, "Test Case 7 Failed"

    print("All test cases passed! WOOHOO")


def test_build_huffman_tree():
    # Test case 1: Simple case with only two symbols
    freq = {2: 6, 3: 4}
    t = build_huffman_tree(freq)
    result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    assert t == result, "Test Case 1 Failed"

    # Test case 2: Three symbols with different frequencies
    freq = {2: 6, 3: 4, 7: 5}
    t = build_huffman_tree(freq)
    result = HuffmanTree(None, HuffmanTree(2), HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    assert t == result, "Test Case 2 Failed"

    # Test case 3: Single symbol, should create a tree with a dummy symbol
    symbol = 5
    freq = {symbol: 6}
    t = build_huffman_tree(freq)
    dummy_symbol = (symbol + 1) % 256
    dummy_tree = HuffmanTree(dummy_symbol)
    result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    assert t.left == result.left or t.right == result.left, "Test Case 3 Failed"


    # Test case 5: Large input with same frequency for all symbols
    freq = {i: 5 for i in range(100)}
    t = build_huffman_tree(freq)
    assert t is not None, "Test Case 5 Failed"


    print("All test cases passed!")

# Run the test cases

def test_get_codes():
    # Test Case 1: Simple Tree (Basic Case)
    tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    d = get_codes(tree)
    assert d == {3: "0", 2: "1"}, f"Test Case 1 Failed: {d}"

    # # Test Case 2: Tree with More Symbols
    # tree = HuffmanTree(None, HuffmanTree(4),
    #                    HuffmanTree(None, HuffmanTree(3), HuffmanTree(None, HuffmanTree(2), HuffmanTree(1))))
    # d = get_codes(tree)
    # assert d == {4: "00", 3: "01", 2: "10", 1: "11"}, f"Test Case 2 Failed: {d}"

    # Test Case 3: Single Node Tree (Edge Case)
    tree = HuffmanTree(5)  # Single node with symbol 5
    d = get_codes(tree)
    assert d == {5: ""}, f"Test Case 3 Failed: {d}"

    # # Test Case 4: Larger Tree with Multiple Levels
    # tree = HuffmanTree(None, HuffmanTree(1), HuffmanTree(None, HuffmanTree(3), HuffmanTree(4)))
    # d = get_codes(tree)
    # assert d == {1: "00", 3: "01", 4: "10", 2: "11"}, f"Test Case 4 Failed: {d}"

    # Test Case 5: Empty Tree (Edge Case)
    # Since an empty tree is invalid for Huffman, this test case shouldn't happen in practice.
    # If you're testing for an empty tree, it should raise an exception.
    try:
        tree = None  # Empty tree
        d = get_codes(tree)  # Should raise an exception
        assert False, "Test Case 5 Failed: Expected an exception for an empty tree"
    except Exception as e:
        assert isinstance(e, AttributeError), f"Test Case 5 Failed: Unexpected exception type {type(e)}"


def test_number_nodes():
    # Test case 1: Simple tree with two internal nodes
    left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    tree = HuffmanTree(None, left, right)

    number_nodes(tree)

    assert tree.left.number == 0, "Test Case 1 Failed: Left node number should be 0"
    assert tree.right.number == 1, "Test Case 1 Failed: Right node number should be 1"
    assert tree.number == 2, "Test Case 1 Failed: Root node number should be 2"

    # Test case 2: Tree with single node
    tree = HuffmanTree(5)

    number_nodes(tree)

    assert tree.number == 0, "Test Case 2 Failed: Single node tree should have number 0"

    # Test case 3: Larger tree with multiple internal nodes
    left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(4))
    right = HuffmanTree(None, HuffmanTree(5), HuffmanTree(6))
    tree = HuffmanTree(None, left, right)

    number_nodes(tree)

    assert tree.left.number == 0, "Test Case 3 Failed: Left node number should be 0"
    assert tree.right.number == 1, "Test Case 3 Failed: Right node number should be 1"
    assert tree.number == 2, "Test Case 3 Failed: Root node number should be 2"

    # Test case 4: Tree with more depth
    left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(None, HuffmanTree(4), HuffmanTree(5)))
    right = HuffmanTree(6)
    tree = HuffmanTree(None, left, right)

    number_nodes(tree)

    assert tree.left.number == 0, "Test Case 4 Failed: Left node number should be 0"
    assert tree.right.number == 1, "Test Case 4 Failed: Right node number should be 1"
    assert tree.number == 2, "Test Case 4 Failed: Root node number should be 2"

    # Test case 5: Empty tree
    tree = None

    number_nodes(tree)

    # No assertion, just ensure no error occurs for empty tree

    print("All test cases passed!")


def test_avg_length():
    # Test case 1: Simple case with a small tree
    freq = {3: 2, 2: 7, 9: 1}
    left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    right = HuffmanTree(9)
    tree = HuffmanTree(None, left, right)

    result = avg_length(tree, freq)

    # (2*2 + 7*2 + 1*1) / (2 + 7 + 1) = 1.9
    assert abs(result - 1.9) < 1e-6, f"Test Case 1 Failed: Expected 1.9, got {result}"

    # Test case 2: All symbols have the same frequency
    freq = {1: 5, 2: 5, 3: 5}
    left = HuffmanTree(None, HuffmanTree(1), HuffmanTree(2))
    right = HuffmanTree(3)
    tree = HuffmanTree(None, left, right)

    result = avg_length(tree, freq)

    # Each symbol should have a code of length 2 since they all have the same frequency
    assert abs(result - 2.0) < 1e-6, f"Test Case 2 Failed: Expected 2.0, got {result}"

    # Test case 3: One symbol with high frequency, others with low frequency
    freq = {1: 100, 2: 1, 3: 1}
    left = HuffmanTree(None, HuffmanTree(1), HuffmanTree(2))
    right = HuffmanTree(3)
    tree = HuffmanTree(None, left, right)

    result = avg_length(tree, freq)

    # The most frequent symbol should have the shortest code (length 1)
    # while the least frequent ones should have longer codes (length 2)
    # So the weighted average should be less than 2
    assert abs(result - 1.02) < 1e-6, f"Test Case 3 Failed: Expected ~1.02, got {result}"

    # Test case 4: Only one symbol in the frequency dict
    freq = {10: 1000}
    tree = HuffmanTree(10)

    result = avg_length(tree, freq)

    # Only one symbol, the average length should be 0 since there are no bits to encode
    assert abs(result - 0.0) < 1e-6, f"Test Case 4 Failed: Expected 0.0, got {result}"

    # Test case 5: Empty frequency dictionary (invalid case, should handle gracefully)
    try:
        freq = {}
        tree = None  # No tree is possible for an empty frequency dict
        result = avg_length(tree, freq)
        print("Test Case 5 Failed: Should not handle empty dictionary")
    except TypeError:
        print("Test Case 5 Passed: Correctly handled empty input")

    print("All test cases passed!")

def test_compress_bytes():
    # Test case 1: Simple case with small input
    d = {0: "0", 1: "10", 2: "11"}
    text = bytes([1, 2, 1, 0])
    result = compress_bytes(text, d)
    assert result == bytes([184]), "Test Case 1 Failed"
    assert [byte_to_bits(byte) for byte in result] == ['10111000'], "Test Case 1 Failed"

    # Test case 2: Text with multiple symbols and varying bit lengths
    text = bytes([1, 2, 1, 0, 2])
    result = compress_bytes(text, d)
    assert [byte_to_bits(byte) for byte in result] == ['10111001', '10000000'], "Test Case 2 Failed"

    # Test case 3: Single byte input (edge case)
    text = bytes([0])
    d = {0: "0"}
    result = compress_bytes(text, d)
    assert result == bytes([0]), "Test Case 3 Failed"
    assert [byte_to_bits(byte) for byte in result] == ['0'], "Test Case 3 Failed"

    # Test case 4: Input with repeated symbols
    text = bytes([2, 2, 2, 2, 2])  # multiple occurrences of symbol 2
    d = {2: "11"}
    result = compress_bytes(text, d)
    assert [byte_to_bits(byte) for byte in result] == ['11100000'], "Test Case 4 Failed"

    # Test case 5: Larger input with different symbols
    d = {0: "0", 1: "10", 2: "11", 3: "100"}
    text = bytes([1, 2, 1, 3, 0, 2, 1, 3])
    result = compress_bytes(text, d)
    assert [byte_to_bits(byte) for byte in result] == ['1011100110101000'], "Test Case 5 Failed"

    # Test case 6: Empty input (edge case)
    text = bytes([])
    d = {0: "0", 1: "10", 2: "11"}
    result = compress_bytes(text, d)
    assert result == bytes([]), "Test Case 6 Failed"

    print("All test cases passed!")

# Helper function to convert byte to bit string for validation
def byte_to_bits(byte):
    return f"{byte:08b}"

def test_tree_to_bytes():
    # Test case 1: Simple case with a small tree (two leaf nodes)
    left = HuffmanTree(3, None, None)
    right = HuffmanTree(2, None, None)
    tree = HuffmanTree(None, left, right)
    number_nodes(tree)
    result = list(tree_to_bytes(tree))
    assert result == [0, 3, 0, 2], f"Test Case 1 Failed: {result}"

    # Test case 2: Tree with one internal node and a leaf node
    left = HuffmanTree(3, None, None)
    right = HuffmanTree(5, None, None)
    tree = HuffmanTree(None, left, right)
    number_nodes(tree)
    result = list(tree_to_bytes(tree))
    assert result == [0, 3, 0, 5], f"Test Case 2 Failed: {result}"

    # Test case 3: Tree with more complex structure
    left = HuffmanTree(None, HuffmanTree(3, None, None), HuffmanTree(2, None, None))
    right = HuffmanTree(5)
    tree = HuffmanTree(None, left, right)
    number_nodes(tree)
    result = list(tree_to_bytes(tree))
    assert result == [0, 3, 0, 2, 1, 0, 0, 5], f"Test Case 3 Failed: {result}"

    # Test case 4: Tree built from a frequency dictionary
    text = b"helloworld"
    tree = build_huffman_tree(build_frequency_dict(text))
    number_nodes(tree)
    result = list(tree_to_bytes(tree))
    expected_result = [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,
                       1, 3, 1, 2, 1, 4]
    assert result == expected_result, f"Test Case 4 Failed: {result}"

    # Test case 5: Edge case with an empty tree (should return an empty byte string)
    empty_tree = None
    result = list(tree_to_bytes(empty_tree))
    assert result == [], f"Test Case 5 Failed: {result}"

    # Test case 6: Tree with only one leaf node
    tree = HuffmanTree(5)
    number_nodes(tree)
    result = list(tree_to_bytes(tree))
    assert result == [0, 5], f"Test Case 6 Failed: {result}"

    print("All test cases passed!")

def test_generate_tree_general():
    # Test case 1: Simple case with three nodes, where the root has two children
    lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), ReadNode(1, 1, 1, 0)]
    tree = generate_tree_general(lst, 2)
    expected_tree = HuffmanTree(None,
                                HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)),
                                HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    assert tree == expected_tree, f"Test Case 1 Failed: {tree}"

    # Test case 2: Tree with two nodes, where one is a leaf node and the other is an internal node
    lst = [ReadNode(0, 8, 0, 3), ReadNode(1, 1, 1, 0)]
    tree = generate_tree_general(lst, 1)
    expected_tree = HuffmanTree(None,
                                HuffmanTree(8, None, None),
                                HuffmanTree(None, HuffmanTree(1, None, None), HuffmanTree(3, None, None)))
    assert tree == expected_tree, f"Test Case 2 Failed: {tree}"

    # Test case 3: Tree with all leaf nodes, and the root node combines them
    lst = [ReadNode(0, 7, 0, 8), ReadNode(0, 4, 0, 3), ReadNode(1, 1, 1, 0)]
    tree = generate_tree_general(lst, 2)
    expected_tree = HuffmanTree(None,
                                HuffmanTree(None, HuffmanTree(4, None, None), HuffmanTree(3, None, None)),
                                HuffmanTree(None, HuffmanTree(7, None, None), HuffmanTree(8, None, None)))
    assert tree == expected_tree, f"Test Case 3 Failed: {tree}"

    # Test case 4: Edge case with only one node (a single leaf node)
    lst = [ReadNode(0, 5, 0, 0)]
    tree = generate_tree_general(lst, 0)
    expected_tree = HuffmanTree(5)
    assert tree == expected_tree, f"Test Case 4 Failed: {tree}"

    # Test case 5: Complex case with a larger tree structure
    lst = [
        ReadNode(0, 1, 0, 2),
        ReadNode(0, 3, 0, 4),
        ReadNode(1, 0, 1, 1),
        ReadNode(0, 5, 0, 6),
        ReadNode(1, 3, 1, 2),
        ReadNode(1, 4, 1, 3)
    ]
    tree = generate_tree_general(lst, 5)
    expected_tree = HuffmanTree(None,
                                HuffmanTree(None, HuffmanTree(1), HuffmanTree(2)),
                                HuffmanTree(None, HuffmanTree(3), HuffmanTree(4)))
    assert tree == expected_tree, f"Test Case 5 Failed: {tree}"

    # Test case 6: Complex tree with multiple internal nodes
    lst = [
        ReadNode(0, 5, 0, 8),
        ReadNode(0, 4, 0, 9),
        ReadNode(1, 1, 1, 0),
        ReadNode(0, 10, 0, 7),
        ReadNode(1, 3, 1, 2)
    ]
    tree = generate_tree_general(lst, 4)
    expected_tree = HuffmanTree(None,
                                HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(7, None, None)),
                                HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(8, None, None)))
    assert tree == expected_tree, f"Test Case 6 Failed: {tree}"

    print("All test cases passed!")


def test_generate_tree_postorder():
    # Test case 1: Simple case with three nodes, where the root has two children
    lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), ReadNode(1, 0, 1, 0)]
    tree = generate_tree_postorder(lst, 2)
    expected_tree = HuffmanTree(None,
                                HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)),
                                HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    assert tree == expected_tree, f"Test Case 1 Failed: {tree}"

    # Test case 2: Tree with two nodes, where one is a leaf node and the other is an internal node
    lst = [ReadNode(0, 8, 0, 3), ReadNode(1, 0, 1, 0)]
    tree = generate_tree_postorder(lst, 1)
    expected_tree = HuffmanTree(None,
                                HuffmanTree(8, None, None),
                                HuffmanTree(None, HuffmanTree(1, None, None), HuffmanTree(3, None, None)))
    assert tree == expected_tree, f"Test Case 2 Failed: {tree}"

    # Test case 3: Tree with all leaf nodes, and the root node combines them
    lst = [ReadNode(0, 7, 0, 8), ReadNode(0, 4, 0, 3), ReadNode(1, 0, 1, 0)]
    tree = generate_tree_postorder(lst, 2)
    expected_tree = HuffmanTree(None,
                                HuffmanTree(None, HuffmanTree(4, None, None), HuffmanTree(3, None, None)),
                                HuffmanTree(None, HuffmanTree(7, None, None), HuffmanTree(8, None, None)))
    assert tree == expected_tree, f"Test Case 3 Failed: {tree}"

    # Test case 4: Edge case with only one node (a single leaf node)
    lst = [ReadNode(0, 5, 0, 0)]
    tree = generate_tree_postorder(lst, 0)
    expected_tree = HuffmanTree(5)
    assert tree == expected_tree, f"Test Case 4 Failed: {tree}"

    # Test case 5: Complex case with a larger tree structure
    lst = [
        ReadNode(0, 1, 0, 2),
        ReadNode(0, 3, 0, 4),
        ReadNode(1, 0, 1, 1),
        ReadNode(0, 5, 0, 6),
        ReadNode(1, 3, 1, 2),
        ReadNode(1, 4, 1, 3)
    ]
    tree = generate_tree_postorder(lst, 5)
    expected_tree = HuffmanTree(None,
                                HuffmanTree(None, HuffmanTree(1), HuffmanTree(2)),
                                HuffmanTree(None, HuffmanTree(3), HuffmanTree(4)))
    assert tree == expected_tree, f"Test Case 5 Failed: {tree}"

    # Test case 6: Complex tree with multiple internal nodes
    lst = [
        ReadNode(0, 5, 0, 8),
        ReadNode(0, 4, 0, 9),
        ReadNode(1, 1, 1, 0),
        ReadNode(0, 10, 0, 7),
        ReadNode(1, 3, 1, 2)
    ]
    tree = generate_tree_postorder(lst, 4)
    expected_tree = HuffmanTree(None,
                                HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(7, None, None)),
                                HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(8, None, None)))
    assert tree == expected_tree, f"Test Case 6 Failed: {tree}"

    print("All test cases passed!")

def test_decompress_bytes():
    # Test case 1: Simple test case with a small string "helloworld"
    tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    number_nodes(tree)
    compressed = compress_bytes(b'helloworld', get_codes(tree))
    decompressed = decompress_bytes(tree, compressed, len(b'helloworld'))
    assert decompressed == b'helloworld', f"Test Case 1 Failed: {decompressed}"

    # Test case 2: Edge case where the input byte string is empty
    tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    number_nodes(tree)
    compressed = compress_bytes(b'', get_codes(tree))
    decompressed = decompress_bytes(tree, compressed, 0)
    assert decompressed == b'', f"Test Case 2 Failed: {decompressed}"

    # Test case 3: Single byte test case with repeated character
    tree = build_huffman_tree(build_frequency_dict(b'aaaa'))
    number_nodes(tree)
    compressed = compress_bytes(b'aaaa', get_codes(tree))
    decompressed = decompress_bytes(tree, compressed, len(b'aaaa'))
    assert decompressed == b'aaaa', f"Test Case 3 Failed: {decompressed}"

    # Test case 4: Larger text input "thisisaverylongtext"
    tree = build_huffman_tree(build_frequency_dict(b'thisisaverylongtext'))
    number_nodes(tree)
    compressed = compress_bytes(b'thisisaverylongtext', get_codes(tree))
    decompressed = decompress_bytes(tree, compressed, len(b'thisisaverylongtext'))
    assert decompressed == b'thisisaverylongtext', f"Test Case 4 Failed: {decompressed}"

    # Test case 5: Check for a case with only one symbol (e.g., 'a' repeated multiple times)
    tree = build_huffman_tree(build_frequency_dict(b'aaaaaaaa'))
    number_nodes(tree)
    compressed = compress_bytes(b'aaaaaaaa', get_codes(tree))
    decompressed = decompress_bytes(tree, compressed, len(b'aaaaaaaa'))
    assert decompressed == b'aaaaaaaa', f"Test Case 5 Failed: {decompressed}"

    # Test case 6: A random string of different characters
    text = b'abracadabra'
    tree = build_huffman_tree(build_frequency_dict(text))
    number_nodes(tree)
    compressed = compress_bytes(text, get_codes(tree))
    decompressed = decompress_bytes(tree, compressed, len(text))
    assert decompressed == text, f"Test Case 6 Failed: {decompressed}"

    # Test case 7: Checking edge case with larger size (random text)
    text = b'This is a larger test case to ensure correctness!'
    tree = build_huffman_tree(build_frequency_dict(text))
    number_nodes(tree)
    compressed = compress_bytes(text, get_codes(tree))
    decompressed = decompress_bytes(tree, compressed, len(text))
    assert decompressed == text, f"Test Case 7 Failed: {decompressed}"

    print("All test cases passed!")

def test_improve_tree():
    # Test case 1: Simple case with a small tree
    left = HuffmanTree(None, HuffmanTree(99, None, None), HuffmanTree(100, None, None))
    right = HuffmanTree(None, HuffmanTree(101, None, None), HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    tree = HuffmanTree(None, left, right)
    freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    # Before improvement, calculate average length
    avg_before = avg_length(tree, freq)
    assert abs(avg_before - 2.49) < 0.01, f"Test Case 1 Failed: {avg_before}"

    # Improve the tree
    improve_tree(tree, freq)
    # After improvement, calculate average length
    avg_after = avg_length(tree, freq)
    assert abs(avg_after - 2.31) < 0.01, f"Test Case 1 Failed: {avg_after}"

    # Test case 2: Tree with only one leaf node (edge case)
    tree = HuffmanTree(97)  # Only one node in the tree
    freq = {97: 10}
    avg_before = avg_length(tree, freq)
    assert abs(avg_before - 0) < 0.01, f"Test Case 2 Failed: {avg_before}"

    # Test case 3: Case with two leaf nodes
    left = HuffmanTree(97)
    right = HuffmanTree(98)
    tree = HuffmanTree(None, left, right)
    freq = {97: 5, 98: 3}
    avg_before = avg_length(tree, freq)
    assert abs(avg_before - 2) < 0.01, f"Test Case 3 Failed: {avg_before}"

    # Test case 4: Tree with multiple nodes where frequencies are already sorted
    left = HuffmanTree(None, HuffmanTree(101, None, None), HuffmanTree(100, None, None))
    right = HuffmanTree(None, HuffmanTree(99, None, None), HuffmanTree(98, None, None))
    tree = HuffmanTree(None, left, right)
    freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    avg_before = avg_length(tree, freq)
    assert abs(avg_before - 2.49) < 0.01, f"Test Case 4 Failed: {avg_before}"

    # Test case 5: After improvement, check that average length is decreased
    improve_tree(tree, freq)
    avg_after = avg_length(tree, freq)
    assert avg_after < avg_before, f"Test Case 5 Failed: {avg_after}"

    # Test case 6: Tree with identical frequencies
    tree = HuffmanTree(None, HuffmanTree(97), HuffmanTree(98))
    freq = {97: 10, 98: 10}
    avg_before = avg_length(tree, freq)
    assert abs(avg_before - 1) < 0.01, f"Test Case 6 Failed: {avg_before}"

    # Test case 7: Tree with more complex structure
    tree = HuffmanTree(None, HuffmanTree(None, HuffmanTree(97), HuffmanTree(99)), HuffmanTree(None, HuffmanTree(100), HuffmanTree(101)))
    freq = {97: 5, 98: 3, 99: 2, 100: 1, 101: 7}
    avg_before = avg_length(tree, freq)
    assert abs(avg_before - 2.5) < 0.01, f"Test Case 7 Failed: {avg_before}"

    # Improve tree and check average length after improvement
    improve_tree(tree, freq)
    avg_after = avg_length(tree, freq)
    assert avg_after < avg_before, f"Test Case 7 Failed: {avg_after}"

    print("All test cases passed!")


if __name__ == "__main__":
    pytest.main(["test_huffman_properties_basic.py"])
