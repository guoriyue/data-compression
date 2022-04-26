"""Fano Coding
Shannon coding is a basic prefix free coding. There is some confusion in literature about Shannon, Fano and
Shannon-Fano codes, e.g. see Naming section of https://en.wikipedia.org/wiki/Shannon–Fano_coding.

This document uses Fano coding as described in the wiki article above.
"""
from typing import Any, Tuple
from utils.bitarray_utils import BitArray
from utils.test_utils import get_random_data_block, try_lossless_compression
from compressors.prefix_free_compressors import PrefixFreeTree, PrefixFreeEncoder, PrefixFreeDecoder, BinaryNode
from core.prob_dist import ProbabilityDist


class FanoTree(PrefixFreeTree):
    def __init__(self, prob_dist):
        super().__init__(prob_dist)
        # sort the probability distribution
        self.sorted_prob_dist = ProbabilityDist.get_sorted_prob_dist(prob_dist.prob_dict)
        # initialize root node of Fano Tree
        self.root_node = BinaryNode(id=None)

        # build tree returns the root node
        super().__init__(root_node=self.build_fano_tree(self.root_node, self.sorted_prob_dist))


    @staticmethod
    def criterion(dict_item):
        """used for finding the symbol at which cumulative probability is nearest to 0.5
        This symbol is used to split between left and right trees of Fano. We split right starting this symbol
        to get most balanced tree.
        """
        key, value = dict_item
        return abs(value - 0.5)

    @staticmethod
    def split_prob_dist_into_two(norm_sort_prob_dist):
        """
        Given a normalized and sorted probability distribution, split it into two sets of approximately equal
        probabilities as described in Fano's algorithm.
        """
        # Get cumulative probability dict
        cumulative_prob_dict = norm_sort_prob_dist.cumulative_prob_dict

        # Split the tree into left and right nodes based on cumulative probability and recursively build
        # FanoTree for these trees
        recursive_prob_dicts = {
            'left': {},
            'right': {}
        }

        # all alphabets which are nearest to 0.5 cumulative probability in sorted list are put to left tree and rest
        # to right, such that we have a most balanced probabilistic tree subsets.
        min_diff_code = min(cumulative_prob_dict.items(), key=FanoTree.criterion)[0]
        curr_dict = recursive_prob_dicts['left']
        for s, cum_prob in cumulative_prob_dict.items():
            if s == min_diff_code:
                curr_dict = recursive_prob_dicts['right']
            curr_dict.update({s: norm_sort_prob_dist.probability(s)})

        return recursive_prob_dicts


    @staticmethod
    def build_fano_tree(root_node, norm_sort_prob_dist) -> BinaryNode:
        """recursively build Fano Tree"""
        # split the symbols into left and right half symbols such that the probability weights in each set is
        # most balanced
        recursive_prob_dicts = FanoTree.split_prob_dist_into_two(norm_sort_prob_dist)

        # Call recursion
        # if only 1 symbol in either left or right tree, just assign it as a child and we don't have to call recursion
        for branch, curr_dict in recursive_prob_dicts.items():
            norm_prob_dict = ProbabilityDist.normalize_prob_dict(curr_dict)

            # if pointer is None, the object is not updated
            # https://stackoverflow.com/questions/55777748/updating-none-value-does-not-reflect-in-the-object
            if root_node.right_child is None: root_node.right_child = BinaryNode(id=None)
            if root_node.left_child is None: root_node.left_child = BinaryNode(id=None)

            child = root_node.left_child if branch == 'left' else root_node.right_child
            if len(curr_dict) == 1:
                child.id = list(curr_dict)[0]
            else:
                child.id = None
                FanoTree.build_fano_tree(child, norm_prob_dict)

        return root_node


class FanoEncoder(PrefixFreeEncoder):
    """
    PrefixFreeEncoder already has a encode_block function to encode the symbols once we define a encode_symbol function
    for the particular compressor.
    PrefixFreeTree provides get_encoding_table given a PrefixFreeTree
    """

    def __init__(self, prob_dist: ProbabilityDist):
        tree = FanoTree(prob_dist)
        self.encoding_table = tree.get_encoding_table()

    def encode_symbol(self, s):
        return self.encoding_table[s]


class FanoDecoder(PrefixFreeDecoder):
    """
    PrefixFreeDecoder already has a decode_block function to decode the symbols once we define a decode_symbol function
    for the particular compressor.
    PrefixFreeTree provides decode_symbol given a PrefixFreeTree
    """

    def __init__(self, prob_dist: ProbabilityDist):
        self.tree = FanoTree(prob_dist)

    def decode_symbol(self, encoded_bitarray: BitArray) -> Tuple[Any, BitArray]:
        decoded_symbol, num_bits_consumed = self.tree.decode_symbol(encoded_bitarray)
        return decoded_symbol, num_bits_consumed


def test_fano_coding():
    NUM_SAMPLES = 2000
    distributions = [
        ProbabilityDist({"A": 0.5, "B": 0.5}),
        ProbabilityDist({"A": 0.3, "B": 0.3, "C": 0.4}),
        ProbabilityDist({"A": 0.5, "B": 0.25, "C": 0.12, "D": 0.13}),
        ProbabilityDist({"A": 0.9, "B": 0.1})
    ]
    expected_codewords = [
        {"A": BitArray('0'), "B": BitArray('1')},
        {"A": BitArray('10'), "B": BitArray('11'), "C": BitArray('0')},
        {"A": BitArray('0'), "B": BitArray('10'), "C": BitArray('111'), "D": BitArray('110')},
        {"A": BitArray('0'), "B": BitArray('1')}
    ]

    def test_end_to_end(prob_dist, num_samples):
        """
        Test if decoding of (encoded symbol) results in original
        """
        # generate random data
        data_block = get_random_data_block(prob_dist, num_samples, seed=0)

        # create encoder decoder
        encoder = FanoEncoder(prob_dist)
        decoder = FanoDecoder(prob_dist)

        # perform compression
        is_lossless, _, _ = try_lossless_compression(data_block, encoder, decoder)
        assert is_lossless, "Lossless compression failed"

    def test_encoded_symbol(prob_dist, expected_codeword_dict):
        """
        test if the encoded symbol is as expected
        """
        encoder = FanoEncoder(prob_dist)
        for s in prob_dist.prob_dict.keys():
            assert encoder.encode_symbol(s) == expected_codeword_dict[s]

    for i, prob_dist in enumerate(distributions):
        test_end_to_end(prob_dist, NUM_SAMPLES)
        test_encoded_symbol(prob_dist, expected_codeword_dict=expected_codewords[i])
