"""Shannon Tree Encoder/Decoder
HW1 Q4
"""
from typing import Any, Tuple
from scl.utils.bitarray_utils import BitArray
from scl.utils.test_utils import get_random_data_block, try_lossless_compression
from scl.compressors.prefix_free_compressors import (
    PrefixFreeEncoder,
    PrefixFreeDecoder,
    PrefixFreeTree,
)
from scl.core.prob_dist import ProbabilityDist, get_avg_neg_log_prob
import math
from scl.utils.bitarray_utils import uint_to_bitarray, bitarray_to_uint
from bitarray import bitarray
import random
from string import ascii_uppercase

class ShannonTreeEncoder(PrefixFreeEncoder):
    """
    PrefixFreeEncoder already has a encode_block function to encode the symbols once we define a encode_symbol function
    for the particular compressor.
    """

    def __init__(self, prob_dist: ProbabilityDist):
        self.prob_dist = prob_dist
        self.encoding_table = ShannonTreeEncoder.generate_shannon_tree_codebook(self.prob_dist)

    @classmethod
    def generate_shannon_tree_codebook(cls, prob_dist):
        """
        :param prob_dist: ProbabilityDist object
        :return: codebook: dictionary mapping symbols to bitarrays
        """

        # sort the probability distribution in decreasing probability
        sorted_prob_dist = ProbabilityDist.get_sorted_prob_dist(
            prob_dist.prob_dict, descending=True
        )
        codebook = {}

        ############################################################
        # ADD CODE HERE
        # NOTE: 
        # - The utility functions ProbabilityDist.neg_log_probability
        # - scl.utils.bitarray_utils.uint_to_bitarray and scl.utils.bitarray_utils.bitarry_to_uint might be useful

        def compute_length(prob):
            return math.ceil(-math.log2(prob))
        
        codelengths = [compute_length(sorted_prob_dist.prob_dict[s]) for s in sorted_prob_dist.prob_dict.keys()]

        available_codes = ['0', '1']

        for symbol, length in zip(sorted_prob_dist.prob_dict.keys(), codelengths):
            current_code = available_codes.pop(0)
            while len(current_code) < length:
                available_codes.append(current_code + '0')
                available_codes.append(current_code + '1')
                current_code = available_codes.pop(0)
                
            codebook[symbol] = current_code
        
        print("codebook", codebook)
        return codebook

    def encode_symbol(self, s):
        return self.encoding_table[s]


class ShannonTreeDecoder(PrefixFreeDecoder):

    def __init__(self, prob_dist: ProbabilityDist):
        encoding_table = ShannonTreeEncoder.generate_shannon_tree_codebook(prob_dist)
        print("encoding_table", encoding_table)
        for symbol in encoding_table.keys():
            encoding_table[symbol] = bitarray(encoding_table[symbol])
        self.tree = PrefixFreeTree.build_prefix_free_tree_from_code(encoding_table)

    def decode_symbol(self, encoded_bitarray: BitArray) -> Tuple[Any, BitArray]:
        decoded_symbol, num_bits_consumed = self.tree.decode_symbol(encoded_bitarray)
        return decoded_symbol, num_bits_consumed


class ShannonTableDecoder(PrefixFreeDecoder):

    def __init__(self, prob_dist: ProbabilityDist):
        self.decoding_table, self.codelen_table, self.max_codelen = self.create_decoding_table(prob_dist)

    @staticmethod
    def create_decoding_table(prob_dist: ProbabilityDist):
        """
        :param prob_dist: ProbabilityDist object
        :return:
            decoding_table: dictionary mapping bitarrays to symbols
            codelen_table: dictionary mapping symbols to code-length
            max_codelen: maximum code-length of any symbol in the codebook
        """
        # create the encoding table
        encoding_table = ShannonTreeEncoder.generate_shannon_tree_codebook(prob_dist)
        ############################################################
        # ADD CODE HERE
        # NOTE:
        # - The utility functions ProbabilityDist.neg_log_probability
        # - scl.utils.bitarray_utils.uint_to_bitarray and scl.utils.bitarray_utils.bitarry_to_uint might be useful
        max_codelen = 0
        for symbol in encoding_table.values():
            max_codelen = max(max_codelen, len(symbol))
        padding_table = {}
        for length in range(1, max_codelen):  # This will generate padding for lengths 1, 2, and 3.
            padding_table[length] = [bin(i)[2:].zfill(length) for i in range(2**length)]
        decoding_table = {}
        codelen_table = {x: len(encoding_table[x]) for x in encoding_table.keys()}
        for symbol, code in encoding_table.items():
            padding_length = max_codelen - len(code)
            if padding_length == 0:
                decoding_table[code] = symbol
            else:
                for padding in padding_table[padding_length]:
                    decoding_table[code + padding] = symbol
        print("decoding_table", decoding_table)
        print("codelen_table", codelen_table)
        print("max_codelen", max_codelen)
        ############################################################

        return decoding_table, codelen_table, max_codelen

    def decode_symbol(self, encoded_bitarray: BitArray) -> Tuple[Any, BitArray]:
        # get the padded codeword to be decoded
        padded_codeword = encoded_bitarray[:self.max_codelen]
        if len(encoded_bitarray) < self.max_codelen:
            padded_codeword = padded_codeword + "0" * (self.max_codelen - len(encoded_bitarray))
        
        decoded_symbol = self.decoding_table[padded_codeword.to01()]
        num_bits_consumed = self.codelen_table[decoded_symbol]
        # print("padded_codeword.to01(), num_bits_consumed", padded_codeword.to01(), num_bits_consumed)
        return decoded_symbol, num_bits_consumed


def test_shannon_tree_coding_specific_case():
    # NOTE -> this test must succeed with your implementation
    ############################################################
    # Add the computed expected codewords for distributions presented in part 1 to these list to improve the test
    ############################################################

    distributions = [
        ProbabilityDist({"A": 0.5, "B": 0.5}),
        ProbabilityDist({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}),
        ProbabilityDist({"A": 0.5, "B": 0.25, "C": 0.12, "D": 0.13}),
        ProbabilityDist({"A": 0.9, "B": 0.1})
    ]
    
    expected_codewords = [
        {"A": "0", "B": "1"},
        {"A": "00", "B": "01", "C": "10", "D": "11"},
        {"A": "0", "B": "10", "C": "1110", "D": "110"},
        {"A": "0", "B": "1000"}
    ]
    
    def test_encoded_symbol(prob_dist, expected_codeword_dict):
        """
        test if the encoded symbol is as expected
        :type prob_dist: ProbabilityDist object
        """
        encoder = ShannonTreeEncoder(prob_dist)
        for s in prob_dist.prob_dict.keys():
            assert encoder.encode_symbol(s) == expected_codeword_dict[s]

    for i, prob_dist in enumerate(distributions):
        test_encoded_symbol(prob_dist, expected_codeword_dict=expected_codewords[i])


def generate_distribution(num_chars):
    while True:
        weights = [random.random() for _ in range(num_chars)]
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        
        # avoid generating probabilities that are too small
        # for simplicity, we just random generate again
        flag = True
        for prob in probabilities:
            if prob < 1e-6:
                flag = False
                break
        if flag:
            return dict(zip(ascii_uppercase[:num_chars], probabilities))


def test_shannon_tree_coding_end_to_end():
    NUM_SAMPLES = 2000
    distributions = [
        ProbabilityDist({"A": 0.5, "B": 0.5}),
        ProbabilityDist({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}),
        ProbabilityDist({"A": 0.5, "B": 0.25, "C": 0.12, "D": 0.13}),
        ProbabilityDist({"A": 0.9, "B": 0.1})
    ]
    # # more test cases
    # for t in range(100):
    #     for i in range(1, 27):
    #         distributions.append(ProbabilityDist(generate_distribution(i)))
    def test_end_to_end(prob_dist, num_samples):
        """
        Test if decoding of (encoded symbol) results in original
        """
        # generate random data
        data_block = get_random_data_block(prob_dist, num_samples, seed=0)

        # create encoder decoder
        encoder = ShannonTreeEncoder(prob_dist)
        decoder_tree = ShannonTreeDecoder(prob_dist)
        # decoder_table = ShannonTableDecoder(prob_dist)

        # perform compression
        is_lossless, encode_len, _ = try_lossless_compression(data_block, encoder, decoder_tree)
        assert is_lossless, "Lossless compression failed"

        # avg_log_prob should be close to the avg_codelen
        avg_log_prob = get_avg_neg_log_prob(prob_dist, data_block)
        avg_codelen = encode_len / data_block.size
        assert avg_codelen <= (avg_log_prob + 1), "avg_codelen should be within 1 bit of mean_neg_log_prob"
        print(f"Shannon-tree coding: avg_log_prob={avg_log_prob:.3f}, avg codelen: {avg_codelen:.3f}")

    for i, prob_dist in enumerate(distributions):
        test_end_to_end(prob_dist, NUM_SAMPLES)


def test_shannon_table_coding_end_to_end():
    NUM_SAMPLES = 2000
    distributions = [
        ProbabilityDist({"A": 0.5, "B": 0.5}),
        ProbabilityDist({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}),
        ProbabilityDist({"A": 0.5, "B": 0.25, "C": 0.12, "D": 0.13}),
        ProbabilityDist({"A": 0.9, "B": 0.1})
    ]
    # # more test cases
    # for t in range(100):
    #     for i in range(1, 27):
    #         distributions.append(ProbabilityDist(generate_distribution(i)))

    def test_end_to_end(prob_dist, num_samples):
        """
        Test if decoding of (encoded symbol) results in original
        """
        # generate random data
        data_block = get_random_data_block(prob_dist, num_samples, seed=0)

        # create encoder decoder
        encoder = ShannonTreeEncoder(prob_dist)
        decoder_table = ShannonTableDecoder(prob_dist)

        # perform compression
        is_lossless, encode_len, _ = try_lossless_compression(data_block, encoder, decoder_table)
        # print("data_block", data_block.data_list)
        print("is_lossless, encode_len", is_lossless, encode_len)
        assert is_lossless, "Lossless compression failed"

        # avg_log_prob should be close to the avg_codelen
        avg_log_prob = get_avg_neg_log_prob(prob_dist, data_block)
        avg_codelen = encode_len / data_block.size
        assert avg_codelen <= (avg_log_prob + 1), "avg_codelen should be within 1 bit of mean_neg_log_prob"
        print(f"Shannon-tree coding: avg_log_prob={avg_log_prob:.3f}, avg codelen: {avg_codelen:.3f}")

    for i, prob_dist in enumerate(distributions):
        test_end_to_end(prob_dist, NUM_SAMPLES)
