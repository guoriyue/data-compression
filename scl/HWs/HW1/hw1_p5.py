from typing import Tuple
from scl.core.data_encoder_decoder import DataDecoder, DataEncoder
from scl.core.data_block import DataBlock
import argparse
from scl.core.prob_dist import ProbabilityDist
from scl.utils.bitarray_utils import BitArray, uint_to_bitarray, bitarray_to_uint
from scl.compressors.huffman_coder import HuffmanEncoder, HuffmanDecoder, HuffmanTree
from scl.core.data_stream import Uint8FileDataStream
from scl.core.encoded_stream import EncodedBlockReader, EncodedBlockWriter
import pickle

# constants
BLOCKSIZE = 50_000  # encode in 50 KB blocks

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--decompress", help="decompress", action="store_true")
parser.add_argument("-i", "--input", help="input file", required=True, type=str)
parser.add_argument("-o", "--output", help="output file", required=True, type=str)


def encode_prob_dist(prob_dist: ProbabilityDist) -> BitArray:
    """Encode a probability distribution as a bit array

    Args:
        prob_dist (ProbabilityDist): probability distribution over 0, 1, 2, ..., 255
            (note that some probabilities might be missing if they are 0).

    Returns:
        BitArray: encoded bit array
    """
    prob_dict = prob_dist.prob_dict # dictionary mapping symbols to probabilities
    #########################
    # ADD CODE HERE
    # Following functions from utils can be useful to implement this
    # bits = BitArray(), bits.frombytes(byte_array), uint_to_bitarray
    # print("encode prob_dict", prob_dict)
    # {0: 0.4, 1: 0.2, 255: 0.4}
    
    # class NodeTree(object):
    #     def __init__(self, left=None, right=None):
    #         self.left = left
    #         self.right = right

    #     def children(self):
    #         return self.left, self.right

    #     def __str__(self):
    #         return self.left, self.right
        
    # def huffman_code_tree(node, binString=''):
    #     '''
    #     Function to find Huffman Code
    #     '''
    #     if not isinstance(node, NodeTree):
    #         return {node: binString}
    #     # print("node.children()", node.children())
    #     l, r = node.children()
    #     d = dict()
    #     d.update(huffman_code_tree(l, binString + '0'))
    #     d.update(huffman_code_tree(r, binString + '1'))
    #     return d


    # def make_tree(nodes):
    #     while len(nodes) > 1:
    #         (key1, c1) = nodes[-1]
    #         (key2, c2) = nodes[-2]
    #         nodes = nodes[:-2]
    #         node = NodeTree(key1, key2)
    #         nodes.append((node, c1 + c2))
    #         nodes = sorted(nodes, key=lambda x: (x[1], x[0]), reverse=True)
    #     return nodes[0][0]

    
    
    # freq = sorted(prob_dict.items(), key=lambda x: (x[1], x[0]), reverse=True)
    # node = make_tree(freq)
    # encoding = huffman_code_tree(node)
    
    # for i in encoding:
    #     print(f'{i} : {encoding[i]}')

    # # Serialize probabilities
    # prob_count = len(prob_dict)
    # encoded_probdist_bitarray = BitArray(uint_to_bitarray(prob_count, 8))
    # for i in encoding:
    #     encoded_probdist_bitarray += uint_to_bitarray(i, 8)
    #     encoded_probdist_bitarray += uint_to_bitarray(len(encoding[i]), 8)
    #     encoded_probdist_bitarray += BitArray(encoding[i])

    # print("encoded_probdist_bitarray", encoded_probdist_bitarray)
    # #########################
    
    prob_count = len(prob_dict)
    encoded_probdist_bitarray = BitArray(uint_to_bitarray(prob_count, 128))
    for i in prob_dict:
        encoded_probdist_bitarray += uint_to_bitarray(i, 8)
        encoded_probdist_bitarray += uint_to_bitarray(int(prob_dict[i] * 2**128), 128)


    print("encoded 01 length", len(encoded_probdist_bitarray.to01()))
    print("prob_dict", prob_dict)
    return encoded_probdist_bitarray


def decode_prob_dist(bitarray: BitArray) -> Tuple[ProbabilityDist, int]:
    """Decode a probability distribution from a bit array

    Args:
        bitarray (BitArray): bitarray encoding probability dist followed by arbitrary data

    Returns:
        prob_dist (ProbabilityDist): the decoded probability distribution
        num_bits_read (int): the number of bits read from bitarray to decode probability distribution
    """
    #########################
    # ADD CODE HERE
    # Following functions from utils can be useful to implement this
    # bitarray.tobytes() and bitarray_to_uint()
    
    index = 0
    prob_count = bitarray_to_uint(bitarray[index:index+128])
    index += 128
    prob_dict = {}
    for _ in range(prob_count):
        symbol = bitarray_to_uint(bitarray[index:index+8])
        index += 8
        prob = bitarray_to_uint(bitarray[index:index+128]) / 2**128
        index += 128
        prob_dict[symbol] = prob
    
    # print("prob_dict", prob_dict)
    num_bits_read = index
    print("prob_dict", prob_dict)
    prob_dist = ProbabilityDist(prob_dict)
    
    return prob_dist, num_bits_read
    
    
    
    # index = 0
    # prob_count = bitarray_to_uint(bitarray[index:index+8])
    # index += 8
    # prob_dict = {}
    # for _ in range(prob_count):
    #     symbol = bitarray_to_uint(bitarray[index:index+8])
    #     index += 8
    #     codelen = bitarray_to_uint(bitarray[index:index+8])
    #     index += 8
    #     code = bitarray[index:index+codelen]
    #     index += codelen
    #     prob_dict[symbol] = code.to01()

    # print("prob_dict", prob_dict)
    # num_bits_read = index
    # #########################

    # prob_dist = ProbabilityDist(prob_dict)
    # return prob_dist, num_bits_read


def print_huffman_tree(prob_dist):
    """Print Huffman tree after changing to ASCII symbols"""
    prob_dist = ProbabilityDist({chr(k): prob_dist.prob_dict[k] for k in prob_dist.prob_dict})
    tree = HuffmanTree(prob_dist)
    tree.print_tree()


class HuffmanEmpiricalEncoder(DataEncoder):
    def encode_block(self, data_block: DataBlock):

        # get the empirical distribution of the data block
        prob_dist = data_block.get_empirical_distribution()

        # uncomment below to print Huffman tree
        # print_huffman_tree(prob_dist)

        # create Huffman encoder for the empirical distribution
        huffman_encoder = HuffmanEncoder(prob_dist)
        # encode the data with Huffman code
        encoded_data = huffman_encoder.encode_block(data_block)
        # return the Huffman encoding prepended with the encoded probability distribution
        return encode_prob_dist(prob_dist) + encoded_data

    def encode_file(self, input_file_path: str, encoded_file_path: str, block_size: int):
        """utility wrapper around the encode function using Uint8FileDataStream
        Args:
            input_file_path (str): path of the input file
            encoded_file_path (str): path of the encoded binary file
            block_size (int): choose the block size to be used to call the encode function
        """
        # call the encode function and write to the binary file
        with Uint8FileDataStream(input_file_path, "rb") as fds:
            with EncodedBlockWriter(encoded_file_path) as writer:
                self.encode(fds, block_size=block_size, encode_writer=writer)


class HuffmanEmpiricalDecoder(DataDecoder):
    def decode_block(self, encoded_block: DataBlock):
        # first decode the probability distribution
        prob_dist, num_bits_read_prob_dist_encoder = decode_prob_dist(encoded_block)
        # now create Huffman decoder
        huffman_decoder = HuffmanDecoder(prob_dist)
        # now apply Huffman decoding
        decoded_data, num_bits_read_huffman = huffman_decoder.decode_block(
            encoded_block[num_bits_read_prob_dist_encoder:]
        )
        # verify we read all the bits provided
        assert num_bits_read_huffman + num_bits_read_prob_dist_encoder == len(encoded_block)
        return decoded_data, len(encoded_block)

    def decode_file(self, encoded_file_path: str, output_file_path: str):
        """utility wrapper around the decode function using Uint8FileDataStream
        Args:
            encoded_file_path (str): input binary file
            output_file_path (str): output (text) file to which decoded data is written
        """
        # read from a binary file and decode data and write to a binary file
        with EncodedBlockReader(encoded_file_path) as reader:
            with Uint8FileDataStream(output_file_path, "wb") as fds:
                self.decode(reader, fds)


"""
Helper function to verify roundtrip encoding and decoding of probability distribution.
"""
def helper_encode_decode_prob_dist_roundtrip(prob_dist: ProbabilityDist):
    encoded_bits = encode_prob_dist(prob_dist)
    len_encoded_bits = len(encoded_bits)
    # add some garbage bits in the end to make sure they are not consumed by the decoder
    encoded_bits += BitArray("1010")
    decoded_prob_dist, num_bits_read = decode_prob_dist(encoded_bits)
    assert decoded_prob_dist.prob_dict == prob_dist.prob_dict, "decoded prob dist does not match original"
    assert list(decoded_prob_dist.prob_dict.keys()) == list(prob_dist.prob_dict.keys()), "order of symbols changed"
    assert num_bits_read == len_encoded_bits, "All encoded bits were not consumed by the decoder"

def test_encode_decode_prob_dist():
    # try a simple probability distribution
    helper_encode_decode_prob_dist_roundtrip(ProbabilityDist({0: 0.4, 1: 0.2, 255: 0.4}))

    # try an uglier looking distribution
    helper_encode_decode_prob_dist_roundtrip(ProbabilityDist({0: 0.44156346, 1: 0.23534656, 255: 0.32308998}))

    # reorder the symbols to make sure the implementation preserves the order
    helper_encode_decode_prob_dist_roundtrip(ProbabilityDist({0: 0.4, 255: 0.4, 1: 0.2}))

    # now get a real distribution based on this file
    # load current file as a byte array and get the empirical distribution
    with open(__file__, "rb") as f:
        file_bytes = f.read()
    prob_dist = DataBlock(file_bytes).get_empirical_distribution()
    helper_encode_decode_prob_dist_roundtrip(prob_dist)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.decompress:
        decoder = HuffmanEmpiricalDecoder()
        decoder.decode_file(args.input, args.output)
    else:
        encoder = HuffmanEmpiricalEncoder()
        encoder.encode_file(args.input, args.output, block_size=BLOCKSIZE)
