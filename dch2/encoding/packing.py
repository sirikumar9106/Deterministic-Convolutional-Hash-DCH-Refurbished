"""
dch2.encoding.packing
---------------------
Bit vector manipulation and hex string conversion utilities.
"""

from typing import List


def pack_bits(hash_list: List[List[int]]) -> List[int]:
    """
    Concatenate multiple bit lists into a single flat vector.

    Parameters
    ----------
    hash_list : List[List[int]]
        List of bit vectors to concatenate in order.

    Returns
    -------
    List[int]
        Single flat binary list.
    """
    combined = []
    for bits in hash_list:
        combined.extend(bits)
    return combined


def bits_to_hex(hash_bits: List[int]) -> str:
    """
    Convert a binary bit list into a hexadecimal string.

    Groups bits into nibbles (4 bits each) and converts each
    to its hex digit. Length of output = len(hash_bits) // 4.

    A 256-bit hash produces a 64-character hex string.

    Parameters
    ----------
    hash_bits : List[int]
        Binary list. Length must be divisible by 4.

    Returns
    -------
    str
        Hexadecimal string representation.

    Raises
    ------
    ValueError
        If len(hash_bits) is not divisible by 4.
    """
    if len(hash_bits) % 4 != 0:
        raise ValueError(
            f"Hash bit length must be divisible by 4. "
            f"Got {len(hash_bits)} bits."
        )

    bit_string = ''.join(str(b) for b in hash_bits)
    hex_string = ''

    for i in range(0, len(bit_string), 4):
        nibble     = bit_string[i:i + 4]
        hex_string += hex(int(nibble, 2))[2:]

    return hex_string


def hex_to_bits(hex_string: str) -> List[int]:
    """
    Convert a hexadecimal string back into a binary bit list.

    Inverse of bits_to_hex(). Useful for loading stored hashes
    and comparing against newly generated ones.

    Parameters
    ----------
    hex_string : str
        Hexadecimal string. Each character expands to 4 bits.

    Returns
    -------
    List[int]
        Binary list of length len(hex_string) * 4.
    """
    bits = []
    for char in hex_string:
        value    = int(char, 16)
        nibble   = [(value >> (3 - i)) & 1 for i in range(4)]
        bits.extend(nibble)
    return bits