"""
dch2.encoding — encoding layer.
    quantizer.py — overlapping quadrant local median binarization
    packing.py   — bit concatenation and hex conversion
"""
from dch2.encoding.quantizer import quantize
from dch2.encoding.packing import pack_bits, bits_to_hex, hex_to_bits