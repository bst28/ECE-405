############################ Problem 2 ####################

# (b)
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    # Convert each byte into a single-byte bytes object
    # and try to decode it as UTF-8
    # This is incorrect because UTF-8 characters can use multiple bytes
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
