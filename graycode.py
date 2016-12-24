# via https://en.wikipedia.org/wiki/Gray_code#Converting_to_and_from_Gray_code
# Note: a separate bit should be used to tell a number is negative or not


def binary_to_gray(number):
    """Converts an unsigned binary number into its reflected binary gray code equivalent."""
    return number ^ (number >> 1)


def gray_to_binary(number):
    """Converts a reflected binary gray code number into its binary equivalent."""
    mask = number >> 1
    while mask != 0:
        number ^= mask
        mask >>= 1
    return number


if __name__ == "__main__":
    print(bin(binary_to_gray(2)))
