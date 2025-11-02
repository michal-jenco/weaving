from random import random

def generate_fibonacci_seq(length: int) -> list[int]:
    seq = [1, 1]

    for i in range(length):
        new = seq[-1] + seq[-2]
        seq.append(new)
    return seq

def list_as_binary(input: list[int]) -> list[str]:
    output = []

    for item in input:
        output.append("{0:b}".format(item))
    return output

def pad_str_list(input: list[str]) -> list[str]:
    output = []
    max_length = len(input[-1])

    for item in input:
        output.append(item.rjust(max_length, "0"))
    return output

def pad_str_list_rand_left_right(input: list[str]) -> list[str]:
    output = []
    max_length = len(input[-1])

    for item in input:
        if random() > .5:
            output.append(item.rjust(max_length, "0"))
        else:
            output.append(item.ljust(max_length, "0"))
    return output

if __name__ == '__main__':
    fib_list = generate_fibonacci_seq(500)
    print(fib_list)
    list_fib = list_as_binary(fib_list)
    print(list_fib)
    list_fib = pad_str_list_rand_left_right(list_fib)
    print(list_fib)

    for item in list_fib:
        print(item)
