import re

def isprime(n):
    return re.compile(r'^1?$|^(11+)\1+$').match('1' * n) is None


if __name__ == '__main__':
    x = 3333

    primes = [str(x).ljust(100, "0") for x in range(x) if isprime(x)]

    for prime in primes:
        print("{0:b}".format(prime))