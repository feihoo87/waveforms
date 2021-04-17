import bisect
import functools
import itertools
import random
from math import ceil, floor, log, sqrt
from typing import Iterator, List, Optional, Set


def __sieve(lst: List[int], p_lst: List[int]) -> None:
    """Sieve of Eratosthenes"""
    p = next(lst)
    p_lst.append(p)
    for num in lst:
        if all(num % p != 0 for p in itertools.takewhile(
                lambda x, lim=floor(sqrt(num)): x <= lim, p_lst)):
            p_lst.append(num)


__least_primes: List[int] = [2]

__sieve(iter(range(3, 50000, 2)), __least_primes)

__primes_set: Set[int] = set(__least_primes)


def powMod(a: int, b: int, r: int) -> int:
    """
    power mod

    Args:
        a (int), b (int), r (int)

    Returns:
        int: (a ** b) % r
    """
    ans, buff = 1, a
    while b:
        if b & 1:
            ans = (ans * buff) % r
        buff = (buff * buff) % r
        b >>= 1
    return ans


def _MillerRabin(n: int, a: int) -> bool:
    """
    Miller-Rabin test with base a

    Args:
        n (int): number to test
        a (int): base

    Returns:
        bool: result
    """
    d = n - 1
    while (d & 1) == 0:
        d >>= 1
    t = powMod(a, d, n)
    while d != n - 1 and t != n - 1 and t != 1:
        t = (t * t) % n
        d <<= 1
    return t == n - 1 or (d & 1) == 1


@functools.lru_cache()
def millerRabinTest(q: int) -> bool:
    if q < 1373653:
        return all(_MillerRabin(q, a) for a in [2, 3])
    elif q < 9080191:
        return all(_MillerRabin(q, a) for a in [31, 73])
    elif q < 4759123141:
        return all(_MillerRabin(q, a) for a in [2, 7, 61])
    elif q < 2152302898747:
        return all(_MillerRabin(q, a) for a in [2, 3, 5, 7, 11])
    elif q < 3474749660383:
        return all(_MillerRabin(q, a) for a in [2, 3, 5, 7, 11, 13])
    elif q < 341550071728321:
        return all(_MillerRabin(q, a) for a in [2, 3, 5, 7, 11, 13, 17])
    elif q < 3825123056546413051:
        return all(
            _MillerRabin(q, a) for a in [2, 3, 5, 7, 11, 13, 17, 19, 23])
    elif q < 318665857834031151167461:
        return all(
            _MillerRabin(q, a)
            for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37])
    elif q < 3317044064679887385961981:
        return all(
            _MillerRabin(q, a)
            for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41])
    else:
        bases = random.sample(__primes_set, 20)
        return all(_MillerRabin(q, a) for a in bases)


def is_prime(q: int) -> bool:
    if q in __primes_set:
        return True
    if q % 2 == 0 or q % 3 == 0:
        return False
    if millerRabinTest(q):
        __primes_set.add(q)
        return True
    else:
        return False


class Primes:
    @staticmethod
    def greater_than(x: int) -> Iterator[int]:
        """
        generate primes greater than `x`

        Args:
            x (int): lower boundary

        Yields:
            Iterator[int]: primes
        """
        if x <= 2:
            yield 2
            q = 3
        elif x % 2 == 0:
            q = x + 1
        else:
            q = x
        while True:
            if is_prime(q):
                yield q
            q += 2

    @staticmethod
    def less_than(x: int) -> Iterator[int]:
        """
        generate primes less than x

        Args:
            x (int): upper boundary

        Yields:
            Iterator[int]: primes
        """
        if x <= 2:
            return
        if x == 3:
            yield 2
            return
        if x % 2 == 0:
            q = x - 1
        else:
            q = x - 2
        while True:
            if is_prime(q):
                yield q
            q -= 2
            if q == 1:
                yield 2
                break

    @staticmethod
    def take(count: int) -> Iterator[int]:
        """
        take first `count` primes
        """
        for _, q in zip(range(count), Primes.greater_than(1)):
            yield q

    def __iter__(self) -> Iterator[int]:
        yield from self.greater_than(1)

    def __contains__(self, num: int) -> bool:
        return is_prime(num)


def next_prime(x: int) -> int:
    """下一个质数"""
    if x < __least_primes[-1]:
        index = bisect.bisect(__least_primes, x)
        return __least_primes[index]
    if x % 2 == 0:
        x += 1
    else:
        x += 2
    while True:
        if is_prime(x):
            return x
        else:
            x += 2


def previous_prime(x: int) -> Optional[int]:
    """前一个质数"""
    if x <= 2:
        return None
    if x == 3:
        return 2
    if x % 2 == 0:
        x -= 1
    else:
        x -= 2
    while True:
        if is_prime(x):
            return x
        else:
            x -= 2


def _prime(n: int) -> int:
    a = floor(n * (log(n) + log(log(n)) - 1 + (log(log(n)) - 2.1) / log(n)))
    if a % 2 == 0:
        a += 1
    count = primePi(a)
    if count == n and is_prime(a):
        return a
    while True:
        a = next_prime(a)
        count += 1
        if count == n:
            return a


def prime(n: int) -> int:
    """第 n 个质数"""
    if n <= len(__least_primes):
        return __least_primes[n - 1]

    return _prime(n)


def _Phi(m: int, b: int) -> int:
    if b == 0:
        return floor(m)
    elif b == 1:
        return floor(m) - floor(m / 2)
    elif floor(m) == 0:
        return 0
    elif floor(m) in [1, 2, 3, 4]:
        return 1
    else:
        return _cached_Phi(floor(m), b)


@functools.lru_cache(maxsize=None)
def _cached_Phi(m: int, b: int) -> int:
    ret = m
    for i in range(b):
        ret -= _Phi(floor(m / prime(i + 1)), i)
    return ret


@functools.lru_cache(maxsize=None)
def _primePi(m: int) -> int:
    y = ceil(m**(1 / 3))
    n = primePi(y)

    P2 = 0
    if y % 2 == 0:
        q = y + 1
    else:
        q = y + 2
    while q <= m**(1 / 2):
        if is_prime(q):
            P2 += primePi(m / q) - primePi(q) + 1
        q += 2

    return _Phi(m, n) + n - 1 - P2


def primePi(m: int) -> int:
    if m <= __least_primes[-1]:
        return bisect.bisect(__least_primes, m)

    return _primePi(floor(m))


__all__ = ['Primes', 'prime', 'is_prime', 'next_prime', 'primePi']
