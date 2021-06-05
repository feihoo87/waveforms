from .fit import fitCircle, fitCrossPoint, fitPole, goodnessOfFit, linFit
from .prime import Primes
from .signal import getFTMatrix

try:
    from math import comb
except:

    def comb(n, k):
        """
        Return the number of ways to choose k items from n items without
        repetition and without order.

        Evaluates to n! / (k! * (n - k)!) when k <= n and evaluates to
        zero when k > n.

        Also called the binomial coefficient because it is equivalent to
        the coefficient of k-th term in polynomial expansion of the expression
        (1 + x) ** n.

        Raises TypeError if either of the arguments are not integers.
        Raises ValueError if either of the arguments are negative.
        """
        if not (isinstance(n, int) and isinstance(k, int)):
            raise TypeError('All arguments must be integers.')
        if n < 0 or k < 0:
            raise ValueError('All arguments must be nonnegative.')
        if k > n:
            return 0
        k = min(k, n - k)
        ret = 1
        for i in range(1, k + 1):
            ret *= n - i + 1
            ret //= i
        return ret
