#!usr/bin/env python3
""" A small implementation of binary reduction.
"""

from collections.abc import Set
from typing import TypeVar, Callable, Iterable, Iterator

V = TypeVar('V')

def binary_reduction(
        input : Set[V], 
        predicate : Callable[[Set[V]], bool], 
        progression : Callable[[Iterable[Set[V]], Iterable[V]], Iterator[Set[V]]],
        ) -> Set[V]:
    """A small implementation of binary reduction.


    >>> def debug(x): 
    ...     print(list(x), res := x >= {4, 5})
    ...     return res

    >>> binary_reduction(set(range(0,10)), debug, dumb_progression)
    [] False
    [0, 1, 2, 3, 4] False
    [0, 1, 2, 3, 4, 5, 6, 7] True
    [0, 1, 2, 3, 4, 5, 6] True
    [0, 1, 2, 3, 4, 5] True
    [5] False
    [0, 1, 5] False
    [0, 1, 2, 3, 5] False
    [4, 5] True
    frozenset({4, 5})

    Parameters
    ----------
    input : Set[V]
        A set of input variables

    predicate : Set[V] -> bool
        A function that returns true or false depending on the input set.

    progression : Iterable[Set[V]] x Iterable[V] -> list[Set[v]]
        A function that a list of sets containing the variables.

    Returns
    -------
    A small set of variables that make the predicate true.

    """
    learned_sets = set()
    d = list(progression(frozenset(learned_sets), input))

    def successful_prefix_union(i):
        return predicate(prefix_union(d, i))

    while not predicate(d[0]):
        r = find_smallest(successful_prefix_union, 0, len(d))
        learned_sets.add(d[r])
        d = list(progression(frozenset(learned_sets), prefix_union(d, r)))

    return d[0];

def prefix_union(d : list[Set[V]], i : int) -> Set[V]:
    """Finds the union of all prefixes upto and including i."""
    return frozenset().union(*d[0:i+1])

def dumb_progression(learned_sets, items): 
    """ This progression calculator is very dumb, and only does bare minimal to be 
    a valid progression.

    You should write your progression so that all prefixes represent valid inputs.

    A progression needs to contain at least one element from each learned set
    in the first set of the progression, and then the rest of the the items exactly once
    in the remaining sets.
    """
    chosen = set()
    for l in learned_sets:
        chosen.add(min(l))
    yield frozenset(chosen)
    for i in items:
       if i in chosen: continue
       chosen.add(i)
       yield frozenset({i})

def find_smallest(predicate : Callable[[int], bool], min : int, max : int) -> int:
    """Do a binary search to find the smallest value in the range were the
    predicate is true.

    >>> find_smallest((lambda i: i >= 10), 0, 20)
    10

    >>> find_smallest((lambda i: i >= 10), 0, 10)
    9

    Find a the first name that order before bet
    >>> names = ["a", "b", "be", "c"]
    >>> names[find_smallest((lambda i: names[i] >= "ba"), 0, len(names))]
    'be'

    Parameters
    ----------
    predicate : Callable[[int], bool]
        A function that returns true if the value or higher is true.

    min : int
        The smallest index in the range (inclusive).

    max : int
        The largest index in the range (exclusive).

    Returns
    -------
    The smallest index in the range.

    """
    max -= 1 # make max non-inclusive
    while (min < max):
        if predicate(mid := (min + max >> 1)): 
            max = mid
        else: 
            min = mid + 1
    return min

if __name__ == "__main__":
    import doctest
    doctest.testmod()
