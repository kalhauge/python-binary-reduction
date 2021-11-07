#!usr/bin/env python3
""" A small implementation of binary reduction.
"""

from collections.abc import Set
from dataclasses import dataclass
from typing import TypeVar, Callable, Iterable, Iterator, Tuple


V = TypeVar('V')

def binary_reduction(
        input : Set[V], 
        predicate : Callable[[Set[V]], bool], 
        progression : Callable[[Iterable[Set[V]], Iterable[V]], Iterator[Set[V]]],
        ) -> Set[V]:
    """A small implementation of binary reduction.

    >>> binary_reduction(
    ...     set(range(0,10)), 
    ...     debug_predicate(lambda x: x >= {3, 4}), 
    ...     debug_progression(dumb_progression)
    ... )
    == Progression 0
    == L = [], D = [[], [0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
    0 [] False
    1 [0, 1, 2, 3, 4] True
    2 [0, 1] False
    3 [0, 1, 2, 3] False
    == Progression 1
    == L = [[4]], D = [[4], [0], [1], [2], [3]]
    4 [4] False
    5 [0, 1, 4] False
    6 [0, 1, 2, 4] False
    == Progression 2
    == L = [[3], [4]], D = [[3, 4], [0], [1], [2]]
    7 [3, 4] True
    frozenset({3, 4})

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

    while len(d) > 0 and not predicate(d[0]):
        r = find_smallest(successful_prefix_union, 1, len(d))
        learned_sets.add(d[r])
        d = list(progression(frozenset(learned_sets), prefix_union(d, r)))

    return d[0];

def debug_predicate(fn, max_iterations=None): 
    """ Debug a predicate """
    counter = 0
    def predicate(x):
        nonlocal counter
        if max_iterations and counter >= max_iterations: 
            raise Exception("Out of invocations of the predicate")
        print(counter, list(x), (res := fn(x)))
        counter += 1
        return res

    return predicate;

def prefix_union(d : list[Set[V]], i : int) -> Set[V]:
    """Finds the union of all prefixes upto and including i."""
    return frozenset().union(*d[0:i+1])

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

# Here on lies progressions.

def debug_progression(fn):
    counter = 0
    def progression(learned_sets, items):
        nonlocal counter
        d = list(fn(learned_sets, items))
        print(f"== Progression {counter}")
        print(f"== I = {list(items)}")
        print(f"== L = {[list(l) for l in learned_sets]}"
              f", D = {[list(di) for di in d]}")
        counter += 1
        return d
    return progression

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

@dataclass
class Graph:
    """ A very simple graph represented using an adjecency list, and node
    ides compressed in the range from 0 to max size.
    """
    neighbors : list[list[int]]

    def transpose(self):
        """ Create a transposed graph. Ei. turn all the arrows in the 
        oppesite direction. 

        >>> Graph([[0, 1], [2], [0]]).transpose()
        Graph(neighbors=[[0, 2], [0], [1]])

        """
        transposed = [[] for _ in self.neighbors]
        for i, outs in enumerate(self.neighbors):
            for j in outs: 
                transposed[j].append(i)
        return Graph(transposed)

    def postorder(self, nodes : Iterable[int], visited : list[bool] = None):
        """ Do a depth first serach from the nodes and yield the nodes visited
        in post order. This means that each node in the depth first search tree
        is yielded when all its children have been reported.


        >>> list(Graph([[0, 1], [2], [0]]).postorder([0]))
        [2, 1, 0]

        >>> list(Graph([[0, 1], [2], [0]]).postorder([2]))
        [1, 0, 2]

        Parameters
        ----------

        nodes : Iterator[int]
            An iterator of the nodes that should be part of the search. 
            For a single depth-first search simply used a singleton list

        visited : list[bool] (optional)
            A list that records if a node has been visited or not. Will be 
            generated as none is visited if none is given.

        Yields
        ------
        Nodes reachable from the nodes in postorder.

        """
        visited = visited or [False] * len(self.neighbors)
        stack : list[Tuple[int, bool]] = list(reversed([ (n, False) for n in nodes ]))
        while stack:
            i, post = stack.pop()
            if post: yield i; continue
            if visited[i]: continue
            visited[i] = True
            stack.append((i, True))
            stack.extend((j, False) for j in self.neighbors[i])

    def nodes(self):
        """Yields the nodes in the graph."""
        yield from range(0, len(self.neighbors))

def graph_progression(graph : Graph):
    """ A graph based progression; which takes into account an adjecency list
    of graph dependencies. Here an item A depends on another B, if we know if 
    A is in the output so must B.

    Here is the graph from the original "Binary Reduction of Dependency Graphs" 
    paper.

    >>> example_1 = Graph(
    ...   [ []                     #  0
    ...   , [2, 4, 7]              #  1
    ...   , [2, 7]                 #  2
    ...   , [1, 7]                 #  3
    ...   , [7]                    #  4
    ...   , [3, 5, 6]              #  5
    ...   , [4, 5, 7]              #  6
    ...   , []                     #  7
    ...   , [7, 9, 10, 11, 12]     #  8
    ...   , [10]                   #  9
    ...   , [8]                    # 10
    ...   , [13]                   # 11
    ...   , [13]                   # 12
    ...   , [7, 10, 14]            # 13
    ...   , [8, 10, 13]            # 14
    ...   , [7, 8, 10, 12, 13, 16] # 15
    ...   , [7, 12, 13, 15]        # 16
    ...   ]
    ...   )

    The closures look a little different than the results from the paper. 

    >>> for di in graph_progression(example_1)([], example_1.nodes()):
    ...     print(list(di))
    []
    [7]
    [8, 9, 10, 11, 12, 13, 14]
    [16, 15]
    [4]
    [2]
    [1]
    [3]
    [5, 6]
    [0]

    However, the final results are the same:

    >>> binary_reduction(
    ...     example_1.nodes(), 
    ...     debug_predicate(lambda x: x >= {1}), 
    ...     debug_progression(graph_progression(example_1))
    ... )
    == Progression 0
    == L = [], D = [[], [7], [8, 9, 10, 11, 12, 13, 14], [16, 15], [4], [2], [1], [3], [5, 6], [0]]
    0 [] False
    1 [4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] False
    2 [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] True
    3 [1, 2, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] True
    4 [2, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] False
    == Progression 1
    == L = [[1]], D = [[1, 2, 4, 7]]
    5 [1, 2, 4, 7] True
    frozenset({1, 2, 4, 7})

    And, for a bug in {1, 12}:

    >>> binary_reduction(
    ...     example_1.nodes(), 
    ...     debug_predicate(lambda x: x >= {1, 12}), 
    ...     debug_progression(graph_progression(example_1))
    ... )
    == Progression 0
    == I = []
    == L = [], D = [[], [7], [8, 9, 10, 11, 12, 13, 14], [16, 15], [4], [2], [1], [3], [5, 6], [0]]
    0 [] False
    1 [2, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] False
    2 [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] True
    3 [1, 2, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] True
    == Progression 1
    == I = [1, 2, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    == L = [[1]], D = [[1, 2, 4, 7], [8, 9, 10, 11, 12, 13, 14], [16, 15]]
    4 [1, 2, 4, 7] False
    5 [1, 2, 4, 7, 8, 9, 10, 11, 12, 13, 14] True
    == Progression 2
    == I = [1, 2, 4, 7, 8, 9, 10, 11, 12, 13, 14]
    == L = [[8, 9, 10, 11, 12, 13, 14], [1]], D = [[1, 2, 4, 7, 8, 9, 10, 11, 12, 13, 14]]
    6 [1, 2, 4, 7, 8, 9, 10, 11, 12, 13, 14] True
    frozenset({1, 2, 4, 7, 8, 9, 10, 11, 12, 13, 14})

    """
    import itertools
    transposed = graph.transpose()
 
    # Compute the variable orders as the reversed postorder of the
    # transposed graph. (This is simply the Kosaraju Sharir algorithm for 
    # computing strongly conected components)
    variableorder = list(reversed(list(transposed.postorder(transposed.nodes()))))

    def progression(learned_sets, items): 
        # Mark unincluded items as already visited.
        visited = [ i not in items for i in graph.nodes()]

        # We don't have to think too hard about which elements to add to the 
        # learned sets sice they are all strongly connected.
        yield frozenset(graph.postorder(itertools.chain(*learned_sets), visited))

        # Now go through the variable order and add all reachable nodes from each variable
        for i in variableorder:
            if visited[i]: continue
            yield frozenset(graph.postorder([i], visited))

    return progression

if __name__ == "__main__":
    import doctest
    doctest.testmod()
