class WeightedSimplex:
    def __init__(self, weight, vertices):
        self.weight = weight
        self.vertices = vertices

    def n(self):
        return len(self.vertices) - 1

class Chain:
    def __init__(self, n, combination):
        # We store dimensionality separately rather than leaving it implied by
        # the simplices in the chain so that we can have an empty n-chain.
        self.n = n
        assert all(ws.n() == n for ws in combination)
        self.combination = combination

    def boundary(self):
        new_combination = []
        for _i, weighted_simplex in enumerate(self.combination):
            for j, _vertex in enumerate(weighted_simplex.vertices):
                new_combination.append(
                    WeightedSimplex(
                        (-1)**j * weighted_simplex.weight
                        weighted_simplex.vertices[:j] + weighted_simplex.vertices[j+1:]
                    )
                )
        # TODO PROBABLY: canonicalize orientations?
        return Chain(self.n - 1, new_combination)


# We can thus compute the boundary of any particular chain, but that's not what
# we really want! We want the kernel of the boundary operator ...

# Maybe this is a good opportunity to play more with AI-assisted coding, just
# to save the time?
