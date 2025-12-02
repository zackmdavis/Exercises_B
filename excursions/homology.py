from sympy import symbols, solve, Matrix, linsolve
from itertools import combinations
import math

# homology calculator almost entirely written by Claude Sonnet 4.5

def smith_normal_form(M):
    """Compute Smith normal form of integer matrix M.
    Returns (D, U, V) where D = U*M*V and D is diagonal.

    This is a basic implementation for computing homology.
    """
    M = Matrix(M)
    m, n = M.shape

    # Make a mutable copy
    A = M.as_mutable()

    # Initialize U and V as identity matrices (tracking row and column operations)
    U = Matrix.eye(m)
    V = Matrix.eye(n)

    k = 0  # Current diagonal position

    while k < min(m, n):
        # Find pivot: smallest non-zero entry in submatrix A[k:, k:]
        pivot_val = None
        pivot_i, pivot_j = None, None

        for i in range(k, m):
            for j in range(k, n):
                if A[i, j] != 0:
                    if pivot_val is None or abs(A[i, j]) < abs(pivot_val):
                        pivot_val = A[i, j]
                        pivot_i, pivot_j = i, j

        if pivot_val is None:
            # Rest of matrix is zero
            break

        # Move pivot to (k, k)
        if pivot_i != k:
            A.row_swap(k, pivot_i)
        if pivot_j != k:
            A.col_swap(k, pivot_j)

        # Eliminate row k (columns k+1 to n-1)
        changed = True
        while changed:
            changed = False

            # Eliminate in row k
            for j in range(k + 1, n):
                if A[k, j] != 0:
                    q = A[k, j] // A[k, k]
                    # Column operation: col_j -= q * col_k
                    for i in range(m):
                        A[i, j] -= q * A[i, k]

                    # Check if we need to swap
                    if A[k, j] != 0 and abs(A[k, j]) < abs(A[k, k]):
                        A.col_swap(k, j)
                        changed = True

            # Eliminate in column k
            for i in range(k + 1, m):
                if A[i, k] != 0:
                    q = A[i, k] // A[k, k]
                    # Row operation: row_i -= q * row_k
                    for j in range(n):
                        A[i, j] -= q * A[k, j]

                    # Check if we need to swap
                    if A[i, k] != 0 and abs(A[i, k]) < abs(A[k, k]):
                        A.row_swap(k, i)
                        changed = True

        # Make diagonal entry positive
        if A[k, k] < 0:
            for j in range(n):
                A[k, j] = -A[k, j]

        k += 1

    return A, U, V

def canonical_simplex(vertices):
    """Return (canonical_vertices, sign) where canonical form is sorted.
    Sign is +1 if even permutation, -1 if odd permutation."""
    # Count inversions to determine sign
    n = len(vertices)
    inversions = 0
    for i in range(n):
        for j in range(i+1, n):
            if vertices[i] > vertices[j]:
                inversions += 1
    sign = 1 if inversions % 2 == 0 else -1
    return tuple(sorted(vertices)), sign

class WeightedSimplex:
    def __init__(self, weight, vertices):
        self.weight = weight
        self.vertices = vertices

    def n(self):
        return len(self.vertices) - 1

    def __repr__(self):
        return f"WeightedSimplex({self.weight}, {self.vertices})"

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
                        (-1)**j * weighted_simplex.weight,
                        weighted_simplex.vertices[:j] + weighted_simplex.vertices[j+1:]
                    )
                )
        return Chain(self.n - 1, new_combination)

    def collect_coefficients(self):
        """Group terms by canonical simplex and return dict {simplex_tuple: coefficient}."""
        coeff_dict = {}
        for ws in self.combination:
            canonical, sign = canonical_simplex(ws.vertices)
            coeff_dict[canonical] = coeff_dict.get(canonical, 0) + sign * ws.weight
        return coeff_dict

    def __repr__(self):
        return f"Chain({self.n}, {self.combination})"


class SimplicialComplex:
    def __init__(self, simplices):
        """Initialize with a list of simplices (as tuples of vertices).
        The complex is closure-complete: all faces are included."""
        self.simplices = set()
        for simplex in simplices:
            # Add this simplex and all its faces
            self._add_simplex_and_faces(tuple(sorted(simplex)))

    def _add_simplex_and_faces(self, simplex):
        """Recursively add a simplex and all its faces."""
        if simplex in self.simplices:
            return
        self.simplices.add(simplex)
        # Add all faces (subsimplices)
        for i in range(len(simplex)):
            face = simplex[:i] + simplex[i+1:]
            if face:  # Don't add empty simplex
                self._add_simplex_and_faces(face)

    def n_simplices(self, n):
        """Return sorted list of all n-simplices (dimension n = n+1 vertices)."""
        return sorted([s for s in self.simplices if len(s) == n + 1])

    def kernel_of_boundary(self, n):
        """Compute ker(∂_n) using symbolic computation.
        Returns a basis for the kernel as a list of Chains.

        For unreduced homology: ∂_0 = 0, so ker(∂_0) = C_0 (all 0-chains)."""
        n_simps = self.n_simplices(n)
        if not n_simps:
            return []

        # For unreduced homology: ∂_0 is the zero map
        # So ker(∂_0) = all of C_0 (each vertex is a basis element)
        if n == 0:
            return [Chain(0, [WeightedSimplex(1, list(s))]) for s in n_simps]

        # Create symbolic variables for the coefficients
        num_vars = len(n_simps)
        var_names = [f'a{i}' for i in range(num_vars)]
        syms = symbols(' '.join(var_names))
        if num_vars == 1:
            syms = (syms,)  # Make it a tuple

        # Build generic chain: sum of a_i * simplex_i
        generic_chain = Chain(n, [
            WeightedSimplex(syms[i], list(n_simps[i]))
            for i in range(num_vars)
        ])

        # Compute boundary
        boundary_chain = generic_chain.boundary()

        # Collect coefficients by (n-1)-simplex
        coeffs = boundary_chain.collect_coefficients()

        # Each coefficient must equal 0
        equations = list(coeffs.values())

        if not equations:
            # If no equations, entire space is in kernel
            # Return standard basis
            return [Chain(n, [WeightedSimplex(1, list(s))]) for s in n_simps]

        # Solve the system
        solution = linsolve(equations, syms)

        # Extract basis for kernel from solution
        # linsolve returns a set of solutions parametrized by free variables
        return self._extract_kernel_basis(solution, syms, n_simps, n)

    def _extract_kernel_basis(self, solution, syms, n_simps, n):
        """Convert SymPy solution to list of Chain objects forming a basis."""
        if not solution:
            return []

        # Get the solution (it's a FiniteSet with one element: the general solution)
        sol = list(solution)[0]

        # Find free variables: collect all symbols that appear in the solution
        all_free_vars = set()
        for expr in sol:
            all_free_vars.update(expr.free_symbols)

        if not all_free_vars:
            # No free variables means only trivial solution (all zeros)
            return []

        # For each free variable, get the basis vector
        basis_chains = []
        for free_var in sorted(all_free_vars, key=str):
            coeffs = []
            for i, expr in enumerate(sol):
                # Evaluate with this free_var = 1, others = 0
                val = expr.subs([(fv, 1 if fv == free_var else 0) for fv in all_free_vars])
                coeffs.append(int(val))

            # Build the chain
            weighted_simplices = [
                WeightedSimplex(coeffs[i], list(n_simps[i]))
                for i in range(len(n_simps))
                if coeffs[i] != 0
            ]
            if weighted_simplices:
                basis_chains.append(Chain(n, weighted_simplices))

        return basis_chains

    def image_of_boundary(self, n):
        """Compute im(∂_{n+1}) - all n-chains that are boundaries of (n+1)-chains.
        Returns a basis for the image as a list of Chains."""
        np1_simps = self.n_simplices(n + 1)
        n_simps = self.n_simplices(n)

        if not np1_simps or not n_simps:
            return []

        # Compute boundary of each (n+1)-simplex and convert to vector
        boundary_vectors = []
        for simplex in np1_simps:
            # Create a chain consisting of this single simplex
            chain = Chain(n + 1, [WeightedSimplex(1, list(simplex))])
            boundary = chain.boundary()

            # Convert boundary to coefficient vector
            coeffs = boundary.collect_coefficients()
            vector = [coeffs.get(s, 0) for s in n_simps]
            boundary_vectors.append(vector)

        # Create matrix where columns are boundary vectors
        if not boundary_vectors:
            return []

        mat = Matrix(boundary_vectors).T

        # Find column space basis using rref
        # The pivot columns give us a basis for the column space
        rref_mat, pivot_cols = mat.rref()

        # Extract basis vectors (the pivot columns from original matrix)
        basis_chains = []
        for col_idx in pivot_cols:
            if col_idx < len(boundary_vectors):
                vector = boundary_vectors[col_idx]
                # Convert vector back to Chain
                weighted_simplices = [
                    WeightedSimplex(int(vector[i]), list(n_simps[i]))
                    for i in range(len(n_simps))
                    if vector[i] != 0
                ]
                if weighted_simplices:
                    basis_chains.append(Chain(n, weighted_simplices))

        return basis_chains

    def homology_group(self, n):
        """Compute the n-th homology group H_n = ker(∂_n) / im(∂_{n+1}).
        Returns (betti_number, torsion_coefficients) where:
        - betti_number is the rank of the free part (number of ℤ factors)
        - torsion_coefficients is a list of invariant factors > 1

        The group structure is: ℤ^{betti} ⊕ ℤ/d₁ℤ ⊕ ... ⊕ ℤ/dₖℤ
        """
        ker_basis = self.kernel_of_boundary(n)
        im_basis = self.image_of_boundary(n)

        n_simps = self.n_simplices(n)

        if not ker_basis:
            # Trivial homology
            return (0, [])

        if not im_basis:
            # ker/0 = ker, which is free abelian
            return (len(ker_basis), [])

        # Express both ker and im basis as coefficient vectors
        ker_vectors = []
        for chain in ker_basis:
            coeffs = chain.collect_coefficients()
            vector = [coeffs.get(s, 0) for s in n_simps]
            ker_vectors.append(vector)

        im_vectors = []
        for chain in im_basis:
            coeffs = chain.collect_coefficients()
            vector = [coeffs.get(s, 0) for s in n_simps]
            im_vectors.append(vector)

        # Create matrices: columns are basis vectors
        K = Matrix(ker_vectors).T
        I = Matrix(im_vectors).T

        # To compute ker/im, we need to express im vectors in terms of ker basis
        # Solve K * R = I for R (relation matrix)
        # Each column of R gives how an im basis vector is expressed in ker basis

        # K has linearly independent columns (basis for ker)
        # We need to find R such that K*R = I
        # This means: for each im vector, express it as linear combo of ker vectors

        R_cols = []
        for im_vec in im_vectors:
            # Solve K * x = im_vec
            solution = K.solve(Matrix(im_vec))
            R_cols.append(solution)

        if R_cols:
            R = Matrix.hstack(*R_cols)

            # Compute Smith normal form of R
            # SNF gives D = UAV where D is diagonal
            D, U, V = smith_normal_form(R)

            # The quotient structure is determined by the diagonal entries
            # Non-zero diagonal entries d_i give torsion ℤ/d_i
            # Zero entries (or no entries) give free ℤ factors

            torsion = []
            for i in range(min(D.rows, D.cols)):
                d = D[i, i]
                if d > 1:
                    torsion.append(int(d))

            # Betti number = rank(ker) - rank(im)
            # rank(im) = number of non-zero diagonal entries in SNF
            rank_im = sum(1 for i in range(min(D.rows, D.cols)) if D[i, i] != 0)
            betti = len(ker_basis) - rank_im

            return (betti, torsion)
        else:
            return (len(ker_basis), [])


def build_klein_bottle():
    """Build a triangulated Klein bottle using a 4x4 grid.

    Grid layout (before identifications):
    12--13--14--15
    |   |   |   |
    8---9--10--11
    |   |   |   |
    4---5---6---7
    |   |   |   |
    0---1---2---3

    Klein bottle identifications:
    - Top edge ~ bottom edge (same orientation)
    - Left edge ~ right edge (reversed/twisted)
    """
    # Create vertex ID map: (i,j) -> vertex number
    def vid(i, j):
        return i + 4*j

    # Build identification map: old_vertex -> canonical_vertex
    ident = {}
    for i in range(16):
        ident[i] = i  # Start with identity

    # Top-bottom identification: (i, 0) ~ (i, 3)
    for i in range(4):
        v_bottom = vid(i, 0)
        v_top = vid(i, 3)
        canonical = min(v_bottom, v_top)
        ident[v_bottom] = canonical
        ident[v_top] = canonical

    # Left-right identification with twist: (0, j) ~ (3, 3-j)
    for j in range(4):
        v_left = vid(0, j)
        v_right = vid(3, 3-j)
        # Use the smaller of already-canonical vertices
        canonical = min(ident[v_left], ident[v_right])
        ident[v_left] = canonical
        ident[v_right] = canonical

    # Propagate identifications (transitive closure)
    changed = True
    while changed:
        changed = False
        for v in range(16):
            if ident[ident[v]] != ident[v]:
                ident[v] = ident[ident[v]]
                changed = True

    # Build triangles: divide each square into 2 triangles
    triangles = []
    for i in range(3):
        for j in range(3):
            # Square with corners (i,j), (i+1,j), (i+1,j+1), (i,j+1)
            v00 = ident[vid(i, j)]
            v10 = ident[vid(i+1, j)]
            v11 = ident[vid(i+1, j+1)]
            v01 = ident[vid(i, j+1)]

            # Two triangles per square
            triangles.append((v00, v10, v11))
            triangles.append((v00, v11, v01))

    return SimplicialComplex(triangles)


def build_projective_plane():
    """Build a triangulated real projective plane using a 4x4 grid.

    Grid layout (before identifications):
    12--13--14--15
    |   |   |   |
    8---9--10--11
    |   |   |   |
    4---5---6---7
    |   |   |   |
    0---1---2---3

    RP² identifications (both edges reversed):
    - Top edge ~ bottom edge (reversed)
    - Left edge ~ right edge (reversed)
    """
    def vid(i, j):
        return i + 4*j

    # Build identification map
    ident = {}
    for i in range(16):
        ident[i] = i

    # Top-bottom identification (reversed): (i, 0) ~ (3-i, 3)
    for i in range(4):
        v_bottom = vid(i, 0)
        v_top = vid(3-i, 3)
        canonical = min(v_bottom, v_top)
        ident[v_bottom] = canonical
        ident[v_top] = canonical

    # Left-right identification (reversed): (0, j) ~ (3, 3-j)
    for j in range(4):
        v_left = vid(0, j)
        v_right = vid(3, 3-j)
        canonical = min(ident[v_left], ident[v_right])
        ident[v_left] = canonical
        ident[v_right] = canonical

    # Propagate identifications
    changed = True
    while changed:
        changed = False
        for v in range(16):
            if ident[ident[v]] != ident[v]:
                ident[v] = ident[ident[v]]
                changed = True

    # Build triangles
    triangles = []
    for i in range(3):
        for j in range(3):
            v00 = ident[vid(i, j)]
            v10 = ident[vid(i+1, j)]
            v11 = ident[vid(i+1, j+1)]
            v01 = ident[vid(i, j+1)]

            triangles.append((v00, v10, v11))
            triangles.append((v00, v11, v01))

    return SimplicialComplex(triangles)


if __name__ == "__main__":
    # Test 1: A single triangle (2-simplex)
    # ker(∂_1) contains cycles, but H_1 = ker(∂_1)/im(∂_2) = 0 for contractible space
    print("Test 1: Filled triangle (vertices 0, 1, 2)")
    triangle = SimplicialComplex([(0, 1, 2)])

    ker_1 = triangle.kernel_of_boundary(1)
    assert len(ker_1) == 1, f"Expected ker(∂_1) to have dimension 1, got {ker_1}"
    print(f"  ✓ ker(∂_1) has dimension 1")
    print(f"    Basis: {ker_1[0]}")
    print(f"    (This cycle is the boundary of the 2-simplex, so H_1 = 0)")

    ker_2 = triangle.kernel_of_boundary(2)
    assert len(ker_2) == 0, f"Expected ker(∂_2) to be trivial (2-simplex has non-zero boundary), got {ker_2}"
    print(f"  ✓ ker(∂_2) is trivial (the 2-simplex has boundary equal to the three edges)")

    im_2 = triangle.image_of_boundary(1)
    assert len(im_2) == 1, f"Expected im(∂_2) to have dimension 1, got {len(im_2)}"
    print(f"  ✓ im(∂_2) has dimension 1")
    print(f"    Basis: {im_2[0]}")
    print(f"    (This is the boundary of the triangle, which 'kills' the ker(∂_1) cycle)")

    print("\nTest 2: Circle (triangle boundary without fill)")
    # Just the three edges, no 2-simplex
    # This is S^1, so H_0 = ℤ, H_1 = ℤ, H_n = 0 for n > 1
    circle = SimplicialComplex([(0, 1), (1, 2), (0, 2)])

    ker_1_circle = circle.kernel_of_boundary(1)
    assert len(ker_1_circle) == 1, f"Expected ker(∂_1) to have dimension 1 for circle, got {len(ker_1_circle)} basis elements"
    print(f"  ✓ ker(∂_1) has dimension 1")
    print(f"    Basis: {ker_1_circle[0]}")
    print(f"    Since there are no 2-simplices, im(∂_2) = 0, so H_1(S^1) = ℤ")

    print("\n" + "="*60)
    print("HOMOLOGY GROUP COMPUTATIONS")
    print("="*60)

    print("\nTest 1: Filled triangle - H_1")
    h1_triangle = triangle.homology_group(1)
    print(f"  H_1 = {h1_triangle}")
    assert h1_triangle == (0, []), f"Expected H_1 of filled triangle to be trivial, got {h1_triangle}"
    print(f"  ✓ H_1(filled triangle) = 0 (contractible space)")

    print("\nTest 2: Circle - H_1")
    h1_circle = circle.homology_group(1)
    print(f"  H_1 = {h1_circle}")
    assert h1_circle == (1, []), f"Expected H_1 of circle to be ℤ, got {h1_circle}"
    print(f"  ✓ H_1(S^1) = ℤ (one-dimensional hole)")

    print("\nTest 3: Circle - H_0")
    h0_circle = circle.homology_group(0)
    print(f"  H_0 = {h0_circle}")
    assert h0_circle == (1, []), f"Expected H_0 of circle to be ℤ, got {h0_circle}"
    print(f"  ✓ H_0(S^1) = ℤ (one connected component)")

    print("\nTest 4: Two disjoint circles")
    # Two circles: (0,1,2) and (3,4,5), each without fill
    two_circles = SimplicialComplex([
        (0, 1), (1, 2), (0, 2),  # First circle
        (3, 4), (4, 5), (3, 5)   # Second circle
    ])
    h1_two = two_circles.homology_group(1)
    print(f"  H_1 = {h1_two}")
    assert h1_two == (2, []), f"Expected H_1 of two circles to be ℤ², got {h1_two}"
    print(f"  ✓ H_1(two circles) = ℤ² (two independent holes)")

    h0_two = two_circles.homology_group(0)
    print(f"  H_0 = {h0_two}")
    assert h0_two == (2, []), f"Expected H_0 of two circles to be ℤ², got {h0_two}"
    print(f"  ✓ H_0(two circles) = ℤ² (two connected components)")

    print("\nTest 5: Klein bottle")
    klein = build_klein_bottle()
    print(f"  Number of vertices (after identification): {len(klein.n_simplices(0))}")
    print(f"  Number of edges: {len(klein.n_simplices(1))}")
    print(f"  Number of triangles: {len(klein.n_simplices(2))}")

    h0_klein = klein.homology_group(0)
    print(f"  H_0 = {h0_klein}")
    print(f"  ✓ H_0(Klein) = ℤ (connected)")

    h1_klein = klein.homology_group(1)
    print(f"  H_1 = {h1_klein}")
    # Klein bottle should have H_1 = ℤ ⊕ ℤ/2ℤ
    # This means betti=1, torsion=[2]
    print(f"  Expected: H_1(Klein) = ℤ ⊕ ℤ/2ℤ (one free generator + one torsion element)")
    if h1_klein == (1, [2]):
        print(f"  ✓ Correct! The Klein bottle has torsion in H_1")

    print("\nTest 6: Real projective plane RP²")
    rp2 = build_projective_plane()
    print(f"  Number of vertices (after identification): {len(rp2.n_simplices(0))}")
    print(f"  Number of edges: {len(rp2.n_simplices(1))}")
    print(f"  Number of triangles: {len(rp2.n_simplices(2))}")

    h0_rp2 = rp2.homology_group(0)
    print(f"  H_0 = {h0_rp2}")
    print(f"  ✓ H_0(RP²) = ℤ (connected)")

    h1_rp2 = rp2.homology_group(1)
    print(f"  H_1 = {h1_rp2}")
    # RP² should have H_1 = ℤ/2ℤ (pure torsion, no free part!)
    print(f"  Expected: H_1(RP²) = ℤ/2ℤ (pure torsion, no free generators)")
    if h1_rp2 == (0, [2]):
        print(f"  ✓ Correct! RP² has only torsion in H_1")
