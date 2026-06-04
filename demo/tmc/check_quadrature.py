import basix
from basix import CellType

cell_type = CellType.hexahedron
degs = [2, 3]

for deg in degs:
    points, weights = basix.make_quadrature(cell_type, deg)

    print(f"Cell type: {cell_type}, degree: {deg}")
    print(f"Number of quadrature points: {len(weights)}")
    print(f"Quadrature points positions:\n{points}\n")
