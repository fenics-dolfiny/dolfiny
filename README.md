# dolfiny: high-level and convenience wrappers for DOLFINx

The `dolfiny` package provides a set of high-level and convenience wrappers for 
[DOLFINx](https://github.com/FEniCS/dolfinx), the [FEniCS library](https://www.fenicsproject.org).

While DOLFINx involves many low-level operations - from handling ghosted values 
to interfacing PETSc solvers directly - `dolfiny` serves as a wrapper around the
low-level functionality of DOLFINx and is meant to combine a user-oriented API
with the performance and functionality of DOLFINx, FFCx, Basix and UFL.

This library is written exclusively in Python with optional interfacing 
to user-provided C++ kernels.

# Installation

```
pip3 install dolfiny
```

Certain functionality (see [demos](demo/) and [unit tests](test/)) relies on 
the availability of external packages such as

- [Matplotlib](https://github.com/matplotlib/matplotlib) (plotting),
- [PyVista](https://github.com/pyvista/pyvista) (scientific visualisation), or 
- [cppyy](https://github.com/wlav/cppyy) (dynamic Python/C++ bindings).

Install `dolfiny` with these dependencies by running 
```
pip3 install dolfiny[all]
```

For ARM-based architectures (`aarch64`/`arm64` on Linux) we recommend to fallback to our
custom-compiled binary wheels for `vtk` and `cppyy-cling` by setting the
local package index repository in the respective environment variable 
```
export PIP_INDEX_URL=https://gitlab.uni.lu/api/v4/projects/3415/packages/pypi/simple
```
before calling `pip`.

You may also check the [Dockerfile](docker/Dockerfile) for an up-to-date version of the installation process.

# Docker image

Multi-arch (`amd64` and `arm64`) Docker images with pre-installed `dolfiny` (and dependencies)
are available at [DockerHub](https://hub.docker.com/r/dolfiny/dolfiny).

```
docker pull dolfiny/dolfiny
```

# Documentation

In preparation.

In the meantime please check available [demos](demo/) or [unit tests](test/).

Presentations about `dolfiny` functionality:
- [dolfiny: Convenience wrappers for DOLFINx](https://hdl.handle.net/10993/47422)
  at FEniCS 2021 conference,
- [Nonlinear analysis of thin-walled structures based on tangential differential calculus with FEniCSx](https://hdl.handle.net/10993/54222)
  at FEniCS 2022 conference,
- [Nonlinear local solver](https://hdl.handle.net/10993/54223)
  at FEniCS 2022 conference.
- [Non-standard formulations of 3D elasticity with FEniCSx](https://hdl.handle.net/10993/61399)
  at FEniCS 2024 conference.

# Authors

- Michal Habera, Rafinex, Luxembourg.
- Andreas Zilian, University of Luxembourg, Luxembourg.

# Contributing

We are always looking for contributions and help with `dolfiny`.
If you have ideas, nice applications or code contributions then we would 
be happy to help you get them included.
We ask you to follow the FEniCS Project git workflow.

# Issues and Support

Please use the GitHub issue tracker to report any issues.

# License

`dolfiny` is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

`dolfiny` is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with `dolfiny`. If not, see <http://www.gnu.org/licenses/>.
