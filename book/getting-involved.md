# Getting involved

You enjoy using `dolfiny` and would like to get involved?
Great, we value contributions of all kind, from comments, corrections, extensions to new features.

- For comments and corrections, use the [issue tracker](https://github.com/fenics-dolfiny/dolfiny/issues).
- For code contributions, create a [pull request](https://github.com/fenics-dolfiny/dolfiny/pulls) form a fork ([guide](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)).
- For bigger endeavors, student- or internship projects please contact the authors via email. 

## 🎓 Student & Internship Projects

As part of the study programs at the University of Luxembourg, internships or other arrangements, we offer the possibility to get involved with open source software development of modern finite element solutions.

Projects may address problems ranging across
- modern IT infrastructure (containerized applications, CI/CD pipelines, dependency management),
- software development (efficient implementations, automated testing, adaptation of modern code practices and tooling),
- numerics (non-linear solvers, optimisation algorithms, parallel data structures), and
- civil engineering (models and design of: beam-, truss- or continua).


### 📁 Idea Hub - collection of project ideas

````{grid} 2

```{card}
:header: ➰ Interface to PETSc arc length solver

Simulations of unstable behaviour, such as buckling, require advanced solution strategies.
*Arc length* methods offer a robust alternative to classic Newton solvers.

Currently, `dolfiny` supports (with a custom implementation) an arc length method as presented by https://doi.org/10.1002/nme.1620190902 - see the continuation demos in the repository.

The project aims at interfacing to the PETSc [SNES arc length solver](https://petsc.org/main/manualpages/SNES/SNESNEWTONAL/) and demonstrating its capabilities.

```

```{card}
:header: 🤝 Third-medium contact demo

The *third-medium* contact method, ref. https://doi.org/10.1007/s00466-013-0848-5 and https://doi.org/10.1016/j.cma.2025.117740, gained popularity for its variationally pleasing and differential contact formulation.

Its formulation fits well into the domain description language [`UFL`](https://github.com/fenics/ufl).

The project aims to demonstrate the usage of third medium contact formulations within FEniCS and (possibly) applying it other use cases, such as optimisation problems.
It gives the opportunity to understand a modern research topic and to get involved with modern, coding practices.
```

```{card}
:header: 📝 Your project idea
You have an own idea for a project that you would like to see accomplished?
Suggest it to us!
```

````

## 📢 Opportunities 

```{card}
:header: Unsolicited application
:link: https://www.uni.lu/fstm-en/research-groups/mechanics-and-structural-analysis/
Are you looking for a new academic position related to the here covered applications or alike?
Then you might be interested in joining our *Mechanics and Structural Analysis* group at the [University of Luxembourg](https://www.uni.lu/en/).
For more details on current availability, we are happy to receive your unsolicited application - reach out to the authors directly.
```
