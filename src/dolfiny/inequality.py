import ufl


class Inequality:
    """Stores an inequality as.

        lhs <= rhs

    where lhs is a form and rhs a constant
    """

    def __init__(self, lhs: ufl.Form, rhs: int | float) -> None:
        self._lhs = lhs
        self._rhs = rhs

    def __str__(self):
        return f"{self._lhs.__str__()} <= {self._rhs.__str__()}"

    @property
    def lhs(self) -> ufl.Form:
        return self._lhs

    @property
    def rhs(self) -> int | float:
        return self._rhs

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Inequality) and self._lhs.equals(other.lhs) and self._rhs == other.rhs
        )


# Overwrite <=, >= operations for ufl.Form


def custom_le(form: ufl.Form, other: int | float) -> Inequality:
    return Inequality(form, other)


def custom_ge(form: ufl.Form, other: int | float) -> Inequality:
    """Form >= other iff. -form <= - other."""
    return Inequality(-form, -other)


ufl.Form.__le__ = custom_le  # type: ignore
ufl.Form.__ge__ = custom_ge  # type: ignore
