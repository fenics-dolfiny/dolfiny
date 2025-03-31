from dolfiny.localsolver import LocalSolver


class TAOBlockProblem:
    def __init__(
        self,
        F,
        u: list,
        bcs=[],
        J_form=None,
        nest=False,
        restriction=None,
        prefix=None,
        localsolver: LocalSolver = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
    ):
        pass
