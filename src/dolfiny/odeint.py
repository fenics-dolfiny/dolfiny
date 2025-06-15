import dolfinx
import ufl

import dolfiny

# ruff: noqa: E501


def _copy_entries(source, target):
    """Helper function to copy solution values from the source Function to the target Function."""
    if isinstance(source, list):
        for si, ti in zip(source, target):
            with si.x.petsc_vec.localForm() as locs, ti.x.petsc_vec.localForm() as loct:
                locs.copy(loct)
    else:
        with source.x.petsc_vec.localForm() as locs, target.x.petsc_vec.localForm() as loct:
            locs.copy(loct)


class ODEInt:
    def __init__(self, t, dt, x, xt, **kwargs):
        """Initialises the ODE integrator (single-step-method) for 1st order ODEs.

        Uses underneath the generalised alpha method and its limits.

        Important: This variant ensures that the solution x(t1), xt(t1)
                   fulfills the residual r(t, x, xt) at the end of the time step t1.
                   Properties of the generalised alpha method (controlled by the parameters
                   alpha_f, alpha_m and gamma) remain intact. In addition, the rate of the
                   state converges with 2nd order in time (with appropriate gamma).
                   This is achieved with help of the auxiliary rate xt_aux.

        (1) xt_gamma = (x1 - x0) / dt = [ (1 - gamma) * x0t_aux + gamma * x1t_aux ]
        (2) xt_alpha = (1 - alpha_f) * x0t + alpha_f * x1t = (1 - alpha_m) * x0t_aux + alpha_m * x1t_aux

        K.E. Jansen, C.H. Whiting, G.M. Hulbert. CMAME, 190, 305-319, 2000.
        http://dx.doi.org/10.1016/S0045-7825(00)00203-6

        Crank-Nicolson: [default]
            alpha_f = 1/2, alpha_m = 1/2, gamma = 1/2
        Euler backward:
            alpha_f = 1, alpha_m = 1/2, gamma = 1/2
        Theta:
            alpha_f = theta, alpha_m = 1/2, gamma = 1/2
        Generalised alpha:
            The value of rho can be used to determine the values
            alpha_f = 1 / (1 + rho),
            alpha_m = 1 / 2 * (3 - rho) / (1 + rho),
            gamma = 1 / 2 + alpha_m - alpha_f --> second order

        Parameters
        ----------
        t: Stage time.
        dt: Time step size.
        x: Pointer to function describing the state.
        xt: Pointer to function describing the rate of the state.

        rho: Spectral radius rho_infinity for generalised alpha.
        alpha_f: Specific value for alpha_f.
        alpha_m: Specific value for alpha_m.
        gamma: Specific value for gamma.

        """
        # Set stage time and time step
        self.t = t
        self.dt = dt

        if not isinstance(self.t, dolfinx.fem.Constant):
            raise RuntimeError("No stage time t as dolfinx.Constant provided.")

        if not isinstance(self.dt, dolfinx.fem.Constant):
            raise RuntimeError("No time step dt as dolfinx.Constant provided.")

        # Pointers to state x1 and rate x1t (as function or list of functions)
        self.x1 = x
        self.x1t = xt

        if isinstance(self.x1, list):
            for x1, x1t in zip(self.x1, self.x1t):
                if x1.function_space is not x1t.function_space:
                    raise RuntimeError("Incompatible function spaces for state and rate.")
        else:
            if self.x1.function_space is not self.x1t.function_space:
                raise RuntimeError("Incompatible function spaces for state and rate.")

        # Set state x0
        if isinstance(self.x1, list):
            self.x0 = [x1i.copy() for x1i in self.x1]
        else:
            self.x0 = self.x1.copy()

        # Set rate of state x0t
        if isinstance(self.x1t, list):
            self.x0t = [x1ti.copy() for x1ti in self.x1t]
        else:
            self.x0t = self.x1t.copy()

        # Set *auxiliary* rate of state x1t_aux and x0t_aux
        if isinstance(self.x1t, list):
            self.x1t_aux = [x1ti.copy() for x1ti in self.x1t]
            self.x0t_aux = [x0ti.copy() for x0ti in self.x0t]
        else:
            self.x1t_aux = self.x1t.copy()
            self.x0t_aux = self.x0t.copy()

        # Initialise values of auxiliary state
        dolfiny.odeint._copy_entries(self.x1t, self.x1t_aux)

        # Default values: Crank-Nicolson
        domain = ufl.domain.extract_unique_domain(self.t)
        self.alpha_f = dolfinx.fem.Constant(domain, 0.5)
        self.alpha_m = dolfinx.fem.Constant(domain, 0.5)
        self.gamma = dolfinx.fem.Constant(domain, 0.5)

        # Parameters from given rho
        if "rho" in kwargs:
            self.alpha_f.value = 1.0 / (1.0 + kwargs["rho"])
            self.alpha_m.value = 0.5 * (3.0 - kwargs["rho"]) * self.alpha_f.value
            self.gamma.value = 0.5 + self.alpha_m.value - self.alpha_f.value

        # Parameters directly
        if "alpha_f" in kwargs and "alpha_m" in kwargs and "gamma" in kwargs:
            self.alpha_f.value = kwargs["alpha_f"]
            self.alpha_m.value = kwargs["alpha_m"]
            self.gamma.value = kwargs["gamma"]

    def _derivative_dt(self, x1t_aux, x0t_aux, x0t):
        """Returns the UFL expression for: derivative in time x1t."""
        # return equation (1) solved for x1t
        return (
            1
            / self.alpha_f
            * ((1 - self.alpha_m) * x0t_aux + self.alpha_m * x1t_aux - (1 - self.alpha_f) * x0t)
        )

    def _derivative_dt_aux(self, x1, x0, x0t_aux):
        """Returns the UFL expression for: derivative in time x1t_aux."""
        # return equation (2) solved for x1t_aux
        return 1 / self.gamma * (1 / self.dt * (x1 - x0) - (1 - self.gamma) * x0t_aux)

    def _integral_dt(self, x1, x1t, x0, x0t):
        """Returns the UFL expression for: integral over the time interval int_t0^t1 x(t) dt."""
        # return integrated polynomial of degree 3
        return self.dt / 2 * (x0 + x1) + self.dt**2 / 12 * (x0t - x1t)

    def integral_dt(self, y):
        """Returns the UFL expression for: time integral of given UFL function y registered in ODEInt."""
        if isinstance(self.x1, list):
            if y not in self.x1:
                raise RuntimeError("Given function not registered in ODEInt object.")
            else:
                i = self.x1.index(y)
                return self._integral_dt(self.x1[i], self.x1t[i], self.x0[i], self.x0t[i])
        else:
            if y != self.x1:
                raise RuntimeError("Given function not registered in ODEInt object.")
            else:
                return self._integral_dt(self.x1, self.x1t, self.x0, self.x0t)

    def stage(self, t0=None, dt=None):
        """Stages the processing of the next time step: sets time value (to t1) and initial values."""
        if t0 is not None:
            self.t.value = t0

        if dt is not None:
            self.dt.value = dt

        # Set time
        self.t.value += self.dt.value

        # Store states (set initial values for next time step)
        dolfiny.odeint._copy_entries(self.x1, self.x0)
        dolfiny.odeint._copy_entries(self.x1t, self.x0t)
        dolfiny.odeint._copy_entries(self.x1t_aux, self.x0t_aux)

        return self.t, self.dt

    def update(self):
        """Set rate x1t and auxiliary rate x1t_aux once x1 has been computed."""
        # update x1t_aux
        if isinstance(self.x1t_aux, list):
            for x1, x0, x1t_aux, x0t_aux in zip(self.x1, self.x0, self.x1t_aux, self.x0t_aux):
                dolfiny.interpolation.interpolate(self._derivative_dt_aux(x1, x0, x0t_aux), x1t_aux)
        else:
            dolfiny.interpolation.interpolate(
                self._derivative_dt_aux(self.x1, self.x0, self.x0t_aux), self.x1t_aux
            )

        # update x1t
        if isinstance(self.x1t, list):
            for x1t_aux, x0t_aux, x1t, x0t in zip(self.x1t_aux, self.x0t_aux, self.x1t, self.x0t):
                dolfiny.interpolation.interpolate(self._derivative_dt(x1t_aux, x0t_aux, x0t), x1t)
        else:
            dolfiny.interpolation.interpolate(
                self._derivative_dt(self.x1t_aux, self.x0t_aux, self.x0t), self.x1t
            )

    def discretise_in_time(self, f):
        """Discretises the form f(t, x, xt) in time. The solution fulfills f(t1, x1, x1t) = 0."""
        # Construct expression for x1t_aux
        if isinstance(self.x1t, list):
            x1t_aux = []
            for x1i, x0i, x0ti_aux in zip(self.x1, self.x0, self.x0t_aux):
                x1t_aux.append(self._derivative_dt_aux(x1i, x0i, x0ti_aux))
        else:
            x1t_aux = self._derivative_dt_aux(self.x1, self.x0, self.x0t_aux)

        # Construct expression for x1t
        if isinstance(self.x1t, list):
            x1t = []
            for x1ti_aux, x0ti_aux, x0ti in zip(self.x1t_aux, self.x0t_aux, self.x0t):
                x1t.append(self._derivative_dt(x1ti_aux, x0ti_aux, x0ti))
        else:
            x1t = self._derivative_dt(self.x1t_aux, self.x0t_aux, self.x0t)

        # Replace in form
        f = dolfiny.expression.evaluate(f, self.x1t, x1t)
        f = dolfiny.expression.evaluate(f, self.x1t_aux, x1t_aux)

        return f


class ODEInt2:
    def __init__(self, t, dt, x, xt, xtt, **kwargs):
        """Initialises the ODE integrator (single-step-method) for 2nd order ODEs.

        Uses underneath the generalised alpha method and its limits.

        Important: This variant ensures that the solution x(t1), xt(t1), xtt(t1)
                   fulfills the residual r(t, x, xt, xtt) at the end of the time step t1.
                   Properties of the generalised alpha method (controlled by the parameters
                   alpha_f, alpha_m and gamma) remain intact. In addition, the second derivative
                   of the state converges with 2nd order in time (with appropriate gamma and beta).
                   This is achieved with help of the auxiliary rate of rate xtt_aux.

        (1) xt_beta = (x1 - x0) / dt = x0t + dt * [ (1 / 2 - beta) * x0tt_aux + beta * x1tt_aux ]
        (2) xtt_gamma = (xt1 - xt0) / dt = [ (1 - gamma) * x0tt_aux + gamma * x1tt_aux ]
        (3) xtt_alpha = (1 - alpha_f) * x0tt + alpha_f * x1tt = (1 - alpha_m) * x0tt_aux + alpha_m * x1tt_aux

        J. Chung, G. M. Hulbert. ASME Journal of Applied Mechanics, 60, 371-375, 1993.
        http://dx.doi.org/10.1115/1.2900803

        M. Arnold, O. Brüls. Multibody Syst Dyn, 18, 185-202, 2007.
        http://dx.doi.org/10.1007/s11044-007-9084-0

        Newmark: [default]
            alpha_f = 1, alpha_m = 1, gamma = 1/2, beta = 1/4
        HHT-a:
            alpha_f = a, alpha_m = 1, gamma = 3/2 - a, beta = (2-a)^2 / 4
        WBZ-a:
            alpha_f = 1, alpha_m = a, gamma = a - 1/2, beta = (0+a)^2 / 4
        Generalised alpha:
            The value of rho can be used to determine the values
            alpha_f = 1 / (1 + rho),
            alpha_m = 1 / 2 * (3 - rho) / (1 + rho),
            gamma = 1 / 2 + alpha_m - alpha_f --> second order
            beta = 1 / 4 * (1 + alpha_m - alpha_f)^2 --> unconditionally stable

        Parameters
        ----------
        t: Stage time.
        dt: Time step size.
        x: Pointer to function describing the state.
        xt: Pointer to function describing the rate of the state.
        xtt: Pointer to function describing the rate of rate of the state.

        rho: Spectral radius rho_infinity for generalised alpha.
        alpha_f: Specific value for alpha_f.
        alpha_m: Specific value for alpha_m.
        gamma: Specific value for gamma.
        beta: Specific value for beta.

        """
        # Set stage time and time step
        self.t = t
        self.dt = dt

        if not isinstance(self.t, dolfinx.fem.Constant):
            raise RuntimeError("No stage time t as dolfinx.Constant provided.")

        if not isinstance(self.dt, dolfinx.fem.Constant):
            raise RuntimeError("No time step dt as dolfinx.Constant provided.")

        # Pointers to state x1, rate x1t and rate of rate x1tt (as function or list of functions)
        self.x1 = x
        self.x1t = xt
        self.x1tt = xtt

        if isinstance(self.x1, list):
            for x1, x1t, x1tt in zip(self.x1, self.x1t, self.x1tt):
                if (
                    x1.function_space is not x1t.function_space
                    or x1.function_space is not x1tt.function_space
                ):
                    raise RuntimeError("Incompatible function spaces for state and rate.")
        else:
            if (
                self.x1.function_space is not self.x1t.function_space
                or self.x1.function_space is not self.x1tt.function_space
            ):
                raise RuntimeError("Incompatible function spaces for state and rate.")

        # Set state x0
        if isinstance(self.x1, list):
            self.x0 = [x1i.copy() for x1i in self.x1]
        else:
            self.x0 = self.x1.copy()

        # Set rate of state x0t
        if isinstance(self.x1t, list):
            self.x0t = [x1ti.copy() for x1ti in self.x1t]
        else:
            self.x0t = self.x1t.copy()

        # Set rate of rate of state x0tt
        if isinstance(self.x1tt, list):
            self.x0tt = [x1tti.copy() for x1tti in self.x1tt]
        else:
            self.x0tt = self.x1tt.copy()

        # Set *auxiliary* of rate of rate of state x1tt_aux and x0tt_aux
        if isinstance(self.x1tt, list):
            self.x1tt_aux = [x1tti.copy() for x1tti in self.x1tt]
            self.x0tt_aux = [x0tti.copy() for x0tti in self.x0tt]
        else:
            self.x1tt_aux = self.x1tt.copy()
            self.x0tt_aux = self.x0tt.copy()

        # Initialise values of auxiliary state
        dolfiny.odeint._copy_entries(self.x1tt, self.x1tt_aux)

        # Default values: Newmark
        domain = ufl.domain.extract_unique_domain(self.t)
        self.alpha_f = dolfinx.fem.Constant(domain, 1.0)
        self.alpha_m = dolfinx.fem.Constant(domain, 1.0)
        self.gamma = dolfinx.fem.Constant(domain, 0.5)
        self.beta = dolfinx.fem.Constant(domain, 0.25)

        # Parameters from given rho
        if "rho" in kwargs:
            self.alpha_f.value = 1.0 / (1.0 + kwargs["rho"])
            self.alpha_m.value = (2.0 - kwargs["rho"]) * self.alpha_f.value
            self.gamma.value = 0.5 + self.alpha_m.value - self.alpha_f.value
            self.beta.value = 0.25 * (1.0 + self.alpha_m.value - self.alpha_f.value) ** 2

        # Parameters directly
        if "alpha_f" in kwargs and "alpha_m" in kwargs and "gamma" in kwargs and "beta" in kwargs:
            self.alpha_f.value = kwargs["alpha_f"]
            self.alpha_m.value = kwargs["alpha_m"]
            self.gamma.value = kwargs["gamma"]
            self.beta.value = kwargs["beta"]

    def _derivative_dt(self, x1tt_aux, x0tt_aux, x0t):
        """Returns the UFL expression for: derivative in time x1t."""
        # return equation (2) solved for x1t
        return x0t + self.dt * ((1 - self.gamma) * x0tt_aux + self.gamma * x1tt_aux)

    def _derivative_dt2(self, x1tt_aux, x0tt_aux, x0tt):
        """Returns the UFL expression for: derivative in time x1tt."""
        # return equation (3) solved for x1t
        return (
            (1 - self.alpha_m) * x0tt_aux + self.alpha_m * x1tt_aux - (1 - self.alpha_f) * x0tt
        ) / self.alpha_f

    def _derivative_dt2_aux(self, x1, x0, x0t, x0tt_aux):
        """Returns the UFL expression for: derivative in time x1tt."""
        # return equation (1) solved for x1tt_aux
        return ((x1 - x0) / self.dt**2 - x0t / self.dt - (1 / 2 - self.beta) * x0tt_aux) / self.beta

    def _integral_dt(self, x1, x1t, x1tt, x0, x0t, x0tt):
        """Returns the UFL expression for: integral over the time interval int_t0^t1 x(t) dt."""
        # return integrated polynomial of degree 5
        return (
            self.dt / 2 * (x0 + x1)
            + self.dt**2 / 10 * (x0t - x1t)
            + self.dt**3 / 120 * (x0tt + x1tt)
        )

    def integral_dt(self, y):
        """Returns the UFL expression for: time integral of given UFL function y registered in ODEInt2."""
        if isinstance(self.x1, list):
            if y not in self.x1:
                raise RuntimeError("Given function not registered in ODEInt2 object.")
            else:
                i = self.x1.index(y)
                return self._integral_dt(
                    self.x1[i], self.x1t[i], self.x1tt[i], self.x0[i], self.x0t[i], self.x0tt[i]
                )
        else:
            if y != self.x1:
                raise RuntimeError("Given function not registered in ODEInt2 object.")
            else:
                return self._integral_dt(self.x1, self.x1t, self.x1tt, self.x0, self.x0t, self.x0tt)

    def stage(self, t0=None, dt=None):
        """Stages the processing of the next time step: sets time value (to t1) and initial values."""
        if t0 is not None:
            self.t.value = t0

        if dt is not None:
            self.dt.value = dt

        # Set time
        self.t.value += self.dt.value

        # Store states (set initial values for next time step)
        dolfiny.odeint._copy_entries(self.x1, self.x0)
        dolfiny.odeint._copy_entries(self.x1t, self.x0t)
        dolfiny.odeint._copy_entries(self.x1tt, self.x0tt)
        dolfiny.odeint._copy_entries(self.x1tt_aux, self.x0tt_aux)

        return self.t, self.dt

    def update(self):
        """Set rate x1t, rate2 x1tt and auxiliary rate2 x1tt_aux once x1 has been computed."""
        # update x1tt_aux
        if isinstance(self.x1tt_aux, list):
            for x1, x0, x0t, x1tt_aux, x0tt_aux in zip(
                self.x1, self.x0, self.x0t, self.x1tt_aux, self.x0tt_aux
            ):
                dolfiny.interpolation.interpolate(
                    self._derivative_dt2_aux(x1, x0, x0t, x0tt_aux), x1tt_aux
                )
        else:
            dolfiny.interpolation.interpolate(
                self._derivative_dt2_aux(self.x1, self.x0, self.x0t, self.x0tt_aux), self.x1tt_aux
            )

        # update x1t
        if isinstance(self.x1t, list):
            for x1tt_aux, x0tt_aux, x1t, x0t in zip(
                self.x1tt_aux, self.x0tt_aux, self.x1t, self.x0t
            ):
                dolfiny.interpolation.interpolate(self._derivative_dt(x1tt_aux, x0tt_aux, x0t), x1t)
        else:
            dolfiny.interpolation.interpolate(
                self._derivative_dt(self.x1tt_aux, self.x0tt_aux, self.x0t), self.x1t
            )

        # update x1tt
        if isinstance(self.x1tt, list):
            for x1tt_aux, x0tt_aux, x1tt, x0tt in zip(
                self.x1tt_aux, self.x0tt_aux, self.x1tt, self.x0tt
            ):
                dolfiny.interpolation.interpolate(
                    self._derivative_dt2(x1tt_aux, x0tt_aux, x0tt), x1tt
                )
        else:
            dolfiny.interpolation.interpolate(
                self._derivative_dt2(self.x1tt_aux, self.x0tt_aux, self.x0tt), self.x1tt
            )

    def discretise_in_time(self, f):
        """Discretises the form f(t, x, xt, xtt) in time. The solution fulfills f(t1, x1, x1t, x1tt) = 0."""
        # Construct expression for x1tt_aux
        if isinstance(self.x1tt, list):
            x1tt_aux = []
            for x1i, x0i, x0ti, x0tti_aux in zip(self.x1, self.x0, self.x0t, self.x0tt_aux):
                x1tt_aux.append(self._derivative_dt2_aux(x1i, x0i, x0ti, x0tti_aux))
        else:
            x1tt_aux = self._derivative_dt2_aux(self.x1, self.x0, self.x0t, self.x0tt_aux)

        # Construct expression for x1t
        if isinstance(self.x1t, list):
            x1t = []
            for x1tti_aux, x0tti_aux, x0ti in zip(self.x1tt_aux, self.x0tt_aux, self.x0t):
                x1t.append(self._derivative_dt(x1tti_aux, x0tti_aux, x0ti))
        else:
            x1t = self._derivative_dt(self.x1tt_aux, self.x0tt_aux, self.x0t)

        # Construct expression for x1tt
        if isinstance(self.x1tt, list):
            x1tt = []
            for x1tti_aux, x0tti_aux, x0tti in zip(self.x1tt_aux, self.x0tt_aux, self.x0tt):
                x1tt.append(self._derivative_dt2(x1tti_aux, x0tti_aux, x0tti))
        else:
            x1tt = self._derivative_dt2(self.x1tt_aux, self.x0tt_aux, self.x0tt)

        # Replace in form
        f = dolfiny.expression.evaluate(f, self.x1t, x1t)
        f = dolfiny.expression.evaluate(f, self.x1tt, x1tt)
        f = dolfiny.expression.evaluate(f, self.x1tt_aux, x1tt_aux)

        return f
