# Third-Medium Contact (TMC): Notes and Development status

This document collects TMC formulations available in the literature and track implementation status and numerical experiments performed in `demo/tmc`.

The objectives are:
- Review the main TMC regularizations in chronological order, having them in a unique document along with the source papers;
- Assess advantages and limitations of each approach;
- Identify their suitability to dolfiny demo and in general to FEniCSx implementation.

## TMC formulation
The third-medium (TM) is designed to fill the space between two contacting bodies. Its strain energy is provided in general by two terms.

The first one is a strain energy function that models the material behavior of the third medium. The idea is to have an almost non existent influence when there is no contact and to provide a stiffness which tends to a very high value once the bodies getting close to each other. Traditionally, the Neo-Hookean material law in the Simo-Pister form is used:

$$
W_m = \gamma \int_{\Omega_m}\left[\frac{K}{2}[\ln J]^2+\frac{\mu}{2}\left(J^{-\frac{2}{3}} \operatorname{tr} \boldsymbol{C}-3\right)\right] \mathrm{d} \Omega, \tag{1}
$$

where $\gamma$ is the _relative contact stiffness_ (generally in the order $10^{-6}$ - $10^{-7}$) that ensures the third medium acts as a highly compliant material in the pre-contact phase, and then rapidly stiffens upon contact thanks to the term $\ln J$.

<u>Remark</u>: As discussed in Faltus [1], the volumetric term in (1) can be omitted in 2D plane strain settings since the increase of stiffness coming from the isochoric term as $J \rightarrow 0$, together with the plane strain constraint $F_{33} = 1$, generate an increase of stiffness sufficient to prevent penetration. Therefore, in this case the strain energy density for the third medium reduces to:

$$
W_m^{2D} = \gamma \int_{\Omega_m} \frac{\mu}{2}\left(J^{-\frac{2}{3}} \operatorname{tr} \boldsymbol{C}-3\right) \mathrm{d} \Omega. \tag{2}
$$

The second term is the _regularization_ energy $W_R = \int_{\Omega_m} \Psi_R \, \mathrm{d} \Omega$, which controls mesh distorsions in the third medium. This regularization should provide sufficient stabilization while being minimally intrusive towards the deformation of the third medium. 
Different forms of this contribution have been presented in the literature and they are analysed (in more or less chronological order) below.

## HuHu regularization - Bluhm et al. (2021) [2]
Higher-order deformation modes related to warping and bending of third-medium elements are penalized through the Hessian of the displacement field $\mathbb{H} u$ in the following way:

$$
\Psi_R^{Hu} = \frac{1}{2} k_r \mathbb{H} u \cdot \mathbb{H} u,  \tag{3}
$$

where $k_r$ is a scaling constant which controls the severity of the regularization. 

### Limitations 
- Certain deformation modes, such as <u>bending</u>, are excessively penalized, preventing a correct representation of deformed surfaces in contact (improvement: HuHu-LuLu regularization, see below);
- (at least) Quadratic shape functions are necessary to compute meaningful second order derivatives of the displacement field, preventing the use of both linear and quadratic triangular and tetrahedral elements. 

## HuHu-LuLu regularization - Frederiksen et al. (2025) [3]
To overcome the previous limitation of the HuHu regularization, a new term involving the Laplacian of the displacement field $\mathbb{L} u$ is subtracted from the HuHu regularization:

$$
\Psi_R^{HuLu} = \frac{1}{2} k_r (\mathbb{H} u \cdot \mathbb{H} u
- \frac{1}{\operatorname{tr} {\boldsymbol{I}}} \mathbb{L} u \cdot \mathbb{L} u).  \tag{4}
$$

This serves to reduce the penalization imposed on specific higher-order deformations, such as quadratic compression and bending, while maintaining stability. 

### Limitations 
- Also in this case, finite elements of at least second-order are necessary for effective regularization. Additionally, <u>skew</u> deformations are penalized equally in both HuHu and HuHu-LuLu regularizations (see Table 2 in the paper), necessitating Gauss-Lobatto integration for third-medium elements to mitigate local inversion of corner nodes (see Fig. 3 in the paper).

## First order regularization - Wriggers et al. (2025) [4],[5]
This regularization can be considered an improvement of the one proposed in [1], which was based on the fundamental idea that "..<u>curvature</u> on the element level is the primary problem of unregularized third medium, while stretch deformation gradients and even volume change gradients, on the other hand, are the desired behaviors during the compliant pre-contact phase..". This lead to a curvature $\nabla \boldsymbol{R}$ penalization, where the rotation tensor $\boldsymbol{R}$ was NOT directly computed from the multiplicative split of the deformation gradient $\boldsymbol{F} = \boldsymbol{R} \boldsymbol{U}$, but rather approximated by an update of a rotation matrix $\boldsymbol{Q}$ which is easier to calculate (see [1] for further details). Instead, Wriggers et al. in [4] start from the observation that in 2D the rotation tensor $\boldsymbol{R}$ can be characterized by a single angle $\varphi$: 

$$
\boldsymbol{R}=\left[\begin{array}{ccc}
\cos \varphi & \sin \varphi & 0 \\
-\sin \varphi & \cos \varphi & 0 \\
0 & 0 & 1
\end{array}\right] \tag{5}
$$

and, by using the symmetry of $\boldsymbol{U}$, they arrive to the following explicit expression for the rotation angle:

$$
\tan \varphi=\left[\frac{F_{12}-F_{21}}{F_{11}+F_{22}}\right] \Rightarrow 
\varphi =\arctan \left[\left(F_{12}-F_{21}\right) /\left(F_{11}+F_{22}\right)\right]. \tag{6}
$$

At this point, they propose two alternative regularizations based on (6):

$$
\begin{aligned}
\Psi_{RJ}^{\varphi} &= \frac{\gamma}{2} \alpha_r\left(\|\nabla \varphi\|^2+\|\nabla J\|^2\right), \\ \tag{7}
\Psi_{RJ}^{\text{tan}}&= \frac{\gamma}{2} \alpha_r\left(\left\|\nabla\left[\frac{F_{12}-F_{21}}{F_{11}+F_{22}}\right]\right\|^2+\|\nabla J\|^2\right), 
\end{aligned}
$$

where $\alpha_r$ is again a regularization scaling parameter. Note the additional contribution $\|\nabla J\|^2$ which is necessary since "..due to the larger freedom in element deformation caused by the lack of stretch and shear penalization, elements may approach a deformed state with locally zero volume in certain geometries of the third medium..". And continuing to cite [1], "Overall, uniform rotation, i.e., zero curvature and uniform volume change, are enforced. Nevertheless, significant components of second displacement gradients are still left unpenalized, namely stretch and certain modes of symmetric extension or contraction of the elements, contrary to Bluhm et al. approach [2] of penalizing second displacement gradients as a whole. Thus the compliant behavior of the third medium is improved".

In the follow-up paper [5], the same authors extended this regularization to 3D by first noting that the numerator in $\tan \varphi=\left[\frac{F_{12}-F_{21}}{F_{11}+F_{22}}\right]$ is the skew (shear) part of the deformation gradient, while the denominator is its trace (stretch part). Then, by assuming that _the trace does not play a big role when it comes to severe element distortions_, the denominator can be omitted, ending up with a regularization term involving only the skew-symmetric part of the deformation gradient $\boldsymbol{F}^{\text{skew}} = \frac{1}{2}\left(\boldsymbol{F}-\boldsymbol{F}^T\right)$ as:

$$
\Psi_{RJ}^{3D} = \frac{\gamma}{2} \alpha_r\left(\|\nabla \boldsymbol{F}^{\text{skew}}\|^2+\|\nabla J\|^2\right).  \tag{8}
$$

However, the above regularizations still require at least quadratic ansatz functions to consistently compute second-order derivatives of the displacement field. A formulation which allows to replace the higher-order gradients through the introduction of additional dofs was proposed in the same papers ([4] for 2D and [5] for 3D). This approach, when considering the form $(7)_1$, lead to:

$$
\begin{aligned}
\Psi_R^{\varphi}(\boldsymbol{u}, p) & = \frac{\gamma}{2}\left[\beta_1(\varphi(\boldsymbol{u})-p_1)^2+\alpha_r\|\nabla p_1\|^2\right], \\ \tag{9}
\Psi_J(\boldsymbol{u}, q) & = \frac{\gamma}{2}\left[\beta_2(J(\boldsymbol{u})-q)^2+\alpha_r\|\nabla q\|^2\right],
\end{aligned}
$$

where $p$, $q$ are additional unknowns and the parameters $\beta_1$ and $\beta_2$ can be interpreted as penalty parameters which enforce the approximation of the gradients. Analogously, in 3D the three independent components of $\boldsymbol{F}^{\text{skew}}$ are approximated by the introduction of the field variables $p_i$, leading to:

$$
\begin{aligned}
\Psi_R^{3D}\left(\boldsymbol{u}, p_i\right) &= \frac{\gamma}{2} \sum_{i=1}^{3} \left(\beta_1 \left[f_i^{s k e w}-\frac{1}{d} p_i\right]^2 +\alpha_r \left\|\nabla p_i\right\|^2\right), \\
\Psi_J^{3D}(\boldsymbol{u}, q) &= \frac{\gamma}{2}\left[\beta_2(J(\boldsymbol{u})-q)^2+\alpha_r\|\nabla q\|^2\right],
\end{aligned} \tag{10}
$$

where $d$ is a scaling parameter depending on the characteristic size of the investigated problem. In this way, regularizations (9), (10) allows the use of triangular and quadrilateral finite element discretizations with linear ansatz functions in 2D and the use of tetrahedral and hexahedral finite elements with linear ansatz functions in 3D. The additional scalar fields $p_i$, $q$ are approximated in the same way as the displacement field.

<u>Remark</u>: Citing [5], "Since in a linear element less deformation modes have to be controlled, the regularization term $\nabla J$ can be omitted in 2D problems when using finite elements with linear shape functions. Numerical simulations revealed that this is also the case for the three-dimensional elements. Thus, only the additional fields $p_i$ are required in the third medium which reduces the global computational effort."

### Limitations 
- Evidently, the flexibility in using linear elements comes at the cost of additional unknowns (1 in 2D, 3 in 3D) at each third-medium node to solve for, rapidly increasing the compuational cost for large structures. Some advanced approaches have been developed to overcome this issue, see references in [5].

## Deformation gradient averaging regularization - Faltus et al. (2026) [6]
In this recent paper, Faltus et al. propose either a new form for the regularization term and a different constitutive behavior for the third-medium, specifically the small-strain linear elastic material law expressed as:

$$
W_{\mathrm{ss}}=\frac{1}{2}(\boldsymbol{F}-\boldsymbol{I}): \mathbb{C} :(\boldsymbol{F}-\boldsymbol{I}), \tag{11}
$$

exploiting the symmetries of $\mathbb{C}$ to replace the symmetric displacement field gradient $\boldsymbol{\varepsilon} = \nabla_s \boldsymbol{u}$ with the full gradient $(\boldsymbol{F} − \boldsymbol{I}) = \nabla \boldsymbol{u}$.

As noted by the authors themselves "The validity of this material law for physical materials relies on the standard small-strain assumptions; in a large-strain context it results in a non-objective formulation that produces non-zero energy under pure rotations. Nevertheless, <u>the third medium is fictitious</u> and we therefore adopt this formulation despite this limitation. Given that the stiffness of the third medium is typically at least six orders of magnitude smaller than the stiffness of the bulk material, any spurious energy generated remains negligible unless the third medium undergoes extremely large rotations."

This choice should improve the pre-contact phase as the third-medium now acts as a constant stiffness material, thus removing geometrical nonlinearities from its constitutive response. Of course, the logarithmic contribution has to be retained to provide an asymptotically increasing stiffness under very large compressions to impose contact. The third-medium constitutive response is thus modeled by the following strain energy function:

$$ 
W_m = \gamma \int_{\Omega_m}\left[\frac{K}{2}[\ln J]^2 + 
\frac{1}{2} (\boldsymbol{F}-\boldsymbol{I}): \mathbb{C} :(\boldsymbol{F}-\boldsymbol{I})\right]
\mathrm{d} \Omega, \tag{12}
$$

Concering the regularization term, the authors propose an approach which allows the use of first-order finite elements WITHOUT additional degrees of freedom. This is based on the observation that, at the element level, it is possible to control the level of uniformity of $\boldsymbol{F}$ by penalizing its deviation from the element-wise average value. More in detail, they penalize differences between the deformation gradient at the integration points and the representative deformation gradient of the element $\bar{\boldsymbol{F}}$, taken as its value at the element centroid. This amounts to the following regularization term:

$$
\Psi_{\bar{F}}=\frac{1}{2} \kappa_{\bar{F}}\|\boldsymbol{F}-\bar{\boldsymbol{F}}\|^2, \tag{13}
$$

where $\kappa_{\bar{F}}$ is a penalty parameter. In this way there's no need to compute second gradients. 

### Limitations
- This regularization is only meaningful when multiple integration points with theoretically different values of the deformation gradient tensor exist within the element, preventing the use of linear triangular elements in 2D or linear tetrahedrons in 3D.

## Other approaches in the literature
- Xu et al. (2026) [7] - Incremental rotation gradient regularization, still needs higher-order elements.


## Development status



## References

[1] O. Faltus, M. Horák, M. Doškář, O. Rokoš, Third medium finite element contact formulation for pneumatically actuated systems, Comput. Methods Appl.  Mech. Engrg. 431 (2024) 117262, http://dx.doi.org/10.1016/j.cma.2024.117262.

[2] G.L. Bluhm, O. Sigmund, K. Poulios, Internal contact modeling for finite strain topology optimization, Comput. Mech. 67 (4) (2021) 1099–1114,  http://dx.doi.org/10.1007/s00466-021-01974-x.

[3] A.H. Frederiksen, A. Dalklint, O. Sigmund, K. Poulios, Improved third medium formulation for 3D topology optimization with contact, Comput. Methods  Appl. Mech. Engrg. 436 (2025) http://dx.doi.org/10.1016/j.cma.2024.117595.

[4] P. Wriggers, J. Korelc, P. Junker, A third medium approach for contact using first and second order finite elements, Comput. Methods Appl. Mech. Engrg.  436 (2025) http://dx.doi.org/10.1016/j.cma.2025.117740.

[5] P. Wriggers, J. Korelc, P. Junker, First order finite element formulations for third medium contact, Comput. Mech. 76 (3) (2025) 829–845, http:  //dx.doi.org/10.1007/s00466-025-02628-y.

[6] O. Faltus, M. Amato, M. Horák, Deformation gradient averaging regularization for third medium contact,
Computer Methods in Applied Mechanics and Engineering 458 (2026) 119072, https://doi.org/10.1016/j.cma.2026.119072.

[7] Xu, B.B., Xue, T., Wriggers, P., Three-dimensional third medium contact model for hyperelastic contact and pneumatically actuated systems, 
Journal of the Mechanics and Physics of Solids 213 (2026) 106617,
https://doi.org/10.1016/j.jmps.2026.106617


