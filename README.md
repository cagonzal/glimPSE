# Incompressible Parabolized Stability Equation Solver

A high-performance, parallel solver for analyzing boundary layer stability using incompressible parabolized stability equations (PSE). The solver supports both linear and nonlinear PSE formulations, including specialized cases for Orr-Sommerfeld and Görtler equations.

## Features

- **Comprehensive Stability Analysis**
  - Linear and nonlinear parabolized stability equations
  - Orr-Sommerfeld equation solver for parallel flow analysis
  - Görtler equation solver for flows with significant curvature effects
  - Body-fitted curvilinear coordinate system implementation

- **Numerical Methods**
  - Spatial marching using backward Euler scheme
  - Wall-normal discretization options:
    - Chebyshev collocation
    - Finite difference schemes
  - MPI-based parallelization using mpi4py

## Verification Cases

The solver has been validated against several benchmark cases from literature:

1. Tollmien-Schlichting mode propagation (Bertolotti et al., 1991)
2. Subharmonic and oblique mode propagation (Joslin et al., 1993)
3. Görtler mode propagation (Li and Malik, 1995)
4. N-factor envelope computation for the NLF(01)-0416 airfoil

## Theory

The solver implements the incompressible parabolized stability equations in a body-fitted curvilinear coordinate system. The PSE approach allows for the efficient computation of convectively unstable flows by taking advantage of the predominantly one-directional nature of the instability propagation.


## Citation

If you use this solver in your research, please cite: Gonzalez, C.A., Harris, R.S., & Moin, P. (2024). "glimPSE: an open-source incompressible nonlinear parabolized stability equation solver". Annual Research Briefs, Center for Turbulence Research. 

## References

1. Bertolotti, F. P., Herbert, T., & Spalart, P. R. (1992). Linear and nonlinear stability of the Blasius boundary layer. Journal of Fluid Mechanics, 242, 441-474.
2. Joslin, R. D., Streett, C. L., & Chang, C. L. (1993). Spatial direct numerical simulation of boundary-layer transition mechanisms: Validation of PSE theory. Theoretical and Computational Fluid Dynamics, 4(6), 271-288.
3. Li, F., & Malik, M. R. (1995). Fundamental and subharmonic secondary instabilities of Görtler vortices. Journal of Fluid Mechanics, 297, 77-100.
