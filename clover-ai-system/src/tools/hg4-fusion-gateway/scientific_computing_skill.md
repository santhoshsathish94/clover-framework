# Skill: Scientific & Numerical Research Computing (`SKILL.md`)

When implementing multi-step scientific Python functions (physics, quantum mechanics, chemistry, biology, materials science, numerical PDEs/ODEs, optimization, and linear algebra):

1. **Exact Function Contract (`AGENTS.md`):**
   - Preserve the provided `def <name>(...):` signature, parameter names, default values, and exact return variable count/order from the provided `return ...` template.
   - Return plain `numpy.ndarray`, `float`, `complex`, `int`, or `tuple` matching the docstring — never wrap a single return value in a tuple unless specified, and never print or plot at top level.
   - Use ONLY imports provided in `DEPENDENCIES` (do not import uninstalled libraries, and do not re-define functions from earlier steps; call earlier step functions directly with their exact signatures).

2. **NumPy / SciPy Numerical Precision & API Discipline:**
   - Cast inputs that undergo division or linear algebra to `np.asarray(..., dtype=float)` (or `dtype=complex` when wavefunctions, phases `exp(1j*...)`, or non-Hermitian operators appear) to prevent silent integer floor-division or `ComplexWarning` truncation.
   - Prefer `scipy.integrate.simpson` (with fallback `getattr(scipy.integrate, "simpson", None) or scipy.integrate.simps`), `np.trapezoid` / `np.trapz`, `scipy.linalg.eigh` for Hermitian matrices (sorted ascending eigenvalues!), and `scipy.sparse.linalg.eigsh` with explicit `which=` (`'SA'` for ground state / smallest algebraic).
   - Check broadcasting dimensions explicitly (`[:, None]` vs `[None, :]`), especially on 2D/3D grids (`meshgrid(..., indexing='ij')` vs `'xy'`), FFT wavevectors (`2 * np.pi * np.fft.fftfreq(N, d=dx)`), and Kronecker products (`functools.reduce(np.kron, ops)`).

3. **Boundary, Normalization & Conservation Checks:**
   - Guard singular denominators (`r == 0`, `k == 0` zero-mode in Poisson/spectral solvers) explicitly as specified by the physics background.
   - Verify normalization conventions (`dx` / `dV` quadrature weights on continuous grids vs discrete vector `np.linalg.norm`), periodic vs Dirichlet boundary indices, and radian vs degree units.
