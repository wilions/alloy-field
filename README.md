# AlloyField (`alloy-field`)

Solidification Microstructure, Dendrite Tip Kinetics, Columnar-to-Equiaxed Transition (CET), and Crystallographic Texture Simulation Engine for Integrated Computational Materials Engineering (ICME).

---

## 🏛️ Architecture & Physics

`alloy-field` bridges thermal solidification histories $(G, R, \dot{T})$ and crystal plasticity homogenization (DAMASK / Hill'48):

1. **Dendrite Tip Kinetics (`alloy_field.core.kgt_kinetics`)**:
   * Kurz–Giovanola–Trivedi (KGT) / LGK non-equilibrium tip velocity: $v(\Delta T) = a_1 \Delta T^2 + a_2 \Delta T^3$.
   * Total undercooling decomposition: $\Delta T_{\text{total}} = \Delta T_c + \Delta T_r + \Delta T_t + \Delta T_k$.
   * Primary dendrite arm spacing: $\lambda_1 = 4.3 \left(\frac{\Delta T_0 \Gamma D_L}{k}\right)^{0.25} G^{-0.5} R^{-0.25}$.
2. **Cellular Automata (CA) Microstructure Solver (`alloy_field.core.cellular_automata`)**:
   * Decentered square (2D) / decentered octahedron (3D) crystallographic grain growth on structured Cartesian voxel grids.
   * Preferential growth along cubic $\langle 100 \rangle$ easy-growth directions.
   * Heterogeneous nucleation using continuous Gaussian distribution: $dn/d(\Delta T) = \frac{N_{\text{max}}}{\sqrt{2\pi}\Delta T_\sigma} \exp\left(-\frac{(\Delta T - \Delta T_N)^2}{2 \Delta T_\sigma^2}\right)$.
3. **Columnar-to-Equiaxed Transition (CET) Engine (`alloy_field.core.cet_solver`)**:
   * Hunt's criterion for fully columnar ($G^n/R > A$), mixed, or fully equiaxed ($G^n/R < B$) microstructure regimes.
   * 2D Process maps: $G$ vs. $R$ and cooling rate $\dot{T}$ vs. thermal gradient $G$.
4. **Crystallographic Texture & RVE Generator (`alloy_field.core.texture`)**:
   * Bunge Euler angles $(\phi_1, \Phi, \phi_2)$ per grain.
   * Orientation Distribution Function (ODF) & pole figure evaluation.
   * Direct export of voxelized RVE grids with Euler angle tensors for DAMASK CP-FFT.
