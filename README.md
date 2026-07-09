# Two-Wheeled Bipedal Self-Balancing Robot

**Algorithmic Tutorial — Design, Modeling, and Control**

<!-- Equations in this file render on github.com via GitHub's built-in LaTeX (MathJax) support. -->

This is a **self-contained tutorial** that walks through every algorithmic component of a two-wheeled bipedal self-balancing robot. No external code repository is needed — all equations, pseudocode, intermediate derivations, numerical values, and implementation details are given here explicitly.

Topics covered: link-length optimisation via projected gradient descent, closed-form forward kinematics of a parallel five-bar leg, centre-of-mass computation, reduced-order dynamic modelling, infinite-horizon LQR design, offline gain scheduling over a 2-D lookup table, body-leveling control with force-to-torque mapping, and real-time teleoperation. Each section gives the **why** (rationale), the **what** (formulation), and the **how** (ready-to-implement pseudocode). All numerical parameters match the hardware prototype.

---

## Table of Contents

- [Abstract](#abstract)
- [1. Overview of the System](#1-overview-of-the-system)
- [2. Link-Length Optimisation](#2-link-length-optimisation)
- [3. Closed-Form Forward Kinematics](#3-closed-form-forward-kinematics)
- [4. Centre-of-Mass Computation](#4-centre-of-mass-computation)
- [5. Dynamic Modelling and State-Space Formulation](#5-dynamic-modelling-and-state-space-formulation)
- [6. LQR Controller Design](#6-lqr-controller-design)
- [7. Gain Scheduling](#7-gain-scheduling)
- [8. Force-to-Torque Mapping for Body Leveling](#8-force-to-torque-mapping-for-body-leveling)
- [9. Teleoperation (Remote Control)](#9-teleoperation-remote-control)
- [10. Runtime Architecture](#10-runtime-architecture)
- [Parameter Compendium](#parameter-compendium)

---

## Abstract

This document is a self-contained tutorial that walks through every algorithmic component of the two-wheeled bipedal self-balancing robot described in the accompanying manuscript. No external code repository is needed — all equations, pseudocode, intermediate derivations, numerical values, and implementation details are given here explicitly. Each section begins with the *why* (the design rationale), follows with the *what* (the mathematical formulation), and concludes with *how* (a ready-to-implement pseudocode algorithm). All numerical parameters match the hardware prototype.

---

## 1. Overview of the System

### 1.1 What the robot looks like

The robot consists of a central chassis (body) supported by two parallel five-bar legs. Each leg has:

- **A hip crank** (link $l_5$, driven by motor M1 on the right leg and M3 on the left leg) that controls leg extension and body height.
- **A wheel** (driven by motor M2 on the right and M4 on the left) that provides rolling traction.
- **Passive links** ($l_1, l_2, l_3, l_4$) that constrain the foot to follow an approximately vertical trajectory.

All four motors are identical CAN-bus servo actuators with torque commands saturated at $\pm 8$ N·m. An IMU mounted on the chassis measures roll, pitch, yaw, and their angular rates.

### 1.2 Control architecture at a glance

The system runs two independent control loops simultaneously at $400$ Hz:

1. **Balancing / Locomotion loop** (always active): uses wheel torques $(\tau_L, \tau_R)$ to stabilise the pitch $\theta$, track a forward-velocity reference $\dot{x}_d$, and track a yaw reference $\delta_d$.
2. **Body-leveling loop** (engaged on demand): uses hip forces $(F_{bl}, F_{br})$ to regulate body height $y$ and body roll $\gamma$.

Both loops use infinite-horizon LQR controllers whose gains are precomputed offline over a grid of hip crank angles and stored in lookup tables — no on-line Riccati equation solving is required.

### 1.3 What this tutorial covers

| Section | Module | Input → Output |
|---|---|---|
| 2  | Link-length optimisation | Operating range → optimal $(l_1,\dots,l_5)$ |
| 3  | Forward kinematics | Crank angle → foot position, all joint angles |
| 4  | Centre of mass | Crank angles, body tilt → $l_{cg}$, $\Delta\beta$, $J_{cg}$ |
| 5  | Dynamic modelling | $l_{cg}, J_{cg}$ → state-space $(A, B)$ |
| 6  | LQR gain design | $(A, B)$ + weights → feedback gain $K$ |
| 7  | Gain scheduling | $(\theta_1, \theta_3)$ grid → lookup table of $K$ |
| 8  | Force-to-torque mapping | Leveling force → hip motor torque |
| 9  | Teleoperation | Joystick input → reference $Z_d, Q_d$ |
| 10 | Runtime architecture | All of the above glued together at $400$ Hz |

### 1.4 Notation

- $\alpha_5$ — absolute orientation of the crank $l_5$ in the base frame $\{0\}$ (the single kinematic input of the leg).
- $\alpha_5^{\text{def}}$ — the assembly home offset of $\alpha_5$ (a constant, set at $53.33^\circ$ on the prototype).
- $\theta_1 \triangleq \alpha_5 - \alpha_5^{\text{def}}$ — the crank deviation commanded by the height-regulation loop (also called the crank offset).
- $\alpha_{11}$ — the constant mounting orientation of the body-fixed link $l_1$ (a design constant, $41^\circ$ on the prototype).
- $\alpha_8$ — body tilt angle (pitch), measured by the IMU.
- $\theta$ — inverted-pendulum lean angle (pitch of the CoM line from vertical).
- $\gamma$ — body roll angle; $\delta$ — yaw angle.
- $l_{cg}$ — distance from the wheel-centre midpoint to the overall CoM.
- $\Delta\beta$ — offset angle of the CoM from the wheel-centre vertical (used for feedforward).
- Shorthand: $c_i \triangleq \cos\alpha_i$, $s_i \triangleq \sin\alpha_i$, $c_{i,j} \triangleq \cos(\alpha_i+\alpha_j)$, $s_{i,j} \triangleq \sin(\alpha_i+\alpha_j)$.

---

## 2. Link-Length Optimisation

### 2.1 Why optimise the link lengths?

The parallel five-bar leg is a Hoeckens-linkage variant. For an arbitrary choice of link lengths, the foot trajectory bows horizontally as the crank rotates. This horizontal displacement of the foot translates directly into a horizontal displacement of the centre of mass (CoM), which acts as an *internal disturbance* on the wheel-balancing loop. The balancing controller must then expend wheel torque to compensate for this parasitic motion, reducing the torque budget available for external disturbance rejection.

The key insight is that this problem can be addressed *before* designing the controller: by choosing the link lengths so that the foot trajectory is nearly vertical over the operating range, the parasitic horizontal CoM motion is suppressed at the mechanical-design stage. The controller then only has to reject genuine external disturbances.

### 2.2 Optimisation problem formulation

Let the link lengths be collected in a vector

$$
\boldsymbol{l} \triangleq (l_1,\, l_2,\, l_3,\, l_4,\, l_5).
$$

For a given $\boldsymbol{l}$ and crank angle $\alpha_5$, the forward kinematics (Section 3) yields the foot position ${}^{G}_{0}p(\alpha_5;\boldsymbol{l}) = ({}^{G}_{0}p_x,\, {}^{G}_{0}p_y)$ in closed form. The crank angle is written as

$$
\alpha_5 = \alpha_5^{\text{def}} + \theta_1,
$$

where $\alpha_5^{\text{def}}$ is the fixed assembly home offset ($53.33^\circ$) and $\theta_1$ is the deviation commanded by the height loop. The operating range of $\theta_1$ is $[0^\circ,\, 30^\circ]$, sampled at $m = 31$ uniformly spaced points $\{\alpha_5^{(i)}\}_{i=1}^{m}$ (step $1^\circ$).

Define $g_x^{(i)}(\boldsymbol{l}) \triangleq {}^{G}_{0}p_x(\alpha_5^{(i)};\boldsymbol{l})$. The **cost function** is the variance of the horizontal foot positions about their own mean:

$$
J(\boldsymbol{l}) = \frac{1}{2m}\sum_{i=1}^{m}\bigl[g_x^{(i)}(\boldsymbol{l}) - \bar{g}_x(\boldsymbol{l})\bigr]^{2},
\qquad
\bar{g}_x(\boldsymbol{l}) = \frac{1}{m}\sum_{i=1}^{m} g_x^{(i)}(\boldsymbol{l}).
$$

The divisor $2m$ (rather than $m$) makes the root-mean-square deviation $\sqrt{2J}$ directly interpretable as the standard deviation of the $x$-positions in millimetres.

The optimisation problem is

$$
\min_{\boldsymbol{l}}\; J(\boldsymbol{l}) \quad\text{s.t.}\quad l_j^{\min} \le l_j \le l_j^{\max}\ (j=1,\dots,5),
$$

together with a feasible four-bar assembly at every $\alpha_5^{(i)}$. The box constraints $[0.03,\, 0.30]$ m reflect manufacturability limits. The feasibility constraint means the diagonal $l_8$ must satisfy the triangle inequalities so that $\alpha_3$ is real-valued at every sample.

> **Note.** R1 is the "foot verticality" requirement; R2 (height coverage) is satisfied by the choice of $\theta_1 \in [0^\circ, 30^\circ]$, which on the optimised design produces ${}^{G}_{0}p_y \in [-0.134,\, -0.030]$ m.

### 2.3 Gradient computation

The cost function $J$ is smooth in $\boldsymbol{l}$ because the forward kinematics is smooth and four-bar feasibility is either satisfied or violated as a hard boundary. The gradient is

$$
\frac{\partial J}{\partial l_j} = \frac{1}{m}\sum_{i=1}^{m} \bigl[g_x^{(i)} - \bar{g}_x\bigr]\, \frac{\partial g_x^{(i)}}{\partial l_j},
$$

where $\partial g_x^{(i)}/\partial l_j$ is approximated by central finite differences:

$$
\frac{\partial g_x^{(i)}}{\partial l_j} \approx \frac{g_x^{(i)}(l_j+\delta) - g_x^{(i)}(l_j-\delta)}{2\delta}, \qquad \delta = 10^{-6}\ \text{m}.
$$

Each evaluation of $g_x^{(i)}(l_j \pm \delta)$ requires one call to the forward kinematics.

### 2.4 Projected gradient descent with Armijo backtracking

The update rule is projected gradient descent:

$$
l_j \leftarrow \mathrm{Proj}_{[l_j^{\min},\, l_j^{\max}]}
\left(l_j-\eta\,\frac{\partial J}{\partial l_j}\right),
\quad j=1,\ldots,5.
$$

The learning rate $\eta$ is adapted by a simplified Armijo backtracking line search:

- Start with $\eta_{\text{try}} = \min(2\eta_{\text{prev}},\, \eta_{\max})$, where $\eta_{\max} = 1.0$.
- While $\eta_{\text{try}} > 10^{-14}$:
  - Compute the trial point $\boldsymbol{l}_{\text{try}} = \operatorname{Proj}(\boldsymbol{l} - \eta_{\text{try}}\nabla J)$.
  - If $J(\boldsymbol{l}_{\text{try}}) < J_{\text{curr}}$, accept and set $\eta = \eta_{\text{try}}$.
  - Otherwise, halve $\eta_{\text{try}} \leftarrow \eta_{\text{try}} / 2$.

Convergence is declared when $\|\nabla J\| < 10^{-10}$ or when the cost improvement $|J_{\text{curr}} - J_{\text{new}}| < 10^{-13}\times\max(1, J_{\text{curr}})$. The initial guess is $\boldsymbol{l}_0 = (0.06, 0.10, 0.02, 0.10, 0.10)$ m.

### 2.5 Pseudocode

```text
Algorithm 1: Link-Length Optimisation
Input : l0 (initial guess), θ1_min, θ1_max (operating range)
Output: l* (optimised link lengths)

 1  l ← l0
 2  Θ ← { α5_def + θ1 | θ1 ∈ [θ1_min, θ1_max], step 1° }
 3  J_curr ← Cost(l, Θ)                        # using the variance cost J
 4  η ← 0.01
 5  for k = 1 to 3000 do
 6      for j = 1 to 5 do                      # central-difference gradient
 7          g_j ← ( Cost(l + δ·e_j, Θ) − Cost(l − δ·e_j, Θ) ) / (2δ)
 8      if ‖g‖ < 1e−10 then break
 9      η ← LineSearch(l, g, J_curr, η, Θ)
10      if line search failed then break
11  return l* ← l

function Cost(l, Θ):
 1  m ← |Θ|;   x ← 0_m
 2  for i = 1 to m do
 3      run forward kinematics with α5 = Θ[i], link lengths l
 4      if four-bar assembly infeasible then return 1e3
 5      x[i] ← G_px                            # horizontal foot coordinate
 6  return (1 / 2m) · Σ_i ( x[i] − mean(x) )²

function LineSearch(l, g, J_curr, η, Θ):
 1  η_t ← min(2η, 1.0)
 2  while η_t > 1e−14 do
 3      l_t ← Proj_[0.03, 0.30]( l − η_t·g )
 4      J_t ← Cost(l_t, Θ)
 5      if J_t < J_curr then
 6          l ← l_t;   return η_t
 7      η_t ← η_t / 2
 8  return failed
```

### 2.6 Results on the prototype

Starting from $\boldsymbol{l}_0 = (60, 100, 20, 100, 100)$ mm, the optimiser converges in approximately $3000$ iterations to

$$
\boldsymbol{l}^{\star} = (62.4,\; 101.0,\; 30.3,\; 99.5,\; 98.6)\ \text{mm}.
$$

Key performance indicators:

- Peak-to-peak horizontal deviation: $20.9$ mm → $0.41$ mm (a $51\times$ reduction).
- RMS deviation $\sqrt{2J}$: $5.92$ mm → $0.09$ mm.
- Body-height span: ${}^{G}_{0}p_y \in [-134,\, -30]$ mm.
- CoM offset angle $\Delta\beta$ (Section 4): stays below $0.2^{\circ}$ over the entire workspace.

---

## 3. Closed-Form Forward Kinematics

### 3.1 Why closed-form?

A parallel five-bar linkage is a single-degree-of-freedom closed-chain mechanism: given the crank angle $\alpha_5$, the positions of all remaining joints are uniquely determined (up to an assembly-mode choice). We use the closed-form solution because:

- It is deterministic — no convergence issues, no initial guesses.
- It is differentiable in closed form, enabling analytical Jacobians for the force mapping (Section 8).
- It is fast enough to evaluate inside the optimisation loop (Section 2) where it is called thousands of times.

### 3.2 Kinematic topology

The leg is a planar five-bar linkage with hip pivot $A$ (fixed to the body) and wheel centre $G$. It is organised into two chains:

- **Active chain** $A \to E \to G$: the crank $l_5$ (driven by the hip motor) and the shank $l_4$. Joint $E$ is passive.
- **Passive chain** $A \to B \to D \to E$: the body-fixed link $l_1$ and the coupler pair $l_2, l_3$, forming a four-bar loop $ABDE$ that determines the angle at $E$ once $\alpha_5$ is fixed.

Links $l_3$ and $l_4$ form a single rigid body pivoting about $E$ (printed as one piece on the prototype). The base frame $\{0\}$ is attached at $A$ with its $x$-axis along the body's longitudinal direction. All angles are measured in $\{0\}$.

### 3.3 Solving the four-bar loop $ABDE$

**Step 1 — Diagonal of $\triangle ABE$.** With $A=(0,0)$, $B=(l_1\cos\alpha_{11},\, l_1\sin\alpha_{11})$ known from the fixed mounting angle $\alpha_{11}$, and $E=(l_5\cos\alpha_5,\, l_5\sin\alpha_5)$ known from the crank angle, the law of cosines in $\triangle ABE$ gives the diagonal length:

$$
l_8^{2} = \|\overrightarrow{BE}\|^{2} = l_1^{2} + l_5^{2} - 2\,l_1 l_5\cos(\alpha_5 - \alpha_{11}).
$$

**Step 2 — The angle $\alpha_3$ via circle intersection.** Joint $D$ must satisfy two distance constraints simultaneously: $\|D - B\| = l_2$ (circle centred at $B$, radius $l_2$) and $\|D - E\| = l_3$ (circle centred at $E$, radius $l_3$). For two circles with centres $C_1=(x_1,y_1)$, $C_2=(x_2,y_2)$ and radii $r_1, r_2$, define

$$
d = \sqrt{(x_2-x_1)^{2}+(y_2-y_1)^{2}}, \qquad a = \frac{r_1^{2}-r_2^{2}+d^{2}}{2d}, \qquad h = \sqrt{r_1^{2}-a^{2}}.
$$

If $d > r_1 + r_2$ (too far apart) or $d < |r_1 - r_2|$ (one inside the other), the assembly is physically impossible. Otherwise, the two intersection points are

$$
P_{1,2} = \Bigl(x_1 + \tfrac{a}{d}(x_2-x_1) \mp \tfrac{h}{d}(y_2-y_1),\;\; y_1 + \tfrac{a}{d}(y_2-y_1) \pm \tfrac{h}{d}(x_2-x_1)\Bigr).
$$

Apply this with $C_1 = B$, $r_1 = l_2$, $C_2 = E$, $r_2 = l_3$. The two solutions correspond to the two assembly modes; on the prototype the leg housing physically excludes one mode, so we always select the second intersection point $D_2$. Once $D$ is chosen, the angle $\alpha_3$ is the angle between $\overrightarrow{ED}$ and $\overrightarrow{EA}$:

$$
\alpha_3 = \arccos\!\left( \frac{\overrightarrow{ED}\cdot\overrightarrow{EA}}{\|\overrightarrow{ED}\|\;\|\overrightarrow{EA}\|} \right),
$$

with the argument clamped to $[-1, 1]$ for numerical safety.

**Step 3 — Wheel-centre position.** The wheel centre $G$ lies on line $EG$ (link $l_4$), rigidly connected to $ED$ (link $l_3$). The orientation of $\overrightarrow{EG}$ in $\{0\}$ is $\alpha_3 + \alpha_5$, yielding

$$
{}^{G}_{0}p = \begin{bmatrix} l_5 c_5 + l_4\,c_{3,5} \\ l_5 s_5 + l_4\,s_{3,5} \end{bmatrix}.
$$

**Step 4 — Orientation of $l_2$ (needed for CoM).** The absolute orientation of $\overrightarrow{BD}$ is $\alpha_{11} + \alpha_{12}$, where

$$
\alpha_{12} = \pi - \arccos\frac{l_1^{2}+l_8^{2}-l_5^{2}}{2\,l_1 l_8} - \arccos\frac{l_2^{2}+l_8^{2}-l_3^{2}}{2\,l_2 l_8}.
$$

### 3.4 Pseudocode

```text
Algorithm 2: Forward Kinematics — Single Leg
Input : θ1 (crank offset, rad), α5_def, α11, link lengths l1..l5
Output: G_p (foot position), α3, α12

 1  α5 ← α5_def + |θ1|                          # absolute crank angle
 2  α9 ← α5 − α11
 3  // Diagonal of triangle ABE
 4  l8² ← l1² + l5² − 2·l1·l5·cos(α5 − α11)
 5  if l8² ≤ 0 then error "infeasible"
 6  // Circle intersection: B(l1,0) radius l2, E radius l3
 7  d ← sqrt( (l5·c5 − l1)² + (l5·s5)² )
 8  if d > l2 + l3 or d < |l2 − l3| then error "no intersection"
 9  a ← (l2² − l3² + d²) / (2d)
10  h ← sqrt(l2² − a²)
11  x_m ← l1 + a·(l5·c5 − l1)/d
12  y_m ← a·l5·s5 / d
13  D ← ( x_m − h·l5·s5/d , y_m + h·(l5·c5 − l1)/d )   # second intersection
14  // Angle α3
15  ED ← D − (l5·c5, l5·s5)
16  EA ← (−l5·c5, −l5·s5)
17  cosφ ← (ED·EA) / (‖ED‖·‖EA‖);   clamp to [−1, 1]
18  α3 ← arccos(cosφ)
19  // Wheel-centre position
20  G_px ← l5·c9 − l4·(sin α3·sin α9 − cos α3·cos α9)
21  G_py ← l4·(cos α3·sin α9 + cos α9·sin α3) + l5·s9
22  // Angle α12 (for CoM computation)
23  α10 ← arccos( (l1² + l2² − l8²) / (2·l1·l2) )
24  α12 ← π − α10
25  return G_p, α3, α12
```

> **Note.** On the prototype the active chain uses a slightly different angular convention than the passive chain. The quantity $\alpha_9 = \alpha_5 - \alpha_{11}$ re-aligns the crank angle to the body-fixed frame. On the physical robot, $\alpha_{11} = 41^\circ$ and $\alpha_8 = 151.45^\circ$ in the code.

---

## 4. Centre-of-Mass Computation

### 4.1 Why compute the CoM in closed form?

The balancing controller (Section 6) treats the robot as an inverted pendulum whose pivot is the wheel-contact point and whose "bob" is the overall CoM. The pendulum length $l_{cg}$ and the CoM offset angle $\Delta\beta$ directly parameterise the linearised dynamics. If these can be computed analytically from the two hip crank angles $(\theta_1, \theta_3)$ and the body tilt $\alpha_8$, then:

- The height-regulation loop can predict how the CoM moves when it changes $\theta_1, \theta_3$, and feed this prediction forward to the balancing loop.
- The balancing loop can compensate for any residual $\Delta\beta$ by adjusting its pitch reference.

This is the "analytic separability" that makes the decoupled two-LQR architecture possible.

### 4.2 Mass model

| Body | Mass (kg) | Local CoM $(\xi, \eta)$ (m) | Inertia about own CoM (kg·m²) |
|---|---|---|---|
| Chassis ($b$)      | 2.254 | $(0.01835\,c_8,\ 0.01835\,s_8)$ | 0.006316 |
| Crank ($l_5$)      | 0.062 | $(0.03815,\ 0.00331)$ | 0.000120 |
| Shank ($l_{34}$)   | 0.612 | $(0.075,\ 0.002)$ | 0.001839 |
| Coupler ($l_2$)    | 0.071 | $(0.03788,\ 0.00330)$ | 0.000111 |
| Wheel ($w$)        | 0.565 | at wheel centre | 0.001243 |

### 4.3 Computing each link's CoM in $\{0\}$

For a link whose proximal joint is at $p_i^{0}$ in $\{0\}$ and whose local orientation is given by $R(\phi)=\begin{bmatrix}\cos\phi&-\sin\phi\\\sin\phi&\cos\phi\end{bmatrix}$, the CoM in $\{0\}$ is

$$
p_{m_i}^{0} = R(\phi)\begin{bmatrix}\xi_i\\\eta_i\end{bmatrix} + p_i^{0}.
$$

Concretely, for each link on one leg (right or left, with crank angle $\alpha_5$):

$$
\begin{aligned}
p_{m_5}^{0} &= \begin{bmatrix} \xi_5 c_9 - \eta_5 s_9 \\ \xi_5 s_9 + \eta_5 c_9 \end{bmatrix}, \\
p_{m_4}^{0} &= \begin{bmatrix} l_5 c_9 + \xi_4\,c_{3,9} - \eta_4\,s_{3,9} \\ l_5 s_9 + \xi_4\,s_{3,9} + \eta_4\,c_{3,9} \end{bmatrix}, \\
p_{m_2}^{0} &= \begin{bmatrix} l_1 c_{11} + \xi_2\,c_{11,12} - \eta_2\,s_{11,12} \\ l_1 s_{11} + \xi_2\,s_{11,12} + \eta_2\,c_{11,12} \end{bmatrix}.
\end{aligned}
$$

Here $\alpha_9 = \alpha_5 - \alpha_{11}$, and $l_{34}$ denotes the rigid $l_3 + l_4$ assembly.

### 4.4 Total CoM and derived quantities

The total CoM (everything except the two wheels) is the mass-weighted sum over both legs:

$$
p_R^{0} = \frac{1}{M}\sum_{i\in\{b,l_5,l_4,l_2\}} m_i\left(p_{m_i}^{0,R}+p_{m_i}^{0,L}\right), \qquad M = m_b + 2(m_{l_5}+m_{l_4}+m_{l_2}).
$$

The wheel-centre midpoint is

$$
p_G^{0} = \tfrac{1}{2}\bigl({}^{G}_{0}p_R + {}^{G}_{0}p_L\bigr).
$$

Three quantities are passed to the controller:

1. **Effective pendulum length:** $l_{cg} = \|p_R^{0} - p_G^{0}\|$.
2. **CoM offset angle:** $\Delta\beta = \mathrm{atan2}(\Delta p_x,\, \Delta p_y)$, where $\Delta p = p_R^{0} - p_G^{0}$.
3. **Moment of inertia about the CoM** via the parallel-axis theorem:

$$
J_{cg} = \sum_{i} J_i^{\text{(local)}} + \sum_i m_i\,\bigl\|p_{m_i}^{0} - p_R^{0}\bigr\|^{2},
$$

where the sum runs over both legs and the body. The offset angle $\Delta\beta$ is used as the pitch reference (with a minus sign) so that the equilibrium places the CoM directly above the contact line. Using $\mathrm{atan2}$ rather than $\arctan$ preserves correctness when $\Delta p_y \to 0$ on tilted ground.

### 4.5 Pseudocode

```text
Algorithm 3: Centre of Mass, l_cg, Δβ, and J_cg
Input : θ1, θ3 (hip crank offsets, rad), α8 (body tilt), system constants
Output: l_cg, Δβ, J_cg

 1  // For each leg • ∈ {R, L}:
 2  for • in {R, L} do
 3      G_p•, α3•, α12• ← ForwardKinematics(θ•)        # Algorithm 2
 4      α9• ← α5_def + |θ•| − α11
 5      compute p_m5^(0,•), p_m4^(0,•), p_m2^(0,•)      # using the CoM formulas
 6  p_mb^(0) ← ( l_AC·cos α8 , l_AC·sin α8 )
 7  // Mass-weighted sum:
 8  p_R^(0) ← (1/M)·( m_b·p_mb + Σ_• [ m_l5·p_m5^(0,•) + m_l4·p_m4^(0,•) + m_l2·p_m2^(0,•) ] )
 9  p_G^(0) ← ½·( G_pR + G_pL )
10  Δp ← p_R^(0) − p_G^(0)
11  l_cg ← ‖Δp‖
12  Δβ ← atan2(Δp_x, Δp_y)
13  // Moment of inertia (parallel-axis):
14  J_cg ← [ J_b + m_b·‖p_mb − p_R^(0)‖² ]
15  for • in {R, L} do
16      J_cg += J_l5 + m_l5·‖p_m5^(0,•) − p_R^(0)‖²
17      J_cg += J_l4 + m_l4·‖p_m4^(0,•) − p_R^(0)‖²
18      J_cg += J_l2 + m_l2·‖p_m2^(0,•) − p_R^(0)‖²
19  return l_cg, Δβ, J_cg, G_pR, G_pL
```

---

## 5. Dynamic Modelling and State-Space Formulation

### 5.1 Why two models instead of one?

The full 3D dynamics involve at least 8 DOF (two wheel rotations, pitch, yaw, roll, body height, two hip angles) and are strongly coupled. A single monolithic model would be high-dimensional with poorly conditioned matrices, making LQR tuning unintuitive.

The key observation: the coupling between the balancing/forward/yaw subsystem and the height/roll subsystem is *weak* — they are linked only through the configuration-dependent scalars $l_{cg}$, $J_{cg}$, and $\Delta\beta$, all available in closed form (Section 4). We therefore build **two separate** low-order models:

- **Balancing model:** 6th-order, actuated by wheel torques $(\tau_L, \tau_R)$.
- **Body-leveling model:** 4th-order, actuated by hip forces $(F_{bl}, F_{br})$.

Both are individually controllable and well-conditioned for LQR design.

### 5.2 Wheel dynamics

For the right wheel, the translational (Newton) and rotational (Euler) equations are

$$
m_w\ddot{x}_R = f_R - H_R, \qquad \tau_R - f_R\,r = J_w\,\dot{\omega}_R,
$$

where $f_R$ is ground friction, $H_R$ the horizontal reaction force from the body, $\omega_R$ the wheel angular speed, and $J_w$ the wheel inertia. Under rolling without slipping, $\ddot{x}_R = r\,\dot{\omega}_R$, eliminating $f_R$ and $\dot\omega_R$:

$$
\Bigl(m_w+\frac{J_w}{r^{2}}\Bigr)\ddot{x}_R = \frac{\tau_R}{r} - H_R, \qquad \Bigl(m_w+\frac{J_w}{r^{2}}\Bigr)\ddot{x}_L = \frac{\tau_L}{r} - H_L.
$$

### 5.3 Body pitch dynamics (inverted pendulum)

Define the common forward coordinate $x = (x_R + x_L)/2$. The sprung mass $m_{rot} = m_b + 2(m_{l_5}+m_{l_4}+m_{l_2})$ translates with the wheel contact and rotates about the overall CoM with inertia $J_{cg}$. With $\theta$ the lean angle from vertical:

$$
\begin{aligned}
m_{rot}\,\frac{d^{2}}{dt^{2}}\bigl(x+l_{cg}\sin\theta\bigr) &= H_L+H_R, \\
m_{rot}\,\frac{d^{2}}{dt^{2}}\bigl(l_{cg}\cos\theta\bigr) &= V_L+V_R-m_{rot}g, \\
J_{cg}\,\ddot{\theta} &= (V_R+V_L)\,l_{cg}\sin\theta - (H_R+H_L)\,l_{cg}\cos\theta - (\tau_R+\tau_L).
\end{aligned}
$$

### 5.4 Longitudinal–pitch coupling and decoupling

Averaging the two wheel equations and eliminating $H_L + H_R$ yields the coupling between forward acceleration and pitch:

$$
\Bigl(2m_w+\frac{2J_w}{r^{2}}+m_{rot}\Bigr)\ddot{x} = \frac{\tau_R+\tau_L}{r} - m_{rot}l_{cg}\bigl(\ddot{\theta}\cos\theta - \dot{\theta}^{2}\sin\theta\bigr).
$$

Introduce the shorthands

$$
\begin{aligned}
a_1 &= \frac{1}{r\,(2m_w+2J_w/r^{2}+m_{rot})}, & a_2 &= -\frac{m_{rot}l_{cg}}{2m_w+2J_w/r^{2}+m_{rot}}, & a_3 &= -a_2, \\
b_1 &= \frac{m_{rot}l_{cg}\,g}{J_{cg}+m_{rot}l_{cg}^{2}}, & b_2 &= -\frac{m_{rot}l_{cg}}{J_{cg}+m_{rot}l_{cg}^{2}}, & b_3 &= -\frac{1}{J_{cg}+m_{rot}l_{cg}^{2}}.
\end{aligned}
$$

The coupling becomes $\ddot{x} = a_1(\tau_R+\tau_L) + a_2\ddot{\theta}\cos\theta + a_3\dot{\theta}^{2}\sin\theta$, and substituting into the moment balance yields $\ddot{\theta} = b_1\sin\theta + b_2\ddot{x}\cos\theta + b_3(\tau_R+\tau_L)$. With $\Delta_\theta \triangleq 1 - a_2 b_2\cos^{2}\theta$, solving the $2\times2$ system gives closed-form accelerations $\ddot{x} = f_x(\theta,\dot\theta,\tau_R{+}\tau_L)$ and $\ddot{\theta} = f_\theta(\theta,\dot\theta,\tau_R{+}\tau_L)$ — exact, with no small-angle assumptions.

### 5.5 Yaw dynamics

Differential ground reactions produce a yaw moment about the vertical axis:

$$
J_{\delta}\,\ddot{\delta} = \frac{D}{2}\,(H_L-H_R),
$$

where $D$ is the wheel track and $J_{\delta}$ the yaw inertia. Using $\ddot{\delta} = (\ddot{x}_L - \ddot{x}_R)/D$ and eliminating the reaction forces:

$$
\ddot{\delta} = \frac{\tau_L-\tau_R}{r\bigl(m_w D + \frac{J_w}{r^{2}}D + \frac{2J_{\delta}}{D}\bigr)}.
$$

### 5.6 Linearisation about the upright equilibrium

Collect the states into

$$
Z = (x,\; \dot{x},\; \theta,\; \dot{\theta},\; \delta,\; \dot{\delta})^{\top}, \qquad U = (\tau_L,\; \tau_R)^{\top}.
$$

Linearising the nonlinear dynamics $\dot{Z} = f(Z, U)$ about $\bar{Z} = (x, 0, 0, 0, \delta, 0)^{\top}$, $\bar{U} = (0,0)^{\top}$ (i.e. evaluating the Jacobians at $\theta = 0$, $\dot\theta = 0$) gives

$$
A = \begin{bmatrix}
0 & 1 & 0 & 0 & 0 & 0\\
0 & 0 & -\frac{a_2 b_1}{a_2 b_2-1} & 0 & 0 & 0\\
0 & 0 & 0 & 1 & 0 & 0\\
0 & 0 & -\frac{b_1}{a_2 b_2-1} & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 1\\
0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix},
\qquad
B = \begin{bmatrix}
0 & 0\\
-\frac{a_1+a_2 b_3}{a_2 b_2-1} & -\frac{a_1+a_2 b_3}{a_2 b_2-1}\\
0 & 0\\
-\frac{b_3+a_1 b_2}{a_2 b_2-1} & -\frac{b_3+a_1 b_2}{a_2 b_2-1}\\
0 & 0\\
\frac{1}{r\eta} & -\frac{1}{r\eta}
\end{bmatrix},
$$

with $\eta \triangleq D m_w + \frac{2J_{\delta}}{D} + \frac{D J_w}{r^{2}}$.

### 5.7 Body-leveling model

On uneven ground, the vertical forces $F_{bl}$ and $F_{br}$ regulate body height $y$ and body roll $\gamma$:

$$
F_{bl}+F_{br} - m_b g = m_b\,\ddot{y}, \qquad \frac{L_b}{2}\,(F_{br}-F_{bl}) = I_b\,\ddot{\gamma},
$$

where $I_b$ is the body roll inertia and $L_b$ the lateral leg spacing. With state $Q = (y, \dot{y}, \gamma, \dot{\gamma})^{\top}$ and input $(F_{bl}, F_{br})^{\top}$, the system is already linear:

$$
\dot{Q} = \underbrace{\begin{bmatrix}0&1&0&0\\0&0&0&0\\0&0&0&1\\0&0&0&0\end{bmatrix}}_{A_h} Q + \underbrace{\begin{bmatrix}0&0\\1/m_b&1/m_b\\0&0\\-L_b/(2I_b)&L_b/(2I_b)\end{bmatrix}}_{B_h} \begin{bmatrix}F_{bl}\\F_{br}\end{bmatrix} - \begin{bmatrix}0\\g\\0\\0\end{bmatrix}.
$$

The pair $(A_h, B_h)$ is controllable (controllability matrix has rank 4), so height and roll can be independently regulated by the two legs.

---

## 6. LQR Controller Design

### 6.1 Infinite-horizon LQR: the basics

For a linear time-invariant system $\dot{e} = A e + B u$ where $e$ is the tracking error, the infinite-horizon LQR problem finds the state-feedback gain $K$ minimising

$$
\mathcal{J} = \int_{0}^{\infty}\bigl(e^{\top}Q\,e + u^{\top}R\,u\bigr)\,dt,
$$

with $Q \succeq 0$ and $R \succ 0$. The solution is

$$
u = -K e, \qquad K = R^{-1} B^{\top} P,
$$

where $P \succ 0$ is the unique stabilising solution of the Continuous Algebraic Riccati Equation (CARE):

$$
A^{\top}P + PA - P B R^{-1} B^{\top} P + Q = 0.
$$

In practice the CARE is solved numerically via the Schur-vector method (`scipy.linalg.solve_continuous_are` or MATLAB's `lqr`).

### 6.2 Balancing / locomotion LQR

The linearised model is $(A, B)$ from Section 5.6, with state $Z = (x, \dot{x}, \theta, \dot{\theta}, \delta, \dot{\delta})^{\top}$. The reference is

$$
Z_d = (x_d,\; \dot{x}_d,\; \theta_d,\; 0,\; \delta_d,\; 0)^{\top},
$$

where $\dot{x}_d$ and $\delta_d$ come from the remote operator and $\theta_d = -\Delta\beta$ (placing the CoM above the contact line). The wheel-torque law is

$$
\begin{bmatrix}\tau_L\\\tau_R\end{bmatrix} = -K_1\,(Z-Z_d).
$$

**Weight selection rationale:**

- The pitch error $\theta - \theta_d$ receives the highest penalty ($q_{33} = 300$) — upright posture is the primary task.
- Forward velocity error $\dot{x} - \dot{x}_d$ and forward position $x - x_d$ receive moderate penalties ($q_{11} = 10$, $q_{22} = 1$).
- Pitch rate $\dot{\theta}$ is penalised to damp oscillations ($q_{44} = 15$).
- Yaw and yaw rate receive light penalties ($q_{55} = 2$, $q_{66} = 1$).
- Control effort is penalised symmetrically on the two wheels ($R_1 = \operatorname{diag}(1.5,\, 1.5)$).

$$
Q_1 = \mathrm{diag}(10,\; 1,\; 300,\; 15,\; 2,\; 1), 
\qquad 
R_1 = \mathrm{diag}(1.5,\; 1.5).
$$

### 6.3 Body-leveling LQR

The body-leveling model $(A_h, B_h)$ from Section 5.7 is stabilised by a second LQR with gravity feedforward:

$$
\begin{bmatrix}F_{bl}\\F_{br}\end{bmatrix} = K_2\,(Q_d - Q) + \begin{bmatrix}m_b g/2\\ m_b g/2\end{bmatrix},
$$

where $Q_d = (y_d, 0, \gamma_d, 0)^{\top}$, with $y_d$ the commanded body height and $\gamma_d$ the desired roll (zero for self-levelling, or equal to the measured ground slope for adaptation). The prototype uses

$$
Q_2 = \mathrm{diag}(50000,\; 500,\; 5000,\; 100), 
\qquad 
R_2 = I_2.
$$

The height error receives the highest penalty ($q_{11} = 5\times10^{4}$), followed by roll error ($q_{33} = 5000$).

### 6.4 Pseudocode

```text
Algorithm 4: LQR Gain Computation for One Posture
Input : θ1, θ3 (hip offsets), system constants, weight matrices Q, R
Output: K1 (wheel-torque gain) or K2 (body-leveling gain)

 1  // 1. Compute inertial quantities:
 2  l_cg, Δβ, J_cg, G_pR, G_pL ← ComputeCoM(θ1, θ3)     # Algorithm 3
 3  // 2. Compute coefficients:
 4  m_rot ← m_b + 2·(m_l5 + m_l4 + m_l2)
 5  a1, a2, a3 ← (coefficient eqs);   b1, b2, b3 ← (coefficient eqs)
 6  // 3. Assemble linearised state-space:
 7  A, B ← (6×6 A and 6×2 B)
 8  // 4. Solve CARE and compute gain:
 9  solve  Aᵀ·P + P·A − P·B·R⁻¹·Bᵀ·P + Q = 0   for  P ≻ 0
10  K ← R⁻¹·Bᵀ·P
11  return K
```

> **Note.** For the body-leveling LQR, steps 2–4 use $(A_h, B_h, Q_2, R_2)$ instead, and step 1 is skipped because $A_h$ and $B_h$ are configuration-independent.

---

## 7. Gain Scheduling

### 7.1 Why schedule the gains?

The state-space matrices $(A, B)$ depend on $l_{cg}$ and $J_{cg}$, which depend on the hip crank angles $\theta_1$ and $\theta_3$. Changing body height changes $l_{cg}$, which changes the eigenvalues of $A$, which changes the optimal LQR gain. A single fixed gain is valid only near the height for which it was computed. Similarly, the Jacobian transpose used in the force mapping (Section 8) varies with $\alpha_5$.

The solution is **gain scheduling**: precompute everything over a dense grid of $(\theta_1, \theta_3)$, store the results in a lookup table (LUT), and do a constant-time table read at runtime instead of solving the Riccati equation on-line.

### 7.2 Grid construction (offline)

The two hip crank offsets span $[0^\circ,\, 35^\circ]$ independently, sampled at $0.5^\circ$ increments:

$$
|\Theta| = 71 \;\Rightarrow\; 71\times71 = 5041\ \text{nodes}.
$$

At each node $(\theta_1^{\text{deg}}, \theta_3^{\text{deg}})$ we compute and store:

- $K_1$ — the $2\times6$ wheel-torque LQR gain.
- $K_2$ — the $2\times4$ body-leveling LQR gain.
- $\Delta\beta$ — the CoM offset angle.
- ${}^{G}_{0}p_R$, ${}^{G}_{0}p_L$ — foot positions (used to compute body height for height feedback).
- $J_{\text{foot}}^R$, $J_{\text{foot}}^L$ — the $2\times1$ foot Jacobians $\partial\,{}^{G}_{0}p/\partial\alpha_5$, used for force-to-torque mapping.
- $l_{cg}$, $J_{cg}$ — for diagnostics.

The grid is serialised to disk for loading at boot time.

### 7.3 Online lookup

At runtime, the measured hip crank angles $(\theta_1, \theta_3)$ are snapped to the nearest $0.5^\circ$ grid point:

$$
\tilde{\theta} = \mathrm{round}(\theta \times 2)\,/\,2.
$$

This replaces an on-line Riccati solve (tens of milliseconds on an embedded processor) with a dictionary lookup (nanoseconds).

```text
Algorithm 5: Offline Gain-Scheduling Table Generation
Input : grid Θ = {0, 0.5, 1, ..., 35} (degrees)
Output: T (lookup table, |T| = 5041)

 1  T ← { }                                    # empty dictionary
 2  for θ1_deg in Θ do
 3      for θ3_deg in Θ do
 4          θ1_rad ← θ1_deg · π/180
 5          θ3_rad ← θ3_deg · π/180
 6          l_cg, Δβ, J_cg, G_pR, G_pL ← ComputeCoM(θ1_rad, θ3_rad)
 7          K1 ← LQR(θ1_rad, θ3_rad)            # Algorithm 4
 8          K2 ← BodyLQR()                      # same, with A_h, B_h, Q2, R2
 9          J_foot^R ← ∂G_pR / ∂α5              # analytical Jacobian
10          J_foot^L ← ∂G_pL / ∂α5
11          T[(θ1_deg, θ3_deg)] ← { K1, K2, Δβ, G_pR, G_pL, J_foot^R, J_foot^L }
12  return T
```

---

## 8. Force-to-Torque Mapping for Body Leveling

### 8.1 Why is mapping needed?

The body-leveling LQR (Section 6) outputs desired vertical forces $(F_{bl}, F_{br})$ in the gravity-aligned (world) frame. To produce these forces, the hip motors must apply torques at the crank joints. The foot (wheel centre) is the point through which the leg transmits force to the ground; by the principle of virtual work, the mapping from a foot-referenced force to a hip torque is given by the transpose of the foot Jacobian.

### 8.2 Step 1 — Rotate the force into the leg frame

The body-leveling forces act vertically in the world frame. The leg is tilted by $\Delta\beta$ relative to the world vertical, so we rotate by $\Delta\beta$ in the $x$–$y$ plane:

$$
\begin{bmatrix}
F_{x\bullet}\\
F_{y\bullet}
\end{bmatrix}
=
\mathrm{rotz}(\Delta\beta)^{-1}
\begin{bmatrix}
0\\
F_{b\bullet}
\end{bmatrix},
\qquad \bullet\in\{l,r\}.
$$

where $\operatorname{rotz}(\Delta\beta)^{-1} = \begin{bmatrix}\cos\Delta\beta & \sin\Delta\beta \\ -\sin\Delta\beta & \cos\Delta\beta\end{bmatrix}$ is the inverse (transpose) of the elementary $z$-axis rotation.

### 8.3 Step 2 — Project through the foot Jacobian transpose

The foot position is a function of the single crank input: ${}^{G}_{0}p = {}^{G}_{0}p(\alpha_5)$. Its derivative is the foot Jacobian:

$$
{}^{G}_{0}J = \frac{\partial\,{}^{G}_{0}p}{\partial\alpha_5} = \begin{bmatrix} \partial\,{}^{G}_{0}p_x / \partial\alpha_5 \\ \partial\,{}^{G}_{0}p_y / \partial\alpha_5 \end{bmatrix}.
$$

Because $\alpha_3$ itself depends on $\alpha_5$ through the four-bar loop, the Jacobian must account for this chain rule. The analytical derivation (rather than numerical differentiation) is faster and avoids finite-difference noise entering the torque commands. By the principle of virtual work, the hip torque required to produce a foot force $(F_{x\bullet}, F_{y\bullet})$ is

$$
\tau_{\bullet} = {}^{G}_{0}J_{\bullet}^{\top}\begin{bmatrix}F_{x\bullet}\\F_{y\bullet}\end{bmatrix} = \frac{\partial\,{}^{G}_{0}p_x}{\partial\alpha_5}\,F_{x\bullet} + \frac{\partial\,{}^{G}_{0}p_y}{\partial\alpha_5}\,F_{y\bullet}.
$$

### 8.4 Step 3 — Saturate

All motor commands are software-clamped to $[-8,\, 8]$ N·m to respect the actuator hardware limits.

```text
Algorithm 6: Force-to-Torque Mapping (Single Leg)
Input : F_b• (body-leveling force), Δβ, J_foot (foot Jacobian)
Output: τ• (hip torque, N·m)

 1  // 1. Resolve into leg frame:
 2  F_x ←  F_b• · sin Δβ
 3  F_y ←  F_b• · cos Δβ
 4  // 2. Project through Jacobian transpose:
 5  τ ← J_foot[0]·F_x + J_foot[1]·F_y
 6  // 3. Clamp to hardware limits:
 7  τ ← clamp(τ, −8.0, 8.0)
 8  return τ
```

---

## 9. Teleoperation (Remote Control)

### 9.1 Joystick-to-reference mapping

The operator uses a wireless gamepad with two analogue joysticks and several buttons. The joystick axes are read as unsigned 8-bit integers in $[0, 255]$ with centre at $127$. The mapping to robot references is:

| Joystick | Robot reference | Formula |
|---|---|---|
| Left-Y (LY)  | Forward speed $\dot{x}_d$ (m/s) | $-(LY-127)/127 \times (0.6 + \text{addv})$ |
| Left-X (LX)  | Yaw rate $\dot\delta_d$ (rad/s) | $-(LX-127)/127 \times 0.8$ |
| Right-Y (RY) | Body height $y_d$ (m) | interpolated: RY ↑ ⇒ taller |
| Right-X (RX) | Desired roll $\gamma_d$ (deg) | $(RX-127)/127 \times 5^{\circ}$ |

The speed gain `addv` is a runtime-adjustable bias (default $0$, adjustable via the triangle/cross buttons between $-0.4$ and $+0.2$) that lets the operator fine-tune the maximum forward speed.

### 9.2 Button-to-action mapping

| Button | Action |
|---|---|
| UP       | Enable balancing (activate both LQR loops) |
| DOWN     | Disable balancing and hip control (safe stop) |
| LEFT     | Set both hip cranks to a fixed angle (manual leg control) |
| RIGHT    | Release manual leg control, resume LQR |
| triangle | Increment speed gain `addv` by $+0.001$ |
| cross    | Decrement speed gain `addv` by $-0.001$ |
| circle   | Set yaw rate to $+5$ rad/s (turn left) |
| box      | Set yaw rate to $-5$ rad/s (turn right) |
| R + L    | Trigger a jump manoeuvre |

### 9.3 Reference integration

Velocity references are integrated to position references at each control step:

$$
x_d \leftarrow x_d + \dot{x}_d\,\Delta t, \qquad \delta_d \leftarrow \delta_d + \dot\delta_d\,\Delta t,
$$

where $\Delta t = 1/400$ s. The pitch reference is $\theta_d = -\Delta\beta$ (not set by the operator), and the roll reference $\gamma_d$ is either zero (self-levelling) or set by the right joystick X-axis.

---

## 10. Runtime Architecture

### 10.1 Thread model and timing

The on-board software runs three concurrent threads plus the main control loop:

| Thread | Task | Rate | Interface |
|---|---|---|---|
| IMU reader    | Parse serial IMU frames | ~400 Hz | UART |
| Motor reader  | Poll CAN-bus encoders   | ~400 Hz | CAN |
| Remote reader | Parse gamepad UART data | ~100 Hz | UART |
| **Main loop** | **Control + dispatch**  | **400 Hz** | — |

All threads use mutex locks to protect the shared state variables (motor angles, speeds, IMU attitude, remote input).

### 10.2 Initialisation sequence

1. Load the two precomputed lookup tables from disk into RAM.
2. Open serial connections to the IMU and the gamepad receiver.
3. Open the CAN bus and configure the four actuators for position-control mode.
4. Command both hip cranks to their home position ($\theta_1 \approx -15^\circ$, $\theta_3 \approx +15^\circ$).
5. Wait for the operator to press the UP button (which transitions the state machine to "active").

### 10.3 State machine

The main loop is governed by a simple state variable `start_num`:

- `start_num = 0` or `−1` — **Idle.** All motors are passive or position-controlled. The robot can be manually positioned.
- `start_num = 1` — **Transition.** Wheel motors are switched to torque mode and the LQR gains are read from the table. Immediately transitions to state 3.
- `start_num = 3` — **Active LQR control.** The full main loop runs.

```text
Algorithm 7: On-Board Runtime Main Loop
Input : T1 (wheel LUT, 5041 entries), T2 (body LUT, 5041 entries)

 1  // Initialise
 2  ẋ_d, δ̇_d, y_d, γ_d, addv ← 0;   start_num ← 0
 3  load T1, T2 from disk
 4  start IMU, CAN, and UART reader threads
 5  while true do
 6      // Read sensors (protected by mutexes):
 7      θ, θ̇, γ, γ̇, δ, δ̇ ← IMU
 8      θ1, θ3 ← hip crank angles
 9      ω_R, ω_L ← wheel speeds via encoders
10      parsed ← most recent remote input
11      if parsed.type = button then
12          execute button action
13      else if parsed.type = joystick then
14          ẋ_d, δ̇_d, y_d, γ_d ← map joystick axes        # Section 9
15          x_d ← x_d + ẋ_d·Δt;   δ_d ← δ_d + δ̇_d·Δt
16      if start_num = 1 then                              # transition
17          switch wheels to torque mode;   start_num ← 3
18      else if start_num = 3 then                         # active LQR
19          // Snap to grid and fetch gains:
20          θ̃1 ← round(|θ1|·2) / 2
21          θ̃3 ← round(|θ3|·2) / 2
22          K1, K2, Δβ, G_pR, G_pL, J_foot^R, J_foot^L ← T[(θ̃1, θ̃3)]
23          // Wheel LQR:
24          Z   ← ( x_d, (ω_R + ω_L)·r/2, θ, θ̇, δ, δ̇ )ᵀ
25          Z_d ← ( x_d, ẋ_d, −Δβ, 0, δ_d, δ̇_d )ᵀ
26          (τ_L, τ_R)ᵀ ← −K1·(Z − Z_d)
27          τ_L, τ_R ← clamp(τ_L,R, −8, 8)
28          // Body-leveling LQR:
29          y  ← −(G_pR_y + G_pL_y) / 2                    # height from foot positions
30          ẏ  ← −(v_R_y + v_L_y) / 2                      # vertical velocity
31          Q   ← ( y, ẏ, γ, γ̇ )ᵀ
32          Q_d ← ( y_d, 0, γ_d, 0 )ᵀ
33          (F_bl, F_br)ᵀ ← −K2·(Q − Q_d) + ( m_b·g/2, m_b·g/2 )ᵀ
34          // Map forces to hip torques:
35          τ_l ← ForceToTorque(F_bl, Δβ, J_foot^L)        # Algorithm 6
36          τ_r ← ForceToTorque(F_br, Δβ, J_foot^R)        # Algorithm 6
37          // Dispatch via CAN:
38          Motor 1 (right hip)   ← −τ_r;   Motor 3 (left hip)   ← +τ_l
39          Motor 2 (right wheel) ← −τ_R;   Motor 4 (left wheel) ← +τ_L
40      else
41          idle: zero torque or position hold
42      log state at 100 Hz (every 4th iteration)
```

> **Note.** Motor sign conventions: on the prototype, the right-hip and right-wheel motors (IDs 1 and 2) are mounted with opposite orientation to the left side (IDs 3 and 4), so the software applies sign reversals when dispatching torques to match the common "positive forward" convention used in the derivation. These signs are set at the CAN layer and are not part of the algorithmic logic.

---

## Parameter Compendium

Complete parameter set used in all algorithms (prototype values).

**Optimised link lengths**

| Symbol | Value |
|---|---|
| $l_1$ | 62.4 mm |
| $l_2$ | 101.0 mm |
| $l_3$ | 30.3 mm |
| $l_4$ | 99.5 mm |
| $l_5$ | 98.6 mm |

**Masses**

| Symbol | Value |
|---|---|
| $m_b$ | 2.254 kg |
| $m_{rot}$ | 3.744 kg |
| $m_{l_2}$ | 0.071 kg |
| $m_{l_5}$ | 0.062 kg |
| $m_{l_4}$ | 0.612 kg |
| $m_w$ | 0.565 kg |

**Geometry**

| Symbol | Value |
|---|---|
| $r$ | 62.25 mm |
| $D$ | 240.4 mm |
| $L_b$ | 125 mm |
| $l_{AC}$ | 18.35 mm |

**Inertias**

| Symbol | Value |
|---|---|
| $J_w$ | $1.243\times10^{-3}$ kg·m² |
| $J_{\delta}$ | $2.854\times10^{-2}$ kg·m² |
| $J_b$ | $6.316\times10^{-3}$ kg·m² |
| $I_b$ | $6.436\times10^{-3}$ kg·m² |
| $J_{l_2}$ | $1.11\times10^{-4}$ kg·m² |
| $J_{l_5}$ | $1.20\times10^{-4}$ kg·m² |
| $J_{l_4}$ | $1.84\times10^{-3}$ kg·m² |

**Angular constants**

| Symbol | Value |
|---|---|
| $\alpha_5^{\text{def}}$ | $53.33^\circ$ |
| $\alpha_{11}$ | $41^\circ$ |
| $\alpha_8$ (home) | $151.45^\circ$ |
| $\theta_1, \theta_3$ range | $[0^\circ, 35^\circ]$ |
| Grid step | $0.5^\circ$ |

**Controller weights**

| Symbol | Value |
|---|---|
| $Q_1$ | $\mathrm{diag}(10, 1, 300, 15, 2, 1)$ |
| $R_1$ | $\mathrm{diag}(1.5,\, 1.5)$ |
| $Q_2$ | $\mathrm{diag}(50000, 500, 5000, 100)$ |
| $R_2$ | $I_2$ |

**Hardware limits & rates**

| Symbol | Value |
|---|---|
| $\tau_{\max}$ (all motors) | $\pm 8$ N·m |
| Control rate | 400 Hz |
| Logging rate | 100 Hz |

---

**A note on angular conventions.** The manuscript keeps $\alpha_{11}$ (the mounting orientation of $l_1$) and $\alpha_8$ (the body tilt) as symbolic quantities; on the prototype the assembly constant is $\alpha_{11} = 41^\circ$. The auxiliary angle $\alpha_9 = \alpha_5 - \alpha_{11}$, which appears throughout the kinematic expressions, simply re-expresses the crank angle in the base frame. The fixed value $\alpha_8 = 151.45^\circ$ in the deployed code is the body-frame orientation at which the chassis CoM is evaluated.
