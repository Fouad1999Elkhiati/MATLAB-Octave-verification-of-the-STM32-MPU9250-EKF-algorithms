# MATLAB/Octave verification of the STM32 MPU9250 EKF algorithms

This adds two self-contained verification scripts for the algorithms
described in the paper (`attitude_ekf_magcal_paper.pdf`). Both are
direct 1:1 ports of the C source (`ekf.c`, `magcal.c`,
`Gauss_elimination.c`) into MATLAB/Octave, tested against synthetic
ground truth so correctness can be checked without any STM32 hardware.

They were written and validated in **GNU Octave 8.4** (a free,
MATLAB-language-compatible interpreter) and require no toolboxes
beyond base MATLAB/Octave. They also run unmodified in MATLAB itself
(tested syntax is R2016b+ compatible: local functions inside a script
file).

## Files

- **`mekf_verification.m`** — verifies the 6-state multiplicative EKF
  (`ekf.c`).
  - Simulates a true attitude trajectory from a known, time-varying
    angular-rate profile and a known constant gyro bias.
  - Synthesizes noisy gyro/accel/mag samples from that ground truth,
    including randomly injected linear-acceleration spikes to
    exercise the accelerometer's `|a| ≈ 1g` gate.
  - Runs the ported filter starting from a deliberately wrong initial
    attitude (30° error) and zero bias.
  - Reports steady-state attitude error (RMS/max, in degrees) and
    final gyro-bias estimation error, with a pass/fail check
    (< 1° RMS attitude error, < 0.003 rad/s bias error).
  - Saves two plots: `mekf_verification_euler.png` (true vs.
    estimated roll/pitch/yaw and the attitude-error time series) and
    `mekf_verification_bias.png` (bias convergence).

  **Result on the last run:** 30° → 0.65° RMS steady-state attitude
  error, bias recovered to within 0.0012 rad/s.

- **`magcal_verification.m`** — verifies the ellipsoid hard/soft-iron
  fit (`magcal.c`) and its Gauss–Jordan solver (`Gauss_elimination.c`).
  Three parts:
  - **Part A** checks the ported Gauss–Jordan elimination/inversion
    routine against MATLAB's built-in solver on 200 random 6×6
    SPD systems (relative error ~1e-14 — floating-point-exact
    agreement).
  - **Part B** generates ~1500 near-uniformly distributed unit
    vectors (Fibonacci-sphere sampling, emulating a full-orientation
    calibration sweep), distorts them with a known hard-iron offset
    and soft-iron per-axis scale, runs the identical
    accumulate/solve/apply pipeline as `magcal_add`/`magcal_solve`/
    `magcal_apply`, and compares the recovered offset/scale against
    ground truth. Saves `magcal_verification_spheres.png` showing the
    raw distorted ellipsoid next to the calibrated unit sphere.
  - **Part C** feeds a poorly-distributed calibration sweep (samples
    clustered in a 10° cone) and confirms the fit is correctly
    flagged as degenerate (hyperboloid / singular normal matrix),
    matching the safeguards in `magcal_solve`.

  **Result on the last run:** offset recovered to within 0.01 raw
  units, scale to within 0.03%, calibrated-vector norm 1.0000 ± 0.003,
  and the degenerate sweep was correctly rejected.

## How to run

```bash
# Octave (free, no license needed)
octave-cli mekf_verification.m
octave-cli magcal_verification.m

# MATLAB
matlab -batch "mekf_verification"
matlab -batch "magcal_verification"
# or just open each file and press Run
```

Both scripts are headless-safe (figures are created with
`'visible','off'` and saved directly to PNG), so they work the same
way in a CI job, a terminal, or an interactive desktop session.

## What "verification" means here

These scripts do not re-run the STM32 firmware itself (that would
require flashing hardware and a physical calibration rig). Instead
they verify the *mathematics*: every line of `ekf_predict`,
`vector_update`, `magcal_add`, `magcal_solve`, and the Gauss–Jordan
solver has a corresponding line in these `.m` files, and the ported
algorithm is checked against independently generated ground truth and
against MATLAB's own trusted linear-algebra primitives. This confirms
the equations derived in the paper are (a) internally consistent, (b)
correctly transcribed from the C source, and (c) numerically well
behaved on realistic synthetic data — the same kind of check you would
do before trusting the estimator on real sensor data.
