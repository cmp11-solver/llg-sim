# llg-sim

A two-dimensional micromagnetic solver implementing the Landau–Lifshitz–Gilbert
equation with adaptive mesh refinement and composite multigrid demagnetising-field
computation.  Written in Rust.

Developed as part of MPhys project CMP11.

## Features

- Full LLG dynamics with Dormand–Prince RK45 adaptive integration and
  Bogacki–Shampine RK23 damping-only relaxation
- Effective fields: exchange, uniaxial anisotropy, interfacial DMI (Néel-type),
  Zeeman, demagnetising field via FFT convolution of the Newell tensor
- Block-structured adaptive mesh refinement with temporal subcycling and
  dynamic regridding
- Composite multigrid demagnetising field with direct Newell near-field
  correction on AMR patch hierarchies
- Validated against MuMax3 on µMAG Standard Problems 2 and 4
- Antidot geometry benchmark demonstrating composite multigrid accuracy
  at shaped boundaries
- Vortex gyration benchmark with frequency extraction validated against
  the Guslienko analytical prediction

## Build

Requires `rustc` + `cargo`.  Python 3 with `numpy` and `matplotlib` for
post-processing scripts.
```bash
git clone https://github.com/cmp11-solver/llg-sim.git
cd llg-sim
cargo build --release
cargo test
```

## Reproducing report results

Each benchmark binary documents its configuration and expected outputs in a
header comment.  Outputs are written to `out/` and `runs/` (gitignored).
MuMax3 reference data is expected in `mumax_outputs/` for comparison scripts.

**Standard Problem 4 & 2 (Section 3)**
```bash
cargo run --release --bin st_problems -- sp4 a
cargo run --release --bin st_problems -- sp2
```

**Antidot composite multigrid benchmark (Section 4.1)**
```bash
LLG_DEMAG_COMPOSITE_VCYCLE=1 LLG_COMPOSITE_MAX_CYCLES=1 \
  LLG_CV_BASE_NX=256 \
  cargo run --release --bin bench_composite_vcycle -- --plots
```

**Antidot crossover sweep**
```bash
LLG_DEMAG_COMPOSITE_VCYCLE=1 LLG_COMPOSITE_MAX_CYCLES=1 \
  LLG_CV_SWEEP_SIZES=32,64,96,128,256,512 \
  cargo run --release --bin bench_composite_vcycle -- --sweep --mg-only
```

**Vortex gyration (Section 4.2)**
```bash
LLG_AMR_SMOOTH_ALPHA=1.0 LLG_DEMAG_COMPOSITE_VCYCLE=1 \
  LLG_COMPOSITE_MAX_CYCLES=1 LLG_GYR_TIME_NS=2 \
  cargo run --release --bin bench_vortex_gyration -- --amr-only --skip-cfft --plots
```

**Component-level validation (Appendix A)**
```bash
cargo run --release --bin macrospin_fmr
cargo run --release --bin macrospin_anisotropy
cargo run --release --bin bloch_dmi -- dmi=1e-4
cargo run --release --bin bloch_dmi -- dmi=-1e-4
cargo run --release --bin demag_cube
cargo run --release --bin demag_2d
```

> **Note on runtime.**
   > The AMR benchmarks (anti-dot sweep, vortex gyration) can take tens of
   > minutes to several hours on a standard computer. Component-level tests
   > in Appendix A run in seconds. `RAYON_NUM_THREADS` controls parallelism.

## License

MIT
