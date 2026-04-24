// tests/demag_methods.rs
//
// Integration tests for the demag dispatcher wiring:
// - When DemagMethod::FftUniform is selected, `effective_field::demag` should route to
//   `effective_field::demag_fft_uniform`.
// - When DemagMethod::PoissonMG is selected, `effective_field::demag` should route to
//   `effective_field::demag_poisson_mg`.
//
// These tests are about *wiring*, not about proving physical equivalence between FFT and MG.

use llg_sim::effective_field::{demag, demag_fft_uniform, demag_poisson_mg};
use llg_sim::grid::Grid2D;
use llg_sim::params::{DemagMethod, Material};
use llg_sim::vector_field::VectorField2D;

use std::sync::{Mutex, OnceLock};

static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

#[test]
fn demag_dispatch_routes_to_fft_uniform() {
    let _guard = ENV_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
    configure_env_for_demag_tests();

    let grid = Grid2D::new(8, 8, 5e-9, 5e-9, 1e-9);
    let mut m = VectorField2D::new(grid);
    init_pattern(&mut m);

    let mat = Material {
        ms: 8.0e5,
        a_ex: 0.0,
        k_u: 0.0,
        easy_axis: [0.0, 0.0, 1.0],
        dmi: None,
        demag: true,
        demag_method: DemagMethod::FftUniform,
    };

    let mut b_dispatch = VectorField2D::new(grid);
    let mut b_direct = VectorField2D::new(grid);

    demag::compute_demag_field(&grid, &m, &mut b_dispatch, &mat);
    demag_fft_uniform::compute_demag_field(&grid, &m, &mut b_direct, &mat);

    let max_abs = max_abs_diff(&b_dispatch, &b_direct);
    assert!(
        max_abs < 1e-12,
        "dispatcher vs direct FFT mismatch: max_abs={}",
        max_abs
    );
}

#[test]
fn demag_dispatch_routes_to_poisson_mg() {
    let _guard = ENV_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
    configure_env_for_demag_tests();

    let grid = Grid2D::new(8, 8, 5e-9, 5e-9, 1e-9);
    let mut m = VectorField2D::new(grid);
    init_pattern(&mut m);

    let mat = Material {
        ms: 8.0e5,
        a_ex: 0.0,
        k_u: 0.0,
        easy_axis: [0.0, 0.0, 1.0],
        dmi: None,
        demag: true,
        demag_method: DemagMethod::PoissonMG,
    };

    let mut b_dispatch = VectorField2D::new(grid);
    let mut b_direct = VectorField2D::new(grid);

    demag::compute_demag_field(&grid, &m, &mut b_dispatch, &mat);
    demag_poisson_mg::compute_demag_field_poisson_mg(&grid, &m, &mut b_direct, &mat);

    let max_abs = max_abs_diff(&b_dispatch, &b_direct);
    assert!(
        max_abs < 1e-12,
        "dispatcher vs direct MG mismatch: max_abs={}",
        max_abs
    );
}

fn configure_env_for_demag_tests() {
    unsafe {
        std::env::remove_var("LLG_DEMAG_METHOD");
        std::env::set_var("LLG_DEMAG_MG_WARM_START", "0");
        std::env::set_var("LLG_DEMAG_MG_PAD_FACTOR_XY", "2");
        std::env::set_var("LLG_DEMAG_MG_NVAC_Z", "2");
        std::env::set_var("LLG_DEMAG_MG_VCYCLES", "2");
        std::env::remove_var("LLG_DEMAG_MG_TOL_ABS");
    }
}

fn init_pattern(m: &mut VectorField2D) {
    let nx = m.grid.nx;
    let ny = m.grid.ny;
    let two_pi = 2.0 * std::f64::consts::PI;

    for j in 0..ny {
        for i in 0..nx {
            let x = (i as f64 + 0.5) / (nx as f64);
            let y = (j as f64 + 0.5) / (ny as f64);

            let mut mx: f64 = (two_pi * x).cos();
            let mut my: f64 = (two_pi * y).sin();
            let mut mz: f64 = 0.2;

            let n = (mx * mx + my * my + mz * mz).sqrt().max(1e-30);
            mx /= n;
            my /= n;
            mz /= n;

            let id = m.idx(i, j);
            m.data[id][0] = mx;
            m.data[id][1] = my;
            m.data[id][2] = mz;
        }
    }
}

fn max_abs_diff(a: &VectorField2D, b: &VectorField2D) -> f64 {
    assert_eq!(a.data.len(), b.data.len());
    let mut max_abs: f64 = 0.0;
    for (va, vb) in a.data.iter().zip(b.data.iter()) {
        let dx = (va[0] - vb[0]).abs();
        let dy = (va[1] - vb[1]).abs();
        let dz = (va[2] - vb[2]).abs();
        max_abs = max_abs.max(dx.max(dy).max(dz));
    }
    max_abs
}
