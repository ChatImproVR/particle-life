use cimvr_common::glam::Vec3;

use crate::query_accel::QueryAccelerator;
use crate::{SimConfig, SimState};

pub struct NewtonConfig {
    /// Time step
    pub dt: f32,
    /// Velocity damping rate
    pub damping: f32,
}

/// Calculates total force, assuming unit mass (m = 1)
pub fn total_force(i: usize, state: &SimState, cfg: &SimConfig) -> Vec3 {
    let mut f = Vec3::ZERO;

    for neighbor in state.accel.query_neighbors(&state.pos, i, state.pos[i]) {
        let a = state.pos[i];
        let b = state.pos[neighbor];

        // The vector pointing from a to b
        let diff = b - a;

        // Distance is capped
        let dist = diff.length();

        // Accelerate towards b
        let normal = diff.normalize();
        let behav = cfg.get_behaviour(state.colors[i], state.colors[neighbor]);
        let accel = normal * behav.force(dist);

        // Unit mass (m = 1)
        f += accel;
    }

    f
}

pub fn newton_step(state: &mut SimState, cfg: &SimConfig, newton: &NewtonConfig) {
    let len = state.pos.len();

    for i in 0..len {
        let total_accel = total_force(i, state, cfg);

        let vel = state.vel[i] + total_accel * newton.dt;

        // Dampen velocity
        let vel = vel * (1. - newton.damping);

        state.vel[i] = vel;
        state.pos[i] += vel * newton.dt;
    }
}

impl Default for NewtonConfig {
    fn default() -> Self {
        Self {
            damping: 0.1,
            dt: 2e-3,
        }
    }
}
