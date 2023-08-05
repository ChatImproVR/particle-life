use cimvr_common::glam::Vec3;

use crate::{SimState, SimConfig};
use crate::query_accel::QueryAccelerator;

pub struct NewtonConfig {
    /// Time step
    pub dt: f32,
    /// Velocity damping rate
    pub damping: f32,
}

pub fn newton_step(state: &mut SimState, cfg: &SimConfig, newton: &NewtonConfig) {
    state.accel = QueryAccelerator::new(&state.pos, cfg.max_interaction_radius());

    let len = state.pos.len();

    for i in 0..len {
        let mut total_accel = Vec3::ZERO;
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
            let accel = normal * behav.force(dist) / dist;
            total_accel += accel;
        }

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
            dt: 1e-3,
        }
    }
}
