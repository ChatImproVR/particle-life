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

    for neighbor in state.query.query_neighbors(&state.pos, i, state.pos[i]) {
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

pub struct NewtonVariableConfig {
    /// Time step
    pub dt: f32,
    /// Adjustment factor for substep time
    pub sub_dt: f32,
    /// Maximum number of substeps per particle
    pub max_steps: usize,
    /// Velocity damping rate (TODO: remove me??)
    pub damping: f32,
}

pub fn newton_step_variable_dt(
    state: &mut SimState,
    cfg: &SimConfig,
    newton: &NewtonVariableConfig,
) {
    let len = state.pos.len();

    let mut time = vec![0.0; len];

    for i in 0..len {
        let total_accel = total_force(i, state, cfg);

        let vel = state.vel[i] + total_accel * newton.dt;

        // Dampen velocity
        let vel = vel * (1. - newton.damping);

        state.vel[i] = vel;
        state.pos[i] += vel * newton.dt;
    }
}

/// Extrapolates the position and velocity for a given particle
fn extrapolate(state: &SimState, idx: usize, time: &[f32], global_time: f32) -> (Vec3, Vec3) {
    let dt = global_time - time[idx];
    let pos = state.pos[idx] + dt * state.vel[idx] + state.accel[idx] * dt.powi(2) / 2.;
    let vel = state.vel[idx] + dt * state.accel[idx];
    (pos, vel)
}

/// Calculate the appropriate delta time for one particle.
/// May return f32::MAX
fn calculate_delta_time(
    state: &SimState,
    time: &[f32],
    idx: usize,
    global_time: f32,
    newton: &NewtonVariableConfig,
) -> f32 {
    let mut min_dt = None;

    for j in state.query.query_neighbors(&state.pos, idx, state.pos[idx]) {
        // TODO: Perform one sqrt total by comparing r^2/v^2 between particles
        let (predict_pos, predict_vel) = extrapolate(state, j, time, global_time);
        let distance_sq = predict_pos.distance_squared(state.pos[idx]);
        let rel_vel_sq = predict_vel.distance_squared(state.pos[idx]);
        if rel_vel_sq > 0. {
            let dt = (distance_sq / rel_vel_sq).sqrt();
            if let Some(min) = min_dt {
                if dt > min {
                    continue;
                }
            }

            min_dt = Some(dt);
        }
    }

    min_dt.unwrap_or(newton.min_dt()).max(newton.min_dt())
}

impl Default for NewtonVariableConfig {
    fn default() -> Self {
        Self {
            dt: 2e-3,
            sub_dt: 1.,
            max_steps: 100,
            damping: 0.1,
        }
    }
}

impl NewtonVariableConfig {
    pub fn min_dt(&self) -> f32 {
        self.dt / self.max_steps as f32
    }
}
