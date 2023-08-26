use std::collections::BinaryHeap;

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
    let mut pq: BinaryHeap<TimeIndex> = (0..state.pos.len())
        .map(|i| TimeIndex(calculate_delta_time(state, &time, i, 0.0, newton), i))
        .collect();

    while let Some(TimeIndex(next_time, idx)) = pq.pop() {
        let global_time = next_time.min(newton.dt);
        let dt = global_time - time[idx];

        let new_accel = total_force_extrapolate(idx, state, cfg, &time, global_time);

        let prev_pos = state.pos[idx];
        state.pos[idx] = state.pos[idx]
            + state.vel[idx]* dt
            + state.accel[idx]* dt.powi(2) / 2.;

        state.query.replace_point(idx, prev_pos, state.pos[idx]);

        state.vel[idx]= state.vel[idx]
            + state.vel[idx]* dt
            + (state.accel[idx]+ new_accel) * dt / 2.;

        state.accel[idx] = new_accel;

        time[idx] = global_time;

        let next_dt = calculate_delta_time(state, &time, idx, global_time, newton);

        if global_time < newton.dt {
            pq.push(TimeIndex(global_time + next_dt, idx));
        }
    }

    for vel in &mut state.vel {
        // Dampen velocity
        *vel *= 1. - newton.damping;
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
            max_steps: 10,
            damping: 0.1,
        }
    }
}

impl NewtonVariableConfig {
    pub fn min_dt(&self) -> f32 {
        self.dt / self.max_steps as f32
    }
}

#[derive(Clone, Copy)]
struct TimeIndex(f32, usize);

impl PartialOrd for TimeIndex {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let TimeIndex(other_dt, _) = other;
        let TimeIndex(self_dt, _) = self;
        other_dt.partial_cmp(self_dt)
    }
}

impl Ord for TimeIndex {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(&other)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl PartialEq for TimeIndex {
    fn eq(&self, other: &Self) -> bool {
        let TimeIndex(_, other_idx) = other;
        let TimeIndex(_, self_idx) = self;
        self_idx == other_idx
    }
}

impl Eq for TimeIndex {}

/// Calculates total force, assuming unit mass (m = 1)
pub fn total_force_extrapolate(i: usize, state: &SimState, cfg: &SimConfig, time: &[f32], global_time: f32) -> Vec3 {
    let mut f = Vec3::ZERO;

    for neighbor in state.query.query_neighbors(&state.pos, i, state.pos[i]) {
        let a = state.pos[i];
        let (predict_pos, predict_vel) = extrapolate(state, neighbor, time, global_time);

        // The vector pointing from a to b
        let diff = predict_pos - a;

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


