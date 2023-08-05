use crate::{rng, SimConfig, SimState};
use cimvr_common::{
    glam::{Vec3},
};
use cimvr_engine_interface::{prelude::*};
use rand::prelude::*;
use rand_distr::Normal;

pub struct MonteCarloConfig {
    pub temperature: f32,
    pub walk_sigma: f32,
    pub substeps: usize,
}

pub fn mcmc_step(state: &mut SimState, cfg: &SimConfig, mcmc: &MonteCarloConfig) {
    let ref mut rng = rng();

    // Pick a particle
    let idx = rng.gen_range(0..state.pos.len());

    // Perterb it
    let original = state.pos[idx];
    let mut candidate = original;
    let normal = Normal::new(0.0, mcmc.walk_sigma).unwrap();
    candidate.x += normal.sample(rng);
    candidate.y += normal.sample(rng);

    // Calculate the candidate change in energy
    let old_energy = energy_due_to(idx, original, state, cfg);
    let new_energy = energy_due_to(idx, candidate, state, cfg);
    let delta_e = new_energy - old_energy;

    // Decide whether to accept the change
    let probability = (-delta_e / mcmc.temperature).exp();
    //let probability = (-delta_e).exp();
    if probability > rng.gen_range(0.0..=1.0) {
        state.pos[idx] = candidate;
        state.accel.replace_point(idx, original, candidate);
    }
}

pub fn energy_due_to(idx: usize, pos: Vec3, state: &SimState, cfg: &SimConfig) -> f32 {
    let mut energy = 0.;

    let my_color = state.colors[idx];

    for neighbor in state.accel.query_neighbors(&state.pos, idx, pos) {
        let distance = state.pos[neighbor].distance(pos);
        let behav = cfg.get_behaviour(my_color, state.colors[neighbor]);

        let potential = behav.potential(distance);
        energy += potential;
    }
    energy
}

impl Default for MonteCarloConfig {
    fn default() -> Self {
        Self {
            temperature: 0.001,
            walk_sigma: 0.01,
            substeps: 1500,
        }
    }
}
