use crate::{newton::total_force, rng, SimConfig, SimState};
use cimvr_common::glam::Vec3;
use cimvr_engine_interface::prelude::*;
use rand::prelude::*;
use rand_distr::Normal;

pub struct MonteCarloConfig {
    pub temperature: f32,
    pub walk_sigma: f32,
    pub substeps: usize,
}

pub fn mcmc_step(
    state: &mut SimState,
    cfg: &SimConfig,
    mcmc: &MonteCarloConfig,
    pseudo_newtonian: bool,
) {
    for _ in 0..mcmc.substeps {
        let ref mut rng = rng();

        // Pick a particle
        let idx = rng.gen_range(0..state.pos.len());

        // Perterb it
        let original = state.pos[idx];
        let mut candidate = original;
        let f = total_force(idx, state, cfg);

        let mut sigma = mcmc.walk_sigma;
        if pseudo_newtonian {
            sigma *= f.length() * mcmc.walk_sigma;
            sigma = sigma.min(mcmc.walk_sigma);
        }

        let normal = Normal::new(0.0, sigma).unwrap();
        candidate.x += normal.sample(rng);
        candidate.y += normal.sample(rng);
        candidate.z += normal.sample(rng);

        if pseudo_newtonian {
            candidate += f * mcmc.walk_sigma;
        }

        // Calculate the candidate change in energy
        let old_energy = energy_due_to(idx, original, state, cfg);
        let new_energy = energy_due_to(idx, candidate, state, cfg);
        let delta_e = new_energy - old_energy;

        // Decide whether to accept the change
        let probability = (-delta_e / mcmc.temperature).exp();
        //let probability = (-delta_e).exp();
        if probability > rng.gen_range(0.0..=1.0) {
            state.accel.replace_point(idx, original, candidate);
            state.pos[idx] = candidate;
        }
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
            walk_sigma: 0.001,
            substeps: 1500,
        }
    }
}
