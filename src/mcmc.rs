use cimvr_common::{
    glam::Vec2,
    render::{Mesh, MeshHandle, Primitive, Render, UploadMesh, Vertex},
    ui::{egui::DragValue, GuiInputMessage, GuiTab},
    Transform,
};
use cimvr_engine_interface::{make_app_state, pcg::Pcg, pkg_namespace, prelude::*};
use crate::query_accel::QueryAccelerator;
use rand::prelude::*;
use rand_distr::Normal;

pub struct MonteCarloConfig {
    pub temperature: f32,
    pub walk_sigma: f32,
    pub substeps: usize,
}

/*
    pub fn new(n: usize, accel_radius: f32, rules: Ruleset, temperature: f32, walk_sigma: f32) -> Self {
        let mut rng = rng();

        let s = 1.;

        let positions = (0..n)
            .map(|_| Vec2::new(rng.gen_range(-s..=s), rng.gen_range(-s..=s)))
            .collect();

        let types = (0..n)
            .map(|_| rng.gen_range(0..rules.particles.len() as u8))
            .collect();

        let state = State { positions, types };

        let accel = QueryAccelerator::new(&state.positions, accel_radius);

        Self {
            state,
            accel,
            temperature,
            walk_sigma,
            rules,
        }
    }

    pub fn step(&mut self) {
        let ref mut rng = rng();

        // Pick a particle
        let idx = rng.gen_range(0..self.state.positions.len());

        // Perterb it
        let original = self.state.positions[idx];
        let mut candidate = original;
        let normal = Normal::new(0.0, self.walk_sigma).unwrap();
        candidate.x += normal.sample(rng);
        candidate.y += normal.sample(rng);

        // Calculate the candidate change in energy
        let old_energy = self.energy_due_to(idx, original);
        let new_energy = self.energy_due_to(idx, candidate);
        let delta_e = new_energy - old_energy;

        // Decide whether to accept the change
        let probability = (-delta_e / self.temperature).exp();
        //let probability = (-delta_e).exp();
        if probability > rng.gen_range(0.0..=1.0) {
            self.state.positions[idx] = candidate;
            self.accel.replace_point(idx, original, candidate);
        }
    }

    pub fn energy_due_to(&self, idx: usize, pos: Vec2) -> f32 {
        let mut energy = 0.;

        let my_ruleset = self.state.types[idx] as usize;
        let my_ruleset = &self.rules.particles[my_ruleset];

        for neighbor in self.accel.query_neighbors(&self.state.positions, idx, pos) {
            let distance = self.state.positions[neighbor].distance(pos);
            let potential = self.state.types[neighbor] as usize;
            let potential = my_ruleset.interactions[potential];

            let potential = potential.eval(distance);
            energy += potential;
        }
        energy
    }
}
*/

impl Default for MonteCarloConfig {
    fn default() -> Self {
        Self {
            temperature: 0.001, 
            walk_sigma: 0.01,
            substeps: 1500,
        }
    }
}
