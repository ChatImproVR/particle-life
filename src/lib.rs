use cimvr_common::glam::Vec3;
use cimvr_engine_interface::pcg::Pcg;
use query_accel::QueryAccelerator;

mod newton;
mod mcmc;
mod client;
mod query_accel;
use rand::prelude::*;

#[derive(Clone)]
pub struct SimState {
    /// Positions
    pub pos: Vec<Vec3>,
    /// Velocities. May or may not be used, depending on the integrator
    pub vel: Vec<Vec3>,
    /// Particle types, corresponding to colors
    pub colors: Vec<u8>,
    /// Query accelerator, tracking particle positions
    pub accel: QueryAccelerator,
}

/// Display colors and physical behaviour coefficients
#[derive(Clone, Debug)]
pub struct SimConfig {
    /// Colors of each type
    pub colors: Vec<[f32; 3]>,
    /// Behaviour matrix
    pub behaviours: Vec<Behaviour>,
}

pub type ParticleType = u8;

#[derive(Clone, Copy, Debug)]
pub struct Behaviour {
    /// Magnitude of the default repulsion force
    pub default_repulse: f32,
    /// Zero point between default repulsion and particle interaction (0 to 1)
    pub inter_threshold: f32,
    /// Interaction peak strength
    pub inter_strength: f32,
    /// Maximum distance of particle interaction (0 to 1)
    pub inter_max_dist: f32,
}

impl Behaviour {
    /// Returns the force on this particle
    ///
    /// Distance is in the range `0.0..=1.0`
    pub fn force(&self, dist: f32) -> f32 {
        if dist < self.inter_threshold {
            let f = dist / self.inter_threshold;
            (1. - f) * -self.default_repulse
        } else if dist > self.inter_max_dist {
            0.0
        } else {
            let x = dist - self.inter_threshold;
            let x = x / (self.inter_max_dist - self.inter_threshold);
            let x = x * 2. - 1.;
            let x = 1. - x.abs();
            x * self.inter_strength
        }
    }

    /// Potential, which falls to zero over distance
    pub fn potential(&self, dist: f32) -> f32 {
        let d = (self.inter_max_dist - self.inter_threshold) / 2.;

        let repulse_r = dist.clamp(0., self.inter_threshold);
        let z_to_peak_r = (dist - self.inter_threshold).clamp(0., d);
        let peak_end_r = (dist - self.inter_threshold - d).clamp(0., d);

        let mut u =
            self.default_repulse * (repulse_r.powi(2) / 2. / self.inter_threshold - repulse_r);
        u += (self.inter_strength / d) * (z_to_peak_r.powi(2) / 2.);
        u -= (self.inter_strength / d) * ((peak_end_r - d).powi(2) / 2.);

        u -= (self.inter_strength * d - self.inter_threshold * self.default_repulse) / 2.;

        u
    }
}

impl SimState {
    pub fn new_uniform_cube(cfg: &SimConfig, n: usize, radius: f32) -> Self {
        let mut rng = rng();

        let pos: Vec<Vec3> = (0..n)
            .map(|_| {
                Vec3::new(
                    rng.gen_range(-radius..=radius),
                    rng.gen_range(-radius..=radius),
                    rng.gen_range(-radius..=radius),
                )
            })
            .collect();

        let types = (0..n)
            .map(|_| rng.gen_range(0..cfg.colors.len() as u8))
            .collect();

        let vel = vec![Vec3::ZERO; n];

        let accel = QueryAccelerator::new(&pos, cfg.max_interaction_radius());

        Self {
            pos,
            vel,
            colors: types,
            accel,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_behaviour() {
        let behav = Behaviour {
            default_repulse: 1.0,
            inter_threshold: 0.25,
            inter_strength: 3.0,
            inter_max_dist: 0.75,
        };

        assert_eq!(behav.force(0.), -behav.default_repulse);
        assert_eq!(behav.force(behav.inter_threshold), 0.0);
        assert_eq!(behav.force(0.25 + 0.125), behav.inter_strength / 2.);
        assert_eq!(behav.force(0.5), behav.inter_strength);
        assert_eq!(behav.force(behav.inter_max_dist), 0.0);
        assert_eq!(behav.force(0.85), 0.0);
    }
}

impl Default for Behaviour {
    fn default() -> Self {
        Self {
            default_repulse: 10.,
            inter_threshold: 0.05,
            inter_strength: 1.,
            inter_max_dist: 0.2,
        }
    }
}

impl SimConfig {
    pub fn max_interaction_radius(&self) -> f32 {
        self.behaviours
            .iter()
            .map(|b| b.inter_max_dist)
            .max_by(|a, b| a.total_cmp(b))
            .unwrap()
    }

    pub fn get_behaviour(&self, a: ParticleType, b: ParticleType) -> Behaviour {
        let idx = a as usize * self.colors.len() + b as usize;
        self.behaviours[idx]
    }

    fn random() -> Self {
        let mut rng = rng();
        let n = rng.gen_range(2..=5);

    let colors: Vec<[f32; 3]> = (0..n).map(|_| hsv_to_rgb(rng.gen_range(0.0..=360.0), 1., 1.)).collect();
    let behaviours = (0..n * n)
        .map(|_| {
            let mut behav = Behaviour::default();
            behav.inter_strength = rng.gen_range(-15.0..=15.0);
            behav
        })
        .collect();

        Self {
            behaviours,
            colors,
        }
    }
}

fn rng() -> SmallRng {
    let u = ((Pcg::new().gen_u32() as u64) << 32) | Pcg::new().gen_u32() as u64;
    SmallRng::seed_from_u64(u)
}

/// https://gist.github.com/fairlight1337/4935ae72bcbcc1ba5c72
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let c = v * s; // Chroma
    let h_prime = (h / 60.0) % 6.0;
    let x = c * (1.0 - ((h_prime % 2.0) - 1.0).abs());
    let m = v - c;

    let (mut r, mut g, mut b);

    if 0. <= h_prime && h_prime < 1. {
        r = c;
        g = x;
        b = 0.0;
    } else if 1.0 <= h_prime && h_prime < 2.0 {
        r = x;
        g = c;
        b = 0.0;
    } else if 2.0 <= h_prime && h_prime < 3.0 {
        r = 0.0;
        g = c;
        b = x;
    } else if 3.0 <= h_prime && h_prime < 4.0 {
        r = 0.0;
        g = x;
        b = c;
    } else if 4.0 <= h_prime && h_prime < 5.0 {
        r = x;
        g = 0.0;
        b = c;
    } else if 5.0 <= h_prime && h_prime < 6.0 {
        r = c;
        g = 0.0;
        b = x;
    } else {
        r = 0.0;
        g = 0.0;
        b = 0.0;
    }

    r += m;
    g += m;
    b += m;

    [r, g, b]
}
