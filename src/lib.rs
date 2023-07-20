use cimvr_common::{
    glam::Vec3,
    render::{CameraComponent, Mesh, MeshHandle, Primitive, Render, UploadMesh, Vertex},
    ui::{
        egui::{color_picker::color_edit_button_rgb, DragValue, Grid, Slider, Ui},
        GuiInputMessage, GuiTab,
    },
    vr::{ControllerEvent, VrUpdate},
    Transform,
};
use cimvr_engine_interface::{
    dbg, make_app_state, pcg::Pcg, pkg_namespace, prelude::*, println, FrameTime,
};
mod sim;
use query_accel::QueryAccelerator;
use sim::*;
mod query_accel;

const SIM_OFFSET: Vec3 = Vec3::new(0., 1., 0.);

// All state associated with client-side behaviour
struct ClientState {
    sim: SimState,
    time: f32,
    last_left_pos: Vec3,
    last_right_pos: Vec3,
    ui: GuiTab,
    dt: f32,
    selected_field: Field,
    constrain_2d: bool,
    show_debug: bool,
    pause: bool,
    deepest: usize,
}

fn new_sim_state(io: &mut EngineIo) -> SimState {
    let mut aa = Behaviour::default();
    aa.inter_threshold = 0.05;

    let mut rand = || io.random() as u64 as f32 / u64::MAX as f32;

    let n = 5;

    let colors: Vec<[f32; 3]> = (0..n).map(|_| hsv_to_rgb(rand() * 360., 1., 1.)).collect();
    let behaviours = (0..n * n)
        .map(|_| aa.with_inter_strength((rand() * 2. - 1.) * 15.))
        .collect();

    // NOTE: We are using the println defined by cimvr_engine_interface here, NOT the standard library!
    let palette = SimConfig {
        colors,
        behaviours,
        /*
        colors: vec![
            [0.1, 1., 0.],
            [1., 0.1, 0.],
            [102. / 256., 30. / 256., 131. / 256.],
        ],
        behaviours: vec![
            aa.with_inter_strength(3.),
            aa.with_inter_strength(-1.5),
            aa.with_inter_strength(1.),
            aa.with_inter_strength(2.),
            aa.with_inter_strength(1.),
            aa.with_inter_strength(1.),
            aa.with_inter_strength(50.),
            aa.with_inter_strength(50.),
            aa.with_inter_strength(-100.),
        ],
        */
        damping: 150.,
    };

    SimState::new(&mut Pcg::new(), palette, 4_000)
}

const SIM_RENDER_ID: MeshHandle = MeshHandle::new(pkg_namespace!("Simulation"));
const DEBUG_RENDER_ID: MeshHandle = MeshHandle::new(pkg_namespace!("Debug"));

impl UserState for ClientState {
    // Implement a constructor
    fn new(io: &mut EngineIo, sched: &mut EngineSchedule<Self>) -> Self {
        let sim = new_sim_state(io);

        io.create_entity()
            .add_component(Transform::identity().with_position(SIM_OFFSET))
            .add_component(Render::new(SIM_RENDER_ID).primitive(Primitive::Points))
            .build();

        io.create_entity()
            .add_component(Transform::identity().with_position(SIM_OFFSET))
            .add_component(Render::new(DEBUG_RENDER_ID).primitive(Primitive::Lines))
            .build();

        sched.add_system(Self::update).build();

        sched
            .add_system(Self::interaction)
            .query(
                "Camera",
                Query::new()
                    .intersect::<Transform>(Access::Read)
                    .intersect::<CameraComponent>(Access::Read),
            )
            .subscribe::<FrameTime>()
            .subscribe::<VrUpdate>()
            .build();
        sched.add_system(Self::update).build();

        sched
            .add_system(Self::update_ui)
            .subscribe::<GuiInputMessage>()
            .build();

        let ui = GuiTab::new(io, "Particle life");

        Self {
            show_debug: false,
            selected_field: Field::InterStrength,
            dt: 1e-3,
            ui,
            sim,
            time: 0.,
            last_left_pos: Vec3::ZERO,
            last_right_pos: Vec3::ZERO,
            constrain_2d: false,
            pause: false,
            deepest: 0,
        }
    }
}

#[derive(PartialEq)]
pub enum Field {
    /// Magnitude of the default repulsion force
    DefaultRepulse,
    /// Zero point between default repulsion and particle interaction (0 to 1)
    InterThreshold,
    /// Interaction peak strength
    InterStrength,
    /// Maximum distance of particle interaction (0 to 1)
    InterMaxDist,
}

fn config_ui(ui: &mut Ui, config: &mut SimConfig, selected_field: &mut Field) {
    ui.horizontal(|ui| {
        ui.selectable_value(selected_field, Field::DefaultRepulse, "Default repulsion");
        ui.selectable_value(
            selected_field,
            Field::InterThreshold,
            "Interaction threshold",
        );
    });
    ui.horizontal(|ui| {
        ui.selectable_value(selected_field, Field::InterStrength, "Interaction Strength");
        ui.selectable_value(
            selected_field,
            Field::InterMaxDist,
            "Interaction max distance",
        );
    });

    let len = config.colors.len();
    Grid::new(pkg_namespace!("Particle Life Grid")).show(ui, |ui| {
        // Top row
        ui.label("Life");
        for color in &mut config.colors {
            color_edit_button_rgb(ui, color);
        }
        ui.end_row();
        // Grid
        for (row_idx, color) in config.colors.iter_mut().enumerate() {
            color_edit_button_rgb(ui, color);
            for column in 0..len {
                let behav = &mut config.behaviours[column + row_idx * len];
                match selected_field {
                    Field::InterStrength => {
                        ui.add(DragValue::new(&mut behav.inter_strength).speed(1e-2))
                    }
                    Field::InterMaxDist => ui.add(
                        DragValue::new(&mut behav.inter_max_dist)
                            .clamp_range(0.0..=1.0)
                            .speed(1e-2),
                    ),
                    Field::DefaultRepulse => {
                        ui.add(DragValue::new(&mut behav.default_repulse).speed(1e-2))
                    }
                    Field::InterThreshold => ui.add(
                        DragValue::new(&mut behav.inter_threshold)
                            .clamp_range(0.0..=1.0)
                            .speed(1e-2),
                    ),
                };
            }
            ui.end_row();
        }
    });

    ui.add(DragValue::new(&mut config.damping).prefix("Damping: "));
}

impl ClientState {
    fn update_ui(&mut self, io: &mut EngineIo, _query: &mut QueryResult) {
        let mut randomize = false;

        self.ui.show(io, |ui| {
            ui.add(Slider::new(&mut self.dt, 0.0..=1e-3));
            config_ui(ui, self.sim.config_mut(), &mut self.selected_field);

            ui.checkbox(&mut self.constrain_2d, "Constrain to 2D");
            if self.constrain_2d {
                project_to_2d(&mut self.sim);
            }

            ui.checkbox(&mut self.show_debug, "Debug");

            ui.checkbox(&mut self.pause, "Pause");

            randomize |= ui.button("Randomize").clicked();

            let deepest = self.sim.query_accel().tiles().map(|(_, b)| b.len()).max().unwrap_or(0);
            ui.label(format!("Deepest bucket: {}", deepest));
            self.deepest = self.deepest.max(deepest);
            ui.label(format!("Deepest bucket ever: {}", self.deepest));
        });

        //dbg!(&debug_upload_mesh.mesh.vertices);
        //dbg!(debug_upload_mesh.mesh.vertices.len());

        if randomize {
            self.sim = new_sim_state(io);
        }
    }

    fn interaction(&mut self, io: &mut EngineIo, query: &mut QueryResult) {
        let mut camera_transf = Transform::identity();
        for entity in query.iter("Camera") {
            camera_transf = query.read::<Transform>(entity);
        }

        if let Some(VrUpdate {
            left_controller,
            right_controller,
            ..
        }) = io.inbox_first()
        {
            for (controller, last) in [
                (left_controller, &mut self.last_left_pos),
                (right_controller, &mut self.last_right_pos),
            ] {
                if let Some(aim) = controller.aim {
                    let pos = aim.pos + camera_transf.pos - SIM_OFFSET;

                    let diff = pos - *last;
                    let mag = (diff.length() * 48.).powi(2);

                    self.sim.move_neighbors(pos, diff.normalize() * mag);
                    *last = pos;
                }

                if controller.events.contains(&ControllerEvent::Menu(
                    cimvr_common::vr::ElementState::Released,
                )) {
                    self.sim = new_sim_state(io);
                }
            }
        }
    }

    fn update(&mut self, io: &mut EngineIo, _query: &mut QueryResult) {
        if !self.pause {
            self.sim.step(self.dt);
        }

        let mesh = draw_particles(&self.sim, self.time);
        io.send(&UploadMesh {
            mesh,
            id: SIM_RENDER_ID,
        });

        let mut debug_upload_mesh = UploadMesh {
            mesh: Mesh::new(),
            id: DEBUG_RENDER_ID,
        };
        if self.show_debug {
            debug_upload_mesh = UploadMesh {
                mesh: query_accel_buckets(&self.sim.query_accel()),
                id: DEBUG_RENDER_ID,
            };
        }

        io.send(&debug_upload_mesh);

        self.time += self.dt;
    }
}

// All state associated with server-side behaviour
struct ServerState;

impl UserState for ServerState {
    // Implement a constructor
    fn new(_io: &mut EngineIo, _sched: &mut EngineSchedule<Self>) -> Self {
        println!("Hello, server!");
        Self
    }
}

// Defines entry points for the engine to hook into.
// Calls new() for the appropriate state.
make_app_state!(ClientState, ServerState);

fn draw_particles(sim: &SimState, _time: f32) -> Mesh {
    let mut vertices = vec![];
    let indices = (0..sim.particles().len() as u32).collect();

    for particle in sim.particles().iter() {
        let color = sim.config().colors[particle.color as usize];

        let vertex = Vertex {
            pos: particle.pos.to_array(),
            uvw: color,
        };

        vertices.push(vertex);
    }

    Mesh { vertices, indices }
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

fn project_to_2d(state: &mut SimState) {
    for p in state.particles_mut() {
        p.pos.y = 0.;
    }
}

fn query_accel_buckets(query_accel: &QueryAccelerator) -> Mesh {
    let mut mesh = Mesh::new();
    let color = [0.1; 3];
    let radius = query_accel.radius();
    for (index, _indices) in query_accel.tiles() {
        let corner = Vec3::from(index.map(|f| f as f32 * radius));
        add_cube(&mut mesh, corner, radius, color);
    }

    mesh
}

fn add_cube(mesh: &mut Mesh, corner: Vec3, width: f32, color: [f32; 3]) {
    let mut vert = |offset: [f32; 3]| {
        mesh.push_vertex(Vertex::new(
            (corner + Vec3::from(offset) * width).into(),
            color,
        ))
    };

    let a = vert([-1., -1., -1.]);
    let b = vert([-1., -1., 1.]);
    let c = vert([-1., 1., -1.]);
    let d = vert([-1., 1., 1.]);

    let e = vert([1., -1., 1.]);
    let f = vert([1., -1., -1.]);
    let g = vert([1., 1., 1.]);
    let h = vert([1., 1., -1.]);

    mesh.push_indices(&[
        a, b, c, d, e, f, g, h, a, c, b, d, e, g, f, h, a, f, b, e, c, h, d, g,
    ]);
}

// TODO: Pause feature
