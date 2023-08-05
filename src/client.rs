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

use crate::{
    newton::{newton_step, NewtonConfig},
    query_accel::QueryAccelerator,
    SimConfig, SimState,
};

const SIM_OFFSET: Vec3 = Vec3::new(0., 1., 0.);

#[derive(Copy, Clone, PartialEq)]
enum Integrator {
    Newton,
    MonteCarlo,
}

// All state associated with client-side behaviour
struct ClientState {
    state: SimState,
    time: f32,
    last_left_pos: Vec3,
    last_right_pos: Vec3,
    ui: GuiTab,
    selected_field: Field,
    constrain_2d: bool,
    show_debug: bool,
    pause: bool,
    deepest: usize,

    n: usize,
    cfg: SimConfig,

    integrator: Integrator,
    newton: NewtonConfig,
}

const SIM_RENDER_ID: MeshHandle = MeshHandle::new(pkg_namespace!("Simulation"));
const DEBUG_RENDER_ID: MeshHandle = MeshHandle::new(pkg_namespace!("Debug"));

impl UserState for ClientState {
    // Implement a constructor
    fn new(io: &mut EngineIo, sched: &mut EngineSchedule<Self>) -> Self {
        io.create_entity()
            .add_component(Transform::identity().with_position(SIM_OFFSET))
            .add_component(Render::new(SIM_RENDER_ID).primitive(Primitive::Points))
            .build();

        io.create_entity()
            .add_component(Transform::identity().with_position(SIM_OFFSET))
            .add_component(Render::new(DEBUG_RENDER_ID).primitive(Primitive::Lines))
            .build();

        sched.add_system(Self::update).build();

        /*
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
        */

        sched.add_system(Self::update).build();

        sched
            .add_system(Self::update_ui)
            .subscribe::<GuiInputMessage>()
            .build();

        let ui = GuiTab::new(io, "Particle life");

        let n = 5_000;

        let cfg = SimConfig::random();

        let state = SimState::new_uniform_cube(&cfg, n, 1.);

        let newton = NewtonConfig::default();

        Self {
            show_debug: false,
            selected_field: Field::InterStrength,
            newton,
            n,
            cfg,
            ui,
            state,
            integrator: Integrator::Newton,
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
}

impl ClientState {
    fn update_ui(&mut self, io: &mut EngineIo, _query: &mut QueryResult) {
        let mut reset_particles = false;

        self.ui.show(io, |ui| {
            config_ui(ui, &mut self.cfg, &mut self.selected_field);

            /*
            ui.checkbox(&mut self.constrain_2d, "Constrain to 2D");
            if self.constrain_2d {
                project_to_2d(&mut self.state);
            }
            */

            ui.checkbox(&mut self.show_debug, "Debug");

            ui.checkbox(&mut self.pause, "Pause");

            if ui.button("Randomize").clicked() {
                self.cfg = SimConfig::random();
                reset_particles = true;
            }

            reset_particles |= ui.button("Reset particles").clicked();

            /*
            let deepest = self.state.accel.tiles().map(|(_, b)| b.len()).max().unwrap_or(0);
            ui.label(format!("Deepest bucket: {}", deepest));
            self.deepest = self.deepest.max(deepest);
            ui.label(format!("Deepest bucket ever: {}", self.deepest));
            */

            ui.separator();
            ui.horizontal(|ui| {
                ui.label("Integrator: ");
                ui.selectable_value(&mut self.integrator, Integrator::Newton, "Newton");
                ui.selectable_value(&mut self.integrator, Integrator::MonteCarlo, "Monte Carlo");
            });
            match self.integrator {
                Integrator::Newton => {
                    ui.add(Slider::new(&mut self.newton.dt, 0.0..=1e-2));
                    ui.add(DragValue::new(&mut self.newton.damping).prefix("Damping: ").speed(1e-2));
                }
                Integrator::MonteCarlo => {}
            }
        });

        //dbg!(&debug_upload_mesh.mesh.vertices);
        //dbg!(debug_upload_mesh.mesh.vertices.len());

        if reset_particles {
            self.state = SimState::new_uniform_cube(&self.cfg, self.n, 1.);
        }
    }

    /*
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
    */

    fn update(&mut self, io: &mut EngineIo, _query: &mut QueryResult) {
        if !self.pause {
            match self.integrator {
                Integrator::Newton => newton_step(&mut self.state, &self.cfg, &self.newton),
                Integrator::MonteCarlo => (),
            }
        }

        let mesh = draw_particles(&self.state, &self.cfg);
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
                mesh: query_accel_buckets(&self.state.accel),
                id: DEBUG_RENDER_ID,
            };
        }

        io.send(&debug_upload_mesh);
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

fn draw_particles(state: &SimState, cfg: &SimConfig) -> Mesh {
    let mut vertices = vec![];
    let n = state.pos.len();

    let indices = (0..n as u32).collect();

    for i in 0..n {
        let color = cfg.colors[state.colors[i] as usize];

        let vertex = Vertex {
            pos: state.pos[i].to_array(),
            uvw: color,
        };

        vertices.push(vertex);
    }

    Mesh { vertices, indices }
}

/*
fn project_to_2d(state: &mut SimState) {
    for p in state.particles_mut() {
        p.pos.y = 0.;
    }
}
*/

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

    let a = vert([0., 0., 0.]);
    let b = vert([0., 0., 1.]);
    let c = vert([0., 1., 0.]);
    let d = vert([0., 1., 1.]);

    let e = vert([1., 0., 1.]);
    let f = vert([1., 0., 0.]);
    let g = vert([1., 1., 1.]);
    let h = vert([1., 1., 0.]);

    mesh.push_indices(&[
        a, b, c, d, e, f, g, h, a, c, b, d, e, g, f, h, a, f, b, e, c, h, d, g,
    ]);
}
