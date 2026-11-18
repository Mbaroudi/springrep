use std::f64::consts::PI;

use super::*;
use crate::{
    check,
    math::{Frame, Vec3},
    measure,
    primitive::{block, cylinder, sphere},
};

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

// A ball slid along a line sweeps a capsule: a cylinder of its radius along the slide, with half a
// ball at each end.
#[test]
fn a_ball_slid_along_a_line_is_a_capsule() {
    let (r, d) = (1.5, Vec3::new(3.0, -2.0, 4.0));
    let ball = sphere(
        &Frame::WORLD.translated(Vec3::new(1.0, 2.0, 3.0)),
        r,
        "ball",
    )
    .unwrap();
    let b = solid(&ball, &Motion::Slide(d), "sw").unwrap();
    assert!(check::check(&b).is_empty());
    let l = d.norm();
    let (area, volume) = measure::area_and_volume(&b);
    assert!(
        rel(volume, 4.0 / 3.0 * PI * r.powi(3) + PI * r * r * l) < 1e-9,
        "{volume}"
    );
    assert!(
        rel(area, 4.0 * PI * r * r + 2.0 * PI * r * l) < 1e-9,
        "{area}"
    );
}

// A convex body slid by `d` sweeps its own volume and the prism of its shadow along `d`: for the
// unit cube the shadow along a unit `u` has area |u.x| + |u.y| + |u.z|, so the volume is
// 1 + |d.x| + |d.y| + |d.z|.
#[test]
fn a_cube_slid_aslant_adds_the_prism_of_its_shadow() {
    let d = Vec3::new(2.0, 1.0, -3.0);
    let cube = block(&Frame::WORLD, Vec3::new(1.0, 1.0, 1.0), "cube").unwrap();
    let b = solid(&cube, &Motion::Slide(d), "sw").unwrap();
    assert!(check::check(&b).is_empty());
    assert!(
        rel(measure::volume(&b), 7.0) < 1e-12,
        "{}",
        measure::volume(&b)
    );
}

// A torus slid along its own axis sweeps itself and the prism of its shadow, the annulus between
// radii R − r and R + r: 2π²Rr² + 4πRr·L. It grazes the slide along its top and bottom circles.
#[test]
fn a_torus_slid_along_its_axis_adds_the_prism_of_its_annulus() {
    let (big, small, l) = (3.0, 1.0, 5.0);
    let frame = Frame::new(Vec3::new(1.0, 0.0, 2.0), Vec3::new(1.0, 1.0, 1.0), Vec3::X).unwrap();
    let ring = crate::primitive::torus(&frame, big, small, "ring").unwrap();
    let b = solid(&ring, &Motion::Slide(frame.z * l), "sw").unwrap();
    assert!(check::check(&b).is_empty());
    let want = 2.0 * PI * PI * big * small * small + 4.0 * PI * big * small * l;
    assert!(
        rel(measure::volume(&b), want) < 1e-9,
        "{} {want}",
        measure::volume(&b)
    );
}

// A leaning drum pushed straight down through a level plane: the parts of its sweep below and
// above the plane, each cut off by the plane across its swept rims, make up the whole.
#[test]
fn a_slid_drum_cut_by_a_level_plane_keeps_its_volume() {
    let lean = Frame::new(Vec3::new(0.0, 0.0, 3.0), Vec3::new(0.5, 0.0, 0.8), Vec3::X).unwrap();
    let drum = cylinder(&lean, 1.5, 2.0, "drum").unwrap();
    let b = solid(&drum, &Motion::Slide(Vec3::new(0.0, 0.0, -8.0)), "sw").unwrap();
    let below = block(
        &Frame::WORLD.translated(Vec3::new(-8.0, -8.0, -8.0)),
        Vec3::new(16.0, 16.0, 8.0),
        "below",
    )
    .unwrap();
    let above = block(
        &Frame::WORLD.translated(Vec3::new(-8.0, -8.0, 0.0)),
        Vec3::new(16.0, 16.0, 8.0),
        "above",
    )
    .unwrap();
    let low = crate::boolean::intersection(&b, &below).unwrap();
    let high = crate::boolean::intersection(&b, &above).unwrap();
    let whole = measure::volume(&b);
    assert!(rel(measure::volume(&low) + measure::volume(&high), whole) < 1e-9);
}

// A cylinder of radius r and height h, its axis along z, casts a shadow along a unit u of area
// π r² |u.z| + 2 r h |u × z|: the ellipse of each end and the rectangle of its side between.
#[test]
fn a_cylinder_slid_aslant_adds_the_prism_of_its_shadow() {
    let (r, h, d) = (1.0, 2.0, Vec3::new(3.0, 0.0, 4.0));
    let drum = cylinder(&Frame::WORLD, r, h, "drum").unwrap();
    let b = solid(&drum, &Motion::Slide(d), "sw").unwrap();
    assert!(check::check(&b).is_empty());
    let u = d.unit().unwrap();
    let shadow = PI * r * r * u.z.abs() + 2.0 * r * h * u.cross(Vec3::Z).norm();
    let want = PI * r * r * h + d.norm() * shadow;
    assert!(
        rel(measure::volume(&b), want) < 1e-9,
        "{} {want}",
        measure::volume(&b)
    );
}
