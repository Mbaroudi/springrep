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

/// The turn about the z axis through `angle`, from where the body is drawn.
fn about_z(angle: f64) -> Motion {
    Motion::Turn {
        origin: Vec3::ZERO,
        axis: Vec3::Z,
        from: 0.0,
        to: angle,
    }
}

// Where each circle about the axis meets the body in one arc, a turn through θ adds to the body θ
// times the moment about the axis of its shadow on a half-plane through the axis, ∫ρ dA (Pappus).
// A ball of radius r whose centre is R from the axis casts a disc: πr²R.
#[test]
fn a_ball_turned_about_a_line_adds_a_piece_of_torus() {
    let (r, big, angle) = (1.0, 4.0, 1.2);
    let ball = sphere(
        &Frame::WORLD.translated(Vec3::new(big, 0.0, 0.5)),
        r,
        "ball",
    )
    .unwrap();
    let b = solid(&ball, &about_z(angle), "sw").unwrap();
    assert!(check::check(&b).is_empty());
    let want = 4.0 / 3.0 * PI * r.powi(3) + angle * PI * r * r * big;
    assert!(
        rel(measure::volume(&b), want) < 1e-9,
        "{} {want}",
        measure::volume(&b)
    );
}

// The axis passes through the ball, so the circles about it that the ball holds whole stay whole,
// and the rest each meet it in one arc the turn widens by θ: the same πr²Rθ is added. The axis
// pierces the sphere where its grazing great circle passes, which the turn holds still.
#[test]
fn a_ball_turned_about_a_line_through_it_adds_a_piece_of_torus() {
    let (r, big, angle) = (1.0, 0.5, 1.0);
    let ball = sphere(
        &Frame::WORLD.translated(Vec3::new(big, 0.0, 0.0)),
        r,
        "ball",
    )
    .unwrap();
    let b = solid(&ball, &about_z(angle), "sw").unwrap();
    assert!(check::check(&b).is_empty());
    let want = 4.0 / 3.0 * PI * r.powi(3) + angle * PI * r * r * big;
    assert!(
        rel(measure::volume(&b), want) < 1e-9,
        "{} {want}",
        measure::volume(&b)
    );
}

// The unit cube from (1, 0, 0) turned about the z axis: its shadow spans radii 1 to √5 at every
// height, a moment of (5 − 1)/2.
#[test]
fn a_cube_turned_about_a_line_beside_it() {
    // Past half a turn the faces the turn carries round lie beyond the axis from where they start.
    for angle in [0.9, 4.0] {
        let cube = block(
            &Frame::WORLD.translated(Vec3::X),
            Vec3::new(1.0, 1.0, 1.0),
            "cube",
        )
        .unwrap();
        let b = solid(&cube, &about_z(angle), "sw").unwrap();
        assert!(check::check(&b).is_empty());
        assert!(
            rel(measure::volume(&b), 1.0 + 2.0 * angle) < 1e-9,
            "{angle} {}",
            measure::volume(&b)
        );
    }
}

// The same cube turned a whole turn fills the ring between radii 1 and √5: 4π.
#[test]
fn a_cube_turned_a_whole_turn_is_a_ring() {
    let cube = block(
        &Frame::WORLD.translated(Vec3::X),
        Vec3::new(1.0, 1.0, 1.0),
        "cube",
    )
    .unwrap();
    let b = solid(&cube, &about_z(2.0 * PI), "sw").unwrap();
    assert!(check::check(&b).is_empty());
    assert!(
        rel(measure::volume(&b), 4.0 * PI) < 1e-9,
        "{}",
        measure::volume(&b)
    );
}

// A drum of radius 1 and length ½ along y, its axis 5 from the z axis, turned about it: at height z
// its shadow runs from 5 − s to √((5 + s)² + ¼), s = √(1 − z²), a moment of ∫(20s + ¼)/2 dz =
// 5π + ¼. Its side grazes the turn along curves that are neither lines nor circles.
#[test]
fn a_drum_turned_about_a_line_square_to_its_axis() {
    let angle = 0.7;
    let frame = Frame::new(Vec3::new(5.0, 0.0, 0.0), Vec3::Y, Vec3::X).unwrap();
    let drum = cylinder(&frame, 1.0, 0.5, "drum").unwrap();
    let b = solid(&drum, &about_z(angle), "sw").unwrap();
    assert!(check::check(&b).is_empty());
    let want = PI * 0.5 + angle * (5.0 * PI + 0.25);
    assert!(
        rel(measure::volume(&b), want) < 1e-9,
        "{} {want}",
        measure::volume(&b)
    );
}

// A disc of radius 2 and thickness ½ along y, whose rim passes through the z axis, turned a
// quarter turn about it: the turn holds that point of the rim still. At height z its shadow runs
// from 2 − s to √((2 + s)² + ¼), s = √(4 − z²), a moment of ∫(8s + ¼)/2 dz = 8π + ½.
#[test]
fn a_disc_turned_about_a_line_through_its_rim() {
    let angle = PI / 2.0;
    let frame = Frame::new(Vec3::new(2.0, 0.0, 0.0), Vec3::Y, Vec3::X).unwrap();
    let disc = cylinder(&frame, 2.0, 0.5, "disc").unwrap();
    let b = solid(&disc, &about_z(angle), "sw").unwrap();
    assert!(check::check(&b).is_empty());
    let want = 2.0 * PI + angle * (8.0 * PI + 0.5);
    assert!(
        rel(measure::volume(&b), want) < 1e-9,
        "{} {want}",
        measure::volume(&b)
    );
}

// The box a ≤ x ≤ b, |y| ≤ c, 0 ≤ z ≤ h turned through θ about the z axis. Its faces square to x
// each graze the turn along y = 0 and cross themselves turned, so it is built in parts. At radius ρ
// it spans one arc out to √(b² + c²), which the turn widens by θ, except past b, where it spans two
// arcs ±(acos(b/ρ), asin(c/ρ)) that the turn joins into one, widening it by θ + 2 acos(b/ρ). With
// ∫ρ acos(b/ρ) dρ from b to √(b² + c²) = (φ(b² + c²) − bc)/2, φ = atan(c/b), the area swept is
// 2c(b − a) + θ(b² + c² − a²)/2 + φ(b² + c²) − bc.
#[test]
fn a_box_turned_about_a_line_beside_it_is_built_in_parts() {
    let (a, b, c, h, angle) = (10.0, 13.0, 2.0, 3.0, PI / 3.0);
    let corner = Frame::WORLD.translated(Vec3::new(a, -c, 0.0));
    let cube = block(&corner, Vec3::new(b - a, 2.0 * c, h), "box").unwrap();
    let swept = solid(&cube, &about_z(angle), "sw").unwrap();
    assert!(check::check(&swept).is_empty());
    let phi = (c / b).atan();
    let r2 = b * b + c * c;
    let want = h * (2.0 * c * (b - a) + angle * (r2 - a * a) / 2.0 + phi * r2 - b * c);
    let got = measure::volume(&swept);
    assert!(rel(got, want) < 1e-9, "{got} {want}");
}

/// The right-hand screw about the z axis that rises `lead` a turn.
fn screw_z(lead: f64) -> crate::sweep::Helix {
    crate::sweep::Helix {
        origin: Vec3::ZERO,
        axis: Vec3::Z,
        rise: lead / (2.0 * PI),
        sign: 1.0,
    }
}

// A ball screwed along a helix no tighter than itself sweeps the tube about its centre's helix,
// π r² times the helix's length θ√(R² + c²), and a ball at each end.
#[test]
fn a_ball_screwed_along_a_helix_is_a_tube() {
    let (r, big, lead, angle) = (1.0, 5.0, 6.0, 4.0);
    let ball = sphere(
        &Frame::WORLD.translated(Vec3::new(big, 0.0, 0.0)),
        r,
        "ball",
    )
    .unwrap();
    let screw = screw_z(lead);
    let motion = Motion::Screw {
        screw,
        from: 0.0,
        to: angle,
    };
    let b = solid(&ball, &motion, "sw").unwrap();
    assert!(check::check(&b).is_empty());
    let length = angle * (big * big + screw.rise * screw.rise).sqrt();
    let want = 4.0 / 3.0 * PI * r.powi(3) + PI * r * r * length;
    assert!(
        rel(measure::volume(&b), want) < 1e-9,
        "{} {want}",
        measure::volume(&b)
    );
}

// The bar 4 ≤ x ≤ 12, |y| ≤ 8, |z| ≤ 1 screwed one turn of lead 6 about the z axis, within the
// cylinder of radius 5. A point at radius ρ is in the bar at turn t while its angle from the bar's
// is within a = acos(4/ρ), so at each angle the turn covers heights 2 + 2ca long, c the rise a
// radian, or two such runs 2 + c(a ± φ) long within a of where the bar starts. Over the turn that is
// 4π + 4πca + 4a a unit of ρ dρ: 18π from radius 4 to 5, and with ∫ρ acos(4/ρ) dρ = 8(25φ₁/16 −
// 3/4), φ₁ = acos(4/5), (4πc + 4)·8(25φ₁/16 − 3/4) = 200φ₁ − 96 more.
#[test]
fn a_bar_screwed_a_turn_cuts_a_square_thread() {
    let bar = block(
        &Frame::WORLD.translated(Vec3::new(4.0, -8.0, -1.0)),
        Vec3::new(8.0, 16.0, 2.0),
        "bar",
    )
    .unwrap();
    let motion = Motion::Screw {
        screw: screw_z(6.0),
        from: 0.0,
        to: 2.0 * PI,
    };
    let b = solid(&bar, &motion, "sw").unwrap();
    assert!(check::check(&b).is_empty());
    let rod = cylinder(
        &Frame::WORLD.translated(Vec3::new(0.0, 0.0, -2.0)),
        5.0,
        10.0,
        "rod",
    )
    .unwrap();
    let thread = crate::boolean::intersection(&b, &rod).unwrap();
    let want = 18.0 * PI + 200.0 * 0.8f64.acos() - 96.0;
    let got = measure::volume(&thread);
    assert!(rel(got, want) < 1e-9, "{got} {want}");
}

// Two slides at once are the slide by their sum: the unit cube slid by (2, 1, 0) adds the prism
// of its shadow along the slide, 2 + 1. Built as a motion whose velocity may change.
#[test]
fn a_cube_slid_two_ways_at_once_adds_the_prism_of_its_shadow() {
    let cube = block(&Frame::WORLD, Vec3::new(1.0, 1.0, 1.0), "cube").unwrap();
    let motion = Motion::Then(
        Box::new(Motion::Slide(Vec3::new(2.0, 0.0, 0.0))),
        Box::new(Motion::Slide(Vec3::new(0.0, 1.0, 0.0))),
    );
    let b = solid(&cube, &motion, "sw").unwrap();
    assert!(check::check(&b).is_empty());
    assert!(
        rel(measure::volume(&b), 4.0) < 1e-9,
        "{}",
        measure::volume(&b)
    );
}

// A frame riding a circle about the z axis turns about the axis as it goes, so a cube it carries
// sweeps what the cube turned through the same angle does. The unit cube from (1, ½, 0) casts a
// square on a half-plane through the axis from radius √(5/4) to radius 5/2: a moment of 5/2.
#[test]
fn a_cube_carried_round_a_circle_is_a_cube_turned() {
    let angle = 0.9;
    let cube = block(
        &Frame::WORLD.translated(Vec3::new(1.0, 0.5, 0.0)),
        Vec3::new(1.0, 1.0, 1.0),
        "cube",
    )
    .unwrap();
    let circle = crate::geom::Curve::Circle(crate::geom::Circle {
        frame: Frame::WORLD,
        radius: 3.0,
    });
    let motion = Motion::Along {
        path: circle,
        range: (0.0, angle),
        up: Vec3::Z,
    };
    let b = solid(&cube, &motion, "sw").unwrap();
    assert!(check::check(&b).is_empty());
    let got = measure::volume(&b);
    assert!(rel(got, 1.0 + 2.5 * angle) < 1e-7, "{got}");
}

// Slid along x in its own plane, a torus about z grazes along its whole meridians at u = ±90°,
// where the rate `cos u cos v` vanishes for every v. Under a motion whose velocity may change that
// is refused, not built wrong.
#[test]
fn a_torus_grazing_along_a_whole_meridian_is_refused() {
    let ring = crate::primitive::torus(&Frame::WORLD, 2.0, 0.5, "ring").unwrap();
    let motion = Motion::Then(
        Box::new(Motion::Slide(Vec3::new(1.0, 0.0, 0.0))),
        Box::new(Motion::Slide(Vec3::new(2.0, 0.0, 0.0))),
    );
    assert!(matches!(
        solid(&ring, &motion, "sw"),
        Err(crate::Error::Unsupported(_))
    ));
}
