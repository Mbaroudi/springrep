use super::section::Section;
use crate::{
    check,
    math::{Frame, Point3, Vec3},
    measure,
    primitive::torus,
    sweep::{Motion, solid},
};

// A torus's depth has a closed form: the tube's radius less the distance from its middle circle.
#[test]
fn a_torus_section_reads_the_depth_of_a_tube() {
    let frame = Frame::new(Vec3::new(1.0, -2.0, 0.5), Vec3::new(0.3, 0.4, 1.0), Vec3::X).unwrap();
    let (big, small) = (3.0, 1.0);
    let ring = torus(&frame, big, small, "ring").unwrap();
    let section = Section::of(&ring).unwrap();
    for k in 0..500 {
        let a = k as f64 * 0.731;
        let p = Point3::new(
            4.0 * (a * 1.3).sin(),
            4.0 * (a * 0.7).cos(),
            2.0 * (a * 2.1).sin(),
        ) + frame.origin;
        let l = frame.to_local(p);
        let rho = (l.x * l.x + l.y * l.y).sqrt();
        let want = small - ((rho - big).powi(2) + l.z * l.z).sqrt();
        let got = section.depth(p);
        assert!((got - want).abs() <= 1e-12, "{got} {want} at {p:?}");
    }
}

/// The volume a torus (radii `big`, `small`, about `frame`) sweeps slid straight down by `drop`,
/// by the midpoint rule over upright lines `h` apart: along each, the stretches inside the torus,
/// each from its closed form `(|q|² + R² − r²)² = 4R²(|q|² − (q·a)²)`, run down by `drop` and
/// joined.
fn slid_torus_volume(frame: &Frame, big: f64, small: f64, drop: f64, h: f64) -> f64 {
    let inside = |p: Point3| {
        let l = frame.to_local(p);
        let rho = (l.x * l.x + l.y * l.y).sqrt();
        (rho - big).powi(2) + l.z * l.z < small * small
    };
    let reach = big + small;
    let (z0, z1) = (frame.origin.z - reach, frame.origin.z + reach);
    let n = ((2.0 * reach) / h).ceil() as i64;
    let steps = 400;
    let mut total = 0.0;
    for i in 0..n {
        for j in 0..n {
            let x = frame.origin.x - reach + h * (i as f64 + 0.5);
            let y = frame.origin.y - reach + h * (j as f64 + 0.5);
            let at = |k: usize| z0 + (z1 - z0) * k as f64 / steps as f64;
            let mut ends = Vec::new();
            let mut last = inside(Point3::new(x, y, at(0)));
            for k in 1..=steps {
                let now = inside(Point3::new(x, y, at(k)));
                if now != last {
                    let (mut lo, mut hi) = (at(k - 1), at(k));
                    for _ in 0..50 {
                        let m = 0.5 * (lo + hi);
                        if inside(Point3::new(x, y, m)) == last {
                            lo = m;
                        } else {
                            hi = m;
                        }
                    }
                    ends.push(0.5 * (lo + hi));
                }
                last = now;
            }
            let mut length = 0.0;
            let mut cur: Option<(f64, f64)> = None;
            for w in ends.chunks(2) {
                let (lo, hi) = (w[0] - drop, w[1]);
                cur = match cur {
                    Some((a, b)) if lo <= b => Some((a, b.max(hi))),
                    Some((a, b)) => {
                        length += b - a;
                        Some((lo, hi))
                    }
                    None => Some((lo, hi)),
                };
            }
            if let Some((a, b)) = cur {
                length += b - a;
            }
            total += length;
        }
    }
    total * h * h
}

// A ring leaning 65° pushed straight down: past a lean of atan √((R − r)/r) its inner outline, seen
// along the push, folds into two swallowtails, and the faces it sweeps run into each other there.
// The trimmed sweep is a valid solid of the volume its upright lines add up to.
#[test]
fn a_ring_pushed_past_its_fold_is_trimmed_where_it_folds() {
    let lean = 65f64.to_radians();
    let frame = Frame::new(
        Vec3::new(0.0, 0.0, 4.0),
        Vec3::new(lean.sin(), 0.0, lean.cos()),
        Vec3::X,
    )
    .unwrap();
    let ring = torus(&frame, 3.0, 1.0, "ring").unwrap();
    let b = solid(&ring, &Motion::Slide(Vec3::new(0.0, 0.0, -8.0)), "push").unwrap();
    assert!(check::check(&b).is_empty());
    let volume = measure::volume(&b);
    let want = slid_torus_volume(&frame, 3.0, 1.0, 8.0, 0.02);
    assert!((volume - want).abs() <= 1e-4 * want, "{volume} {want}");
}
