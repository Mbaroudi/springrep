//! Changes to a solid's faces: faces moved, tilted or taken out, a solid hollowed, grown by a
//! distance or stretched, and the slab a face sweeps. Each builds exact tools and combines them
//! with the booleans, blends and face moves.

use std::collections::HashMap;
use std::f64::consts::{PI, TAU};

use crate::{
    blend,
    blend::shaped::plane_of,
    boolean,
    error::{Error, Result},
    geom::{Curve, Surface},
    math::{Frame, Point3, Vec2, Vec3},
    pick::convex_edges,
    profile::{NamedSegment, Profile, ProfileLoop, Segment},
    sweep::revolve,
    tol,
    topo::{Brep, FaceId, FaceName},
};

mod affine;
mod delete;
pub(crate) mod tweak;

pub use affine::{Affine, affine};
pub use delete::delete_faces;
pub use tweak::{draft, move_faces, move_faces_by, offset_faces};

fn named(segment: Segment, name: &str) -> NamedSegment {
    NamedSegment::new(segment, name)
}

/// Which side of a solid's faces a shell's wall lies on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wall {
    /// Inside the solid: its outside stays where it is.
    Inward,
    /// Outside the solid: its faces become the wall's inside.
    Outward,
}

/// `s` hollowed to a wall `thickness` thick, open through the faces of `open`. The cavity's faces
/// are `<feature>.inner`. An inward wall leaves every face of `s` in place with its names. An
/// outward wall's outside is new and carries no name, and each face of `s` is the wall's inside,
/// keeping its names beside `inner`.
pub fn shell(s: &Brep, open: &[FaceId], thickness: f64, wall: Wall, feature: &str) -> Result<Brep> {
    if thickness.is_nan() || thickness <= tol::LINEAR {
        return Err(Error::InvalidInput(
            "a shell's wall must be thicker than the tolerance".into(),
        ));
    }
    let inner = FaceName::new(feature, FaceName::INNER);
    // Every face moved `by` along its outward normal, the open ones `open_by`.
    let moved = |by: f64, open_by: f64| {
        let by: HashMap<FaceId, f64> = s
            .face_ids()
            .map(|f| (f, if open.contains(&f) { open_by } else { by }))
            .filter(|&(_, d)| d != 0.0)
            .collect();
        tweak::offset_faces(s, &by)
    };
    // The open faces stay where they are: moved out, they would carry the cut past their plane
    // into a wall that stands beside the opening. The cut lies on them there and leaves no skin.
    match wall {
        Wall::Inward => {
            let mut cavity = moved(-thickness, 0.0)?;
            for f in cavity.face_ids().collect::<Vec<_>>() {
                cavity.set_face_names(f, std::iter::once(inner.clone()).collect());
            }
            boolean::difference(s, &cavity)
        }
        Wall::Outward => {
            let mut outer = moved(thickness, 0.0)?;
            for f in outer.face_ids().collect::<Vec<_>>() {
                if !open.contains(&f) {
                    outer.set_face_names(f, Default::default());
                }
            }
            let mut cavity = s.clone();
            for f in cavity.face_ids().collect::<Vec<_>>() {
                let mut names = cavity.face(f).names.clone();
                names.insert(inner.clone());
                cavity.set_face_names(f, names);
            }
            boolean::difference(&outer, &cavity)
        }
    }
}

/// The points within `distance` of `s`: every face moved out that far, a hole that closes on the
/// way followed there, each convex edge rounded at `distance` (which leaves a ball's patch at each
/// convex corner), and over each end of a hole that closed, the dimple its rim leaves.
pub fn offset(s: &Brep, distance: f64, feature: &str) -> Result<Brep> {
    if distance == 0.0 {
        return Ok(s.clone());
    }
    if distance < 0.0 {
        return Err(Error::Unsupported("an inward offset".into()));
    }
    let by: HashMap<FaceId, f64> = s.face_ids().map(|f| (f, distance)).collect();
    let moved = tweak::offset_faces(s, &by)?;
    let edges = convex_edges(&moved)?;
    let mut out = if edges.is_empty() {
        moved
    } else {
        blend::fillet_edges_mitred(&moved, &edges, distance, feature)?
    };
    for f in s.face_ids() {
        let face = s.face(f);
        let Surface::Cylinder(c) = &face.surface else {
            continue;
        };
        if !face.reversed || c.radius >= distance {
            continue;
        }
        for &lp in &face.loops {
            for &co in &s.lp(lp).coedges {
                let e = s.coedge(co).edge;
                let Some(Curve::Circle(rim)) = &s.edge(e).curve else {
                    continue;
                };
                let Some(g) = s
                    .coedge_ids()
                    .filter(|&o| s.coedge(o).edge == e)
                    .map(|o| s.coedge_face(o))
                    .find(|&g| g != f)
                else {
                    continue;
                };
                let (_, up) = plane_of(s, g)?;
                out = boolean::difference(
                    &out,
                    &dimple(rim.frame.origin, up, c.radius, distance, feature)?,
                )?;
            }
        }
    }
    Ok(out)
}

/// What the offset by `d` leaves open over a hole's rim of radius `r` about `o`, opening along
/// `up`: the hole's cylinder beyond the rim, farther than `d` from it. In the meridian (ρ out from
/// the axis, z up from the rim's plane), the region above the arc of radius `d` about the rim.
fn dimple(o: Point3, up: Vec3, r: f64, d: f64, feature: &str) -> Result<Brep> {
    let q = (d * d - r * r).sqrt();
    let from = q.atan2(-r);
    let line = |a: [f64; 2], b: [f64; 2]| Segment::Line {
        a: Vec2::new(a[0], a[1]),
        b: Vec2::new(b[0], b[1]),
    };
    let meridian = Profile {
        loops: vec![ProfileLoop {
            segments: vec![
                named(
                    Segment::Arc {
                        center: Vec2::new(r, 0.0),
                        radius: d,
                        start: from,
                        sweep: PI / 2.0 - from,
                    },
                    "ring",
                ),
                named(line([r, d], [r, d + 1.0]), "wall"),
                named(line([r, d + 1.0], [0.0, d + 1.0]), "top"),
                named(line([0.0, d + 1.0], [0.0, q]), "axis"),
            ],
        }],
    };
    let f = Frame::from_axis(o, up)?;
    revolve(&meridian, &Frame::new(o, -f.y, f.x)?, o, f.z, TAU, feature)
}

/// The solid `faces` sweep moving from 0 to `thickness` along their outward normals, inward where
/// it is negative: `s` with them moved, less `s`, or `s` less `s` with them moved in.
pub fn thicken(s: &Brep, faces: &[FaceId], thickness: f64) -> Result<Brep> {
    let by: HashMap<FaceId, f64> = faces.iter().map(|&f| (f, thickness)).collect();
    let moved = tweak::offset_faces(s, &by)?;
    if thickness > 0.0 {
        boolean::difference(&moved, s)
    } else {
        boolean::difference(s, &moved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{boolean::half_space, measure, sweep::extrude};

    /// A prism on the polygon `pts`, `h` tall, its sides named `e0`, `e1`, … from the first edge.
    fn prism(pts: &[[f64; 2]], h: f64) -> Brep {
        let pts: Vec<Vec2> = pts.iter().map(|p| Vec2::new(p[0], p[1])).collect();
        let names: Vec<String> = (0..pts.len()).map(|i| format!("e{i}")).collect();
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        extrude(&Profile::polygon(&pts, &names), &Frame::WORLD, h, "p").unwrap()
    }

    fn face(b: &Brep, name: &str) -> FaceId {
        let n = FaceName::new("p", name);
        b.face_ids()
            .find(|&f| b.face(f).names.contains(&n))
            .unwrap()
    }

    fn close(got: f64, want: f64) {
        assert!(
            (got - want).abs() < 1e-7 * want.max(1.0),
            "{got} against {want}"
        );
    }

    #[test]
    fn a_block_offset_is_the_block_rounded_by_the_distance() {
        // 4 × 3 × 2 offset by 0.5: V + S·d + π(a + b + c)·d² + (4/3)π·d³, Steiner's formula.
        let b = prism(&[[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]], 2.0);
        let d = 0.5;
        let want = 24.0 + 52.0 * d + PI * 9.0 * d * d + 4.0 / 3.0 * PI * d.powi(3);
        close(measure::volume(&offset(&b, d, "o").unwrap()), want);
    }

    #[test]
    fn an_l_s_rounded_edges_swing_round_its_concave_edge() {
        // An L of arms 8 × 3, 5 tall, every convex edge rounded 1: the union of the balls of
        // radius 1 inside it. The two arm boxes rounded, less their overlap (the corner column
        // rounded, with (1 − π/4)·r² short of a square along the concave edge, and 4r³/3 of that
        // at its ends), and at each end of the concave edge what the ball keeps swinging round
        // it, (1 − π/4)·r³·(5/3 − π/2). A box rounded r all over: the inner box, its faces moved
        // out r, its edges' quarter cylinders and its corners' eighth balls.
        // Its area likewise, the swing's torus, (π/2)·r²(π/2 − 1), standing for two cylinder
        // pieces of r²(π/2 − 1), and the flat face keeping the corner's r × r square less its
        // quarter disk. The arms' widths, depths, heights and radii include draws of the
        // conformance suite's L-prisms the swing once failed on.
        let cases = [
            (8.0, 3.0, 8.0, 5.0, 1.0),
            (
                6.199379767061393,
                3.2446824245098926,
                5.944930609453334,
                2.787400445599377,
                0.983805266315506,
            ),
            (
                8.410642646708611,
                2.4606909192521025,
                9.679346455567444,
                7.230068216963447,
                0.2821417462255193,
            ),
            (
                9.564764789551234,
                3.3401708771362455,
                6.900089620242228,
                3.7297726032339753,
                0.7396202116839262,
            ),
            (
                7.342452085970791,
                2.923584026607261,
                6.1264770305629295,
                2.2167560977768446,
                0.8678845173029267,
            ),
            (
                5.52965122853034,
                2.429579012791682,
                9.404723647837528,
                5.938985170662157,
                0.6834351663091055,
            ),
            (
                6.3990454944784565,
                2.3841696916601958,
                8.181436493454777,
                4.839043994693185,
                0.7102451360532922,
            ),
            (
                6.538663752237994,
                3.1564303512067906,
                9.224051200212097,
                5.94676849583945,
                1.2821630460001838,
            ),
        ];
        for (w, t, d, h, r) in cases {
            let b = prism(&[[0.0, 0.0], [w, 0.0], [w, t], [t, t], [t, d], [0.0, d]], h);
            let out = blend::fillet_edges(&b, &convex_edges(&b).unwrap(), r, "fil").unwrap();
            assert!(
                crate::check::check(&out).is_empty(),
                "{:?}",
                crate::check::check(&out)
            );
            let rounded = |x: f64, y: f64, z: f64| {
                let (a, b, c) = (x - 2.0 * r, y - 2.0 * r, z - 2.0 * r);
                (
                    a * b * c
                        + 2.0 * r * (a * b + b * c + c * a)
                        + PI * r * r * (a + b + c)
                        + 4.0 / 3.0 * PI * r.powi(3),
                    2.0 * (a * b + b * c + c * a) + 2.0 * PI * r * (a + b + c) + 4.0 * PI * r * r,
                )
            };
            let k = 1.0 - PI / 4.0;
            let ((va, sa), (vb, sb), (vc, sc)) =
                (rounded(w, t, h), rounded(t, d, h), rounded(t, t, h));
            let core_h = h - 2.0 * r;
            let volume = va + vb - vc - k * (core_h * r * r + 4.0 / 3.0 * r.powi(3))
                + 2.0 * k * (5.0 / 3.0 - PI / 2.0) * r.powi(3);
            let area = sa + sb
                - sc
                - k * (2.0 * core_h * r + 4.0 * r * r)
                - 2.0 * ((2.0 - PI / 2.0) * (PI / 2.0 - 1.0) - k) * r * r;
            close(measure::volume(&out), volume);
            close(measure::area(&out), area);
            // Written and read back, the swing's torus, whose pole lies on its boundary, is
            // the same solid.
            let text = crate::step::write(&out, "l").unwrap();
            let back = crate::step::read(text.as_bytes(), "l").unwrap();
            assert!(back.rejected.is_empty(), "{:?}", back.rejected);
            close(measure::volume(&back.brep), volume);
        }
    }

    #[test]
    fn an_l_prism_offset_turned_away_is_the_same_size() {
        let (w, t, h) = (9.536367478693311, 2.7154798898513985, 6.344052519590756);
        let b = prism(
            &[[0.0, 0.0], [w, 0.0], [w, t], [t, t], [t, h], [0.0, h]],
            5.222922150702244,
        );
        let d = 0.5756566176826041;
        let here = measure::volume(&offset(&b, d, "o").unwrap());
        let mut moved = b.clone();
        moved.transform(&crate::math::Transform {
            r: [
                [
                    -0.2124025946267396,
                    -0.9641070956655728,
                    -0.15931932049542274,
                ],
                [
                    0.9494953687642673,
                    -0.1650876859267214,
                    -0.26684190197674823,
                ],
                [0.2309625131726732, -0.20795086930006934, 0.9504802751589895],
            ],
            t: Vec3::new(-45.51695804202156, 41.045298753592036, -21.51869167292315),
        });
        close(measure::volume(&offset(&moved, d, "o").unwrap()), here);
    }

    #[test]
    fn a_notch_s_faces_taken_out_leave_the_block_whole() {
        let b = prism(
            &[
                [0.0, 0.0],
                [6.0, 0.0],
                [6.0, 4.0],
                [4.0, 4.0],
                [4.0, 3.0],
                [2.0, 3.0],
                [2.0, 4.0],
                [0.0, 4.0],
            ],
            1.0,
        );
        let notch = [face(&b, "e3"), face(&b, "e4"), face(&b, "e5")];
        close(measure::volume(&delete_faces(&b, &notch).unwrap()), 24.0);
    }

    fn feature_faces(b: &Brep, feature: &str) -> Vec<FaceId> {
        b.face_ids()
            .filter(|&f| b.face(f).names.iter().any(|n| n.feature == feature))
            .collect()
    }

    /// `b` with the faces of `feature` deleted: its volume and area.
    fn deleted(b: &Brep, feature: &str) -> (f64, f64) {
        let out = delete_faces(b, &feature_faces(b, feature)).unwrap();
        assert!(
            crate::check::check(&out).is_empty(),
            "{:?}",
            crate::check::check(&out)
        );
        (measure::volume(&out), measure::area(&out))
    }

    #[test]
    fn a_boss_and_a_concave_round_deleted_take_material_away() {
        use crate::primitive::cylinder;
        let at = |z: f64| Frame::new(Point3::new(0.0, 0.0, z), Vec3::Z, Vec3::X).unwrap();
        // A plate of radius 8 and 4 thick with a boss of radius 3 and 8 tall standing on it.
        let plate = cylinder(&at(-4.0), 8.0, 4.0, "plate").unwrap();
        let boss = cylinder(&at(0.0), 3.0, 8.0, "boss").unwrap();
        let solid = boolean::union(&plate, &boss).unwrap();
        let (v, a) = deleted(&solid, "boss");
        close(v, 256.0 * PI);
        close(a, 192.0 * PI);
        // An L of outline area 150 and perimeter 70, 8 tall, its inside corner rounded 2.
        let pts = [
            [0.0, 0.0],
            [20.0, 0.0],
            [20.0, 5.0],
            [5.0, 5.0],
            [5.0, 15.0],
            [0.0, 15.0],
        ];
        let l = prism(&pts, 8.0);
        let rounded = blend::fillet(&l, &[("p.e2", "p.e3")], 2.0, "round").unwrap();
        let (v, a) = deleted(&rounded, "round");
        close(v, 150.0 * 8.0);
        close(a, 2.0 * 150.0 + 70.0 * 8.0);
    }

    #[test]
    fn deleted_flats_close_where_the_surfaces_round_them_bulge_past_their_edges() {
        use crate::primitive::{cone, cylinder, sphere};
        // A shaft of radius 5 and 20 long with |x| > 4 milled away: whole again, 500π.
        let shaft = cylinder(&Frame::WORLD, 5.0, 20.0, "shaft").unwrap();
        let right = half_space(&shaft, Point3::new(4.0, 0.0, 0.0), Vec3::X).unwrap();
        let left = half_space(&shaft, Point3::new(-4.0, 0.0, 0.0), -Vec3::X).unwrap();
        let milled = boolean::difference_all(&shaft, &[&right, &left]).unwrap();
        let (v, a) = deleted(&milled, "half");
        close(v, 500.0 * PI);
        close(a, 250.0 * PI);
        // A ball of radius 5 cut flat at z = 3: its flat's edge lies in its own plane.
        let ball = sphere(&Frame::WORLD, 5.0, "ball").unwrap();
        let above = half_space(&ball, Point3::new(0.0, 0.0, 3.0), Vec3::Z).unwrap();
        let (v, a) = deleted(&boolean::difference(&ball, &above).unwrap(), "half");
        close(v, 500.0 * PI / 3.0);
        close(a, 100.0 * PI);
        // A frustum of radii 6 and 3, 6 tall, its top rim rounded 1: (π·6/3)(6² + 6·3 + 3²).
        let frustum = cone(&Frame::WORLD, 6.0, 3.0, 6.0, "f").unwrap();
        let rounded = blend::fillet(&frustum, &[("f.top", "f.side")], 1.0, "round").unwrap();
        let (v, a) = deleted(&rounded, "round");
        close(v, 126.0 * PI);
        close(a, 45.0 * PI + 9.0 * PI * 45f64.sqrt());
    }

    #[test]
    fn a_rounded_box_s_top_rim_deleted_runs_its_corner_cylinders_up_only() {
        // The box 20 × 12 × 8, its upright edges rounded 3, then its top rim rounded 1. Deleted,
        // the rim leaves the box with its outline a rectangle with quarter circles of radius 3 at
        // its corners: area 20·12 − (4 − π)·3², perimeter 2(14 + 6) + 2π·3.
        let body = crate::primitive::block(&Frame::WORLD, Vec3::new(20.0, 12.0, 8.0), "b");
        let uprights = [
            ("b.u_min", "b.v_min"),
            ("b.v_min", "b.u_max"),
            ("b.u_max", "b.v_max"),
            ("b.v_max", "b.u_min"),
        ];
        let walls = blend::fillet(&body.unwrap(), &uprights, 3.0, "corner").unwrap();
        let rounded = blend::fillet(&walls, &[("b.end", "b.v_min")], 1.0, "rim").unwrap();
        let outline = 240.0 - (4.0 - PI) * 9.0;
        let (v, a) = deleted(&rounded, "rim");
        close(v, 8.0 * outline);
        close(a, 2.0 * outline + 8.0 * (40.0 + 6.0 * PI));
    }

    #[test]
    fn a_face_thickened_is_a_slab_on_it() {
        // The 4 × 2 side of a 4 × 3 × 2 block thickened 0.5 outward, then 1 inward.
        let b = prism(&[[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]], 2.0);
        let side = face(&b, "e0");
        close(measure::volume(&thicken(&b, &[side], 0.5).unwrap()), 4.0);
        close(measure::volume(&thicken(&b, &[side], -1.0).unwrap()), 8.0);
    }
}
