//! The points within a distance of chosen faces, edges and vertices of a solid, as the union of a
//! piece for each (Rossignac and Requicha 1986): the slab a face sweeps moving along its normal,
//! the tube of points about an edge, and the ball about a vertex. Every point near the solid is
//! near some face, edge or vertex at its foot, so the union over all of them, with the solid,
//! is the solid grown; each piece is exact, and the booleans resolve where pieces meet.

use std::collections::{BTreeSet, HashMap};
use std::f64::consts::{PI, TAU};

use crate::{
    boolean,
    error::{Error, Result},
    geom::{Curve, Surface},
    math::{Frame, Point3, Vec2, Vec3},
    primitive::{cylinder, sphere},
    profile::{
        NamedSegment, Profile, ProfileLoop, Segment,
        region::{Geom, Loop, Op, Region, Seg},
    },
    sweep::{extrude, revolve},
    tol,
    topo::{Brep, EdgeId, FaceId, FaceName, VertexId},
};

fn unsupported<T>(why: &str) -> Result<T> {
    Err(Error::Unsupported(why.to_string()))
}

/// The solid of the points swept by `f` moving `by` along its outward normal (inward where `by`
/// is negative), every face of it named as `f` is.
pub(crate) fn slab(s: &Brep, f: FaceId, by: f64, feature: &str) -> Result<Brep> {
    let mut out = match &s.face(f).surface {
        Surface::Plane(_) => plane_slab(s, f, by, feature)?,
        Surface::Cylinder(_) | Surface::Cone(_) | Surface::Sphere(_) | Surface::Torus(_) => {
            turned_slab(s, f, by, feature)?
        }
        other => {
            return unsupported(&format!(
                "the points near a {} face: its slab is not built",
                other.kind()
            ));
        }
    };
    let names = s.face(f).names.clone();
    for g in out.face_ids().collect::<Vec<_>>() {
        out.set_face_names(g, names.clone());
    }
    Ok(out)
}

/// The profile segment `curve` makes over `(t0, t1)`, run from `t0`, drawn in `fr`'s xy plane.
fn segment_on(fr: &Frame, curve: &Curve, (t0, t1): (f64, f64)) -> Result<Segment> {
    let flat = |p: Point3| {
        let l = fr.to_local(p);
        Vec2::new(l.x, l.y)
    };
    Ok(match curve {
        Curve::Line(_) => Segment::Line {
            a: flat(curve.eval(t0)),
            b: flat(curve.eval(t1)),
        },
        Curve::Circle(c) => {
            let center = flat(c.frame.origin);
            let sense = c.frame.z.dot(fr.z).signum();
            let start = flat(curve.eval(t0)) - center;
            let sweep = (t1 - t0) * sense;
            if (sweep.abs() - TAU).abs() <= tol::ANGULAR {
                // A whole circle runs from angle 0: its loop is that circle alone.
                Segment::Circle {
                    center,
                    radius: c.radius,
                    ccw: sweep > 0.0,
                }
            } else {
                Segment::Arc {
                    center,
                    radius: c.radius,
                    start: start.y.atan2(start.x),
                    sweep,
                }
            }
        }
        Curve::Ellipse(e) => {
            let sense = e.frame.z.dot(fr.z).signum();
            let m = fr.dir_to_local(e.frame.x);
            Segment::Ellipse {
                center: flat(e.frame.origin),
                rotation: m.y.atan2(m.x),
                major: e.major,
                minor: e.minor,
                start: t0 * sense,
                sweep: (t1 - t0) * sense,
            }
        }
        other => {
            return unsupported(&format!(
                "the slab of a plane face bounded by a {} edge",
                other.kind()
            ));
        }
    })
}

/// A plane face's slab: the face extruded along its normal.
fn plane_slab(s: &Brep, f: FaceId, by: f64, feature: &str) -> Result<Brep> {
    let face = s.face(f);
    let Surface::Plane(p) = &face.surface else {
        unreachable!("a plane face")
    };
    let outward = if face.reversed { -p.frame.z } else { p.frame.z };
    let fr = Frame::new(p.frame.origin, outward * by.signum(), p.frame.x)?;
    let mut loops = Vec::new();
    for &lp in &face.loops {
        let mut segments = Vec::new();
        for &c in &s.lp(lp).coedges {
            let co = s.coedge(c);
            let x = s.edge(co.edge);
            let Some(curve) = &x.curve else { continue };
            let range = if co.reversed {
                (x.range.1, x.range.0)
            } else {
                x.range
            };
            segments.push(NamedSegment::new(segment_on(&fr, curve, range)?, "slab"));
        }
        if !segments.is_empty() {
            loops.push(ProfileLoop { segments });
        }
    }
    // The outer loop first: it bounds the most area.
    loops.sort_by(|a, b| b.signed_area2().abs().total_cmp(&a.signed_area2().abs()));
    extrude(&Profile { loops }, &fr, by.abs(), feature)
}

/// Where a face of revolution lies in its meridian half plane, `(ρ, h)` out from and along its
/// axis: a line (a cylinder's or cone's) or a circle (a sphere's or torus's).
enum Meridian {
    Line { a: Vec2, dir: Vec2 },
    Circle { c: Vec2, r: f64 },
}

/// A face of revolution's slab: the region its meridian sweeps moving along its normal, clipped
/// at the axis, turned a whole turn. A point the sweep carries past the axis lies nearer the face
/// on the other side, so the clip loses nothing that is near the face at its foot.
fn turned_slab(s: &Brep, f: FaceId, by: f64, feature: &str) -> Result<Brep> {
    let face = s.face(f);
    let surface = &face.surface;
    let (sf, meridian) = match surface {
        Surface::Cylinder(c) => (
            c.frame,
            Meridian::Line {
                a: Vec2::new(c.radius, 0.0),
                dir: Vec2::new(0.0, 1.0),
            },
        ),
        Surface::Cone(c) => {
            let (sa, ca) = c.half_angle.sin_cos();
            (
                c.frame,
                Meridian::Line {
                    a: Vec2::new(c.radius, 0.0),
                    dir: Vec2::new(sa, ca),
                },
            )
        }
        Surface::Sphere(sp) => (
            sp.frame,
            Meridian::Circle {
                c: Vec2::ZERO,
                r: sp.radius,
            },
        ),
        Surface::Torus(t) => (
            t.frame,
            Meridian::Circle {
                c: Vec2::new(t.major, 0.0),
                r: t.minor,
            },
        ),
        _ => unreachable!("a surface of revolution"),
    };
    let half_plane = |p: Point3| {
        let l = sf.to_local(p);
        Vec2::new(l.x.hypot(l.y), l.z)
    };
    // The face must be bounded by parallels about its axis, its seam and its poles alone: it turns
    // a whole turn, between the parallels' places on the meridian.
    let mut uses: HashMap<EdgeId, usize> = HashMap::new();
    for &lp in &face.loops {
        for &c in &s.lp(lp).coedges {
            *uses.entry(s.coedge(c).edge).or_default() += 1;
        }
    }
    let mut ends: Vec<Vec2> = Vec::new();
    for (&e, &n) in &uses {
        let x = s.edge(e);
        let at = match &x.curve {
            None => half_plane(s.vertex(x.start).point),
            Some(Curve::Circle(c))
                if c.frame.z.cross(sf.z).norm() <= tol::ANGULAR
                    && sf
                        .to_local(c.frame.origin)
                        .x
                        .hypot(sf.to_local(c.frame.origin).y)
                        <= tol::LINEAR =>
            {
                Vec2::new(c.radius, (c.frame.origin - sf.origin).dot(sf.z))
            }
            // A seam runs along the meridian, its ends on the parallels.
            Some(_) if n == 2 => continue,
            Some(_) => {
                return unsupported(
                    "the slab of a face of revolution bounded by other than its parallels",
                );
            }
        };
        if !ends.iter().any(|q| q.dist(at) <= tol::LINEAR) {
            ends.push(at);
        }
    }
    let dom = boolean::face_domain(s, f);
    let Some(uv) = dom.interior_point() else {
        return unsupported("the slab of a face with no inside");
    };
    let inner = half_plane(surface.eval(uv.x, uv.y));
    let n3 = surface.normal(uv.x, uv.y);
    let radial = {
        let l = sf.to_local(surface.eval(uv.x, uv.y));
        sf.x * l.x + sf.y * l.y
    };
    let n = Vec2::new(n3.dot(radial.unit().unwrap_or(sf.x)), n3.dot(sf.z))
        * if face.reversed {
            -by.signum()
        } else {
            by.signum()
        };
    let t = by.abs();
    let mut pieces: Vec<Region> = Vec::new();
    match meridian {
        Meridian::Line { a, dir } => {
            let along = |p: Vec2| (p - a).dot(dir);
            let (Some(lo), Some(hi)) = (
                ends.iter().map(|&p| along(p)).min_by(f64::total_cmp),
                ends.iter().map(|&p| along(p)).max_by(f64::total_cmp),
            ) else {
                return unsupported("the slab of an unbounded face");
            };
            if hi - lo <= tol::LINEAR {
                return unsupported("the slab of a face with no length along its meridian");
            }
            let (p, q) = (a + dir * lo, a + dir * hi);
            let m = Vec2::new(-dir.y, dir.x);
            let m = if m.dot(n) > 0.0 { m } else { -m };
            pieces.push(Region::polygon(&[p, q, q + m * t, p + m * t]));
        }
        Meridian::Circle { c, r } => {
            let angle = |p: Vec2| (p.y - c.y).atan2(p.x - c.x);
            let mid = angle(inner);
            let mut cuts: Vec<f64> = ends.iter().map(|&p| angle(p)).collect();
            // The arc between the ends it runs between: from the last end before the inside to
            // the first after it, counter-clockwise; round the whole circle where one end or none.
            let ahead = |a: f64| (a - mid).rem_euclid(TAU);
            cuts.sort_by(|x, y| ahead(*x).total_cmp(&ahead(*y)));
            let (from, sweep) = match cuts.len() {
                0 => (mid, TAU),
                1 => (cuts[0], TAU),
                _ => {
                    let to = mid + ahead(cuts[0]);
                    let from = mid - (TAU - ahead(cuts[cuts.len() - 1]));
                    (from, to - from)
                }
            };
            let out = n.dot(inner - c) > 0.0;
            let (r0, r1) = if out { (r, r + t) } else { (r - t, r) };
            // Pieces of at most half a turn, so each is simply connected.
            let k = (sweep / PI).ceil().max(1.0) as usize;
            for i in 0..k {
                let a0 = from + sweep * i as f64 / k as f64;
                let a1 = from + sweep * (i + 1) as f64 / k as f64;
                let arc = |radius: f64, x: f64, y: f64| Geom::Arc {
                    c,
                    r: radius,
                    a0: x,
                    sweep: y - x,
                };
                let polar = |radius: f64, a: f64| c + Vec2::new(a.cos(), a.sin()) * radius;
                let geoms = if r0 > tol::LINEAR {
                    vec![
                        arc(r1, a0, a1),
                        Geom::Line {
                            a: polar(r1, a1),
                            b: polar(r0, a1),
                        },
                        arc(r0, a1, a0),
                        Geom::Line {
                            a: polar(r0, a0),
                            b: polar(r1, a0),
                        },
                    ]
                } else {
                    vec![
                        arc(r1, a0, a1),
                        Geom::Line {
                            a: polar(r1, a1),
                            b: c,
                        },
                        Geom::Line {
                            a: c,
                            b: polar(r1, a0),
                        },
                    ]
                };
                let segs = geoms
                    .into_iter()
                    .map(|geom| Seg {
                        geom,
                        names: BTreeSet::new(),
                    })
                    .collect();
                pieces.push(Region {
                    loops: vec![Loop { segs }],
                });
            }
        }
    }
    let plane = Frame::new(sf.origin, -sf.y, sf.x)?;
    let mut parts = Vec::new();
    for piece in pieces {
        for profile in clipped(&piece)? {
            parts.push(revolve(&profile, &plane, sf.origin, sf.z, TAU, feature)?);
        }
    }
    let refs: Vec<&Brep> = parts.iter().collect();
    boolean::union_all(&refs)
}

/// The profiles of `region` where `ρ ≥ 0`, the side of the axis it turns about.
fn clipped(region: &Region) -> Result<Vec<Profile>> {
    let (mut lo, mut hi) = (
        Vec2::new(f64::INFINITY, f64::INFINITY),
        Vec2::new(f64::NEG_INFINITY, f64::NEG_INFINITY),
    );
    for s in region.segs() {
        for k in 0..=16 {
            let p = s.geom.at(k as f64 / 16.0);
            lo = Vec2::new(lo.x.min(p.x), lo.y.min(p.y));
            hi = Vec2::new(hi.x.max(p.x), hi.y.max(p.y));
        }
    }
    if lo.x >= -tol::LINEAR {
        return region.profiles();
    }
    if hi.x <= tol::LINEAR {
        return Ok(Vec::new());
    }
    let pad = 1.0 + (hi - lo).x.abs() + (hi - lo).y.abs();
    let half = Region::polygon(&[
        Vec2::new(0.0, lo.y - pad),
        Vec2::new(hi.x + pad, lo.y - pad),
        Vec2::new(hi.x + pad, hi.y + pad),
        Vec2::new(0.0, hi.y + pad),
    ]);
    region.combine(&half, Op::Intersect)?.profiles()
}

/// The solid of the points within `r` of edge `e`'s curve, its ends cut square: a cylinder about
/// a line, and about a circle the disk of radius `r` turned along it, clipped at its axis.
pub(crate) fn tube(s: &Brep, e: EdgeId, r: f64, feature: &str) -> Result<Brep> {
    let x = s.edge(e);
    let mut out = match &x.curve {
        Some(Curve::Line(l)) => {
            let from = l.origin + l.dir * x.range.0;
            let fr = Frame::from_axis(from, l.dir)?;
            cylinder(&fr, r, x.range.1 - x.range.0, feature)?
        }
        Some(Curve::Circle(c)) => {
            let (t0, t1) = x.range;
            let x0 = c.frame.radial(t0);
            let plane = Frame::new(c.frame.origin, -c.frame.z.cross(x0), x0)?;
            let disk = Region {
                loops: vec![Loop {
                    segs: vec![Seg {
                        geom: Geom::Arc {
                            c: Vec2::new(c.radius, 0.0),
                            r,
                            a0: 0.0,
                            sweep: TAU,
                        },
                        names: BTreeSet::new(),
                    }],
                }],
            };
            let angle = (t1 - t0).min(TAU);
            let mut parts = Vec::new();
            for profile in clipped(&disk)? {
                parts.push(revolve(
                    &profile,
                    &plane,
                    c.frame.origin,
                    c.frame.z,
                    angle,
                    feature,
                )?);
            }
            let refs: Vec<&Brep> = parts.iter().collect();
            boolean::union_all(&refs)?
        }
        Some(other) => {
            return unsupported(&format!(
                "the points near a {} edge: its tube is not built",
                other.kind()
            ));
        }
        None => return unsupported("the tube of an edge with no curve"),
    };
    let name: BTreeSet<FaceName> = std::iter::once(FaceName::new(feature, "round")).collect();
    for g in out.face_ids().collect::<Vec<_>>() {
        out.set_face_names(g, name.clone());
    }
    Ok(out)
}

/// The ball of radius `r` about `p`, its face named `<feature>.round`.
pub(crate) fn ball(p: Point3, r: f64, feature: &str) -> Result<Brep> {
    let mut out = sphere(&Frame::WORLD.translated(p - Point3::ZERO), r, feature)?;
    let name: BTreeSet<FaceName> = std::iter::once(FaceName::new(feature, "round")).collect();
    for g in out.face_ids().collect::<Vec<_>>() {
        out.set_face_names(g, name.clone());
    }
    Ok(out)
}

/// The vertices of `s` whose balls the tubes about `edges` and the slabs leave uncovered: those
/// where the edges of `edges` do not just run on into each other tangentially, and the apexes of
/// cones `apex` says are corners.
pub(crate) fn corners(s: &Brep, edges: &[EdgeId], apex: impl Fn(FaceId) -> bool) -> Vec<VertexId> {
    let mut leaving: HashMap<VertexId, Vec<Vec3>> = HashMap::new();
    for &e in edges {
        let x = s.edge(e);
        let Some(c) = &x.curve else { continue };
        let (_, d0) = c.d1(x.range.0);
        let (_, d1) = c.d1(x.range.1);
        leaving.entry(x.start).or_default().push(d0);
        leaving.entry(x.end).or_default().push(-d1);
    }
    let mut out: BTreeSet<VertexId> = BTreeSet::new();
    for (v, dirs) in leaving {
        let smooth = dirs.len() == 2
            && dirs[0]
                .unit()
                .zip(dirs[1].unit())
                .is_some_and(|(a, b)| a.dot(b) <= -1.0 + tol::ANGULAR);
        if !smooth {
            out.insert(v);
        }
    }
    for c in s.coedge_ids() {
        let x = s.edge(s.coedge(c).edge);
        if x.curve.is_none() && apex(s.coedge_face(c)) {
            out.insert(x.start);
        }
    }
    out.into_iter().collect()
}
