//! The body cut along its characteristic curves into the pieces of face the motion carries outward
//! and the rest.

use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;

use super::grazing::{self, Arrival, Rate};
use crate::{
    Error, Result,
    boolean::{
        arrange, coedge_trim_with,
        domain::{Domain, Trim},
        face_domain,
        sew::PieceGeom,
    },
    check::coedge_params,
    geom::{Curve, Surface},
    math::{Point3, Vec2, Vec3},
    measure, tol,
    topo::{Brep, CoedgeId, EdgeId, FaceId, FaceName},
};

/// A piece of one of the body's faces, on one side of its characteristic curves.
pub(super) struct Part {
    pub surface: Surface,
    pub reversed: bool,
    pub names: BTreeSet<FaceName>,
    pub loops: Vec<Vec<Trim>>,
    /// Whether the motion carries it outward.
    pub front: bool,
}

/// The body cut along its characteristic curves: its vertices (the body's own first), its edge
/// pieces, and its face pieces, whose trims name the pieces.
pub(super) struct Cut {
    pub pool: Vec<Point3>,
    pub pieces: Vec<PieceGeom>,
    pub parts: Vec<Part>,
}

/// Samples along each coedge where the rate is looked for roots.
const SAMPLES: usize = 64;

/// Where a characteristic curve meets the boundary: a point on a coedge, at its edge's parameter. A
/// seam's two coedges are two places on the face's boundary, a period apart.
#[derive(Clone, Copy, Debug)]
struct Meet {
    coedge: CoedgeId,
    t: f64,
}

/// The rate's sign at a sample, or 0 where it is within `eps` of zero.
fn sign(g: f64, eps: f64) -> i8 {
    if g > eps {
        1
    } else if g < -eps {
        -1
    } else {
        0
    }
}

/// The roots of the rate along the loops of face `f`: where its sign changes along the boundary,
/// pinned by bisection, or a single sample where it is zero between opposite signs.
fn boundary_roots(b: &Brep, f: FaceId, rate: &Rate, eps: f64) -> Vec<Meet> {
    let mut out = Vec::new();
    for &lp in &b.face(f).loops {
        // The loop's samples in order: each coedge's, without its last, which is the next one's
        // first.
        let mut samples: Vec<(CoedgeId, f64, Vec2, f64)> = Vec::new();
        for &c in &b.lp(lp).coedges {
            let e = b.edge(b.coedge(c).edge);
            let pieces = e
                .curve
                .as_ref()
                .map_or(0, |cv| cv.breakpoints(e.range.0, e.range.1).len());
            let n = SAMPLES * (pieces + 1);
            let ts = coedge_params(b, c, n);
            let uvs = b.coedge_uvs(c, &ts);
            for k in 0..n {
                samples.push((c, ts[k], uvs[k], rate.at(uvs[k])));
            }
        }
        let n = samples.len();
        let signs: Vec<i8> = samples.iter().map(|s| sign(s.3, eps)).collect();
        if signs.iter().all(|&s| s == 0) {
            continue;
        }
        for i in 0..n {
            let j = (i + 1) % n;
            let (a, z) = (&samples[i], &samples[j]);
            // The stretch from one sample to the next lies on the first one's coedge, which ends
            // where the next coedge starts.
            if signs[i] != 0 && signs[j] != 0 && signs[i] != signs[j] {
                let to = if a.0 == z.0 {
                    (z.1, z.2)
                } else {
                    let end = end_param(b, a.0);
                    (end, b.coedge_uvs(a.0, &[a.1, end])[1])
                };
                out.push(Meet {
                    coedge: a.0,
                    t: bisect_on(b, a.0, rate, (a.1, a.2, a.3), to),
                });
            }
            // A lone zero between opposite signs.
            if signs[i] == 0 {
                let before = signs[(i + n - 1) % n];
                let after = signs[j];
                if before != 0 && after != 0 && before != after {
                    out.push(Meet {
                        coedge: samples[i].0,
                        t: samples[i].1,
                    });
                }
            }
        }
    }
    out
}

/// The parameter a coedge starts at.
fn start_param(b: &Brep, c: CoedgeId) -> f64 {
    let e = b.edge(b.coedge(c).edge);
    if b.coedge(c).reversed {
        e.range.1
    } else {
        e.range.0
    }
}

/// The parameter a coedge ends at.
fn end_param(b: &Brep, c: CoedgeId) -> f64 {
    let e = b.edge(b.coedge(c).edge);
    if b.coedge(c).reversed {
        e.range.0
    } else {
        e.range.1
    }
}

/// The root of the rate along coedge `c` between `a` (parameter, uv, rate) and `z` (parameter, uv),
/// by bisection on the edge parameter.
fn bisect_on(b: &Brep, c: CoedgeId, rate: &Rate, a: (f64, Vec2, f64), z: (f64, Vec2)) -> f64 {
    let e = b.edge(b.coedge(c).edge);
    let Some(curve) = &e.curve else {
        return a.0;
    };
    let (mut lo, mut hi) = (a.0, z.0);
    let mut hint = a.1;
    let s0 = a.2.signum();
    for _ in 0..80 {
        let m = 0.5 * (lo + hi);
        if m == lo || m == hi {
            break;
        }
        let uv = rate.surface.project_near(curve.eval(m), hint);
        hint = uv;
        if rate.at(uv).signum() == s0 {
            lo = m;
        } else {
            hi = m;
        }
    }
    0.5 * (lo + hi)
}

/// Cuts `b` along the curves where `field`, the velocity of its points, lies in its faces.
pub(super) fn cut(b: &Brep, field: &dyn Fn(Point3) -> Vec3) -> Result<Cut> {
    let size = measure::bbox(b).diagonal();
    let speed = b
        .vertex_ids()
        .map(|v| field(b.vertex(v).point).norm())
        .fold(0.0, f64::max)
        .max(field(measure::bbox(b).center()).norm());
    let eps = tol::ANGULAR * speed.max(f64::MIN_POSITIVE);
    let faces: Vec<FaceId> = b.face_ids().collect();
    let doms: Vec<Domain> = faces.iter().map(|&f| face_domain(b, f)).collect();
    let rates: Vec<Rate> = faces
        .iter()
        .map(|&f| Rate {
            surface: &b.face(f).surface,
            out: if b.face(f).reversed { -1.0 } else { 1.0 },
            field,
            size,
        })
        .collect();
    let still: Vec<bool> = doms
        .iter()
        .zip(&rates)
        .map(|(d, r)| {
            let mut pts = d.interior_points();
            pts.extend(d.samples().iter().flatten().flatten().copied());
            pts.iter().all(|&uv| r.at(uv).abs() <= eps)
        })
        .collect();
    // Every face's roots on its boundary, as places on edges.
    let mut meets: Vec<Vec<Meet>> = Vec::with_capacity(faces.len());
    for (i, &f) in faces.iter().enumerate() {
        meets.push(if still[i] {
            Vec::new()
        } else {
            boundary_roots(b, f, &rates[i], eps)
        });
    }
    let mut pool: Vec<Point3> = b.vertex_ids().map(|v| b.vertex(v).point).collect();
    let mut splits: HashMap<EdgeId, Vec<(f64, usize)>> = HashMap::new();
    // A place on an edge as a vertex: an end of the edge where it lies on one, an earlier split
    // where it lies on one, else a new vertex.
    let mut place = |edge: EdgeId, t: f64, pool: &mut Vec<Point3>| -> (usize, Point3) {
        let e = b.edge(edge);
        let curve = e.curve.as_ref().expect("roots lie on edges with curves");
        let p = curve.eval(t);
        for v in [e.start, e.end] {
            let q = b.vertex(v).point;
            if q.dist(p) <= 10.0 * tol::LINEAR {
                return (v.index(), q);
            }
        }
        let list = splits.entry(edge).or_default();
        for &(_, vi) in list.iter() {
            if pool[vi].dist(p) <= 10.0 * tol::LINEAR {
                return (vi, pool[vi]);
            }
        }
        pool.push(p);
        list.push((t, pool.len() - 1));
        (pool.len() - 1, p)
    };
    // Each face's roots as vertices, with their parameters on the face, and the curves traced
    // between them.
    struct Traced {
        curve: Curve,
        range: (f64, f64),
        ends: (usize, usize),
        uvs: (Vec2, Vec2),
    }
    let mut traced: Vec<Vec<Traced>> = Vec::with_capacity(faces.len());
    for (i, &f) in faces.iter().enumerate() {
        // Each place a curve meets the boundary: a vertex where the boundary passes it, so one
        // on a seam is two, a period apart.
        let mut roots: Vec<(Point3, Vec2)> = Vec::new();
        let mut ids: Vec<usize> = Vec::new();
        for m in &meets[i] {
            let (vi, p) = place(b.coedge(m.coedge).edge, m.t, &mut pool);
            let uv = b.coedge_uvs(m.coedge, &[start_param(b, m.coedge), m.t])[1];
            let seen = ids
                .iter()
                .zip(&roots)
                .any(|(&v, r)| v == vi && (r.1 - uv).norm() <= 1e-9 * (1.0 + uv.norm()));
            if !seen {
                roots.push((p, uv));
                ids.push(vi);
            }
        }
        let mut used = vec![false; roots.len()];
        let mut curves = Vec::new();
        for s in 0..roots.len() {
            if used[s] {
                continue;
            }
            used[s] = true;
            let (path, arrival) = grazing::trace(&rates[i], &doms[i], &roots, s)?;
            let end = match arrival {
                Arrival::Root(k) => {
                    if used[k] && k != s {
                        return Err(Error::Numerical(
                            "two characteristic curves end at one point".into(),
                        ));
                    }
                    used[k] = true;
                    ids[k]
                }
                Arrival::Boundary(p) => {
                    let (edge, t) = nearest_edge(b, f, p)?;
                    place(edge, t, &mut pool).0
                }
            };
            let (curve, range) = grazing::fit(&rates[i], &path)?;
            let uvs = (path.uvs[0], *path.uvs.last().expect("a trace has points"));
            if ids[s] == end && curve.period().is_none() {
                // A curve that closes through a seam is two edges, so no edge both starts and ends
                // at one vertex on a curve that does not close by its own period.
                let mid = 0.5 * (range.0 + range.1);
                let p = curve.eval(mid);
                pool.push(p);
                let m = pool.len() - 1;
                let half = rates[i]
                    .surface
                    .project_near(p, path.uvs[path.uvs.len() / 2]);
                curves.push(Traced {
                    curve: curve.clone(),
                    range: (range.0, mid),
                    ends: (ids[s], m),
                    uvs: (uvs.0, half),
                });
                curves.push(Traced {
                    curve,
                    range: (mid, range.1),
                    ends: (m, end),
                    uvs: (half, uvs.1),
                });
                continue;
            }
            curves.push(Traced {
                curve,
                range,
                ends: (ids[s], end),
                uvs,
            });
        }
        traced.push(curves);
    }
    // The edge pieces between splits, each edge's in order along it.
    let mut pieces: Vec<PieceGeom> = Vec::new();
    let mut of_edge: HashMap<EdgeId, Vec<(usize, f64, f64)>> = HashMap::new();
    for e in b.edge_ids() {
        let edge = b.edge(e);
        let mut cuts = splits.remove(&e).unwrap_or_default();
        cuts.sort_by(|x, y| x.0.total_cmp(&y.0));
        let mut stops = vec![(edge.range.0, edge.start.index())];
        stops.extend(cuts);
        stops.push((edge.range.1, edge.end.index()));
        let curve = edge.curve.clone().map(Rc::new);
        let list = of_edge.entry(e).or_default();
        for w in stops.windows(2) {
            list.push((pieces.len(), w[0].0, w[1].0));
            pieces.push(PieceGeom {
                curve: curve.clone(),
                range: (w[0].0, w[1].0),
                v0: w[0].1,
                v1: w[1].1,
                tol: tol::LINEAR,
            });
        }
    }
    let mut parts = Vec::new();
    for (i, &f) in faces.iter().enumerate() {
        let face = b.face(f);
        let boundary: Vec<Vec<Trim>> = face
            .loops
            .iter()
            .map(|&lp| {
                let mut trims = Vec::new();
                for &c in &b.lp(lp).coedges {
                    let co = b.coedge(c);
                    let mut list = of_edge[&co.edge].clone();
                    if co.reversed {
                        list.reverse();
                    }
                    let mut ts: Vec<f64> = Vec::with_capacity(list.len() + 1);
                    for &(_, t0, t1) in &list {
                        ts.push(if co.reversed { t1 } else { t0 });
                    }
                    let last = list.last().expect("an edge has a piece");
                    ts.push(if co.reversed { last.1 } else { last.2 });
                    let uvs = b.coedge_uvs(c, &ts);
                    for (k, &(pi, _, _)) in list.iter().enumerate() {
                        let pg = &pieces[pi];
                        let (v0, v1) = if co.reversed {
                            (pg.v1, pg.v0)
                        } else {
                            (pg.v0, pg.v1)
                        };
                        trims.push(coedge_trim_with(
                            b,
                            c,
                            pi,
                            pg.curve.clone(),
                            (ts[k], ts[k + 1]),
                            (uvs[k], uvs[k + 1]),
                            (v0, v1),
                        ));
                    }
                }
                trims
            })
            .collect();
        let mut cuts = Vec::new();
        for tc in &traced[i] {
            let curve = Rc::new(tc.curve.clone());
            pieces.push(PieceGeom {
                curve: Some(curve.clone()),
                range: tc.range,
                v0: tc.ends.0,
                v1: tc.ends.1,
                tol: tol::LINEAR,
            });
            cuts.push(Trim {
                piece: pieces.len() - 1,
                forward: true,
                curve: Some(curve),
                pole: pool[tc.ends.0],
                t0: tc.range.0,
                t1: tc.range.1,
                map: crate::boolean::domain::UvMap::Projected,
                uv0: tc.uvs.0,
                uv1: tc.uvs.1,
                v0: tc.ends.0,
                v1: tc.ends.1,
            });
        }
        let regions = arrange::split(
            &face.surface,
            face.reversed,
            boundary,
            cuts,
            &|_| tol::LINEAR,
            &|_| false,
        )?;
        for loops in regions {
            let front = if still[i] {
                false
            } else {
                let dom = Domain::new(face.surface.clone(), face.reversed, loops.clone());
                let uv = dom.interior_point().ok_or_else(|| {
                    Error::Numerical(
                        "a piece of a face cut along its grazing curve is empty".into(),
                    )
                })?;
                rates[i].at(uv) > eps
            };
            parts.push(Part {
                surface: face.surface.clone(),
                reversed: face.reversed,
                names: face.names.clone(),
                loops,
                front,
            });
        }
    }
    Ok(Cut {
        pool,
        pieces,
        parts,
    })
}

/// The edge of face `f` nearest `p`, and the parameter on it of the point nearest `p`.
fn nearest_edge(b: &Brep, f: FaceId, p: Point3) -> Result<(EdgeId, f64)> {
    let mut best: Option<(f64, EdgeId, f64)> = None;
    for &lp in &b.face(f).loops {
        for &c in &b.lp(lp).coedges {
            let e = b.coedge(c).edge;
            let edge = b.edge(e);
            let Some(curve) = &edge.curve else { continue };
            let t = curve.project(p, edge.range.0, edge.range.1);
            let d = curve.eval(t).dist(p);
            if best.is_none_or(|x| d < x.0) {
                best = Some((d, e, t));
            }
        }
    }
    match best {
        Some((d, e, t)) if d <= 1e-6 * (1.0 + p.norm()) => Ok((e, t)),
        _ => Err(Error::Numerical(
            "a characteristic curve leaves its face away from every edge".into(),
        )),
    }
}
