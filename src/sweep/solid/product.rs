//! The swept solid of a body under a steady motion: the boundary of the region the motion carries
//! outward, times the motion, capped by the rest of the body at the start and that region at the
//! end.

use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;

use super::cut::Cut;
use super::sheet::Steady;
use crate::{
    Error, Result,
    boolean::{
        domain::{Trim, UvMap},
        sew::{self, OutFace, PieceGeom},
    },
    geom::Curve,
    math::{Point3, Transform, Vec3},
    tol,
    topo::{Brep, FaceName},
};

/// The trims of `loops` moved by `t`, their pieces and vertices renumbered by `piece` and
/// `vertex`, each curve the one `curves` holds for its new piece.
fn moved_loops(
    loops: &[Vec<Trim>],
    t: &Transform,
    piece: &dyn Fn(usize) -> usize,
    vertex: &dyn Fn(usize) -> usize,
    curves: &[Option<Rc<Curve>>],
) -> Vec<Vec<Trim>> {
    loops
        .iter()
        .map(|l| {
            l.iter()
                .map(|tr| {
                    let p = piece(tr.piece);
                    Trim {
                        piece: p,
                        curve: curves[p].clone(),
                        pole: t.point(tr.pole),
                        v0: vertex(tr.v0),
                        v1: vertex(tr.v1),
                        ..tr.clone()
                    }
                })
                .collect()
        })
        .collect()
}

/// Whether the face `curve` sweeps over `range` folds: somewhere inside the range the motion runs
/// along the curve, so the face's normal, the velocity crossed with the curve's direction,
/// vanishes and turns over. Past such a point the envelope lies inside the solid.
fn folds(motion: &Steady, curve: &Curve, range: (f64, f64)) -> bool {
    let n = 256;
    let mut last: Option<Vec3> = None;
    // An end may be where the motion runs along the curve: a characteristic curve ends there.
    for k in 1..n {
        let t = range.0 + (range.1 - range.0) * k as f64 / n as f64;
        let (p, d) = curve.d1(t);
        let v = motion.velocity(p);
        let normal = v.cross(d);
        if normal.norm() <= 1e-12 * v.norm() * d.norm() {
            continue;
        }
        if last.is_some_and(|l| l.dot(normal) <= 0.0) {
            return true;
        }
        last = Some(normal);
    }
    false
}

/// The solid `cut` sweeps under `motion`, its swept faces named after `feature`: the pieces of face
/// the motion carries outward at the end, the rest at the start, and what the curves between them
/// sweep. With `only`, the solid just those pieces of face sweep instead, each at both ends, which
/// the body at the start is united with.
pub(super) fn product(
    cut: &Cut,
    motion: &Steady,
    feature: &str,
    only: Option<&[usize]>,
) -> Result<Brep> {
    let (pose0, pose1) = (motion.pose0(), motion.pose1());
    let n = cut.pool.len();
    let np = cut.pieces.len();
    // A vertex the motion holds still is one vertex at both ends, and has no path.
    let still: Vec<bool> = cut.pool.iter().map(|&p| motion.holds(p)).collect();
    let mut pool: Vec<Point3> = cut.pool.iter().map(|&p| pose0.point(p)).collect();
    pool.extend(cut.pool.iter().map(|&p| pose1.point(p)));
    let at1 = |v: usize| if still[v] { v } else { v + n };
    // A piece of edge the motion holds still, on a turn's axis, is one edge at both ends and
    // sweeps nothing.
    let held: Vec<bool> = cut
        .pieces
        .iter()
        .map(|pg| {
            still[pg.v0]
                && still[pg.v1]
                && pg
                    .curve
                    .as_ref()
                    .is_none_or(|c| motion.holds(c.eval(0.5 * (pg.range.0 + pg.range.1))))
        })
        .collect();
    let piece1 = |p: usize| if p < np && held[p] { p } else { p + np };
    let mut pieces: Vec<PieceGeom> = Vec::with_capacity(2 * np);
    for (t, end) in [(&pose0, false), (&pose1, true)] {
        for pg in &cut.pieces {
            let shift = |v: usize| if end { at1(v) } else { v };
            pieces.push(PieceGeom {
                curve: pg.curve.as_ref().map(|c| Rc::new(c.transformed(t))),
                range: pg.range,
                v0: shift(pg.v0),
                v1: shift(pg.v1),
                tol: pg.tol,
            });
        }
    }
    let curves: Vec<Option<Rc<Curve>>> = pieces.iter().map(|p| p.curve.clone()).collect();
    let mut faces: Vec<OutFace> = Vec::new();
    // The parts whose sweep is built: those carried outward, or the ones `only` names.
    let swept = |pi: usize| match only {
        None => cut.parts[pi].front,
        Some(set) => set.contains(&pi),
    };
    for (pi, part) in cut.parts.iter().enumerate() {
        if swept(pi) {
            faces.push(OutFace {
                surface: part.surface.transformed(&pose1),
                reversed: part.reversed,
                names: part.names.clone(),
                loops: moved_loops(&part.loops, &pose1, &piece1, &at1, &curves),
                memo: None,
                side: Some(1),
            });
        }
        if swept(pi) == only.is_some() {
            let mut face = OutFace {
                surface: part.surface.transformed(&pose0),
                reversed: part.reversed,
                names: part.names.clone(),
                loops: moved_loops(&part.loops, &pose0, &|p| p, &|v| v, &curves),
                memo: None,
                side: Some(0),
            };
            if only.is_some() {
                face.flip();
            }
            faces.push(face);
        }
    }
    // Each piece's uses by the parts: which part, and whether its trim runs along the piece.
    let mut uses: HashMap<usize, Vec<(usize, bool)>> = HashMap::new();
    for (pi, part) in cut.parts.iter().enumerate() {
        for tr in part.loops.iter().flatten() {
            uses.entry(tr.piece).or_default().push((pi, tr.forward));
        }
    }
    let mut keys: Vec<usize> = uses.keys().copied().collect();
    keys.sort_unstable();
    // The path of each vertex the swept faces end on, made once and shared by the two faces
    // beside it.
    let mut paths: HashMap<usize, Vec<usize>> = HashMap::new();
    let names: BTreeSet<FaceName> = [FaceName::new(feature, FaceName::SIDE)].into();
    // A path is cut at each whole turn of a screw, as a screwed profile's are, so no edge winds
    // round the axis more than once.
    let stops = motion.stops();
    for k in keys {
        let us = &uses[&k];
        let inside: Vec<&(usize, bool)> = us.iter().filter(|u| swept(u.0)).collect();
        if us.len() != 2 || inside.len() != 1 || held[k] {
            continue;
        }
        let forward = inside[0].1;
        let pg = &cut.pieces[k];
        let Some(curve) = pg.curve.as_ref() else {
            continue;
        };
        // The piece runs from `a` to `b` with the outward-moving part on its left.
        let (a, b) = if forward {
            (pg.v0, pg.v1)
        } else {
            (pg.v1, pg.v0)
        };
        let (ta, tb) = if forward {
            (pg.range.0, pg.range.1)
        } else {
            (pg.range.1, pg.range.0)
        };
        let placed = curve.transformed(&pose0);
        if folds(motion, &placed, pg.range) {
            return Err(Error::Unsupported(
                "a solid sweep whose envelope folds: the motion runs along a curve of the body's \
                 outline"
                    .into(),
            ));
        }
        let sheet = motion.sheet(&placed, cut.whole[k], pg.range)?;
        let span = sheet.span;
        let mut path =
            |v: usize, pieces: &mut Vec<PieceGeom>, pool: &mut Vec<Point3>| -> Result<Vec<usize>> {
                if let Some(ids) = paths.get(&v) {
                    return Ok(ids.clone());
                }
                if still[v] {
                    // A pole of the face, its own: its row of points at the vertex is the vertex.
                    pieces.push(PieceGeom {
                        curve: None,
                        range: (0.0, span),
                        v0: v,
                        v1: v,
                        tol: tol::LINEAR,
                    });
                    return Ok(vec![pieces.len() - 1]);
                }
                let curve = Rc::new(motion.path(pose0.point(cut.pool[v]))?);
                let mut ids = Vec::with_capacity(stops.len() - 1);
                let mut from = v;
                for (i, w) in stops.windows(2).enumerate() {
                    let to = if i + 2 == stops.len() {
                        at1(v)
                    } else {
                        pool.push(curve.eval(w[1]));
                        pool.len() - 1
                    };
                    pieces.push(PieceGeom {
                        curve: Some(curve.clone()),
                        range: (w[0], w[1]),
                        v0: from,
                        v1: to,
                        tol: tol::LINEAR,
                    });
                    ids.push(pieces.len() - 1);
                    from = to;
                }
                paths.insert(v, ids.clone());
                Ok(ids)
            };
        let pa = path(a, &mut pieces, &mut pool)?;
        let pb = path(b, &mut pieces, &mut pool)?;
        let cap = |piece: usize, s: f64, from: f64, to: f64, fwd: bool| {
            let pg = &pieces[piece];
            Trim {
                piece,
                forward: fwd,
                curve: pg.curve.clone(),
                pole: pool[pg.v0],
                t0: from,
                t1: to,
                map: if sheet.affine {
                    UvMap::Affine {
                        origin: sheet.uv(s, 0.0),
                        dir: sheet.across,
                    }
                } else {
                    UvMap::Projected
                },
                uv0: sheet.uv(s, from),
                uv1: sheet.uv(s, to),
                v0: if fwd { pg.v0 } else { pg.v1 },
                v1: if fwd { pg.v1 } else { pg.v0 },
            }
        };
        let side = |ids: &[usize], t: f64, up: bool| -> Vec<Trim> {
            let mut trims: Vec<Trim> = ids
                .iter()
                .map(|&piece| {
                    let pg = &pieces[piece];
                    let (s0, s1) = if up {
                        pg.range
                    } else {
                        (pg.range.1, pg.range.0)
                    };
                    Trim {
                        piece,
                        forward: up,
                        curve: pg.curve.clone(),
                        pole: pool[pg.v0],
                        t0: s0,
                        t1: s1,
                        // A pole has no curve to project: its row of parameters is the straight
                        // one between the row's ends, a single point on a plane.
                        map: if pg.curve.is_none() && !sheet.along_affine {
                            UvMap::Affine {
                                origin: sheet.uv(0.0, t),
                                dir: (sheet.uv(span, t) - sheet.uv(0.0, t)) / span,
                            }
                        } else if !sheet.along_affine {
                            UvMap::Projected
                        } else {
                            UvMap::Affine {
                                origin: sheet.uv(0.0, t),
                                dir: sheet.along,
                            }
                        },
                        uv0: sheet.uv(s0, t),
                        uv1: sheet.uv(s1, t),
                        v0: if up { pg.v0 } else { pg.v1 },
                        v1: if up { pg.v1 } else { pg.v0 },
                    }
                })
                .collect();
            if !up {
                trims.reverse();
            }
            trims
        };
        let mut lp = vec![cap(k, 0.0, ta, tb, forward)];
        lp.extend(side(&pb, tb, true));
        lp.push(cap(k + np, span, tb, ta, !forward));
        lp.extend(side(&pa, ta, false));
        // Material on the left in the parameter plane where the face is not reversed. A small
        // patch in the middle of the face, as the corners of a plane turned past half a turn
        // wind the other way.
        let (sm, tm, ds, dt) = (span / 2.0, (ta + tb) / 2.0, span * 1e-3, (tb - ta) * 1e-3);
        let corners = [
            sheet.uv(sm - ds, tm - dt),
            sheet.uv(sm - ds, tm + dt),
            sheet.uv(sm + ds, tm + dt),
            sheet.uv(sm + ds, tm - dt),
        ];
        let area: f64 = (0..4).map(|i| corners[i].cross(corners[(i + 1) % 4])).sum();
        faces.push(OutFace {
            surface: sheet.surface,
            reversed: area < 0.0,
            names: names.clone(),
            loops: vec![lp],
            memo: None,
            side: None,
        });
    }
    let vtol = vec![tol::LINEAR; pool.len()];
    sew::assemble(faces, &pool, &vtol, &pieces)
}

/// The body where the motion starts, its faces cut along their characteristic curves, so each
/// piece of face a part's sweep starts from is a face of it.
pub(super) fn start(cut: &Cut, motion: &Steady) -> Result<Brep> {
    let pose0 = motion.pose0();
    let pool: Vec<Point3> = cut.pool.iter().map(|&p| pose0.point(p)).collect();
    let pieces: Vec<PieceGeom> = cut
        .pieces
        .iter()
        .map(|pg| PieceGeom {
            curve: pg.curve.as_ref().map(|c| Rc::new(c.transformed(&pose0))),
            ..pg.clone()
        })
        .collect();
    let curves: Vec<Option<Rc<Curve>>> = pieces.iter().map(|p| p.curve.clone()).collect();
    let faces = cut
        .parts
        .iter()
        .map(|part| OutFace {
            surface: part.surface.transformed(&pose0),
            reversed: part.reversed,
            names: part.names.clone(),
            loops: moved_loops(&part.loops, &pose0, &|p| p, &|v| v, &curves),
            memo: None,
            side: Some(0),
        })
        .collect();
    let vtol = vec![tol::LINEAR; pool.len()];
    sew::assemble_as_cut(faces, &pool, &vtol, &pieces)
}
