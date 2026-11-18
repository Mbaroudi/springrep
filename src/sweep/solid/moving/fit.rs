//! Surfaces through the points of a map of two parameters: a plane or an extrusion where the points
//! lie on one, else the B-spline that interpolates them on a grid in those parameters (C. de Boor,
//! "A Practical Guide to Splines", 1978, chapter XVII), its grid refined until it lies within
//! [`SHEET_FIT`] of the map halfway between the grid's points.

use super::super::grazing::FIT;
use crate::{
    Error, Result,
    geom::{BSpline, Curve, Extrusion, NurbsSurface, Plane, Surface},
    math::{Frame, Point3, Vec2},
};

/// How near a fitted sheet lies to its map. Tighter than a curve's: where a sheet's two parameter
/// directions run nearly together, a point off it by `d` is read off its parameters by far more.
const SHEET_FIT: f64 = FIT / 10.0;

/// The surface through `map` over the rectangle from `lo` to `hi`, whose points are `size` across,
/// held to the map where `care` says it matters.
pub(super) fn surface(
    map: &dyn Fn(Vec2) -> Point3,
    lo: Vec2,
    hi: Vec2,
    size: f64,
    care: &dyn Fn(Vec2) -> bool,
) -> Result<Surface> {
    let at = |i: usize, j: usize, n: usize| {
        map(Vec2::new(
            lo.x + (hi.x - lo.x) * i as f64 / n as f64,
            lo.y + (hi.y - lo.y) * j as f64 / n as f64,
        ))
    };
    let k = 8;
    let pts: Vec<Point3> = (0..=k)
        .flat_map(|i| (0..=k).map(move |j| (i, j)))
        .map(|(i, j)| at(i, j, k))
        .collect();
    let exact = 1e-11 * size.max(1.0);
    if let Some(plane) = flat(&pts, exact) {
        return Ok(plane);
    }
    // Ruled along y with one direction everywhere: the extrusion of its edge at `lo.y`.
    let w = at(0, k, k) - at(0, 0, k);
    let straight = (0..=k).all(|i| {
        let (a, b) = (at(i, 0, k), at(i, k, k));
        (b - a - w).norm() <= exact
            && (1..k).all(|j| (at(i, j, k) - a.lerp(b, j as f64 / k as f64)).norm() <= exact)
    });
    // Points are found on an extrusion through its curve seen along it, so the curve must not
    // come round over itself so seen: a path winding about the extrusion's direction is fitted.
    if straight
        && let Some(dir) = w.unit()
        && winding(&|x| map(Vec2::new(x, lo.y)), lo.x, hi.x, dir) < std::f64::consts::PI
    {
        let curve = fit_curve(&|x| map(Vec2::new(x, lo.y)), (lo.x, hi.x))?;
        return Ok(Surface::Extrusion(Box::new(Extrusion { curve, dir })));
    }
    let ruled = (0..=k).all(|i| {
        let (a, b) = (at(i, 0, k), at(i, k, k));
        (1..k).all(|j| (at(i, j, k) - a.lerp(b, j as f64 / k as f64)).norm() <= exact)
    });
    let (mut nx, mut ny) = (16usize, if ruled { 1 } else { 4 });
    for _ in 0..12 {
        let s = grid(map, lo, hi, (nx, ny))?;
        let x = |i: usize, n: usize| lo.x + (hi.x - lo.x) * i as f64 / n as f64;
        let y = |j: usize, n: usize| lo.y + (hi.y - lo.y) * j as f64 / n as f64;
        let err = |u: f64, v: f64| {
            let q = Vec2::new(u, v);
            if care(q) {
                s.eval(u, v).dist(map(q))
            } else {
                0.0
            }
        };
        let ex = (0..nx)
            .flat_map(|i| (0..=ny).map(move |j| (i, j)))
            .map(|(i, j)| err(x(2 * i + 1, 2 * nx), y(j, ny)))
            .fold(0.0, f64::max);
        let ey = (0..=nx)
            .flat_map(|i| (0..ny).map(move |j| (i, j)))
            .map(|(i, j)| err(x(i, nx), y(2 * j + 1, 2 * ny)))
            .fold(0.0, f64::max);
        let exy = (0..nx)
            .flat_map(|i| (0..ny).map(move |j| (i, j)))
            .map(|(i, j)| err(x(2 * i + 1, 2 * nx), y(2 * j + 1, 2 * ny)))
            .fold(0.0, f64::max);
        if ex.max(ey).max(exy) <= SHEET_FIT {
            return Ok(Surface::Nurbs(Box::new(s)));
        }
        if ex > SHEET_FIT || (exy > SHEET_FIT && ex >= ey) {
            nx *= 2;
        }
        if !ruled && (ey > SHEET_FIT || (exy > SHEET_FIT && ey > ex)) {
            ny *= 2;
        }
        if nx > 2048 || ny > 256 || nx * ny > 1 << 15 {
            break;
        }
    }
    Err(Error::Numerical(
        "a swept surface's fit did not reach its tolerance".into(),
    ))
}

/// How far the tangent of `c` over `[a, b]`, seen along `dir`, turns in all.
fn winding(c: &dyn Fn(f64) -> Point3, a: f64, b: f64, dir: crate::math::Vec3) -> f64 {
    let n = 256;
    let flat = |p: Point3| p - dir * p.dot(dir);
    let pts: Vec<Point3> = (0..=n)
        .map(|i| flat(c(a + (b - a) * i as f64 / n as f64)))
        .collect();
    let mut total = 0.0;
    for i in 1..n {
        let (d0, d1) = (pts[i] - pts[i - 1], pts[i + 1] - pts[i]);
        let s = d0.cross(d1).dot(dir);
        total += s.atan2(d0.dot(d1)).abs();
    }
    total
}

/// The plane all of `pts` lie on within `exact`, if there is one.
fn flat(pts: &[Point3], exact: f64) -> Option<Surface> {
    let p0 = pts[0];
    let p1 = *pts
        .iter()
        .max_by(|a, b| a.dist(p0).total_cmp(&b.dist(p0)))?;
    let d = (p1 - p0).unit()?;
    let off = |p: Point3| (p - p0) - d * (p - p0).dot(d);
    let p2 = *pts
        .iter()
        .max_by(|a, b| off(**a).norm().total_cmp(&off(**b).norm()))?;
    let n = d.cross(p2 - p0).unit()?;
    if pts.iter().all(|&p| (p - p0).dot(n).abs() <= exact) {
        let frame = Frame::new(p0, n, d).ok()?;
        return Some(Surface::Plane(Plane { frame }));
    }
    None
}

/// The B-spline through `map` on a grid of `n` cells over the rectangle, at the grid's own
/// parameters.
fn grid(
    map: &dyn Fn(Vec2) -> Point3,
    lo: Vec2,
    hi: Vec2,
    n: (usize, usize),
) -> Result<NurbsSurface> {
    let xs: Vec<f64> = (0..=n.0)
        .map(|i| lo.x + (hi.x - lo.x) * i as f64 / n.0 as f64)
        .collect();
    let ys: Vec<f64> = (0..=n.1)
        .map(|j| lo.y + (hi.y - lo.y) * j as f64 / n.1 as f64)
        .collect();
    // Each row along x through its points, then each column of their control points along y.
    let mut rows: Vec<Spline> = Vec::with_capacity(ys.len());
    for &y in &ys {
        let pts: Vec<Point3> = xs.iter().map(|&x| map(Vec2::new(x, y))).collect();
        rows.push(interpolate(&pts, &xs)?);
    }
    let cu = rows[0].ctrl.len();
    let mut cols: Vec<Spline> = Vec::with_capacity(cu);
    for i in 0..cu {
        let pts: Vec<Point3> = rows.iter().map(|r| r.ctrl[i]).collect();
        cols.push(interpolate(&pts, &ys)?);
    }
    let cv = cols[0].ctrl.len();
    let mut ctrl = Vec::with_capacity(cu * cv);
    for c in &cols {
        ctrl.extend_from_slice(&c.ctrl);
    }
    NurbsSurface::new(
        (rows[0].degree, cols[0].degree),
        (rows[0].knots.clone(), cols[0].knots.clone()),
        (cu, cv),
        ctrl,
        None,
    )
}

/// Degree of the splines the grid is interpolated with where it has points enough: a quintic's
/// error falls with the sixth power of the spacing, so a sheet that winds several turns needs far
/// fewer spans than a cubic's.
const DEGREE: usize = 5;

/// A B-spline's degree, knots and control points.
struct Spline {
    degree: usize,
    knots: Vec<f64>,
    ctrl: Vec<Point3>,
}

/// The spline of [`DEGREE`] through `pts` at the increasing `params`, its inner knots at the data
/// sites with the first and last few left out (the not-a-knot ends of de Boor, chapter XIII); a
/// cubic where there are too few points.
fn interpolate(pts: &[Point3], params: &[f64]) -> Result<Spline> {
    let n = pts.len();
    if n <= DEGREE + 1 {
        let c = BSpline::interpolate_at(pts, params)?;
        return Ok(Spline {
            degree: c.degree(),
            knots: c.knots().to_vec(),
            ctrl: c.control_points().to_vec(),
        });
    }
    let p = DEGREE;
    let skip = p.div_ceil(2);
    let mut knots = vec![params[0]; p + 1];
    knots.extend_from_slice(&params[skip..n - skip]);
    knots.extend(std::iter::repeat_n(params[n - 1], p + 1));
    let rows: Vec<(usize, Vec<f64>)> = params.iter().map(|&t| local_basis(p, &knots, t)).collect();
    let rhs: Vec<f64> = pts.iter().flat_map(|q| [q.x, q.y, q.z]).collect();
    let x = crate::numeric::solve_banded(rows, rhs, 3)?;
    let ctrl = x.chunks(3).map(|c| Point3::new(c[0], c[1], c[2])).collect();
    Ok(Spline {
        degree: p,
        knots,
        ctrl,
    })
}

/// The basis functions of `degree` on the clamped `knots` that do not vanish at `t`, and the index
/// of the first (de Boor, chapter X).
fn local_basis(degree: usize, knots: &[f64], t: f64) -> (usize, Vec<f64>) {
    let n = knots.len() - degree - 1;
    // The span holding `t`, the last one at the domain's end.
    let span = (degree..n)
        .rev()
        .find(|&i| knots[i] <= t && knots[i] < knots[i + 1])
        .unwrap_or(degree);
    let mut b = vec![0.0; degree + 1];
    let mut left = vec![0.0; degree + 1];
    let mut right = vec![0.0; degree + 1];
    b[0] = 1.0;
    for j in 1..=degree {
        left[j] = t - knots[span + 1 - j];
        right[j] = knots[span + j] - t;
        let mut saved = 0.0;
        for r in 0..j {
            let tmp = b[r] / (right[r + 1] + left[j - r]);
            b[r] = saved + right[r + 1] * tmp;
            saved = left[j - r] * tmp;
        }
        b[j] = saved;
    }
    (span - degree, b)
}

/// The cubic B-spline through `at` over `range` at its own parameter, its points refined until it
/// lies within [`FIT`] of `at` halfway between them. A cubic, as the points nearest a cubic's
/// extrusion are found along it directly.
pub(super) fn fit_curve(at: &dyn Fn(f64) -> Point3, range: (f64, f64)) -> Result<Curve> {
    let mut n = 16usize;
    while n <= 1 << 14 {
        let xs: Vec<f64> = (0..=n)
            .map(|i| range.0 + (range.1 - range.0) * i as f64 / n as f64)
            .collect();
        let pts: Vec<Point3> = xs.iter().map(|&x| at(x)).collect();
        let c = BSpline::interpolate_at(&pts, &xs)?;
        let worst = (0..n)
            .map(|i| {
                let x = 0.5 * (xs[i] + xs[i + 1]);
                c.eval(x).dist(at(x))
            })
            .fold(0.0, f64::max);
        if worst <= FIT {
            return Ok(Curve::BSpline(c));
        }
        n *= 2;
    }
    Err(Error::Numerical(
        "a swept curve's fit did not reach its tolerance".into(),
    ))
}
