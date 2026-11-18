//! What a curve sweeps under a steady motion, exactly where the motion has a closed form: an
//! extrusion under a slide, a revolution under a turn, and under a screw the rational B-spline whose
//! columns follow its control points' helices.

use std::f64::consts::TAU;

use super::motion::Motion;
use crate::{
    Error, Result,
    geom::{
        Circle, Cone, Curve, Cylinder, Extrusion, Line, NurbsCurve, NurbsSurface, Plane,
        Revolution, Sphere, Surface, Torus,
    },
    loft::rail::{Rail, point},
    math::{Frame, Point3, Transform, Vec2, Vec3},
    sweep::helix::{self, Helix},
    tol,
};

/// A slide, a turn or a screw, each run forward over a positive span.
pub(super) enum Steady {
    Slide {
        by: Vec3,
    },
    /// Turned by `from + τ·angle` about the unit `axis` through `origin`.
    Turn {
        origin: Point3,
        axis: Vec3,
        from: f64,
        angle: f64,
    },
    /// Carried to turn `from + τ·angle` of `screw`, whose axis is a unit vector.
    Screw {
        screw: Helix,
        from: f64,
        angle: f64,
    },
}

/// The face a curve sweeps: its surface, and where on it the curve lies at each step of the
/// motion. The point the motion carries the curve's point at `t` to after `s`, `s` from 0 to
/// `span`, is at `origin + s·along + t·across` in the surface's parameters: exactly where `affine`,
/// and at the curve's ends otherwise. A plane a turn sweeps has cartesian parameters instead, and
/// `turned` places its points.
pub(super) struct Sheet {
    pub surface: Surface,
    pub span: f64,
    pub origin: Vec2,
    pub along: Vec2,
    pub across: Vec2,
    pub affine: bool,
    pub turned: Option<Box<dyn Fn(f64, f64) -> Vec2>>,
}

impl Sheet {
    pub(super) fn uv(&self, s: f64, t: f64) -> Vec2 {
        match &self.turned {
            Some(f) => f(s, t),
            None => self.origin + self.along * s + self.across * t,
        }
    }

    fn affine(surface: Surface, span: f64, along: Vec2, across: Vec2, origin: Vec2) -> Sheet {
        Sheet {
            surface,
            span,
            origin,
            along,
            across,
            affine: true,
            turned: None,
        }
    }
}

/// Whether `curve` over `range` lies in one plane square to the unit `dir`: every point at one
/// height along it.
fn lies_square_to(curve: &Curve, range: (f64, f64), dir: Vec3) -> bool {
    match curve {
        Curve::Line(l) => l.dir.dot(dir).abs() <= tol::ANGULAR,
        Curve::Circle(c) => c.frame.z.cross(dir).norm() <= tol::ANGULAR,
        Curve::Ellipse(c) => c.frame.z.cross(dir).norm() <= tol::ANGULAR,
        _ => {
            let h0 = curve.eval(range.0).dot(dir);
            let off = |t: f64| (curve.eval(t).dot(dir) - h0).abs() <= tol::LINEAR;
            (0..=32).all(|k| off(range.0 + (range.1 - range.0) * k as f64 / 32.0))
                && curve.extremes(dir, range.0, range.1).into_iter().all(off)
        }
    }
}

/// What `curve` sweeps turning through `span` about the unit `axis` through `o`: a cylinder,
/// cone or plane where the curve is a line in a plane through the axis, a torus or sphere where it
/// is a circle in one, else the revolution of the curve. On each, the angle turned is `u`.
fn turned(curve: &Curve, range: (f64, f64), o: Point3, axis: Vec3, span: f64) -> Result<Sheet> {
    let revolution = || {
        Sheet::affine(
            Surface::Revolution(Box::new(Revolution {
                curve: curve.clone(),
                origin: o,
                axis,
            })),
            span,
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::ZERO,
        )
    };
    let height = |p: Point3| (p - o).dot(axis);
    let radial = |p: Point3| (p - o) - axis * height(p);
    // A curve level with the turn turns in its own plane.
    if lies_square_to(curve, range, axis) {
        let at = curve.eval(range.0);
        let plane = Frame::new(o + axis * height(at), axis, axis.any_perpendicular())?;
        let c = curve.clone();
        return Ok(Sheet {
            surface: Surface::Plane(Plane { frame: plane }),
            span,
            origin: Vec2::ZERO,
            along: Vec2::ZERO,
            across: Vec2::ZERO,
            affine: false,
            turned: Some(Box::new(move |s, t| {
                let q = plane.to_local(Transform::rotation(o, axis, s).point(c.eval(t)));
                Vec2::new(q.x, q.y)
            })),
        });
    }
    match curve {
        Curve::Line(l) => {
            if l.dir.cross(axis).dot(l.origin - o).abs() > tol::LINEAR {
                let still = Helix {
                    origin: o,
                    axis,
                    rise: 0.0,
                    sign: 1.0,
                };
                return screwed(curve, range, &still, span);
            }
            // The plane through the axis and the line, its x pointing out to the line.
            let far = [range.0, range.1, 0.5 * (range.0 + range.1)]
                .into_iter()
                .map(|t| radial(l.origin + l.dir * t))
                .max_by(|a, b| a.norm().total_cmp(&b.norm()))
                .expect("three points");
            let Some(x) = far.unit() else {
                return Ok(revolution());
            };
            let frame = Frame::new(o, axis, x)?;
            let (dr, dh) = (l.dir.dot(x), l.dir.dot(axis));
            let (rho, h) = (radial(l.origin).dot(x), height(l.origin));
            if dr.abs() <= tol::ANGULAR {
                let surface = Surface::Cylinder(Cylinder { frame, radius: rho });
                return Ok(Sheet::affine(
                    surface,
                    span,
                    Vec2::new(1.0, 0.0),
                    Vec2::new(0.0, dh),
                    Vec2::new(0.0, h),
                ));
            }
            // The cone through the line's point at 0, `v` the distance along the line from it.
            let cone = Cone {
                frame: Frame::new(o + axis * h, axis, x)?,
                radius: rho,
                half_angle: dr.atan2(dh),
            };
            Ok(Sheet::affine(
                Surface::Cone(cone),
                span,
                Vec2::new(1.0, 0.0),
                Vec2::new(0.0, 1.0),
                Vec2::ZERO,
            ))
        }
        Curve::Circle(c) => {
            let in_plane = c.frame.z.dot(axis).abs() <= tol::ANGULAR
                && (c.frame.origin - o).dot(c.frame.z).abs() <= tol::LINEAR;
            if !in_plane {
                return Ok(revolution());
            }
            let r = radial(c.frame.origin);
            let rho = r.norm();
            let x = if rho > tol::LINEAR {
                r / rho
            } else {
                c.frame.z.cross(axis)
            };
            let frame = Frame::new(o + axis * height(c.frame.origin), axis, x)?;
            // The circle's angle `t` is the tube's angle `phi0 + sign·t`.
            let (cx, cy) = (c.frame.x, c.frame.y);
            let phi0 = cx.dot(axis).atan2(cx.dot(x));
            let sign = (cx.dot(x) * cy.dot(axis) - cx.dot(axis) * cy.dot(x)).signum();
            if rho > tol::LINEAR {
                let surface = Surface::Torus(Torus {
                    frame,
                    major: rho,
                    minor: c.radius,
                });
                return Ok(Sheet::affine(
                    surface,
                    span,
                    Vec2::new(1.0, 0.0),
                    Vec2::new(0.0, sign),
                    Vec2::new(0.0, phi0),
                ));
            }
            // About its own centre a circle sweeps a sphere, whose latitude reads the arc's angle
            // only while the arc stays on one side of the axis.
            let (lo, hi) = (phi0 + sign * range.0, phi0 + sign * range.1);
            let shift = (0.5 * (lo + hi) / TAU).round() * TAU;
            let half = std::f64::consts::FRAC_PI_2;
            if lo.min(hi) - shift < -half - 1e-12 || lo.max(hi) - shift > half + 1e-12 {
                return Ok(revolution());
            }
            let surface = Surface::Sphere(Sphere {
                frame,
                radius: c.radius,
            });
            Ok(Sheet::affine(
                surface,
                span,
                Vec2::new(1.0, 0.0),
                Vec2::new(0.0, sign),
                Vec2::new(0.0, phi0 - shift),
            ))
        }
        _ => Ok(revolution()),
    }
}

impl Steady {
    pub(super) fn of(m: &Motion) -> Option<Steady> {
        match *m {
            Motion::Slide(by) => Some(Steady::Slide { by }),
            Motion::Turn {
                origin,
                axis,
                from,
                to,
            } => {
                let k = axis.unit()?;
                Some(if to >= from {
                    Steady::Turn {
                        origin,
                        axis: k,
                        from,
                        angle: to - from,
                    }
                } else {
                    Steady::Turn {
                        origin,
                        axis: -k,
                        from: -from,
                        angle: from - to,
                    }
                })
            }
            Motion::Screw { screw, from, to } => {
                let k = screw.axis.unit()?;
                let s = Helix { axis: k, ..screw };
                Some(if to >= from {
                    Steady::Screw {
                        screw: s,
                        from,
                        angle: to - from,
                    }
                } else {
                    Steady::Screw {
                        screw: Helix {
                            sign: -s.sign,
                            rise: -s.rise,
                            ..s
                        },
                        from: -from,
                        angle: from - to,
                    }
                })
            }
            _ => None,
        }
    }

    pub(super) fn pose(&self, tau: f64) -> Transform {
        match self {
            Steady::Slide { by } => Transform::translation(*by * tau),
            Steady::Turn {
                origin,
                axis,
                from,
                angle,
            } => Transform::rotation(*origin, *axis, from + angle * tau),
            Steady::Screw { screw, from, angle } => screw.transform(from + angle * tau),
        }
    }

    pub(super) fn pose0(&self) -> Transform {
        self.pose(0.0)
    }

    pub(super) fn pose1(&self) -> Transform {
        self.pose(1.0)
    }

    /// How fast the point at `x` moves where the motion starts, per unit of `τ`.
    pub(super) fn velocity(&self, x: Point3) -> Vec3 {
        match self {
            Steady::Slide { by } => *by,
            Steady::Turn {
                origin,
                axis,
                angle,
                ..
            } => axis.cross(x - *origin) * *angle,
            Steady::Screw { screw, angle, .. } => {
                (screw.axis.cross(x - screw.origin) * screw.sign + screw.axis * screw.rise) * *angle
            }
        }
    }

    /// The line the motion holds still: a turn's axis.
    pub(super) fn held(&self) -> Option<(Point3, Vec3)> {
        match self {
            Steady::Turn { origin, axis, .. } => Some((*origin, *axis)),
            _ => None,
        }
    }

    /// Whether the motion holds the body's point `p` still: a point on a turn's axis.
    pub(super) fn holds(&self, p: Point3) -> bool {
        match self {
            Steady::Turn { origin, axis, .. } => (p - *origin).cross(*axis).norm() <= tol::LINEAR,
            _ => false,
        }
    }

    /// The span of the motion's own parameter on its paths and sheets: the slide's length, or the
    /// angle turned.
    fn span(&self) -> f64 {
        match self {
            Steady::Slide { by } => by.norm(),
            Steady::Turn { angle, .. } | Steady::Screw { angle, .. } => *angle,
        }
    }

    /// The path of the point at `p` where the motion starts, over `[0, span]`.
    pub(super) fn path(&self, p: Point3) -> Result<Curve> {
        match self {
            Steady::Slide { by } => Ok(Curve::Line(Line {
                origin: p,
                dir: by
                    .unit()
                    .ok_or_else(|| Error::InvalidInput("a slide of no length".into()))?,
            })),
            Steady::Turn { origin, axis, .. } => {
                let foot = *origin + *axis * (p - *origin).dot(*axis);
                Ok(Curve::Circle(Circle {
                    frame: Frame::new(foot, *axis, p - foot)?,
                    radius: p.dist(foot),
                }))
            }
            // Turns of one screw add, so the path from where the motion starts is the screw's own
            // from turn 0.
            Steady::Screw { screw, angle, .. } => {
                let q = p - screw.origin;
                let radius = (q - screw.axis * q.dot(screw.axis)).norm();
                let nodes = helix::nodes(*angle, radius);
                let column = helix::column(screw, p, &nodes);
                Ok(Curve::Nurbs(Box::new(NurbsCurve::new(
                    helix::DEGREE,
                    helix::knots(&nodes),
                    column,
                    None,
                )?)))
            }
        }
    }

    /// What `curve`, where the motion starts, sweeps over its `range`.
    pub(super) fn sheet(&self, curve: &Curve, range: (f64, f64)) -> Result<Sheet> {
        let span = self.span();
        match self {
            Steady::Slide { by } => {
                let dir = by
                    .unit()
                    .ok_or_else(|| Error::InvalidInput("a slide of no length".into()))?;
                // A line slides in a plane, and a circle along its axis in a cylinder.
                match curve {
                    Curve::Line(l) => {
                        let frame = Frame::new(l.origin, l.dir.cross(dir), l.dir)?;
                        return Ok(Sheet {
                            surface: Surface::Plane(Plane { frame }),
                            span,
                            origin: Vec2::ZERO,
                            along: Vec2::new(dir.dot(frame.x), dir.dot(frame.y)),
                            across: Vec2::new(1.0, 0.0),
                            affine: true,
                            turned: None,
                        });
                    }
                    Curve::Circle(c) if c.frame.z.cross(dir).norm() <= tol::ANGULAR => {
                        let turn = c.frame.z.dot(dir).signum();
                        let frame = Frame::new(c.frame.origin, dir, c.frame.x)?;
                        return Ok(Sheet {
                            surface: Surface::Cylinder(Cylinder {
                                frame,
                                radius: c.radius,
                            }),
                            span,
                            origin: Vec2::ZERO,
                            along: Vec2::new(0.0, 1.0),
                            across: Vec2::new(turn, 0.0),
                            affine: true,
                            turned: None,
                        });
                    }
                    _ => {}
                }
                Ok(Sheet {
                    surface: Surface::Extrusion(Box::new(Extrusion {
                        curve: curve.clone(),
                        dir,
                    })),
                    span,
                    origin: Vec2::ZERO,
                    along: Vec2::new(0.0, 1.0),
                    across: Vec2::new(1.0, 0.0),
                    affine: true,
                    turned: None,
                })
            }
            Steady::Turn { origin, axis, .. } => turned(curve, range, *origin, *axis, span),
            Steady::Screw { screw, angle, .. } => screwed(curve, range, screw, *angle),
        }
    }
}

/// What `curve` sweeps over its `range` carried through `angle` of `screw` from where it is: the
/// rational B-spline whose columns follow the helices of its control points, within `SCREW_FIT`
/// of them, `u` the turn and `v` the curve's rational parameter on `[0, 1]`.
fn screwed(curve: &Curve, range: (f64, f64), screw: &Helix, angle: f64) -> Result<Sheet> {
    let rail = Rail::of_curve(curve, range.0, range.1)?;
    let radius = rail
        .pw
        .iter()
        .map(|&h| {
            let q = point(h) - screw.origin;
            (q - screw.axis * q.dot(screw.axis)).norm()
        })
        .fold(0.0, f64::max);
    let nodes = helix::nodes(angle, radius);
    let cols: Vec<Vec<Point3>> = rail
        .pw
        .iter()
        .map(|&h| helix::column(screw, point(h), &nodes))
        .collect();
    let nu = cols[0].len();
    let nv = cols.len();
    let mut net = Vec::with_capacity(nu * nv);
    for row in 0..nu {
        net.extend(cols.iter().map(|c| c[row]));
    }
    let w: Vec<f64> = rail.pw.iter().map(|h| h[3]).collect();
    let rational = w.iter().any(|&x| (x - 1.0).abs() > 1e-15);
    let weights = rational.then(|| (0..nu).flat_map(|_| w.iter().copied()).collect());
    let surface = Surface::Nurbs(Box::new(NurbsSurface::new(
        (helix::DEGREE, rail.degree),
        (helix::knots(&nodes), rail.knots.clone()),
        (nu, nv),
        net,
        weights,
    )?));
    let width = range.1 - range.0;
    Ok(Sheet {
        surface,
        span: angle,
        origin: Vec2::new(0.0, -range.0 / width),
        along: Vec2::new(1.0, 0.0),
        across: Vec2::new(0.0, 1.0 / width),
        affine: matches!(curve, Curve::Line(_)),
        turned: None,
    })
}
