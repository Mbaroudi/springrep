//! What a curve sweeps under a steady motion, exactly where the motion has a closed form: an
//! extrusion under a slide, a revolution under a turn, and under a screw the rational B-spline whose
//! columns follow its control points' helices.

use super::motion::Motion;
use crate::{
    Error, Result,
    geom::{
        Circle, Curve, Cylinder, Extrusion, Line, NurbsCurve, NurbsSurface, Plane, Revolution,
        Surface,
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
/// `span`, is at `origin + s·along + t·across` in the surface's parameters, exactly where `affine`
/// and at the curve's ends otherwise.
pub(super) struct Sheet {
    pub surface: Surface,
    pub span: f64,
    pub origin: Vec2,
    pub along: Vec2,
    pub across: Vec2,
    pub affine: bool,
}

impl Sheet {
    pub(super) fn uv(&self, s: f64, t: f64) -> Vec2 {
        self.origin + self.along * s + self.across * t
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
                })
            }
            Steady::Turn { origin, axis, .. } => Ok(Sheet {
                surface: Surface::Revolution(Box::new(Revolution {
                    curve: curve.clone(),
                    origin: *origin,
                    axis: *axis,
                })),
                span,
                origin: Vec2::ZERO,
                along: Vec2::new(1.0, 0.0),
                across: Vec2::new(0.0, 1.0),
                affine: true,
            }),
            Steady::Screw { screw, angle, .. } => {
                let rel = *screw;
                let rail = Rail::of_curve(curve, range.0, range.1)?;
                let radius = rail
                    .pw
                    .iter()
                    .map(|&h| {
                        let q = point(h) - rel.origin;
                        (q - rel.axis * q.dot(rel.axis)).norm()
                    })
                    .fold(0.0, f64::max);
                let nodes = helix::nodes(*angle, radius);
                let cols: Vec<Vec<Point3>> = rail
                    .pw
                    .iter()
                    .map(|&h| helix::column(&rel, point(h), &nodes))
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
                    span,
                    origin: Vec2::new(0.0, -range.0 / width),
                    along: Vec2::new(1.0, 0.0),
                    across: Vec2::new(0.0, 1.0 / width),
                    affine: matches!(curve, Curve::Line(_)),
                })
            }
        }
    }
}
