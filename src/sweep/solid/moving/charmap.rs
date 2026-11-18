//! Where a face of the body grazes the motion at each τ, as a curve of one coordinate `x`: on a
//! plane, the line `A x + B y + C = 0` by the place along it from its point nearest the face's
//! middle; on a cylinder, cone or extrusion, whose rate is affine along `v`, the surface's first
//! parameter `u`, the curve's `v` at each `u` read from that closed form (doc/design.md, "Finding
//! the curves").

use super::Ctx;
use crate::{
    geom::Surface,
    math::{Point3, Vec2, Vec3},
};

#[derive(Clone, Copy, Debug)]
enum Kind {
    Line,
    Linear,
}

/// The grazing curve of face `face` over the span.
pub(super) struct CharMap<'c, 'a> {
    ctx: &'c Ctx<'a>,
    face: usize,
    kind: Kind,
    centre: Vec2,
}

impl<'c, 'a> CharMap<'c, 'a> {
    /// The grazing curve of face `f`, if its rate has a closed form this follows.
    pub(super) fn of(ctx: &'c Ctx<'a>, f: usize) -> Option<CharMap<'c, 'a>> {
        let face = &ctx.faces[f];
        let kind = match face.surface {
            Surface::Plane(_) => Kind::Line,
            Surface::Cylinder(_) | Surface::Cone(_) | Surface::Extrusion(_) => Kind::Linear,
            _ => return None,
        };
        Some(CharMap {
            ctx,
            face: f,
            kind,
            centre: (face.dom.lo + face.dom.hi) * 0.5,
        })
    }

    fn surface(&self) -> &Surface {
        &self.ctx.faces[self.face].surface
    }

    /// The rate the face moves outward at `uv` at `tau`.
    fn rate(&self, tau: f64, uv: Vec2) -> f64 {
        let s = self.surface();
        let p = s.eval(uv.x, uv.y);
        s.normal(uv.x, uv.y).dot(self.ctx.twist(tau).at(p)) * self.ctx.faces[self.face].out
    }

    /// A plane's line at `tau`: its point nearest the face's middle and its direction.
    fn line(&self, tau: f64) -> Option<(Vec2, Vec2)> {
        let c = self.rate(tau, Vec2::ZERO);
        let a = self.rate(tau, Vec2::new(1.0, 0.0)) - c;
        let b = self.rate(tau, Vec2::new(0.0, 1.0)) - c;
        let l = a.hypot(b);
        let g = a * self.centre.x + b * self.centre.y + c;
        if l <= 1e-12 * (g.abs() + c.abs()) || l <= 1e-300 {
            return None;
        }
        let m = Vec2::new(a / l, b / l);
        Some((self.centre - m * (g / l), m.perp()))
    }

    /// The curve's point at `x` at `tau`, in the face's parameters.
    pub(super) fn uv(&self, tau: f64, x: f64) -> Option<Vec2> {
        match self.kind {
            Kind::Line => {
                let (foot, dir) = self.line(tau)?;
                Some(foot + dir * x)
            }
            Kind::Linear => {
                let dom = &self.ctx.faces[self.face].dom;
                let (v0, v1) = (dom.lo.y, dom.hi.y);
                let g0 = self.rate(tau, Vec2::new(x, v0));
                let g1 = self.rate(tau, Vec2::new(x, v1));
                if (g1 - g0).abs() <= 1e-300 {
                    return None;
                }
                Some(Vec2::new(x, v0 - g0 * (v1 - v0) / (g1 - g0)))
            }
        }
    }

    /// The curve's point at `x` at `tau`, on the body.
    pub(super) fn point(&self, tau: f64, x: f64) -> Option<Point3> {
        let uv = self.uv(tau, x)?;
        Some(self.surface().eval(uv.x, uv.y))
    }

    /// The coordinate of the face's point `uv`, on the curve at `tau`.
    pub(super) fn coord(&self, tau: f64, uv: Vec2) -> Option<f64> {
        match self.kind {
            Kind::Line => {
                let (foot, dir) = self.line(tau)?;
                Some((uv - foot).dot(dir))
            }
            Kind::Linear => Some(uv.x),
        }
    }

    /// The range of the coordinate the face spans.
    pub(super) fn range(&self) -> (f64, f64) {
        match self.kind {
            Kind::Line => {
                let s = 2.0 * self.ctx.size.max(1.0);
                (-s, s)
            }
            Kind::Linear => {
                let dom = &self.ctx.faces[self.face].dom;
                (dom.lo.x, dom.hi.x)
            }
        }
    }

    /// The rate's change along the velocity at the curve's point `uv` at `tau`, less its change
    /// with `tau` there: the gradient of the rate along the way the curve's point on the body moves
    /// relative to the body, `V` plus the curve's own drift over the face (which takes up `−∂g/∂τ`
    /// as the rate stays zero on it). Positive where the swept face leaves the curve toward the
    /// side of the face the motion carries inward.
    pub(super) fn rise(&self, tau: f64, uv: Vec2) -> f64 {
        let dt = 1e-6 * (self.ctx.span.1 - self.ctx.span.0);
        let drift = (self.rate(tau + dt, uv) - self.rate(tau - dt, uv)) / (2.0 * dt);
        self.along_velocity(tau, uv) - drift
    }

    fn along_velocity(&self, tau: f64, uv: Vec2) -> f64 {
        let s = self.surface();
        let (p, su, sv) = s.d1(uv.x, uv.y);
        let vel = self.ctx.twist(tau).at(p);
        // The velocity in the face's parameters, by least squares on its two tangents.
        let (a, b, c) = (su.dot(su), su.dot(sv), sv.dot(sv));
        let (r, t) = (vel.dot(su), vel.dot(sv));
        let det = a * c - b * b;
        if det.abs() <= 1e-300 {
            return 0.0;
        }
        let du = (c * r - b * t) / det;
        let dv = (a * t - b * r) / det;
        let h = 1e-6;
        let gu = (self.rate(tau, uv + Vec2::new(h, 0.0)) - self.rate(tau, uv - Vec2::new(h, 0.0)))
            / (2.0 * h);
        let gv = (self.rate(tau, uv + Vec2::new(0.0, h)) - self.rate(tau, uv - Vec2::new(0.0, h)))
            / (2.0 * h);
        du * gu + dv * gv
    }

    /// The face's outward normal at `uv`.
    pub(super) fn normal(&self, uv: Vec2) -> Vec3 {
        self.surface().normal(uv.x, uv.y) * self.ctx.faces[self.face].out
    }
}
