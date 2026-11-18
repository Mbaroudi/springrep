//! Everything a solid passes through as it moves rigidly (doc/design.md, "Sweeps of solids"): the
//! body at the start, and what the region of its surface that the motion carries outward sweeps.
//! That region's boundary is its characteristic curves, where the velocity lies in the surface,
//! and the sharp edges between it and the rest; under a slide, a turn or a screw it is fixed on the
//! body, and the solid's boundary is that boundary swept, capped by the rest of the body at the
//! start and the region at the end.

mod cut;
mod grazing;
mod motion;
mod product;
mod sheet;
#[cfg(test)]
mod tests;

pub use motion::Motion;

use crate::{Error, Result, error::invalid, topo::Brep};

/// The solid `body` passes through as `motion` carries it, `τ` from 0 to 1: the union of
/// `M(τ)·body`. The faces the body's faces are cut from keep their names; each face a curve sweeps
/// is `feature.side`.
pub fn solid(body: &Brep, motion: &Motion, feature: &str) -> Result<Brep> {
    motion.validate()?;
    let mut solids = body.solid_ids();
    let (Some(s), None) = (solids.next(), solids.next()) else {
        return invalid("a sweep moves one solid");
    };
    if body.solid(s).shells.len() != 1 {
        return Err(Error::Unsupported("a sweep of a solid with a void".into()));
    }
    let (Some(steady), Some(twist)) = (sheet::Steady::of(motion), motion.steady()) else {
        return Err(Error::Unsupported(
            "a sweep along a motion whose velocity changes on the body".into(),
        ));
    };
    let field = move |p| twist.at(p);
    let cut = cut::cut(body, &field)?;
    product::product(&cut, &steady, feature)
}
