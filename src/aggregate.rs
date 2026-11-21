//! Path-ordered accumulation of input adjoints in caller-owned storage.
use std::{error::Error, fmt};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum AccumulationError {
    Shape,
    NonFinite { input: usize },
}

impl fmt::Display for AccumulationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape => f.write_str("gradient and total lengths differ"),
            Self::NonFinite { input } => write!(f, "nonfinite gradient sum at input {input}"),
        }
    }
}
impl Error for AccumulationError {}

/// Add one borrowed VJP to the running totals. Call in path order to
/// reproduce sequential f64 summation. A failed addition preserves all totals.
/// No per-path result storage or additional allocation is required.
pub fn accumulate_gradient(totals: &mut [f64], gradient: &[f64]) -> Result<(), AccumulationError> {
    if totals.len() != gradient.len() {
        return Err(AccumulationError::Shape);
    }
    let finite = totals
        .iter()
        .zip(gradient)
        .fold(true, |ok, (&t, &g)| ok & (t + g).is_finite());
    if !finite {
        let input = totals
            .iter()
            .zip(gradient)
            .position(|(&t, &g)| !(t + g).is_finite())
            .unwrap();
        return Err(AccumulationError::NonFinite { input });
    }
    for (total, &g) in totals.iter_mut().zip(gradient) {
        *total += g;
    }
    Ok(())
}
