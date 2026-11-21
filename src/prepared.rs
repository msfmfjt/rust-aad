//! Immutable, path-major input storage for repeated evaluation.
use crate::{AadError, CompiledGraph, SumChain, Workspace};

enum Storage {
    Full(Vec<f64>),
    Grid { words: Vec<i16>, unit: f64 },
}

/// Owns input rows and losslessly packs values lying on a common binary grid.
/// Packing stores only inputs. The same rows can be used with different graphs and seeds.
///
/// ```
/// use rust_aad_core::{GraphBuilder, PreparedInputs};
///
/// let mut builder = GraphBuilder::new();
/// let x = builder.input();
/// let square = builder.mul(x, x)?;
/// let graph = builder.compile(&[square])?;
/// let inputs = PreparedInputs::new(2, 1, vec![1.25, -2.5])?;
/// let mut worker = inputs.replay(&graph)?;
/// assert_eq!(worker.vjp(1, &[0.5])?, &[-2.5]);
/// assert_eq!(worker.outputs(), &[6.25]);
/// # Ok::<(), rust_aad_core::AadError>(())
/// ```
pub struct PreparedInputs {
    paths: usize,
    width: usize,
    storage: Storage,
}

impl PreparedInputs {
    /// Takes ownership of a path-major matrix. Values that cannot be represented
    /// exactly, including negative zero and nonfinite values, retain their f64 bits.
    /// Nonfinite inputs are reported when their row is evaluated.
    pub fn new(paths: usize, width: usize, inputs: Vec<f64>) -> Result<Self, AadError> {
        if paths.checked_mul(width) != Some(inputs.len()) {
            return Err(AadError::InputShape);
        }
        let storage = match pack(&inputs) {
            Some((words, unit)) => Storage::Grid { words, unit },
            None => Storage::Full(inputs),
        };
        Ok(Self {
            paths,
            width,
            storage,
        })
    }

    /// Number of bytes used by the stored input values, excluding capacity and metadata.
    pub fn storage_bytes(&self) -> usize {
        match &self.storage {
            Storage::Full(xs) => std::mem::size_of_val(xs.as_slice()),
            Storage::Grid { words, .. } => std::mem::size_of_val(words.as_slice()),
        }
    }

    /// Binds a compiled graph to these inputs and creates per-worker scratch space.
    pub fn replay<'a>(&'a self, graph: &'a CompiledGraph) -> Result<InputReplay<'a>, AadError> {
        if self.width != graph.input_count() {
            return Err(AadError::InputShape);
        }
        let grid_sum = match &self.storage {
            Storage::Grid { .. } => graph
                .straight_line
                .as_ref()
                .and_then(|s| s.sum_chain.as_ref())
                .filter(|c| c.accepts_integer_sum()),
            Storage::Full(_) => None,
        };
        let scratch = match self.storage {
            Storage::Grid { .. } => vec![0.0; self.width],
            Storage::Full(_) => Vec::new(),
        };
        Ok(InputReplay {
            inputs: self,
            graph,
            grid_sum,
            work: graph.workspace(),
            scratch,
        })
    }
}

fn pack(xs: &[f64]) -> Option<(Vec<i16>, f64)> {
    // A magnitude maximum also detects infinities and all NaN payloads.
    let largest = xs.iter().map(|x| x.to_bits() & (u64::MAX >> 1)).max()?;
    let exponent = (largest >> 52) as i32 - 1023;
    // Both the quantum and reciprocal are normal. The largest possible sum of
    // 65,536 coefficients (the graph's node limit) times the quantum stays finite.
    let shift = if largest == 0 { 0 } else { exponent - 14 };
    if !(-990..=990).contains(&shift) {
        return None;
    }
    let unit = f64::from_bits(((1023 + shift) as u64) << 52);
    let inv = f64::from_bits(((1023 - shift) as u64) << 52);
    // Check a block before appending it. Accumulating bit differences lets the
    // compiler convert and validate independent elements in vector lanes.
    let mut words = Vec::new();
    let mut encoded = [0i16; 64];
    for block in xs.chunks(encoded.len()) {
        let mut different = 0;
        for (dest, &x) in encoded.iter_mut().zip(block) {
            *dest = (x * inv) as i16;
            different |= (f64::from(*dest) * unit).to_bits() ^ x.to_bits();
        }
        if different != 0 {
            return None;
        }
        if words.is_empty() {
            words.reserve_exact(xs.len());
        }
        words.extend_from_slice(&encoded[..block.len()]);
    }
    Some((words, unit))
}

/// A per-worker replay over immutable inputs. Evaluation returns a borrowed
/// output or gradient vector and allocates no additional memory.
pub struct InputReplay<'a> {
    inputs: &'a PreparedInputs,
    graph: &'a CompiledGraph,
    grid_sum: Option<&'a SumChain>,
    work: Workspace,
    scratch: Vec<f64>,
}

// Graphs contain at most 2^16 inputs. Every prefix fits in i32, so all
// ordered additions on this grid are exactly representable in binary64.
#[inline]
fn grid_total(words: &[i16], unit: f64) -> f64 {
    f64::from(words.iter().map(|&v| i32::from(v)).sum::<i32>()) * unit
}

impl InputReplay<'_> {
    /// Primal outputs from the most recent successful call to `evaluate` or
    /// `vjp`. Only consume these values after the call returns `Ok`.
    pub fn outputs(&self) -> &[f64] {
        &self.work.output
    }

    fn range(&self, path: usize) -> Result<std::ops::Range<usize>, AadError> {
        if path >= self.inputs.paths {
            return Err(AadError::InputShape);
        }
        let start = path * self.inputs.width;
        Ok(start..start + self.inputs.width)
    }

    /// Evaluate the outputs for a zero-based path index.
    pub fn evaluate(&mut self, path: usize) -> Result<&[f64], AadError> {
        let range = self.range(path)?;
        let row = match &self.inputs.storage {
            Storage::Full(xs) => &xs[range],
            Storage::Grid { words, unit } => {
                let words = &words[range];
                if let Some((value, _)) = self
                    .grid_sum
                    .and_then(|c| c.apply_sum(grid_total(words, *unit)))
                {
                    self.work.output[0] = value;
                    return Ok(&self.work.output);
                }
                expand(words, *unit, &mut self.scratch)
            }
        };
        self.graph.evaluate(row, &mut self.work)
    }

    /// Evaluate a path and compute its seeded vector-Jacobian product.
    pub fn vjp(&mut self, path: usize, seeds: &[f64]) -> Result<&[f64], AadError> {
        if seeds.len() != self.graph.output_count() {
            return Err(AadError::SeedShape);
        }
        let range = self.range(path)?;
        let row = match &self.inputs.storage {
            Storage::Full(xs) => &xs[range],
            Storage::Grid { words, unit } => {
                let words = &words[range];
                if let Some((value, grad)) = self
                    .grid_sum
                    .and_then(|c| c.vjp_from_sum(grid_total(words, *unit), seeds[0]))
                {
                    self.work.output[0] = value;
                    return Ok(self.work.uniform_adjoints(grad));
                }
                expand(words, *unit, &mut self.scratch)
            }
        };
        self.graph.vjp(row, seeds, &mut self.work)
    }
}

fn expand<'a>(words: &[i16], unit: f64, scratch: &'a mut [f64]) -> &'a [f64] {
    for (x, &word) in scratch.iter_mut().zip(words) {
        *x = f64::from(word) * unit;
    }
    scratch
}
