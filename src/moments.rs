//! Shared pass over inputs for an ordered sum and sum of squares.
use crate::AadError;
use crate::blocks::BlockValue;
use std::ops::Range;

pub(super) struct Moments {
    pub inputs: Range<usize>,
    pub first_node: usize,
    pub sum_initial: usize,
    pub square_initial: usize,
    pub sum_output: usize,
    pub square_output: usize,
}

impl Moments {
    #[cfg(feature = "simd")]
    pub(super) fn compact(&self, slots: &[usize]) -> Self {
        Self {
            inputs: self.inputs.clone(),
            first_node: self.first_node,
            sum_initial: slots[self.sum_initial],
            square_initial: slots[self.square_initial],
            sum_output: slots[self.sum_output],
            square_output: slots[self.square_output],
        }
    }
    pub(super) fn forward<T: BlockValue>(&self, inputs: &[T::Stored], p: &mut [T]) -> bool {
        let (mut sum, mut squares) = (p[self.sum_initial], p[self.square_initial]);
        for &raw in &inputs[self.inputs.clone()] {
            let x = T::from(raw);
            sum = sum + x;
            squares = squares + x * x;
        }
        p[self.sum_output] = sum;
        p[self.square_output] = squares;
        sum.finite() && squares.finite()
    }
    pub(super) fn reverse<T: BlockValue>(
        &self,
        inputs: &[T::Stored],
        adj: &mut [T],
        gradients: &mut [T::Stored],
        sum_node: usize,
        square_node: usize,
    ) -> Result<(), AadError> {
        let square_seed = adj[self.square_output];
        if !square_seed.finite() {
            return Err(AadError::NonFiniteResult { node: square_node });
        }
        adj[self.square_initial] = adj[self.square_initial] + square_seed;
        let sum_seed = adj[self.sum_output];
        if !sum_seed.finite() {
            return Err(AadError::NonFiniteResult { node: sum_node });
        }
        let gradients = &mut gradients[self.inputs.clone()];
        let mut finite = true;
        for (gradient, &raw) in gradients.iter_mut().zip(&inputs[self.inputs.clone()]) {
            let product = square_seed * T::from(raw);
            let result = ((T::ZERO + product) + product) + sum_seed;
            *gradient = result.into();
            finite &= result.finite();
        }
        if !finite {
            let index = gradients
                .iter()
                .rposition(|&g| !T::from(g).finite())
                .unwrap();
            return Err(AadError::NonFiniteResult {
                node: self.first_node + index,
            });
        }
        adj[self.sum_initial] = adj[self.sum_initial] + sum_seed;
        Ok(())
    }
}
