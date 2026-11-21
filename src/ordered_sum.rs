//! Reproduce a left-to-right f64 sum, using an integer reduction when exact.
pub(crate) fn ordered_sum(initial: f64, values: &[f64]) -> f64 {
    const HIDDEN: u64 = 1 << 52;
    const HEADER: u64 = 0xfff0_0000_0000_0000;
    let n = values.len();
    // A binary64 significand has 53 bits. At most 2^11 such unsigned
    // integers fit in a u64 sum. Small rows use the ordinary fold.
    if initial == 0.0 && (16..=2048).contains(&n) {
        let first = values[0].to_bits();
        let header = first & HEADER;
        let exponent = (header >> 52) & 0x7ff;
        // Matching sign and exponent place all terms on one binary grid.
        // If ceil(log2(n)) low bits are zero, every prefix fits in 53
        // significant bits, so the ordered additions are exact up to overflow.
        let mask = HEADER | (n.next_power_of_two() as u64 - 1);
        if exponent != 0 && exponent != 0x7ff && (first ^ header) & mask == 0 {
            let (total, different) = values.iter().fold((0u64, 0u64), |(total, diff), &x| {
                let bits = x.to_bits();
                (total.wrapping_add(bits), diff | (bits ^ header))
            });
            if different & mask == 0 {
                // Raw bit sums wrap modulo 2^64. Removing n headers and
                // restoring n implicit leading ones yields the exact integer.
                let mantissa = total.wrapping_sub((header - HIDDEN).wrapping_mul(n as u64));
                let scale = if exponent > 52 {
                    f64::from_bits((exponent - 52) << 52)
                } else {
                    f64::from_bits(1 << (exponent - 1))
                };
                return (mantissa as f64 * scale).copysign(values[0]);
            }
        }
    }
    values.iter().fold(initial, |s, x| s + x)
}

#[cfg(test)]
mod tests {
    use super::ordered_sum;

    fn check(initial: f64, values: &[f64]) {
        let expected = values.iter().fold(initial, |a, x| a + x);
        let actual = ordered_sum(initial, values);
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn integer_reductions_and_fallbacks_match_every_ordered_sum_bit() {
        let mut state = 1_u64;
        for n in [
            0usize, 1, 15, 16, 17, 63, 64, 65, 120, 1023, 1024, 1025, 2048, 2049,
        ] {
            for exponent in [0_u64, 1, 2, 51, 52, 53, 1022, 1023, 2037, 2045, 2046] {
                for sign in [0_u64, 1 << 63] {
                    let mut row: Vec<_> = (0..n)
                        .map(|_| {
                            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                            let significand =
                                (state >> 12) & !(n.max(1).next_power_of_two() as u64 - 1);
                            f64::from_bits(sign | exponent << 52 | significand)
                        })
                        .collect();
                    check(0.0, &row);
                    check(-0.0, &row);
                    if n < 16 {
                        continue;
                    }
                    for exception in [
                        f64::from_bits(row[n - 1].to_bits() | 1),
                        -row[0],
                        0.0,
                        -0.0,
                        1.0,
                        f64::INFINITY,
                        f64::NAN,
                    ] {
                        row[n - 1] = exception;
                        check(0.0, &row);
                        check(-0.125, &row);
                    }
                }
            }
        }
    }
}
