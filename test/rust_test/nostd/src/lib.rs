//! What a real library did the first time one was built (libm): it has no
//! `std`, and it turns its own warnings into errors.
//!
//! - `#![no_std]`: a library that is the end of a link needs a panic handler,
//!   so this crate cannot be built as the dynamic library itself. It is a
//!   dependency of the root crate RepliBuild generates, whose `std` has one.
//! - `#![deny(warnings)]`: the unused variable below is a warning, so a build
//!   that does not cap lints fails here. A newer compiler has newer warnings,
//!   and a crate written before it cannot have fixed them.
#![no_std]
#![deny(warnings)]

pub fn triple(x: i32) -> i32 {
    let unused = 7;
    x * 3
}

/// Two values back, in two registers.
pub fn split(x: f64) -> (f64, i32) {
    let whole = x as i32;
    (x - whole as f64, whole)
}

pub fn sum(xs: &[f64]) -> f64 {
    let mut s = 0.0;
    let mut i = 0;
    while i < xs.len() {
        s += xs[i];
        i += 1;
    }
    s
}
