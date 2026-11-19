//! The source of the vendored Rust dumps beside this file (lean_o0.*, lean_o3.*),
//! which test/tools/gen_rust_fixtures.jl builds with rustc. It is the part of
//! test/rust_test/crate that needs nothing from `alloc`, so the crate's compile
//! unit stays small: no String, no Box, no iterator adaptors.

// --- scalars
pub fn add(a: i32, b: i32) -> i32 { a + b }
pub fn flag(b: bool) -> bool { !b }
pub fn char_code(c: char) -> u32 { c as u32 }
pub fn wide(x: u128) -> u128 { x.wrapping_mul(3) }

// --- a record the compiler reorders, by value and by reference
pub struct Rec { pub a: u8, pub b: f64, pub c: u16, pub d: u32 }
pub fn rec_make(a: u8, b: f64, c: u16, d: u32) -> Rec { Rec { a, b, c, d } }
pub fn rec_sum(r: Rec) -> f64 { r.a as f64 + r.b + r.c as f64 + r.d as f64 }
pub fn rec_sum_ref(r: &Rec) -> f64 { r.a as f64 + r.b + r.c as f64 + r.d as f64 }

// --- two scalars
pub struct F2 { pub x: f64, pub y: f64 }
pub struct Two16 { pub a: u16, pub b: u16 }
pub fn f2_make(x: f64, y: f64) -> F2 { F2 { x, y } }
pub fn f2_dot(a: F2, b: F2) -> f64 { a.x * b.x + a.y * b.y }
pub fn two16_swap(t: Two16) -> Two16 { Two16 { a: t.b, b: t.a } }
pub fn pair(a: i32, b: f64) -> (i32, f64) { (a * 2, b * 2.0) }

// --- a small aggregate that is not two scalars, and a zero-sized value
pub struct B3 { pub a: u8, pub b: u8, pub c: u8 }
pub fn b3_rot(x: B3) -> B3 { B3 { a: x.b, b: x.c, c: x.a } }
pub struct Z;
pub fn z(_z: Z, a: i32) -> Z { let _ = a; Z }

// --- fat pointers
pub fn slice_sum(xs: &[f64]) -> f64 {
    let mut s = 0.0;
    let mut i = 0;
    while i < xs.len() { s += xs[i]; i += 1; }
    s
}
pub fn str_len(s: &str) -> usize { s.len() }
pub fn hello() -> &'static str { "Hello, World!" }

// --- enums
pub enum Level { Low, Mid, High }
pub fn level_next(l: Level) -> Level {
    match l { Level::Low => Level::Mid, Level::Mid => Level::High, Level::High => Level::Low }
}
pub enum Shape { Circle(f64), Rect { w: f64, h: f64 }, Empty }
pub fn rect(w: f64, h: f64) -> Shape { Shape::Rect { w, h } }
pub fn unit_circle() -> Shape { Shape::Circle(1.0) }
pub fn area(s: &Shape) -> f64 {
    match s { Shape::Circle(r) => 3.0 * r * r, Shape::Rect { w, h } => w * h, Shape::Empty => 0.0 }
}
pub fn maybe(x: i32) -> Option<i32> { if x >= 0 { Some(x + 1) } else { None } }
pub fn opt_ref_d(r: Option<&Rec>) -> u32 { match r { Some(r) => r.d, None => 0 } }

// --- methods
pub struct Acc { total: f64, count: u32 }
impl Acc {
    pub fn new() -> Acc { Acc { total: 0.0, count: 0 } }
    pub fn push(&mut self, x: f64) { self.total += x; self.count += 1; }
    pub fn mean(&self) -> f64 { if self.count == 0 { 0.0 } else { self.total / self.count as f64 } }
}

// --- a trait, its impl, and a trait object
pub trait Area { fn area(&self) -> f64; }
impl Area for Shape { fn area(&self) -> f64 { area(self) } }
pub fn dyn_area(a: &dyn Area) -> f64 { a.area() }

// --- a panic, and a caller-location parameter DWARF does not list
pub fn checked_div(a: i32, b: i32) -> i32 { if b == 0 { panic!("division by zero in lean") } a / b }
#[track_caller]
pub fn located(x: i32) -> u32 { x as u32 + std::panic::Location::caller().line() }

// --- the crate's own C API, bound by unmangled names
#[repr(C)]
pub struct CPoint { pub x: f64, pub y: f64 }
#[no_mangle]
pub extern "C" fn c_add(a: i32, b: i32) -> i32 { add(a, b) }
#[no_mangle]
pub extern "C" fn cpoint_norm(p: CPoint) -> f64 { p.x * p.x + p.y * p.y }
