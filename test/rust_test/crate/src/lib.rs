//! The Rust front end's fixture crate (test/test_rust_build_live.jl, and the
//! vendored dumps under test/fixtures/rust/, which test/tools/gen_rust_fixtures.jl
//! builds from this file). Each item is here for one shape of call or layout.

use std::sync::atomic::{AtomicU64, Ordering};

// --- scalars -----------------------------------------------------------------
pub fn add(a: i32, b: i32) -> i32 { a + b }
pub fn flag(b: bool) -> bool { !b }
pub fn char_code(c: char) -> u32 { c as u32 }
pub fn wide(x: u128) -> u128 { x.wrapping_mul(3) }

// --- a record the compiler reorders, by value and by reference ------------------
pub struct Rec { pub a: u8, pub b: f64, pub c: u16, pub d: u32 }
pub fn rec_make(a: u8, b: f64, c: u16, d: u32) -> Rec { Rec { a, b, c, d } }
pub fn rec_sum(r: Rec) -> f64 { r.a as f64 + r.b + r.c as f64 + r.d as f64 }
pub fn rec_sum_ref(r: &Rec) -> f64 { r.a as f64 + r.b + r.c as f64 + r.d as f64 }

// --- two scalars: passed as two arguments, returned as a pair --------------------
pub struct F2 { pub x: f64, pub y: f64 }
pub struct Two16 { pub a: u16, pub b: u16 }
pub fn f2_make(x: f64, y: f64) -> F2 { F2 { x, y } }
pub fn f2_dot(a: F2, b: F2) -> f64 { a.x * b.x + a.y * b.y }
pub fn two16_swap(t: Two16) -> Two16 { Two16 { a: t.b, b: t.a } }
pub fn pair(a: i32, b: f64) -> (i32, f64) { (a * 2, b * 2.0) }

// --- a small aggregate that is not two scalars: one integer of its size ----------
pub struct B3 { pub a: u8, pub b: u8, pub c: u8 }
pub fn b3_rot(x: B3) -> B3 { B3 { a: x.b, b: x.c, c: x.a } }

// --- a fixed-size array is its elements, one after another; a slice of records ----
pub fn bytes16(seed: u8) -> [u8; 16] { core::array::from_fn(|i| seed.wrapping_add(i as u8)) }
pub fn bytes16_sum(b: [u8; 16]) -> u32 { b.iter().map(|&x| x as u32).sum() }
pub fn grid(k: f32) -> [[f32; 2]; 3] { [[k, k + 1.0], [k + 2.0, k + 3.0], [k + 4.0, k + 5.0]] }
pub fn opt_bytes(some: bool) -> Option<[u8; 6]> { if some { Some([1, 2, 3, 4, 5, 6]) } else { None } }
pub fn recs_total(rs: &[Rec]) -> f64 { rs.iter().map(|r| r.b + r.d as f64).sum() }

// --- a zero-sized value is not passed at all ------------------------------------
pub struct Z;
pub fn z(_z: Z, a: i32) -> Z { let _ = a; Z }

// --- fat pointers ----------------------------------------------------------------
pub fn slice_sum(xs: &[f64]) -> f64 { xs.iter().sum() }
pub fn str_len(s: &str) -> usize { s.chars().count() }
pub fn hello() -> &'static str { "Hello, World!" }

// --- enums -----------------------------------------------------------------------
pub enum Level { Low, Mid, High }
pub fn level_next(l: Level) -> Level {
    match l { Level::Low => Level::Mid, Level::Mid => Level::High, Level::High => Level::Low }
}
// The standard library's own enum is a signed byte: `Less` is -1.
pub fn compare(a: i32, b: i32) -> std::cmp::Ordering { a.cmp(&b) }
pub fn ordering_sign(o: std::cmp::Ordering) -> i32 { o as i32 }
pub enum Shape { Circle(f64), Rect { w: f64, h: f64 }, Empty }
pub fn rect(w: f64, h: f64) -> Shape { Shape::Rect { w, h } }
pub fn unit_circle() -> Shape { Shape::Circle(1.0) }
pub fn area(s: &Shape) -> f64 {
    match s { Shape::Circle(r) => 3.0 * r * r, Shape::Rect { w, h } => w * h, Shape::Empty => 0.0 }
}
// A niche: `Empty` is the null pointer, and the enum is one word.
pub enum Slot { Empty, Full(Box<Rec>) }
pub fn slot_make(d: u32) -> Slot {
    if d == 0 { Slot::Empty } else { Slot::Full(Box::new(rec_make(1, 2.0, 3, d))) }
}
pub fn slot_d(s: &Slot) -> u32 { match s { Slot::Empty => 0, Slot::Full(r) => r.d } }
pub fn maybe(x: i32) -> Option<i32> { if x >= 0 { Some(x + 1) } else { None } }
pub fn opt_ref_d(r: Option<&Rec>) -> u32 { r.map(|r| r.d).unwrap_or(0) }

// --- one byte that is only ever 0 or 1 is one bit in the call ----------------------
pub enum Side { Left, Right }
pub fn flip(s: Side) -> Side { match s { Side::Left => Side::Right, Side::Right => Side::Left } }
pub fn check(x: i32) -> Result<(), ()> { if x >= 0 { Ok(()) } else { Err(()) } }
pub struct Opts { pub scale: f64, pub on: bool }
pub fn opts(scale: f64) -> Opts { Opts { scale, on: scale > 0.0 } }

// --- owned values: the callee or the caller must drop them -------------------------
pub fn make_greeting(name: &str) -> String { format!("hello {name}") }
pub fn string_len_owned(s: String) -> usize { s.len() }
pub fn boxed(d: u32) -> Box<Rec> { Box::new(rec_make(1, 2.0, 3, d)) }
pub fn unbox_d(b: Box<Rec>) -> u32 { b.d }

// --- owned values the caller holds: each needs Rust's own drop code -----------------
pub fn squares(n: u32) -> Vec<f64> { (0..n).map(|i| (i * i) as f64).collect() }
pub fn vec_sum(v: Vec<f64>) -> f64 { v.iter().sum() }
pub fn vec_push(v: &mut Vec<f64>, x: f64) { v.push(x) }
pub fn recs(n: u32) -> Vec<Rec> { (0..n).map(|i| rec_make(i as u8, i as f64, 3, i)).collect() }
pub fn words(s: &str) -> Vec<String> { s.split_whitespace().map(String::from).collect() }
pub fn join_words(ws: Vec<String>) -> String { ws.join(" ") }
pub fn shout(s: &mut String) { s.make_ascii_uppercase(); s.push('!'); }
pub fn string_ref_len(s: &String) -> usize { s.len() }
pub fn parse_int(s: &str) -> Result<i32, String> {
    s.trim().parse::<i32>().map_err(|e| format!("not a number: {e}"))
}
pub fn find_word(s: &str, k: usize) -> Option<String> { s.split_whitespace().nth(k).map(String::from) }
pub fn name_and_len(s: &str) -> (String, usize) { (s.to_uppercase(), s.len()) }
pub fn opt_box(d: u32) -> Option<Box<Rec>> { if d == 0 { None } else { Some(boxed(d)) } }
pub fn box_d(b: &Rec) -> u32 { b.d }
pub fn slot_take(s: Slot) -> u32 { match s { Slot::Empty => 0, Slot::Full(r) => r.d } }
pub struct Named { pub name: String, pub id: u32 }
pub fn named(name: &str, id: u32) -> Named { Named { name: name.to_string(), id } }
pub fn named_len(n: &Named) -> usize { n.name.len() }
pub fn named_into_name(n: Named) -> String { n.name }

// A type defined in a private module and exported under another path.
mod inner {
    pub struct Token { pub id: u32, pub text: String }
}
pub use inner::Token;
pub fn token(id: u32, text: &str) -> Token { Token { id, text: text.to_string() } }
pub fn token_id(t: &Token) -> u32 { t.id }

// A type that holds no memory and still must be dropped, exactly once.
pub static DROPS: AtomicU64 = AtomicU64::new(0);
pub struct Guard { pub id: u32 }
impl Drop for Guard { fn drop(&mut self) { DROPS.fetch_add(1, Ordering::SeqCst); } }
pub fn guard(id: u32) -> Guard { Guard { id } }
pub fn guard_id(g: &Guard) -> u32 { g.id }
pub fn guard_release(g: Guard) -> u32 { g.id }
pub fn drops() -> u64 { DROPS.load(Ordering::SeqCst) }

// A raw pointer in a record owns nothing by itself: the compiler says so.
pub struct View { pub p: *const f64, pub n: usize }
pub fn view(xs: &[f64]) -> View { View { p: xs.as_ptr(), n: xs.len() } }
pub fn view_len(v: View) -> usize { v.n }

// A type no path outside the crate reaches: nothing can name it to drop it.
mod hidden {
    pub struct Secret { pub v: Vec<u8> }
}
pub fn secret() -> hidden::Secret { hidden::Secret { v: vec![1, 2, 3] } }
pub fn secret_len(s: &hidden::Secret) -> usize { s.v.len() }

// --- methods: `self` is an ordinary first parameter --------------------------------
pub struct Acc { total: f64, count: u32 }
impl Acc {
    pub fn new() -> Acc { Acc { total: 0.0, count: 0 } }
    pub fn push(&mut self, x: f64) { self.total += x; self.count += 1; }
    pub fn mean(&self) -> f64 { if self.count == 0 { 0.0 } else { self.total / self.count as f64 } }
}

// --- a trait, its impl, and trait objects -------------------------------------------
pub trait Area { fn area(&self) -> f64; }
impl Area for Shape { fn area(&self) -> f64 { area(self) } }
pub fn dyn_area(a: &dyn Area) -> f64 { a.area() }
pub fn make_area(k: u32) -> Box<dyn Area> {
    if k == 0 { Box::new(unit_circle()) } else { Box::new(rect(2.0, 3.0)) }
}

// --- a generic exists only where it is instantiated --------------------------------
pub fn largest<T: PartialOrd + Copy>(xs: &[T]) -> T {
    let mut m = xs[0];
    for &x in xs { if x > m { m = x; } }
    m
}
pub fn largest_f64(xs: &[f64]) -> f64 { largest(xs) }

// --- a panic, a static, and a caller-location parameter DWARF does not list ----------
pub fn checked_div(a: i32, b: i32) -> i32 { if b == 0 { panic!("division by zero in rb_fix") } a / b }
pub static COUNTER: AtomicU64 = AtomicU64::new(0);
pub fn count() -> u64 { COUNTER.fetch_add(1, Ordering::SeqCst) }
#[track_caller]
pub fn located(x: i32) -> u32 { x as u32 + std::panic::Location::caller().line() }

// --- the crate's own C API, bound by unmangled names ---------------------------------
#[repr(C)]
pub struct CPoint { pub x: f64, pub y: f64 }
#[no_mangle]
pub extern "C" fn c_add(a: i32, b: i32) -> i32 { add(a, b) }
#[no_mangle]
pub extern "C" fn cpoint_norm(p: CPoint) -> f64 { (p.x * p.x + p.y * p.y).sqrt() }
