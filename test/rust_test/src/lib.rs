// test/rust_test/src/lib.rs — Minimal Rust library for RepliBuild pipeline exploration
//
// Covers the key ABI patterns we need to handle:
//   1. Scalar args + returns (i32, f64, bool)
//   2. repr(C) structs (pass by value, return by value, pointers)
//   3. String interop (CStr, raw pointers)
//   4. Enums (C-compatible)
//   5. Arrays / slices via raw pointers
//   6. Opaque types behind pointers

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

// ── 1. Scalars ──────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[no_mangle]
pub extern "C" fn multiply_f64(a: f64, b: f64) -> f64 {
    a * b
}

#[no_mangle]
pub extern "C" fn is_positive(x: i32) -> bool {
    x > 0
}

// ── 2. repr(C) Structs ─────────────────────────────────────────────────

#[repr(C)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

#[repr(C)]
pub struct Rect {
    pub origin: Point,
    pub size: Point,
}

#[no_mangle]
pub extern "C" fn point_new(x: f64, y: f64) -> Point {
    Point { x, y }
}

#[no_mangle]
pub extern "C" fn point_distance(a: &Point, b: &Point) -> f64 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
}

#[no_mangle]
pub extern "C" fn rect_area(r: &Rect) -> f64 {
    r.size.x * r.size.y
}

// ── 3. String Interop ──────────────────────────────────────────────────

/// Takes a C string, returns its length
#[no_mangle]
pub extern "C" fn string_length(s: *const c_char) -> i32 {
    if s.is_null() {
        return -1;
    }
    unsafe { CStr::from_ptr(s).to_bytes().len() as i32 }
}

/// Greet: returns a heap-allocated C string (caller must free with free_string)
#[no_mangle]
pub extern "C" fn greet(name: *const c_char) -> *mut c_char {
    let name_str = if name.is_null() {
        "world"
    } else {
        unsafe { CStr::from_ptr(name).to_str().unwrap_or("world") }
    };
    let greeting = format!("hello {}", name_str);
    CString::new(greeting).unwrap().into_raw()
}

/// Free a string allocated by greet()
#[no_mangle]
pub extern "C" fn free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe { let _ = CString::from_raw(s); }
    }
}

// ── 4. C-Compatible Enum ────────────────────────────────────────────────

#[repr(C)]
pub enum Color {
    Red = 0,
    Green = 1,
    Blue = 2,
}

#[no_mangle]
pub extern "C" fn color_name(c: Color) -> *const c_char {
    match c {
        Color::Red => c"red".as_ptr(),
        Color::Green => c"green".as_ptr(),
        Color::Blue => c"blue".as_ptr(),
    }
}

// ── 5. Array via raw pointer ────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn sum_array(data: *const i32, len: usize) -> i64 {
    if data.is_null() || len == 0 {
        return 0;
    }
    let slice = unsafe { std::slice::from_raw_parts(data, len) };
    slice.iter().map(|&x| x as i64).sum()
}

// ── 6. Opaque type behind pointer ───────────────────────────────────────

pub struct Counter {
    value: i64,
}

#[no_mangle]
pub extern "C" fn counter_new(initial: i64) -> *mut Counter {
    Box::into_raw(Box::new(Counter { value: initial }))
}

#[no_mangle]
pub extern "C" fn counter_increment(c: *mut Counter) {
    if !c.is_null() {
        unsafe { (*c).value += 1; }
    }
}

#[no_mangle]
pub extern "C" fn counter_get(c: *const Counter) -> i64 {
    if c.is_null() {
        return 0;
    }
    unsafe { (*c).value }
}

#[no_mangle]
pub extern "C" fn counter_free(c: *mut Counter) {
    if !c.is_null() {
        unsafe { let _ = Box::from_raw(c); }
    }
}
