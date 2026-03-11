// test/rust_demo/src/lib.rs
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

#[repr(C)]
pub struct Particle {
    pub x: f32,
    pub y: f32,
    pub mass: f64,
}

#[repr(C)]
pub enum State {
    Inactive = -1,
    Active = 0,
    Destroyed = 1,
}

#[repr(u32)]
pub enum Flags {
    None = 0,
    One = 1,
    HighBit = 0x80000000,
}

// Internal function to create DWARF noise
fn internal_compute() -> Result<String, std::io::Error> {
    let mut v = Vec::new();
    v.push(1);
    v.push(2);
    Ok(format!("computed: {}", v.len()))
}

#[no_mangle]
pub extern "C" fn create_particle(x: f32, y: f32, mass: f64) -> Particle {
    let _ = internal_compute(); // cause some stdlib noise
    Particle { x, y, mass }
}

#[no_mangle]
pub extern "C" fn update_particle(p: *mut Particle, state: State) -> State {
    if !p.is_null() {
        unsafe {
            (*p).x += 1.0;
            (*p).y += 2.0;
        }
    }
    match state {
        State::Inactive => State::Active,
        _ => State::Destroyed,
    }
}

#[no_mangle]
pub extern "C" fn get_status_string(state: State) -> *mut c_char {
    let s = match state {
        State::Inactive => "inactive",
        State::Active => "active",
        State::Destroyed => "destroyed",
    };
    CString::new(s).unwrap().into_raw()
}

#[no_mangle]
pub extern "C" fn check_flags(flag: Flags) -> u32 {
    match flag {
        Flags::None => 0,
        Flags::One => 1,
        Flags::HighBit => 0x80000000,
    }
}

#[no_mangle]
pub extern "C" fn free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe { let _ = CString::from_raw(s); }
    }
}
