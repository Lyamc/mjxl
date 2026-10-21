#![recursion_limit = "256"]

//! Motion JPEG XL: independently coded JPEG XL stills in a timed sample stream.
//!
//! Modular and VarDCT stills come from `o3-jxl`. Dirty-rectangle search, 8×8
//! DCT/IDCT, and canvas paste can run as Vulkan (or DX12/Metal) compute.

pub mod compare;
pub mod container;
pub mod decode;
pub mod error;
pub mod gpu;
pub mod progressive;
pub mod still;
pub mod temporal;
pub mod y4m;

pub use error::{Error, Result};
pub use gpu::Backend;
