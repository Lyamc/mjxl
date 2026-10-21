use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("failed to read JXL file: {0}")]
    ReadJxl(#[from] Box<dyn std::error::Error + Send + Sync>),

    #[error("no frames in image")]
    NoFrames,

    #[error("failed to write PNG: {0}")]
    WritePng(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("encode error: {0}")]
    Encode(String),
}

pub type Result<T> = std::result::Result<T, Error>;
