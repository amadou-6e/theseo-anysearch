use pyo3::{
    exceptions::{PyIOError, PyRuntimeError, PyValueError},
    PyErr,
};

pub(crate) fn runtime(message: impl Into<String>) -> PyErr {
    PyRuntimeError::new_err(message.into())
}

pub(crate) fn invalid_value(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

pub(crate) fn io(message: impl Into<String>) -> PyErr {
    PyIOError::new_err(message.into())
}
