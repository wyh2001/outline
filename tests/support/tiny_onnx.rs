use std::io::Write;

use tempfile::NamedTempFile;

// Build the tiny ONNX model from source so tests do not depend on an opaque binary fixture.
const WIRE_VARINT: u8 = 0;
const WIRE_LENGTH_DELIMITED: u8 = 2;
const TENSOR_FLOAT: i32 = 1;
const ATTRIBUTE_TENSOR: i32 = 4;

/// Temporary-file fixture for [`tiny_matte_model_bytes`].
pub fn tiny_matte_model_file() -> NamedTempFile {
    let mut file = tempfile::Builder::new()
        .suffix(".onnx")
        .tempfile()
        .expect("failed to create temporary ONNX model file");
    let model = tiny_matte_model_bytes();
    file.write_all(&model)
        .expect("failed to write temporary ONNX model file");
    file.flush()
        .expect("failed to flush temporary ONNX model file");
    file
}

/// Encoded fixture for a constant-output ONNX matte model.
///
/// Input: RGB `[1, 3, 2, 2]`; output: matte `[1, 1, 2, 2]`.
pub fn tiny_matte_model_bytes() -> Vec<u8> {
    fn varint(mut value: u64, out: &mut Vec<u8>) {
        while value >= 0x80 {
            out.push((value as u8 & 0x7f) | 0x80);
            value >>= 7;
        }
        out.push(value as u8);
    }

    fn key(field: u32, wire_type: u8, out: &mut Vec<u8>) {
        varint(u64::from((field << 3) | u32::from(wire_type)), out);
    }

    fn int64(field: u32, value: i64, out: &mut Vec<u8>) {
        key(field, WIRE_VARINT, out);
        varint(value as u64, out);
    }

    fn int32(field: u32, value: i32, out: &mut Vec<u8>) {
        key(field, WIRE_VARINT, out);
        varint(value as u64, out);
    }

    fn bytes(field: u32, value: &[u8], out: &mut Vec<u8>) {
        key(field, WIRE_LENGTH_DELIMITED, out);
        varint(value.len() as u64, out);
        out.extend_from_slice(value);
    }

    fn string(field: u32, value: &str, out: &mut Vec<u8>) {
        bytes(field, value.as_bytes(), out);
    }

    fn message(field: u32, value: Vec<u8>, out: &mut Vec<u8>) {
        bytes(field, &value, out);
    }

    fn dimension(value: i64) -> Vec<u8> {
        let mut out = Vec::new();
        int64(1, value, &mut out);
        out
    }

    fn shape(dims: &[i64]) -> Vec<u8> {
        let mut out = Vec::new();
        for &dim in dims {
            message(1, dimension(dim), &mut out);
        }
        out
    }

    fn tensor_type(dims: &[i64]) -> Vec<u8> {
        let mut out = Vec::new();
        int32(1, TENSOR_FLOAT, &mut out);
        message(2, shape(dims), &mut out);
        out
    }

    fn type_proto(dims: &[i64]) -> Vec<u8> {
        let mut out = Vec::new();
        message(1, tensor_type(dims), &mut out);
        out
    }

    fn value_info(name: &str, dims: &[i64]) -> Vec<u8> {
        let mut out = Vec::new();
        string(1, name, &mut out);
        message(2, type_proto(dims), &mut out);
        out
    }

    fn matte_tensor() -> Vec<u8> {
        let mut out = Vec::new();
        for dim in [1, 1, 2, 2] {
            int64(1, dim, &mut out);
        }
        int32(2, TENSOR_FLOAT, &mut out);
        string(8, "matte_values", &mut out);

        let mut raw_data = Vec::new();
        for value in [0.0_f32, 0.25, 0.5, 1.0] {
            raw_data.extend_from_slice(&value.to_le_bytes());
        }
        bytes(9, &raw_data, &mut out);
        out
    }

    fn constant_attribute() -> Vec<u8> {
        let mut out = Vec::new();
        string(1, "value", &mut out);
        message(5, matte_tensor(), &mut out);
        int32(20, ATTRIBUTE_TENSOR, &mut out);
        out
    }

    fn constant_node() -> Vec<u8> {
        let mut out = Vec::new();
        string(2, "matte", &mut out);
        string(4, "Constant", &mut out);
        message(5, constant_attribute(), &mut out);
        out
    }

    fn graph() -> Vec<u8> {
        let mut out = Vec::new();
        message(1, constant_node(), &mut out);
        string(2, "tiny_matte", &mut out);
        message(11, value_info("input", &[1, 3, 2, 2]), &mut out);
        message(12, value_info("matte", &[1, 1, 2, 2]), &mut out);
        out
    }

    fn opset_import() -> Vec<u8> {
        let mut out = Vec::new();
        string(1, "", &mut out);
        int64(2, 13, &mut out);
        out
    }

    let mut out = Vec::new();
    int64(1, 8, &mut out);
    string(2, "outline-core-test", &mut out);
    message(7, graph(), &mut out);
    message(8, opset_import(), &mut out);
    out
}
