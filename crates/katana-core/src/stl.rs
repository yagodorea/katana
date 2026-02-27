use std::io::Cursor;

use byteorder::{LittleEndian, ReadBytesExt};
use nalgebra::{Point3, Vector3};

use crate::mesh::{Mesh, Triangle, MeshSource};

#[derive(Debug, thiserror::Error)]
pub enum StlError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid STL: {0}")]
    Invalid(String),
}

/// Load a mesh from STL data, auto-detecting binary vs ASCII format.
pub fn load_stl(data: &[u8]) -> Result<Mesh, StlError> {
    if is_ascii_stl(data) {
        parse_ascii(data)
    } else {
        parse_binary(data)
    }
}

/// Heuristic to distinguish ASCII from binary STL.
///
/// Binary STL: 80-byte header + 4-byte triangle count + 50 bytes per triangle.
/// ASCII STL: starts with "solid", contains "facet", "vertex", etc.
///
/// Tricky case: binary STL headers can start with "solid" too.
/// We check if the file size matches the expected binary size — if it does,
/// it's almost certainly binary regardless of header content.
fn is_ascii_stl(data: &[u8]) -> bool {
    if !data.starts_with(b"solid") {
        return false;
    }

    // If we have enough bytes for a binary header, check the size heuristic
    if data.len() >= 84 {
        let mut cursor = Cursor::new(&data[80..84]);
        if let Ok(count) = cursor.read_u32::<LittleEndian>() {
            let expected = 84 + count as usize * 50;
            if expected == data.len() {
                return false;
            }
        }
    }

    true
}

// ---------------------------------------------------------------------------
// Binary STL
// ---------------------------------------------------------------------------

fn parse_binary(data: &[u8]) -> Result<Mesh, StlError> {
    if data.len() < 84 {
        return Err(StlError::Invalid("File too small for binary STL".into()));
    }

    let mut cursor = Cursor::new(&data[80..]);
    let num_triangles = cursor.read_u32::<LittleEndian>()? as usize;

    let expected = 84 + num_triangles * 50;
    if data.len() < expected {
        return Err(StlError::Invalid(format!(
            "Expected {} bytes for {} triangles, got {}",
            expected,
            num_triangles,
            data.len()
        )));
    }

    let mut cursor = Cursor::new(&data[84..]);
    let mut triangles = Vec::with_capacity(num_triangles);

    for _ in 0..num_triangles {
        let normal = read_vector3(&mut cursor)?;
        let v0 = read_point3(&mut cursor)?;
        let v1 = read_point3(&mut cursor)?;
        let v2 = read_point3(&mut cursor)?;
        // attribute byte count — unused, skip
        cursor.read_u16::<LittleEndian>()?;

        triangles.push(Triangle {
            vertices: [v0, v1, v2],
            normal,
        });
    }

    Ok(Mesh::new(triangles, MeshSource::StlBinary))
}

fn read_vector3(cursor: &mut Cursor<&[u8]>) -> Result<Vector3<f32>, std::io::Error> {
    Ok(Vector3::new(
        cursor.read_f32::<LittleEndian>()?,
        cursor.read_f32::<LittleEndian>()?,
        cursor.read_f32::<LittleEndian>()?,
    ))
}

fn read_point3(cursor: &mut Cursor<&[u8]>) -> Result<Point3<f32>, std::io::Error> {
    Ok(Point3::new(
        cursor.read_f32::<LittleEndian>()?,
        cursor.read_f32::<LittleEndian>()?,
        cursor.read_f32::<LittleEndian>()?,
    ))
}

// ---------------------------------------------------------------------------
// ASCII STL
// ---------------------------------------------------------------------------

fn parse_ascii(data: &[u8]) -> Result<Mesh, StlError> {
    let text = std::str::from_utf8(data)
        .map_err(|e| StlError::Invalid(format!("Invalid UTF-8: {e}")))?;

    let mut triangles = Vec::new();
    let mut lines = text.lines().map(|l| l.trim());

    // Skip "solid <name>"
    lines.next();

    while let Some(line) = lines.next() {
        if line.starts_with("endsolid") || line.is_empty() {
            break;
        }
        if !line.starts_with("facet normal") {
            continue;
        }

        let normal = parse_floats::<3>(line, 2)
            .map(|f| Vector3::new(f[0], f[1], f[2]))?;

        // "outer loop"
        lines.next();

        let mut vertices = [Point3::origin(); 3];
        for v in &mut vertices {
            let vline = lines
                .next()
                .ok_or_else(|| StlError::Invalid("Unexpected end of file".into()))?;
            let coords = parse_floats::<3>(vline, 1)?;
            *v = Point3::new(coords[0], coords[1], coords[2]);
        }

        // "endloop"
        lines.next();
        // "endfacet"
        lines.next();

        triangles.push(Triangle { vertices, normal });
    }

    Ok(Mesh::new(triangles, MeshSource::StlAscii))
}

/// Parse N floats from a line, skipping `skip` whitespace-separated tokens first.
fn parse_floats<const N: usize>(line: &str, skip: usize) -> Result<[f32; N], StlError> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < skip + N {
        return Err(StlError::Invalid(format!(
            "Expected {} floats after {} tokens in: {line}",
            N, skip
        )));
    }
    let mut result = [0.0f32; N];
    for (i, val) in result.iter_mut().enumerate() {
        *val = parts[skip + i]
            .parse()
            .map_err(|_| StlError::Invalid(format!("Bad float in: {line}")))?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn stl_path(name: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../stls")
            .join(name)
    }

    #[test]
    fn parse_binary_cube() {
        let data = std::fs::read(stl_path("cube.stl")).unwrap();
        let mesh = load_stl(&data).expect("Failed to parse binary STL");

        assert_eq!(mesh.triangles.len(), 12);
        assert!(matches!(mesh.source, MeshSource::StlBinary));

        let (min, max) = mesh.bounding_box();
        assert_eq!(min, Point3::new(-0.5, -0.5, 0.0));
        assert_eq!(max, Point3::new(0.5, 0.5, 1.0));

        let vol = mesh.volume();
        assert!((vol - 1.0).abs() < 1e-6, "Expected volume 1.0, got {vol}");
    }

    #[test]
    fn parse_ascii_block() {
        let data = std::fs::read(stl_path("block100.stl")).unwrap();
        let mesh = load_stl(&data).expect("Failed to parse ASCII STL");

        assert_eq!(mesh.triangles.len(), 12);
        assert!(matches!(mesh.source, MeshSource::StlAscii));

        let (min, max) = mesh.bounding_box();
        assert_eq!(min, Point3::new(0.0, 0.0, 0.0));
        assert_eq!(max, Point3::new(100.0, 100.0, 100.0));

        let vol = mesh.volume();
        assert!((vol - 1_000_000.0).abs() < 1.0, "Expected volume 1000000, got {vol}");
    }

    #[test]
    fn sphere_volume_within_tolerance() {
        let data = std::fs::read(stl_path("sphere.stl")).unwrap();
        let mesh = load_stl(&data).expect("Failed to parse sphere STL");

        // Semi-axes from bounding box: ~19.411, ~19.644, ~20.0
        // Ellipsoid volume = (4/3)π × a × b × c ≈ 31,936
        // The mesh is only 228 triangles, so it under-approximates.
        // We allow 5% tolerance to account for the coarse tessellation.
        let expected = 4.0 / 3.0 * std::f32::consts::PI * 19.411 * 19.644 * 20.0;
        let vol = mesh.volume();
        let error = (vol - expected).abs() / expected;
        assert!(
            error < 0.05,
            "Volume {vol:.1} deviates {:.1}% from expected {expected:.1}",
            error * 100.0
        );
    }
}
