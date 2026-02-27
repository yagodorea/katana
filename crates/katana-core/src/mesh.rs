use nalgebra::{Point3, Vector3};
use std::fmt;

#[derive(Debug, Clone)]
pub enum MeshSource {
    StlBinary,
    StlAscii
}

impl fmt::Display for MeshSource {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MeshSource::StlBinary => write!(f, "StlBinary"),
            MeshSource::StlAscii => write!(f, "StlAscii"),
            
        }
    }
}

#[derive(Debug, Clone)]
pub struct Triangle {
    pub vertices: [Point3<f32>; 3],
    pub normal: Vector3<f32>,
}

// TODO: implement function to check if mesh is watertight
#[derive(Debug, Clone)]
pub struct Mesh {
    pub triangles: Vec<Triangle>,
    pub source: MeshSource,
}

impl Mesh {
    pub fn new(triangles: Vec<Triangle>, source: MeshSource) -> Self {
        Self { triangles, source }
    }

    /// Returns (min_corner, max_corner) of the axis-aligned bounding box.
    pub fn bounding_box(&self) -> (Point3<f32>, Point3<f32>) {
        let mut min = Point3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut max = Point3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);

        for tri in &self.triangles {
            for v in &tri.vertices {
                min.x = min.x.min(v.x);
                min.y = min.y.min(v.y);
                min.z = min.z.min(v.z);
                max.x = max.x.max(v.x);
                max.y = max.y.max(v.y);
                max.z = max.z.max(v.z);
            }
        }

        (min, max)
    }

    /// Compute the volume of a closed (watertight) mesh.
    ///
    /// Uses the divergence theorem: each triangle contributes a signed
    /// tetrahedron volume relative to the origin. The sum's absolute value
    /// is the mesh volume. Only meaningful for closed surfaces.
    pub fn volume(&self) -> f32 {
        let mut vol = 0.0f32;
        for tri in &self.triangles {
            let v0 = &tri.vertices[0].coords;
            let v1 = &tri.vertices[1].coords;
            let v2 = &tri.vertices[2].coords;
            vol += v0.dot(&v1.cross(v2));
        }
        (vol / 6.0).abs()
    }
}
