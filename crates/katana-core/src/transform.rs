use nalgebra::{ Matrix4, Point3, UnitQuaternion, Vector3 };
use rayon::prelude::*;

use crate::mesh::Triangle;

/// Compose a model matrix: translate * (rotate * scale about `pivot`).
pub fn compose(
    translation: Vector3<f32>,
    rotation: UnitQuaternion<f32>,
    scale: Vector3<f32>,
    pivot: Point3<f32>
) -> Matrix4<f32> {
    Matrix4::new_translation(&translation) *
        Matrix4::new_translation(&pivot.coords) *
        rotation.to_homogeneous() *
        Matrix4::new_nonuniform_scaling(&scale) *
        Matrix4::new_translation(&(-pivot.coords))
}

/// Apply `m` to every triangle, recomputing normals from the transformed
/// vertices. Recomputing (rather than rotating the stored normal) stays
/// correct under non-uniform scale; winding (and therefore orientation)
/// is preserved as long as the scale components are positive.
pub fn transform_triangles(triangles: &[Triangle], m: &Matrix4<f32>) -> Vec<Triangle> {
    triangles
        .par_iter()
        .map(|tri| {
            let v0 = m.transform_point(&tri.vertices[0]);
            let v1 = m.transform_point(&tri.vertices[1]);
            let v2 = m.transform_point(&tri.vertices[2]);
            let n = (v1 - v0).cross(&(v2 - v0));
            let len = n.norm();
            // Degenerate triangles keep their old normal rather than NaN.
            let normal = if len > 1e-12 { n / len } else { tri.normal };
            Triangle { vertices: [v0, v1, v2], normal }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::bounding_box_of;
    use std::f32::consts::FRAC_PI_2;

    /// Unit cube [0,1]^3 as 12 triangles with outward normals and CCW winding.
    fn unit_cube() -> Vec<Triangle> {
        let p = |x: f32, y: f32, z: f32| Point3::new(x, y, z);
        let quads: [([Point3<f32>; 4], Vector3<f32>); 6] = [
            // -Z (bottom), viewed from below
            ([p(0., 0., 0.), p(0., 1., 0.), p(1., 1., 0.), p(1., 0., 0.)], -Vector3::z()),
            // +Z (top)
            ([p(0., 0., 1.), p(1., 0., 1.), p(1., 1., 1.), p(0., 1., 1.)], Vector3::z()),
            // -Y (front)
            ([p(0., 0., 0.), p(1., 0., 0.), p(1., 0., 1.), p(0., 0., 1.)], -Vector3::y()),
            // +Y (back)
            ([p(0., 1., 0.), p(0., 1., 1.), p(1., 1., 1.), p(1., 1., 0.)], Vector3::y()),
            // -X (left)
            ([p(0., 0., 0.), p(0., 0., 1.), p(0., 1., 1.), p(0., 1., 0.)], -Vector3::x()),
            // +X (right)
            ([p(1., 0., 0.), p(1., 1., 0.), p(1., 1., 1.), p(1., 0., 1.)], Vector3::x()),
        ];
        let mut tris = Vec::with_capacity(12);
        for (q, n) in quads {
            tris.push(Triangle { vertices: [q[0], q[1], q[2]], normal: n });
            tris.push(Triangle { vertices: [q[0], q[2], q[3]], normal: n });
        }
        tris
    }

    fn identity_quat() -> UnitQuaternion<f32> {
        UnitQuaternion::identity()
    }

    #[test]
    fn identity_leaves_vertices_unchanged() {
        let cube = unit_cube();
        let m = compose(
            Vector3::zeros(),
            identity_quat(),
            Vector3::new(1.0, 1.0, 1.0),
            Point3::origin()
        );
        let out = transform_triangles(&cube, &m);
        for (a, b) in cube.iter().zip(&out) {
            for (va, vb) in a.vertices.iter().zip(&b.vertices) {
                assert!((va - vb).norm() < 1e-6);
            }
            assert!((a.normal - b.normal).norm() < 1e-6);
        }
    }

    #[test]
    fn rotation_90_about_x_maps_aabb() {
        let cube = unit_cube();
        let rot = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), FRAC_PI_2);
        let m = compose(Vector3::zeros(), rot, Vector3::new(1.0, 1.0, 1.0), Point3::origin());
        let out = transform_triangles(&cube, &m);
        let (min, max) = bounding_box_of(&out);
        // (x, y, z) -> (x, -z, y): y ∈ [-1, 0], z ∈ [0, 1]
        assert!((min.x - 0.0).abs() < 1e-5 && (max.x - 1.0).abs() < 1e-5);
        assert!((min.y - -1.0).abs() < 1e-5 && (max.y - 0.0).abs() < 1e-5);
        assert!((min.z - 0.0).abs() < 1e-5 && (max.z - 1.0).abs() < 1e-5);
    }

    #[test]
    fn nonuniform_scale_recomputes_normals() {
        let cube = unit_cube();
        let m = compose(
            Vector3::zeros(),
            identity_quat(),
            Vector3::new(2.0, 1.0, 1.0),
            Point3::origin()
        );
        let out = transform_triangles(&cube, &m);
        for tri in &out {
            assert!((tri.normal.norm() - 1.0).abs() < 1e-5, "normal must stay unit length");
        }
        // A face whose normal was +X must still have exactly +X after axis scaling.
        let x_faces: Vec<_> = out
            .iter()
            .filter(|t| t.vertices.iter().all(|v| (v.x - 2.0).abs() < 1e-5))
            .collect();
        assert_eq!(x_faces.len(), 2);
        for tri in x_faces {
            assert!((tri.normal - Vector3::x()).norm() < 1e-5);
        }
    }

    #[test]
    fn rotation_about_pivot_keeps_center() {
        let cube = unit_cube();
        let pivot = Point3::new(0.5, 0.5, 0.5);
        let rot = UnitQuaternion::from_euler_angles(0.3, 1.1, -0.7);
        let m = compose(Vector3::zeros(), rot, Vector3::new(1.0, 1.0, 1.0), pivot);
        let out = transform_triangles(&cube, &m);
        let (min, max) = bounding_box_of(&out);
        let center = nalgebra::center(&min, &max);
        assert!((center - pivot).norm() < 1e-4, "AABB center moved: {center:?}");
    }

    #[test]
    fn orientation_preserved_under_scale_and_rotation() {
        let cube = unit_cube();
        let rot = UnitQuaternion::from_euler_angles(0.5, -0.4, 1.2);
        let m = compose(
            Vector3::new(3.0, -2.0, 5.0),
            rot,
            Vector3::new(2.0, 3.0, 4.0),
            Point3::new(0.5, 0.5, 0.5)
        );
        let out = transform_triangles(&cube, &m);
        for (orig, new) in cube.iter().zip(&out) {
            let rotated_old = rot * orig.normal;
            assert!(
                new.normal.dot(&rotated_old) > 0.0,
                "normal flipped: old {:?} new {:?}",
                rotated_old,
                new.normal
            );
        }
    }
}
