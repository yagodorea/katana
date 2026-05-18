# Renderer Benchmark

```
./target/release/katana-bench-runner --viewer ./target/release/katana-viewer --stl <stl>
```

## Renderer improvements

| version | description |
|---|---|
| **legacy** | OpenGL / glow renderer (baseline) |
| **wgpu** | Direct wgpu port of the legacy renderer |
| **lookup** | Replace 36-branch vertex shader switch with a uniform-buffer lookup table |
| **passes** | CPU layer culling for line draws (LineBatch); opaque/transparent pipeline split; infill promoted to opaque |
| **packed** | Rhombus instance data packed 48 B → 28 B (f32 dir, palette UBO, half_width moved to FrameUniforms) |

---

## liver.stl (38 k triangles, 998 layers, dense toolpaths) (VS-bound)

| metric | legacy | wgpu | lookup | passes | packed |
|---|--:|--:|--:|--:|--:|
| load ms | 20.4 | 21.1 | 18.4 | 17.4 | 18.4 |
| slice ms | 20.4 | 21.1 | 18.4 | 17.4 | 18.4 |
| offset ms | 2,877.9 | 2,893.5 | 2,733.2 | **2,741.0** | **2,836.5** |
| plan ms | 175.2 | 175.3 | 177.1 | **146.8** | **172.4** |
| **init total ms** | **3,093.9** | **3,111.0** | 2,947.1 | **2,922.6** | **3,045.7** |
| avg fps | 3.1 | 3.4 | 5.0 | **5.7** | **6.0** |
| avg frame ms | 322.6 | 294.1 | 200.0 | **175.4** | **166.7** |

---

## sphere_smooth.stl (29 k triangles, 249 layers) (not VS-bound)

| metric | legacy | wgpu | lookup | passes | packed |
|---|--:|--:|--:|--:|--:|
| load ms | 5.6 | 5.6 | 6.0 | 5.0 | 4.6 |
| slice ms | 5.6 | 5.6 | 6.0 | 5.0 | 4.6 |
| offset ms | 85.9 | 83.1 | 76.0 | **69.9** | **80.8** |
| plan ms | 7.9 | 8.0 | 7.9 | **6.3** | **8.5** |
| **init total ms** | 105.0 | 102.3 | 95.9 | **86.2** | **98.5** |
| avg fps | 31.6 | 32.1 | 30.2 | **34.1** | **30.9** |
| avg frame ms | 31.6 | 31.2 | 33.1 | **29.3** | **32.4** |

---

## block100.stl (12 triangles, 500 layers) (not VS-bound)

| metric | legacy | wgpu | lookup | passes | packed |
|---|--:|--:|--:|--:|--:|
| load ms | 1.2 | 0.8 | 0.8 | 1.2 | 0.9 |
| slice ms | 1.2 | 0.8 | 0.8 | 1.2 | 0.9 |
| offset ms | 3.7 | 4.1 | 4.0 | **4.2** | **3.7** |
| plan ms | 11.6 | 11.5 | 13.4 | **13.6** | **12.4** |
| **init total ms** | 17.7 | **17.2** | 19.0 | 20.2 | **17.9** |
| avg fps | 22.1 | 19.3 | 19.4 | **28.2** | **27.7** |
| avg frame ms | 45.2 | 51.8 | 51.6 | **35.5** | **36.1** |
