# Renderer Benchmark

delta = wgpu - legacy

```
./target/release/katana-bench-runner --viewer ./target/release/katana-viewer --stl stls/liver.stl
```

## liver

| metric | legacy | wgpu | delta |
|---|--:|--:|--:|
| load ms | 20.4 | 21.1 | +0.7 |
| slice ms | 20.4 | 21.1 | +0.7 |
| offset ms | 2,877.9 | 2,893.5 | +15.6 |
| plan ms | 175.2 | 175.3 | +0.1 |
| **init total ms** | **3,093.9** | **3,111.0** | **+17.1** |
| avg fps | 3.1 | 3.4 | +9.7% |
| avg frame ms | 322.6 | 294.1 | -28.5 |

## sphere_smooth

| metric | legacy | wgpu | delta |
|---|--:|--:|--:|
| load ms | 5.6 | 5.6 | 0.0 |
| slice ms | 5.6 | 5.6 | 0.0 |
| offset ms | 85.9 | 83.1 | -2.8 |
| plan ms | 7.9 | 8.0 | +0.1 |
| **init total ms** | **105.0** | **102.3** | **-2.7** |
| avg fps | 31.6 | 32.1 | +1.6% |
| avg frame ms | 31.6 | 31.2 | -0.5 |

## block100

| metric | legacy | wgpu | delta |
|---|--:|--:|--:|
| load ms | 1.2 | 0.8 | -0.4 |
| slice ms | 1.2 | 0.8 | -0.4 |
| offset ms | 3.7 | 4.1 | +0.4 |
| plan ms | 11.6 | 11.5 | -0.1 |
| **init total ms** | **17.7** | **17.2** | **-0.5** |
| avg fps | 22.1 | 19.3 | -12.7% |
| avg frame ms | 45.2 | 51.8 | +6.6 |
