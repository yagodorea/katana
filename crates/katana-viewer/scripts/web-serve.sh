#!/usr/bin/env bash
# Build (and optionally serve) the WASM viewer with real rayon threads.
#
# Setup:
#   rustup toolchain install nightly
#   rustup component add rust-src --toolchain nightly
#   rustup target add wasm32-unknown-unknown --toolchain nightly
#   cargo install wasm-bindgen-cli --version 0.2.125
#
# Usage:
#   ./scripts/web-serve.sh         build + serve on :8080
#   ./scripts/web-serve.sh build   build into dist/ only
set -euo pipefail

export RUSTUP_TOOLCHAIN="${RUSTUP_TOOLCHAIN:-nightly}"
export CARGO_UNSTABLE_BUILD_STD="panic_abort,std"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
TARGET_DIR="$(cd ../.. && pwd)/target/wasm32-unknown-unknown/release"
WASM_RAW="$TARGET_DIR/katana_viewer.wasm"
WASM_PATCHED="$TARGET_DIR/katana_viewer_patched.wasm"
OUT_DIR="dist"
OUT_NAME="katana-viewer"

echo "==> cargo build"
cargo build --release --target wasm32-unknown-unknown --lib

echo "==> patch TLS symbols"
PATCHER="$SCRIPT_DIR/wasm-patcher/target/release/wasm-patcher"
[ -f "$PATCHER" ] || cargo build --release --manifest-path "$SCRIPT_DIR/wasm-patcher/Cargo.toml"
"$PATCHER" "$WASM_RAW" "$WASM_PATCHED"

echo "==> wasm-bindgen"
wasm-bindgen --target web --out-dir "$OUT_DIR" --out-name "$OUT_NAME" --no-typescript "$WASM_PATCHED"

# Memory is imported (--import-memory), so the `wasm.memory` export the JS glue
# expects doesn't exist. Capture the imported Memory and expose it as wasm.memory.
echo "==> patch glue for imported memory"
python3 - "$OUT_DIR/$OUT_NAME.js" <<'PY'
import sys
f = sys.argv[1]
js = open(f).read()
js = js.replace('let WASM_VECTOR_LEN = 0;', 'let WASM_VECTOR_LEN = 0;\nlet __imported_memory;')
js = js.replace(
    'memory: memory || new WebAssembly.Memory({initial:41,maximum:65536,shared:true}),',
    'memory: (__imported_memory = memory || new WebAssembly.Memory({initial:41,maximum:65536,shared:true})),')
js = js.replace('wasm = instance.exports;', 'wasm = instance.exports;\n    if (!wasm.memory) wasm.memory = __imported_memory;')
open(f, 'w').write(js)
PY

if command -v wasm-opt &>/dev/null; then
    echo "==> wasm-opt"
    wasm-opt -Oz --enable-threads --enable-bulk-memory --enable-mutable-globals \
        "$OUT_DIR/${OUT_NAME}_bg.wasm" -o "$OUT_DIR/${OUT_NAME}_bg.wasm"
fi

# coi-serviceworker re-adds COOP/COEP client-side so SharedArrayBuffer works on
# hosts that can't set headers (e.g. GitHub Pages). Harmless when headers exist.
cp "$SCRIPT_DIR/../web/coi-serviceworker.min.js" "$OUT_DIR/"

# Relative paths so it works both at the server root and under a Pages subpath.
cat > "$OUT_DIR/index.html" <<'HTML'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Katana (web version!)</title>
    <script src="./coi-serviceworker.min.js"></script>
    <style>
        html, body { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background: #1a1a2e; }
        canvas { display: block; width: 100%; height: 100%; }
    </style>
</head>
<body>
    <canvas id="katana_canvas"></canvas>
    <script type="module">
        import init from './katana-viewer.js';
        await init({ module_or_path: './katana-viewer_bg.wasm' });
    </script>
</body>
</html>
HTML

echo "==> built $OUT_DIR/"
[ "${1:-}" = "build" ] && exit 0

# SharedArrayBuffer needs cross-origin isolation (COOP/COEP).
echo "==> serving http://localhost:8080"
python3 - "$OUT_DIR" <<'PY'
import http.server, sys
root = sys.argv[1]
class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k): super().__init__(*a, directory=root, **k)
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()
    def guess_type(self, path):
        return 'application/wasm' if path.endswith('.wasm') else super().guess_type(path)
http.server.HTTPServer(('', 8080), Handler).serve_forever()
PY
