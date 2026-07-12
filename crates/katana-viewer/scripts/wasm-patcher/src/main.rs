// Rust's wasm32-unknown-unknown emits a `.tdata` thread-local segment and a
// `__tls_base` global, but not the TLS symbols wasm-bindgen's threading transform
// consumes. This adds them so the transform's per-thread init
// (`__tls_base = malloc(__tls_size, __tls_align); __wasm_init_tls(__tls_base)`)
// actually copies the thread-local template into each thread.
use std::env;
use walrus::ir::{ LocalGet, MemoryInit, Value };
use walrus::{ ConstExpr, Module, ValType };

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <input.wasm> <output.wasm>", args[0]);
        std::process::exit(1);
    }
    let mut module = Module::from_file(&args[1]).expect("load wasm");

    let Some((tdata, tls_size)) = module.data
        .iter()
        .find(|d| d.name.as_deref() == Some(".tdata"))
        .map(|d| (d.id(), d.value.len() as i32)) else {
        eprintln!("no `.tdata` segment found");
        std::process::exit(1);
    };
    let memory = module.memories.iter().next().expect("no memory").id();

    if let Some(g) = module.globals.iter().find(|g| g.name.as_deref() == Some("__tls_base")) {
        if !module.exports.iter().any(|e| e.name == "__tls_base") {
            module.exports.add("__tls_base", g.id());
        }
    }

    let size = module.globals.add_local(
        ValType::I32,
        false,
        false,
        ConstExpr::Value(Value::I32(tls_size))
    );
    module.exports.add("__tls_size", size);
    let align = module.globals.add_local(
        ValType::I32,
        false,
        false,
        ConstExpr::Value(Value::I32(16))
    );
    module.exports.add("__tls_align", align);

    let mut init = walrus::FunctionBuilder::new(&mut module.types, &[ValType::I32], &[]);
    let base = module.locals.add(ValType::I32);
    init.func_body()
        .instr(LocalGet { local: base })
        .i32_const(0)
        .i32_const(tls_size)
        .instr(MemoryInit { memory, data: tdata });
    let init = init.finish(vec![base], &mut module.funcs);
    module.exports.add("__wasm_init_tls", init);

    std::fs::write(&args[2], module.emit_wasm()).expect("write output");
    println!("Patched TLS ({tls_size} bytes) -> {}", args[2]);
}
