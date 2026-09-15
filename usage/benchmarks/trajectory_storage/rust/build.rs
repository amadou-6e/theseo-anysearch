fn main() {
    println!("cargo:rerun-if-changed=../trajectory_bench.proto");
    protobuf_codegen::Codegen::new()
        .pure()
        .includes([".."])
        .input("../trajectory_bench.proto")
        .cargo_out_dir("protos")
        .run_from_script();
}
