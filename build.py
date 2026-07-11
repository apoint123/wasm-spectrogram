import os
import subprocess
import argparse

parser = argparse.ArgumentParser(description="Build wasm-spectrogram")
parser.add_argument("--parallel", action="store_true", help="Build with multi-threading support")
args = parser.parse_args()

cmd = ["wasm-pack", "build", ".", "--target", "web"]

if args.parallel:
    rustflags = (
        "-C target-feature=+atomics,+bulk-memory,+mutable-globals "

        # needed in newer nightly toolchain
        # see https://github.com/rust-lang/rust/pull/147225
        "-C link-arg=--shared-memory "
        "-C link-arg=--max-memory=1073741824 "
        "-C link-arg=--import-memory "
        "-C link-arg=--export=__wasm_init_tls "
        "-C link-arg=--export=__tls_size "
        "-C link-arg=--export=__tls_align "
        "-C link-arg=--export=__tls_base "
        "-C link-arg=--export=__heap_base"
    )
    os.environ["RUSTFLAGS"] = f"{os.environ.get('RUSTFLAGS', '')} {rustflags}".strip()
    
    cmd.extend(["--", "-Z", "build-std=std,panic_abort", "--features", "parallel"])

subprocess.run(cmd, check=True)
