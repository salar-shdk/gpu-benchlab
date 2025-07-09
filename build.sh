#!/bin/bash
set -e

# === Configuration ===
BUILD_DIR="build"
CORES_COUNT=$(($(nproc) - 2))
BACKEND=""
OPTION=""

# === Functions ===

print_help() {
    echo "Usage: ./build.sh [f|c] [--backend=HIP|CUDA]"
    echo
    echo "Options:"
    echo "  -f                  Full rebuild (clean + build)"
    echo "  -c                  Clean build directory"
    echo "  --backend=HIP      Build HIP benchmarks"
    echo "  --backend=CUDA     Build CUDA benchmarks"
    echo "  -h, --help         Show this help message"
    exit 0
}

detect_backend() {
    if [[ -n "$BACKEND" ]]; then return; fi

    if command -v nvidia-smi &> /dev/null; then
        BACKEND="CUDA"
        echo "🟦 Detected NVIDIA GPU: Using CUDA backend"
    elif command -v rocminfo &> /dev/null; then
        BACKEND="HIP"
        echo "🟥 Detected AMD GPU: Using HIP backend"
    else
        echo "❌ Could not detect supported GPU backend (NVIDIA or AMD)."
        echo "Use --backend=HIP or --backend=CUDA to specify manually."
        exit 1
    fi
}

clean() {
    echo "🧹 Cleaning build directory..."
    rm -rf "$BUILD_DIR" build.log
    echo "✅ Cleaning completed."
}

duration() {
    local duration=$((end - start))
    echo | tee -a build.log
    echo "################################################################################" | tee -a build.log
    echo "Build time: ${duration}s" | tee -a build.log
    echo "################################################################################" | tee -a build.log
    echo | tee -a build.log
}

build() {
    detect_backend

    echo "🚀 Starting build for backend: $BACKEND"

    start=$(date +%s)
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    touch ".${BACKEND}"
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_CUDA=$([[ "$BACKEND" == "CUDA" ]] && echo ON || echo OFF) \
        -DENABLE_HIP=$([[ "$BACKEND" == "HIP" ]] && echo ON || echo OFF) \
        2>&1 | tee ../build.log

    make -j "$CORES_COUNT" 2>&1 | tee -a ../build.log

    end=$(date +%s)
    cd ..
    duration
}

parse_args() {
    for arg in "$@"; do
        case $arg in
            -c)
                OPTION="CLEAN"
                ;;
            -f)
                OPTION="FULL"
                ;;
            --backend=*)
                BACKEND="${arg#*=}"
                ;;
            -h|--help)
                print_help
                ;;
            *)
                echo "❌ Unknown argument: $arg"
                print_help
                ;;
        esac
    done
}

# === Main Execution ===

parse_args "$@"

case "$OPTION" in
    "CLEAN")
        clean
        ;;
    "FULL")
        clean
        build
        ;;
    *)
        build
        ;;
esac

