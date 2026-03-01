#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo "Checking for llama.cpp installation..."

# Check OS to handle Windows and Linux differences
OS="$(uname -s)"
case "${OS}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=Mac;;
    CYGWIN*)    machine=Cygwin;;
    MINGW*)     machine=MinGw;;
    MSYS*)      machine=MSYS;;
    *)          machine="UNKNOWN:${OS}"
esac

echo "Operating System detected: $machine"

install_dependencies() {
    echo "Installing build dependencies..."
    if [ "$machine" = "Linux" ]; then
        if command_exists apt-get; then
            sudo apt-get update
            sudo apt-get install -y build-essential curl gcc make cmake git
        elif command_exists dnf; then
            sudo dnf groupinstall -y "Development Tools"
            sudo dnf install -y curl gcc make cmake git
        elif command_exists yum; then
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y curl gcc make cmake git
        else
            echo "[ERROR] Unsupported package manager. Please install gcc, make, cmake, and git manually."
            exit 1
        fi
    elif [[ "$machine" == "MinGw" ]] || [[ "$machine" == "MSYS" ]] || [[ "$machine" == "Cygwin" ]]; then
        if command_exists pacman; then
            pacman -S --noconfirm mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake make git curl
        else
            echo "[ERROR] Please install MSYS2/MinGW packages: gcc, make, cmake, git."
            exit 1
        fi
    else
        echo "[ERROR] Unsupported OS for automatic dependency installation."
        exit 1
    fi
}

build_and_install_llama() {
    echo "Downloading and building llama.cpp..."
    INSTALL_DIR="$HOME/llama.cpp"
    
    if [ ! -d "$INSTALL_DIR" ]; then
        git clone https://github.com/ggerganov/llama.cpp "$INSTALL_DIR"
    fi
    
    cd "$INSTALL_DIR" || exit 1
    echo "Building with CMake for maximum CPU optimization..."
    
    # Optional: If you want to force specific flags like AVX2, AVX512, you can pass them to cmake.
    # By default, CMake on llama.cpp auto-detects host capabilities.
    mkdir -p build
    cd build || exit 1
    cmake .. -DCMAKE_BUILD_TYPE=Release
    cmake --build . --config Release -j $(nproc 2>/dev/null || echo 4)
    
    echo "Build complete."
    
    # Add to path
    BIN_DIR="$INSTALL_DIR/build/bin"
    # Ensure bin dir exists (behavior differs between make and cmake sometimes)
    if [ ! -d "$BIN_DIR" ]; then
        BIN_DIR="$INSTALL_DIR/build"
    fi

    echo "Adding $BIN_DIR to PATH in ~/.bashrc (or equivalent)..."
    
    SHELL_RC="$HOME/.bashrc"
    if [ -f "$HOME/.zshrc" ]; then
        SHELL_RC="$HOME/.zshrc"
    fi
    
    # Add to bashrc if not already there
    if ! grep -q "export PATH=\"\$PATH:$BIN_DIR\"" "$SHELL_RC"; then
        echo "export PATH=\"\$PATH:$BIN_DIR\"" >> "$SHELL_RC"
        echo "[INFO] Added llama.cpp to $SHELL_RC"
        echo "[ACTION REQUIRED] Run 'source $SHELL_RC' or restart your terminal to use the commands."
    fi
    
    # Export for current session
    export PATH="$PATH:$BIN_DIR"
    echo "llama.cpp has been installed to $INSTALL_DIR"
}

# Binaries to look for (newer versions use llama-cli/llama-server, older use main/server)
BINARIES=("llama-server" "llama-cli" "server" "main")
FOUND=false

for bin in "${BINARIES[@]}"; do
    if command_exists "$bin"; then
        echo "[OK] Found $bin at $(command -v "$bin")"
        FOUND=true
        break
    elif command_exists "${bin}.exe"; then
        echo "[OK] Found ${bin}.exe at $(command -v "${bin}.exe")"
        FOUND=true
        break
    fi
done

if [ "$FOUND" = true ]; then
    echo "llama.cpp is already installed on your system."
else
    echo "[INFO] Could not find llama.cpp binaries. Proceeding to download and install..."
    install_dependencies
    build_and_install_llama
    
    # Re-check after installation
    FOUND=false
    for bin in "${BINARIES[@]}"; do
        if command_exists "$bin" || command_exists "${bin}.exe"; then
            FOUND=true
            echo "[SUCCESS] llama.cpp successfully installed and available!"
            break
        fi
    done
    
    if [ "$FOUND" = false ]; then
        echo "[WARNING] Installation seemed to complete, but binaries still aren't in PATH."
    fi
fi

