# Gomoku

A Gomoku (Five in a Row) game written in C++17 with a Dear ImGui GUI. Supports local two-player and AI opponent powered by AlphaZero.

- Mouse-click to place stones on a 15×15 board
- Real-time per-move countdown timer
- Black and white stone rendering on a wood-coloured board
- Game save / load (`.gom` format) and step-through replay
- AI opponent backed by an AlphaZero model running on CUDA

## Requirements

| Dependency | Notes |
|---|---|
| Linux x86_64 | Supported platform |
| CMake ≥ 3.28 | Build system |
| C++17 compiler | GCC 9+ or Clang 10+ |
| CUDA Toolkit 13.0 | Required to build against LibTorch |
| NVIDIA GPU and compatible driver | Required for AI inference with CUDA 13.0 |
| LibTorch 2.11.0 + CUDA 13.0 | Downloaded automatically by CMake |
| OpenGL / libGL / X11 / Wayland dev headers | See below |

CPU-only builds and Windows are not supported.

### Linux — install system packages

**Ubuntu / Debian:**
```bash
sudo apt install cmake build-essential pkg-config libgl1-mesa-dev \
     libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
     libwayland-dev libxkbcommon-dev
```

**Arch / Manjaro:**
```bash
sudo pacman -S cmake base-devel pkg-config mesa libx11 libxrandr libxinerama \
     libxcursor libxi wayland libxkbcommon
```

## Clone & Build

```bash
# 1. Clone with submodules (imgui + glfw live under third_party/)
git clone --recurse-submodules https://github.com/flystar0526/AlphaZero-Gomoku.git
cd AlphaZero-Gomoku

# 2. Configure (choose one)
# Default configuration
cmake -B build
# Release configuration (enables compiler optimizations)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 3. Compile
cmake --build build -j$(nproc)
```

The binary is placed at `build/gomoku`

> **Already cloned without `--recurse-submodules`?**
> ```bash
> git submodule update --init --recursive
> ```

## Run

```bash
./build/gomoku
```

## How to Play

Use the main menu to choose a mode. Everything is mouse-driven:

- **Local Two Player** — click an intersection on the board to place your stone. Black moves first.
- **Player vs AI** — single-player mode where you play Black and the AI responds as White.
- **Replay** — enter a path to a `.gom` file, then use **Prev / Next** (or ← → arrow keys) to step through moves.
- **Settings** — toggle undo and configure the per-move time limit. Changes are saved to `gomoku.cfg`.

### AI Model

**Player vs AI** requires a TorchScript model file at `model/alpha-zero.pt` (relative to the working directory), a CUDA-capable NVIDIA GPU, and a driver compatible with CUDA 13.0. The model always runs on CUDA; initialization fails if CUDA is unavailable.

### Timer

When a per-move time limit is enabled, the countdown is shown in the top bar. If a player's time runs out the opponent wins automatically — no input required.

### Undo

If undo is enabled in Settings, an **Undo** button appears during play and takes back the last two half-moves so the same player moves again. In **Player vs AI** mode, undo also resets the AI's MCTS tree.

## Save Format (`.gom`)

Games are saved as human-readable text files:

```
# Gomoku Game Record
# Format version 1

[metadata]
date = 2026-03-31
duration_sec = 312
total_moves = 23
winner = black
undo_enabled = false
time_limit_enabled = false
time_limit_seconds = 60

[moves]
# num  player  row  col
1      black   7    7
2      white   7    8
```

`row` and `col` are 0-indexed integers. `winner` is `black`, `white`, or `draw`.

## License

MIT — see [LICENSE](LICENSE).
