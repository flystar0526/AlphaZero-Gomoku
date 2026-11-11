# Gomoku

A Gomoku (Five in a Row) game written in C++17 with a Dear ImGui GUI. Supports Player vs Player, Player vs AI, and AI vs AI powered by AlphaZero.

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

### Headless Simulation

```bash
./build/gomoku --simulation 100
```

Runs exactly 100 AI games without initializing GLFW or ImGui. The count must be
a positive integer. AI Player 1 plays Black in odd-numbered games and White in
even-numbered games; AI Player 2 takes the opposite color. An odd count gives
Player 1 one extra game as Black. Both players retain their own `gomoku.cfg`
search settings when switching colors. The model is loaded and warmed up once;
both search trees are reset for each game. Human turn limits do not apply.
The existing CUDA requirement still applies, and the executable still links its
GUI libraries, but simulation does not require a display server.

Each run creates a unique `logs/simulation_<local timestamp>_<microseconds>/`
directory relative to the working directory, and prints its path and per-game
progress to the terminal:

| File | Contents |
|---|---|
| `moves.csv` | Game and move numbers, AI identity, color, zero-based coordinates, total thinking time, Selection / Expansion / Evaluation / Inference / Backup times, completed playouts, inference batches, batched positions, and leaf collisions |
| `games.csv` | Black and White AI identities, winner identity and color, move count, and game duration; a draw has an empty winner identity |
| `summary.txt` | Run status, requested/completed games, model path, AI settings, overall Black/White wins and win rates, and each AI's results, timing, and search counters by color and in total |

All durations are milliseconds measured with a monotonic clock. `think_ms` is
the wall-clock time of the complete `PickMove()` call. Each phase duration is
the sum across that move's MCTS playouts and all search workers, so with more
than one thread it measures CPU time and the phases can add up to more than
`think_ms`. Selection includes traversal, pre-updates, and advancing the search
position; Evaluation is the time a worker waits for its leaf, from queueing it
until the batch result returns; Expansion covers legal moves, prior
normalization, and child creation; Backup covers value propagation. Inference
is the batch evaluator's time inside the model call (feature encoding,
inference, and returning results to CPU) and overlaps Evaluation. Terminal
leaves skip Evaluation and Expansion. Thread start-up and final move selection
also contribute to total thinking time.

`playouts` counts completed simulations, `batches` counts model calls, and
`batched_positions` counts the positions sent in them. `collisions` counts
leaves that another worker expanded first; the worker discards its own
expansion and only backs up its value. The `[search]` summary section reports
the total and per-move mean of each counter, plus `mean_batch_size`
(`batched_positions / batches`, `N/A` when no batch ran).
Model initialization, logging, and synchronizing trees after a move are outside
`think_ms`. Game duration includes move processing, tree synchronization, and
writing move rows, but excludes model initialization and summary writing.

Win rates use `wins / games * 100`, including draws in the denominator. Each
AI/color row uses its actual games in that color; overall Black/White rows use
all completed games. A zero denominator is reported as `N/A`. Timing summaries
report move counts, total, mean, and maximum durations for each metric using
completed games only.

CSV files are flushed and the summary is refreshed after every completed game.
On an inference or logging error, the process exits with a nonzero status,
preserves available logs, and writes `status = failed` with the error when the
summary is writable. An unfinished game can leave move rows in `moves.csv` but
does not contribute to summary statistics. Model initialization errors are
reported before a run directory is created. `--help` prints command-line usage.

## How to Play

Use the main menu to choose a mode. Everything is mouse-driven:

- **Player vs Player** — click an intersection on the board to place your stone. Black moves first.
- **Player vs AI** — single-player mode where you play Black and the AI responds as White.
- **AI vs AI** — watch two AI players play against each other automatically.
- **Replay** — click a `.gom` file in the saved-game list or enter a path, then use **Prev / Next** (or ← → arrow keys) to step through moves. The list shows files directly in `saves/`, relative to the working directory, sorted by filename. It refreshes when Replay opens or when you click **Refresh**, and collapses after a successful load. A failed load preserves the current replay.
- **Settings** — configure undo, the per-move time limit, and independent **AI Player 1** / **AI Player 2** parameters: search limit (`num_playouts` or `think_time_ms`), `c_puct`, `temperature`, `num_sampling_moves`, and the parallel search options described below. Changes are saved to `gomoku.cfg` and apply when the next game starts.

### AI Model

**Player vs AI** and **AI vs AI** require a TorchScript model file at `model/alpha-zero.pt` (relative to the working directory), a CUDA-capable NVIDIA GPU, and a driver compatible with CUDA 13.0. The model always runs on CUDA; initialization fails if CUDA is unavailable.

The bundled model is the three-channel `best_checkpoint.pt` export from the training project. Its single-position input is float32 `[1, 3, 15, 15]`:

| Channel | Contents |
|---|---|
| 0 | Stones belonging to the side to move |
| 1 | Opponent stones |
| 2 | A one at the last move, or all zeros on an empty board |

Coordinates are used directly, with action index `row * 15 + col`; there is no flip or transpose. The model returns a tuple of log-policy `[1, 225]` and value `[1, 1]`, with value measured from the side to move. Older nine-channel models are incompatible. Loading and inference errors are shown in the main menu.

### Parallel Search

Each search runs `num_threads` worker threads over one shared tree (tree parallelization). Every node has its own spinlock guarding its children and their statistics, and a worker never holds two locks at once. Workers that reach an unexpanded leaf put its position into an inference queue and block. A separate batch evaluator thread sends queued positions to the model as one `[B, 3, 15, 15]` batch when `batch_size` positions are waiting, when every active worker is waiting, or when the oldest request has waited `batch_timeout_us`. Threads are started for each search and joined before the move is chosen; one worker is used when `num_threads = 1`.

`parallel_mode` chooses how workers avoid following each other down the same path while results are pending. Each worker pre-updates every node it passes and reverts the pre-update during backup:

| Mode | Config value | Effect on selection |
|---|---|---|
| None | `none` | No pre-update; workers see only completed statistics. |
| Virtual Loss | `virtual_loss` | Each in-flight simulation counts as `virtual_loss` lost visits: `N += vl`, `W -= vl`. |
| WU-PUCT | `wu_puct` | In-flight simulations `O` enter only the exploration term: `Q + c_puct · P · sqrt(N_parent + O_parent) / (1 + N + O)`, where `Q = W / N` is unchanged. |

With one worker, no pre-update is visible during its own selection, so all three modes produce the same search.

`search_limit` ends a search after `num_playouts` simulations (`playouts`) or after `think_time_ms` milliseconds (`think_time`). In think-time mode `num_playouts` is ignored; no new simulations start after the deadline, and in-flight ones finish before the move is chosen.

| Key | Default | Valid range |
|---|---|---|
| `search_limit` | `playouts` | `playouts` or `think_time` |
| `num_playouts` | 400 | > 0 |
| `think_time_ms` | 1000 | > 0 |
| `parallel_mode` | `none` | `none`, `virtual_loss`, or `wu_puct` |
| `virtual_loss` | 1 | finite and > 0 |
| `num_threads` | 1 | 1–64 |
| `batch_size` | 1 | 1–`num_threads` |
| `batch_timeout_us` | 1000 | > 0 |

Settings shows `num_playouts` or `think_time_ms` for the selected search limit and `virtual_loss` only in Virtual Loss mode; hidden values are kept and saved. Each broken rule is listed under the settings, and saving is disabled until all are fixed. If the model fails during a search, the search tree is reset and the error is reported.

### Move Sampling

Each AI samples moves with weights proportional to `visits^(1 / temperature)` after MCTS finishes for the first `num_sampling_moves` moves of a game. The defaults are `temperature = 1` and `num_sampling_moves = 30`. The move limit counts both players together: the first 30 board moves use sampling on AI turns, and move 31 onward selects the most visited move. Lower positive temperatures favor the most visited moves more strongly; higher temperatures spread the probabilities more evenly among visited moves.

Set either parameter to `0` to select the most visited move throughout the game. Temperature must be finite and nonnegative, and the move limit must be between `0` and `225`. If a search has not visited any child yet, selection falls back to the existing prior-based tie breaking. Undo uses the restored board's move count to determine whether sampling applies again.

AI Player 1 and AI Player 2 have independent settings identified separately from stone colors. **Player vs AI** uses AI Player 1, which plays White. **AI vs AI** currently assigns AI Player 1 to Black and AI Player 2 to White; a color-swap control is not yet provided.

Configuration keys use the prefixes `ai_player_1_` and `ai_player_2_`, followed by `num_playouts`, `c_puct`, `temperature`, `num_sampling_moves`, or one of the parallel search keys above (for example, `ai_player_1_temperature`). Missing or invalid keys use their defaults: 400 playouts, `c_puct = 5`, `temperature = 1`, and `num_sampling_moves = 30`. Range rules, including `batch_size <= num_threads`, are checked after the whole file is read, so key order does not matter; each field that breaks a rule is reset to its default. Old `black_ai_*` and `white_ai_*` keys are ignored; saving settings writes only the new keys.

Each AI has an independently seeded random engine, so different games can produce different move sequences, though distinct game records are not guaranteed.

### Core Design

- `Board` owns cells and shared board rules. It has no turn, model or record state.
- `GamePosition` combines a Board with the side to move, last move and result. `PlayMove(action)` requires a legal action in an ongoing position; debug assertions enforce this precondition. Undo and replay share `GamePosition::Rebuild`.
- `Game` handles modes, human-only clocks, input validation, forfeit and move records; search copies positions without move histories. ReplayController validates move sequences after FileManager parses the file.
- `PolicyValueNetwork` owns three-channel feature encoding and the TorchScript backend (defaulting to CUDA for gameplay, with CPU support for headless contract tests). Its header also declares the small `PolicyValueEvaluator` interface used to test MCTS without a GPU. `PolicyValueEvaluator::EvaluateBatch` evaluates several positions at once; its default evaluates them one by one. Each `AIPlayer` owns an MCTS instance and may share an evaluator; `ObserveMove` advances the tree after a move is accepted. Search is parallel internally, but Search, ObserveMove and Reset on one MCTS must run serially, and two searches must not share an evaluator at the same time.

### Tests

CTest targets are built by default (`-DBUILD_TESTING=OFF` disables them):

```bash
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

Core tests cover board rules, feature orientation, undo, replay validation, timers, AI configuration, and MCTS search, including parallel invariants (pre-updates fully reverted, batch limits, worker spreading per mode, think-time limits, and error recovery). Simulation tests cover phase timing, alternating AI identities, color-specific win rates, draws, CSV/summary consistency, and failed runs. CLI tests check help and invalid arguments without a display. The checkpoint contract and two-game simulation smoke tests load the real model on CPU. The model tests also check that batched inference matches single-position inference. The CUDA backend smoke test is explicitly skipped when a GPU is unavailable; this does not enable CPU inference in the game or simulation CLI. The existing Linux/CUDA build requirements still apply.

To check the parallel search for data races, build with ThreadSanitizer in a separate directory and run the core and simulation tests. LibTorch is not instrumented, so the model tests are not meaningful under TSan. Recent kernels need address-space randomization disabled with `setarch -R`:

```bash
cmake -B build-tsan -DGOMOKU_TSAN=ON
cmake --build build-tsan --target gomoku_core_tests gomoku_simulation_tests -j$(nproc)
setarch $(uname -m) -R ctest --test-dir build-tsan -R "^(core|simulation)$" --output-on-failure
```

### Timer

When a per-move time limit is enabled, the countdown is shown in the top bar during human turns. If a player's time runs out the opponent wins automatically — no input required.

### Undo

If undo is enabled in Settings, an **Undo** button appears during human turns and takes back the last two half-moves so the same player moves again. In **Player vs AI** mode, undo also resets the AI's MCTS tree.

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
