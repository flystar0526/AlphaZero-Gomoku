#pragma once

#include <array>
#include <future>
#include <memory>
#include <optional>
#include <string>

#include "ai_player.h"
#include "configuration.h"
#include "game.h"
#include "game_record.h"
#include "replay_controller.h"

struct GLFWwindow;

// Top-level application: owns the GLFW window + ImGui context and drives the
// main render loop.  All game state lives here; navigation is handled by
// switching AppState each frame.
class App {
 public:
  App();
  ~App();

  // Initialises GLFW/ImGui, then runs the event loop until the user quits.
  void Run();

 private:
  enum class AppState {
    kMainMenu,
    kPlaying,
    kGameOver,
    kReplay,
    kSettings,
  };

  // Window / ImGui lifecycle.
  bool Init();
  void Shutdown();
  // Finish and discard an obsolete background result before changing AI state.
  void WaitForAI();
  // Initializes game state and AI players for the selected mode.
  void StartGame(Game::Mode mode);
  // Resets active game/AI state and returns to the main menu.
  void ReturnToMenu();

  // Per-frame orchestration.
  void RenderFrame();

  // Per-state UI panels (called inside the full-screen ImGui window).
  void DrawMainMenu();
  void DrawGame();
  void DrawGameOver();
  void DrawReplay();
  void DrawSettings();

  // ── State ────────────────────────────────────────────────────────────────
  GLFWwindow* window_ = nullptr;
  AppState state_ = AppState::kMainMenu;
  Configuration config_;

  // Active game (alive during kPlaying and kGameOver).
  std::unique_ptr<Game> game_;
  // Sealed record after the game ends.
  std::optional<GameRecord> finished_record_;

  // Save-game dialog (shown inside GameOver panel).
  bool save_attempted_ = false;
  char save_path_buf_[256] = {};
  std::string save_status_msg_;

  // Replay.
  ReplayController replay_;
  int replay_cursor_ = 0;
  char replay_path_buf_[512] = {};
  std::string replay_error_msg_;

  // Settings: temporary copy edited in the Settings panel.
  bool settings_undo_enabled_ = false;
  bool settings_time_enabled_ = false;
  int settings_time_seconds_ = 60;
  std::array<SearchOptions, 2> settings_ai_options_{};

  // ── AI game fields ──────────────────────────────────────────────────────────
  std::array<std::unique_ptr<AIPlayer>, 2> ai_players_;
  std::future<std::optional<Action>> ai_result_;
  std::string ai_error_msg_;
  // Defers returning to menu in AI vs AI until the active search finishes.
  bool return_to_menu_requested_ = false;
};
