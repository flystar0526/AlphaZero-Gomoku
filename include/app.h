#pragma once

#include <array>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "configuration.h"
#include "game_record.h"
#include "match.h"
#include "replay_controller.h"

struct GLFWwindow;

// Owns the application window, UI state, and main render loop.
class App {
 public:
  App();
  ~App();

  // Initializes the window and runs the main loop until exit.
  void Run();

 private:
  enum class AppState {
    kMainMenu,
    kPlaying,
    kGameOver,
    kReplay,
    kSettings,
  };

  struct ReplayFile {
    std::string path;
    std::string filename;
  };

  // Lifecycle and navigation.
  bool Init();
  void Shutdown();
  void WaitForAI();
  void StartGame(Game::Mode mode);
  void ReturnToMenu();

  // UI rendering.
  void RenderFrame();
  void DrawMainMenu();
  void DrawGame();
  void DrawGameOver();
  void DrawReplay();
  void DrawSettings();

  // Replay file management.
  void RefreshReplayFiles();
  void LoadReplay(const std::string& path);

  // ---------------------------------------------------------------------------
  // Application state
  // ---------------------------------------------------------------------------

  GLFWwindow* window_ = nullptr;
  AppState state_ = AppState::kMainMenu;
  Configuration config_;

  std::unique_ptr<Match> match_;
  std::optional<GameRecord> finished_record_;

  // Save-game dialog state.
  bool save_attempted_ = false;
  char save_path_buf_[256] = {};
  std::string save_status_msg_;

  // Replay state.
  ReplayController replay_;
  char replay_path_buf_[512] = {};
  std::string replay_error_msg_;
  std::vector<ReplayFile> replay_files_;
  std::string replay_files_error_msg_;
  bool replay_files_open_ = true;

  // Settings form state.
  bool settings_undo_enabled_ = false;
  bool settings_time_enabled_ = false;
  int settings_time_seconds_ = 60;
  std::array<SearchOptions, 2> settings_ai_options_{};

  // Background AI search state.
  std::future<std::optional<Action>> ai_result_;
  std::string ai_error_msg_;
  bool return_to_menu_requested_ = false;
};
