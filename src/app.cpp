#include "app.h"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "file_manager.h"
#include "grid.h"
#include "imgui.h"
#include "policy_value_network.h"

#include <GLFW/glfw3.h>

#include <cstdio>

// ---------------------------------------------------------------------------
// Lifecycle and navigation
// ---------------------------------------------------------------------------

static void GlfwErrorCallback(int error, const char* description) {
  std::fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

App::App() = default;

App::~App() { Shutdown(); }

bool App::Init() {
  glfwSetErrorCallback(GlfwErrorCallback);
  if (!glfwInit()) return false;

  const char* glsl_version = "#version 130";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

  window_ = glfwCreateWindow(760, 800, "Gomoku", nullptr, nullptr);
  if (!window_) {
    glfwTerminate();
    return false;
  }

  glfwMakeContextCurrent(window_);
  glfwSwapInterval(1);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();

  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.IniFilename = nullptr;

  ImGui::StyleColorsDark();
  ImGuiStyle& style = ImGui::GetStyle();
  style.WindowRounding = 6.0f;
  style.FrameRounding = 4.0f;
  style.ItemSpacing = {10, 8};

  ImGui_ImplGlfw_InitForOpenGL(window_, true);
  ImGui_ImplOpenGL3_Init(glsl_version);
  return true;
}

void App::Shutdown() {
  WaitForAI();
  if (!window_) return;

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window_);
  glfwTerminate();
  window_ = nullptr;
}

void App::WaitForAI() {
  if (ai_result_.valid()) {
    ai_result_.wait();
    ai_result_ = {};
  }
}

void App::StartGame(Game::Mode mode) {
  WaitForAI();

  try {
    std::shared_ptr<PolicyValueEvaluator> evaluator;
    if (mode != Game::Mode::kPlayerVsPlayer) {
      evaluator = std::make_shared<PolicyValueNetwork>("model/alpha-zero.pt");
    }

    match_ = std::make_unique<Match>(config_, mode, std::move(evaluator));
    ai_error_msg_.clear();
    return_to_menu_requested_ = false;
    finished_record_.reset();
    save_attempted_ = false;
    save_status_msg_.clear();
    state_ = AppState::kPlaying;
  } catch (const std::exception& e) {
    ai_error_msg_ = e.what();
    ReturnToMenu();
  }
}

void App::ReturnToMenu() {
  WaitForAI();
  match_.reset();
  finished_record_.reset();
  return_to_menu_requested_ = false;
  state_ = AppState::kMainMenu;
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

void App::Run() {
  if (!Init()) return;

  while (!glfwWindowShouldClose(window_)) {
    glfwPollEvents();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    RenderFrame();

    ImGui::Render();

    int display_w, display_h;
    glfwGetFramebufferSize(window_, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.12f, 0.12f, 0.14f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window_);
  }
}

void App::RenderFrame() {
  ImGuiIO& io = ImGui::GetIO();
  ImGui::SetNextWindowPos({0, 0});
  ImGui::SetNextWindowSize(io.DisplaySize);
  ImGui::SetNextWindowBgAlpha(0.0f);

  ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration |
                           ImGuiWindowFlags_NoMove |
                           ImGuiWindowFlags_NoScrollWithMouse |
                           ImGuiWindowFlags_NoBringToFrontOnFocus;
  ImGui::Begin("##root", nullptr, flags);

  switch (state_) {
    case AppState::kMainMenu:
      DrawMainMenu();
      break;
    case AppState::kPlaying:
      DrawGame();
      break;
    case AppState::kGameOver:
      DrawGameOver();
      break;
    case AppState::kReplay:
      DrawReplay();
      break;
    case AppState::kSettings:
      DrawSettings();
      break;
  }

  ImGui::End();
}

// ---------------------------------------------------------------------------
// Main menu
// ---------------------------------------------------------------------------

void App::DrawMainMenu() {
  const float panel_w = 340.0f;
  const float panel_h = ai_error_msg_.empty() ? 400.0f : 510.0f;

  ImGuiIO& io = ImGui::GetIO();
  ImGui::SetNextWindowPos({(io.DisplaySize.x - panel_w) * 0.5f,
                           (io.DisplaySize.y - panel_h) * 0.5f},
                          ImGuiCond_Always);
  ImGui::SetNextWindowSize({panel_w, panel_h});

  ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration |
                           ImGuiWindowFlags_NoMove |
                           ImGuiWindowFlags_NoScrollbar;
  ImGui::Begin("##menu", nullptr, flags);

  ImGui::SetWindowFontScale(1.8f);
  float title_w = ImGui::CalcTextSize("GOMOKU").x;
  ImGui::SetCursorPosX((panel_w - title_w) * 0.5f);
  ImGui::TextUnformatted("GOMOKU");
  ImGui::SetWindowFontScale(1.0f);
  ImGui::Separator();
  ImGui::Spacing();

  const float btn_w = panel_w - 40.0f;
  auto centred_btn = [&](const char* label) -> bool {
    ImGui::SetCursorPosX(20.0f);
    return ImGui::Button(label, {btn_w, 36.0f});
  };

  if (centred_btn("Player vs Player")) {
    StartGame(Game::Mode::kPlayerVsPlayer);
  }
  ImGui::Spacing();

  if (centred_btn("Player vs AI")) {
    StartGame(Game::Mode::kPlayerVsAI);
  }
  ImGui::Spacing();

  if (centred_btn("AI vs AI")) {
    StartGame(Game::Mode::kAIvsAI);
  }
  ImGui::Spacing();

  if (centred_btn("Replay")) {
    replay_path_buf_[0] = '\0';
    replay_error_msg_.clear();
    replay_files_open_ = true;
    RefreshReplayFiles();
    state_ = AppState::kReplay;
  }
  ImGui::Spacing();

  if (centred_btn("Settings")) {
    settings_undo_enabled_ = config_.undo_enabled();
    settings_time_enabled_ = config_.time_limit_enabled();
    settings_time_seconds_ = config_.time_limit_seconds();
    for (AIPlayerId player : {AIPlayerId::kPlayer1, AIPlayerId::kPlayer2}) {
      settings_ai_options_[static_cast<size_t>(player)] =
          config_.ai_options(player);
    }
    state_ = AppState::kSettings;
  }
  ImGui::Spacing();

  if (centred_btn("Quit")) {
    glfwSetWindowShouldClose(window_, GLFW_TRUE);
  }

  if (!ai_error_msg_.empty()) {
    ImGui::Spacing();
    ImGui::TextWrapped("AI unavailable: %s", ai_error_msg_.c_str());
  }

  ImGui::End();
}

// ---------------------------------------------------------------------------
// In-game
// ---------------------------------------------------------------------------

void App::DrawGame() {
  const Game& game = match_->game();
  const bool spectating = game.mode() == Game::Mode::kAIvsAI;

  if (spectating) {
    ImGui::BeginDisabled(return_to_menu_requested_);
    if (ImGui::Button("Back to Menu", {140, 32})) {
      return_to_menu_requested_ = true;
    }
    ImGui::EndDisabled();
  }

  if (return_to_menu_requested_) {
    if (!ai_result_.valid() ||
        ai_result_.wait_for(std::chrono::seconds(0)) ==
            std::future_status::ready) {
      ReturnToMenu();
      return;
    }
    ImGui::SameLine();
    ImGui::TextUnformatted("Returning to menu after the current AI search...");
  }

  match_->Update();
  if (game.state() != Game::State::kPlaying) {
    finished_record_ = match_->FinalizeRecord();
    state_ = AppState::kGameOver;
    return;
  }

  // Status bar: turn info and remaining time.
  const bool ai_turn = match_->IsAITurn();
  const char* player_name =
      (game.current_player() == Player::kBlack) ? "Black" : "White";

  ImGui::SetWindowFontScale(1.15f);
  ImGui::Text("Move %d  |  %s to play", game.next_move_number(), player_name);
  ImGui::SetWindowFontScale(1.0f);

  const float secs = game.GetSecondsRemaining();
  if (secs >= 0.0f) {
    ImGui::SameLine();
    if (secs <= 10.0f) {
      ImGui::TextColored({1.0f, 0.25f, 0.25f, 1.0f}, "  |  Time: %.1fs", secs);
    } else {
      ImGui::Text("  |  Time: %.1fs", secs);
    }
  }

  if (ai_turn && !return_to_menu_requested_) {
    ImGui::SameLine();
    ImGui::TextColored({0.4f, 0.8f, 1.0f, 1.0f}, "  |  AI is thinking...");
  }
  ImGui::Spacing();

  // Launch or poll the background AI search.
  if (ai_turn && !return_to_menu_requested_) {
    try {
      if (!ai_result_.valid()) {
        ai_result_ = std::async(std::launch::async,
                                [match = match_.get()] { return match->Think(); });
      }
      if (ai_result_.wait_for(std::chrono::seconds(0)) ==
          std::future_status::ready) {
        const auto move = ai_result_.get();
        if (!move || !match_->Play(*move)) {
          throw std::runtime_error("AI returned no legal move");
        }
      }
    } catch (const std::exception& e) {
      ai_error_msg_ = e.what();
      ReturnToMenu();
      return;
    }
  }

  // Board rendering and human move input.
  const float offset_x =
      (ImGui::GetContentRegionAvail().x - Grid::kCanvasSize) * 0.5f;
  if (offset_x > 0) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offset_x);

  const auto last_move = game.position().last_move();
  if (ai_turn) {
    Grid::DrawBoardReadOnly(game.board(), last_move);
  } else {
    const auto clicked = Grid::DrawBoard(game.board(), last_move);
    if (clicked) match_->Play(*clicked);
  }

  if (game.state() != Game::State::kPlaying) {
    finished_record_ = match_->FinalizeRecord();
    state_ = AppState::kGameOver;
    return;
  }

  ImGui::Spacing();
  if (!ai_turn && !match_->IsAITurn()) {
    if (config_.undo_enabled()) {
      if (ImGui::Button("Undo", {100, 32})) match_->Undo();
      ImGui::SameLine();
    }
    if (ImGui::Button("Forfeit", {100, 32})) {
      match_->Forfeit();
      finished_record_ = match_->FinalizeRecord();
      state_ = AppState::kGameOver;
    }
  }
}

// ---------------------------------------------------------------------------
// Game over
// ---------------------------------------------------------------------------

void App::DrawGameOver() {
  const Game& game = match_->game();

  ImGui::SetWindowFontScale(1.15f);
  if (game.winner().has_value()) {
    const char* name = (*game.winner() == Player::kBlack) ? "Black" : "White";
    if (game.state() == Game::State::kTimeout)
      ImGui::Text("%s wins by timeout!", name);
    else if (game.state() == Game::State::kForfeit)
      ImGui::Text("%s wins by forfeit!", name);
    else
      ImGui::Text("%s wins!", name);
  } else {
    ImGui::TextUnformatted("Draw — the board is full!");
  }
  ImGui::SetWindowFontScale(1.0f);
  ImGui::Spacing();

  float avail_x = ImGui::GetContentRegionAvail().x;
  float offset_x = (avail_x - Grid::kCanvasSize) * 0.5f;
  if (offset_x > 0) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offset_x);
  Grid::DrawBoardReadOnly(game.board(), game.position().last_move());

  ImGui::Spacing();

  // Save record prompt and file path input.
  if (!save_attempted_) {
    ImGui::TextUnformatted("Save game record?");
    ImGui::SameLine();
    if (ImGui::Button("Save")) {
      save_attempted_ = true;

      std::time_t t = std::time(nullptr);
      std::tm* tm_buf = std::localtime(&t);
      std::ostringstream ss;
      ss << "saves/" << std::put_time(tm_buf, "%Y%m%d_%H%M%S") << ".gom";
      std::string default_path = ss.str();
      std::snprintf(save_path_buf_, sizeof(save_path_buf_), "%s",
                    default_path.c_str());
    }
    ImGui::SameLine();
    if (ImGui::Button("Skip")) {
      save_attempted_ = true;
      save_path_buf_[0] = '\0';
      save_status_msg_ = "Not saved.";
    }
  } else if (save_path_buf_[0] != '\0' && save_status_msg_.empty()) {
    ImGui::SetNextItemWidth(400.0f);
    ImGui::InputText("File path", save_path_buf_, sizeof(save_path_buf_));
    ImGui::SameLine();
    if (ImGui::Button("Confirm")) {
      std::string path(save_path_buf_);
      std::filesystem::path p(path);
      if (p.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(p.parent_path(), ec);
      }

      if (FileManager::Save(*finished_record_, path)) {
        save_status_msg_ = "Saved to: " + path;
      } else {
        save_status_msg_ = "Failed to save!";
      }
      save_path_buf_[0] = '\0';
    }
  } else if (!save_status_msg_.empty()) {
    ImGui::TextUnformatted(save_status_msg_.c_str());
  }

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  if (ImGui::Button("New Game", {130, 34})) {
    WaitForAI();
    match_->Restart();
    finished_record_.reset();
    save_attempted_ = false;
    save_status_msg_.clear();
    state_ = AppState::kPlaying;
  }
  ImGui::SameLine();
  if (ImGui::Button("Main Menu", {130, 34})) {
    ReturnToMenu();
  }
}

// ---------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------

void App::RefreshReplayFiles() {
  replay_files_.clear();
  replay_files_error_msg_.clear();

  auto paths = FileManager::ListGameFiles("saves");
  if (!paths) {
    replay_files_error_msg_ = "Failed to read saves/.";
    return;
  }

  replay_files_.reserve(paths->size());
  for (auto& path : *paths) {
    std::string filename = std::filesystem::path(path).filename().string();
    replay_files_.push_back({std::move(path), std::move(filename)});
  }
}

void App::LoadReplay(const std::string& path) {
  if (!replay_.Load(path)) {
    replay_error_msg_ = "Failed to load '" + path + "'.";
    return;
  }
  replay_error_msg_.clear();
  replay_files_open_ = false;
}

void App::DrawReplay() {
  ImGui::TextUnformatted("Replay");
  ImGui::SetNextItemOpen(replay_files_open_);
  replay_files_open_ = ImGui::CollapsingHeader("Saved games (saves/)");

  if (replay_files_open_) {
    if (ImGui::Button("Refresh")) RefreshReplayFiles();
    if (!replay_files_error_msg_.empty()) {
      ImGui::TextColored({1.0f, 0.4f, 0.4f, 1.0f}, "%s",
                         replay_files_error_msg_.c_str());
    } else if (replay_files_.empty()) {
      ImGui::TextDisabled("No .gom files in saves/.");
    } else {
      const float list_height = ImGui::GetTextLineHeightWithSpacing() * 4.0f +
                                ImGui::GetStyle().FramePadding.y * 2.0f;
      if (ImGui::BeginListBox("##saved_games", {0.0f, list_height})) {
        for (const auto& file : replay_files_) {
          ImGui::PushID(file.path.c_str());
          if (ImGui::Selectable(file.filename.c_str())) LoadReplay(file.path);
          ImGui::PopID();
        }
        ImGui::EndListBox();
      }
    }
  }

  ImGui::TextUnformatted("Or enter a path to a .gom file:");
  ImGui::SetNextItemWidth(500.0f);
  bool enter_pressed =
      ImGui::InputText("##path", replay_path_buf_, sizeof(replay_path_buf_),
                       ImGuiInputTextFlags_EnterReturnsTrue);

  ImGui::SameLine();
  bool load_clicked = ImGui::Button("Load");

  if (enter_pressed || load_clicked) {
    LoadReplay(replay_path_buf_);
  }

  if (!replay_error_msg_.empty()) {
    ImGui::TextColored({1.0f, 0.4f, 0.4f, 1.0f}, "%s",
                       replay_error_msg_.c_str());
  }

  if (replay_.total_moves() > 0) {
    ImGui::Spacing();

    bool prev = ImGui::Button("< Prev", {90, 30}) ||
                ImGui::IsKeyPressed(ImGuiKey_LeftArrow) ||
                ImGui::IsKeyPressed(ImGuiKey_P);
    ImGui::SameLine();
    bool next = ImGui::Button("Next >", {90, 30}) ||
                ImGui::IsKeyPressed(ImGuiKey_RightArrow) ||
                ImGui::IsKeyPressed(ImGuiKey_N);
    ImGui::SameLine();
    ImGui::Text("Step %d / %d", replay_.step(), replay_.total_moves());

    if (prev) replay_.Seek(replay_.step() - 1);
    if (next) replay_.Seek(replay_.step() + 1);

    const GameMetadata& meta = replay_.record().metadata();
    ImGui::Text("Date: %s  |  Moves: %d  |  Winner: %s", meta.date.c_str(),
                meta.total_moves,
                meta.winner.has_value()
                    ? (*meta.winner == Player::kBlack ? "Black" : "White")
                    : "Draw");

    ImGui::Spacing();

    float avail_x = ImGui::GetContentRegionAvail().x;
    float offset_x = (avail_x - Grid::kCanvasSize) * 0.5f;
    if (offset_x > 0) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offset_x);
    const GamePosition& position = replay_.position();
    Grid::DrawBoardReadOnly(position.board(), position.last_move());
  }

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  if (ImGui::Button("Back to Menu", {140, 32})) {
    replay_ = ReplayController();
    replay_path_buf_[0] = '\0';
    replay_error_msg_.clear();
    state_ = AppState::kMainMenu;
  }
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

void App::DrawSettings() {
  ImGui::SetWindowFontScale(1.2f);
  ImGui::TextUnformatted("Settings");
  ImGui::SetWindowFontScale(1.0f);
  ImGui::Separator();
  ImGui::Spacing();

  ImGui::Checkbox("Enable undo", &settings_undo_enabled_);
  ImGui::Spacing();

  ImGui::Checkbox("Enable per-move time limit", &settings_time_enabled_);
  ImGui::TextDisabled("Applies to human players only.");
  if (settings_time_enabled_) {
    ImGui::SetNextItemWidth(120.0f);
    ImGui::InputInt("Time limit (seconds)", &settings_time_seconds_);
    if (settings_time_seconds_ < 5) settings_time_seconds_ = 5;
  }

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::TextUnformatted("AI search settings");
  ImGui::TextDisabled("Player vs AI uses AI Player 1 settings.");
  ImGui::TextDisabled(
      "Set temperature or num_sampling_moves to 0 to disable sampling.");

  // One column per AI player so both fit without scrolling.
  static constexpr const char* kSearchLimits[] = {"Playouts", "Think time"};
  static constexpr const char* kParallelModes[] = {"None", "Virtual Loss",
                                                   "WU-PUCT"};
  constexpr float kInputWidth = 150.0f;

  bool valid_ai_options = true;
  if (ImGui::BeginTable("ai_players", 2)) {
    for (AIPlayerId player : {AIPlayerId::kPlayer1, AIPlayerId::kPlayer2}) {
      ImGui::TableNextColumn();
      ImGui::PushID(static_cast<int>(player));
      const char* name =
          player == AIPlayerId::kPlayer1 ? "AI Player 1" : "AI Player 2";
      ImGui::TextUnformatted(name);

      auto& options = settings_ai_options_[static_cast<size_t>(player)];
      int search_limit = static_cast<int>(options.search_limit);
      ImGui::SetNextItemWidth(kInputWidth);
      if (ImGui::Combo("search_limit", &search_limit, kSearchLimits,
                       IM_ARRAYSIZE(kSearchLimits))) {
        options.search_limit = static_cast<SearchLimit>(search_limit);
      }
      ImGui::SetNextItemWidth(kInputWidth);
      if (options.search_limit == SearchLimit::kPlayouts) {
        ImGui::InputInt("num_playouts", &options.num_playouts);
      } else {
        ImGui::InputInt("think_time_ms", &options.think_time_ms, 100, 1000);
      }
      ImGui::SetNextItemWidth(kInputWidth);
      ImGui::InputFloat("c_puct", &options.c_puct, 0.1f, 1.0f, "%.6g");
      ImGui::SetNextItemWidth(kInputWidth);
      ImGui::InputFloat("temperature", &options.temperature, 0.1f, 1.0f,
                        "%.6g");
      ImGui::SetNextItemWidth(kInputWidth);
      ImGui::InputInt("num_sampling_moves", &options.num_sampling_moves);

      ImGui::Spacing();
      int parallel_mode = static_cast<int>(options.parallel_mode);
      ImGui::SetNextItemWidth(kInputWidth);
      if (ImGui::Combo("parallel_mode", &parallel_mode, kParallelModes,
                       IM_ARRAYSIZE(kParallelModes))) {
        options.parallel_mode = static_cast<ParallelMode>(parallel_mode);
      }
      if (options.parallel_mode == ParallelMode::kVirtualLoss) {
        ImGui::SetNextItemWidth(kInputWidth);
        ImGui::InputFloat("virtual_loss", &options.virtual_loss, 0.1f, 1.0f,
                          "%.6g");
      }
      ImGui::SetNextItemWidth(kInputWidth);
      ImGui::InputInt("num_threads", &options.num_threads);
      ImGui::SetNextItemWidth(kInputWidth);
      ImGui::InputInt("batch_size", &options.batch_size);
      ImGui::SetNextItemWidth(kInputWidth);
      ImGui::InputInt("batch_timeout_us", &options.batch_timeout_us, 100,
                      1000);

      valid_ai_options = valid_ai_options && options.IsValid();
      ImGui::PopID();
    }
    ImGui::EndTable();
  }

  if (!valid_ai_options) {
    ImGui::PushStyleColor(ImGuiCol_Text, {1.0f, 0.25f, 0.25f, 1.0f});
    ImGui::TextWrapped(
        "Requires num_playouts > 0, think_time_ms in 1..%d, c_puct > 0, "
        "temperature >= 0, num_sampling_moves in 0..%d, virtual_loss > 0, "
        "num_threads in 1..%d, batch_size in 1..num_threads, and "
        "batch_timeout_us in 1..%d.",
        SearchOptions::kMaxThinkTimeMs, kActionCount,
        SearchOptions::kMaxThreads, SearchOptions::kMaxBatchTimeoutUs);
    ImGui::PopStyleColor();
  }

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  ImGui::BeginDisabled(!valid_ai_options);
  if (ImGui::Button("Save & Back", {130, 34})) {
    config_.set_undo_enabled(settings_undo_enabled_);
    config_.set_time_limit_enabled(settings_time_enabled_);
    config_.set_time_limit_seconds(settings_time_seconds_);
    for (AIPlayerId player : {AIPlayerId::kPlayer1, AIPlayerId::kPlayer2}) {
      config_.set_ai_options(
          player, settings_ai_options_[static_cast<size_t>(player)]);
    }
    config_.Save();
    state_ = AppState::kMainMenu;
  }
  ImGui::EndDisabled();

  ImGui::SameLine();
  if (ImGui::Button("Cancel", {100, 34})) {
    state_ = AppState::kMainMenu;
  }
}
