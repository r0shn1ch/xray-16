//-----------------------------------------------------------------------------
// File: x_ray.cpp
//
// Programmers:
// Oles - Oles Shishkovtsov
// AlexMX - Alexander Maksimchuk
//-----------------------------------------------------------------------------
#include "stdafx.h"

#include "x_ray.h"

#include "embedded_resources_management.h"

#include "xrCore/Threading/TaskManager.hpp"
#include "xrNetServer/NET_AuthCheck.h"

#include <fstream>

#if defined(XR_PLATFORM_ANDROID)
#include <glad/gl.h>
#include "Common/d3d9compat.hpp"
#define RENDER_NAMESPACE render_gl
#include "Layers/xrRenderGL/glHW.h"
#undef RENDER_NAMESPACE
#endif

#include "IGame_Persistent.h"
#include "LightAnimLibrary.h"
#include "XR_IOConsole.h"

#if defined(XR_PLATFORM_WINDOWS)
#include "AccessibilityShortcuts.hpp"
#include "Text_Console.h"
#else
#define CTextConsole CConsole
#pragma todo("Implement text console or it's alternative")
#endif

#ifdef XR_PLATFORM_WINDOWS
#include <locale>

#include "DiscordGameSDK/discord.h"
#define USE_DISCORD_INTEGRATION

#include "xrCore/Text/StringConversion.hpp"
#endif

// global variables
constexpr size_t MAX_WINDOW_EVENTS = 32;

#ifdef USE_DISCORD_INTEGRATION
constexpr discord::ClientId DISCORD_APP_ID = 421286728695939072;
#endif

ENGINE_API CInifile* pGameIni = nullptr;
ENGINE_API bool CallOfPripyatMode = false;
ENGINE_API bool ClearSkyMode = false;
ENGINE_API bool ShadowOfChernobylMode = false;

ENGINE_API string512 g_sLaunchOnExit_params{};
ENGINE_API string512 g_sLaunchOnExit_app{};
ENGINE_API string_path g_sLaunchWorkingFolder{};

namespace
{
struct PathIncludePred
{
private:
    const xr_auth_strings_t* ignored;

public:
    explicit PathIncludePred(const xr_auth_strings_t* ignoredPaths) : ignored(ignoredPaths) {}
    bool IsIncluded(pcstr path)
    {
        if (!ignored)
            return true;

        return allow_to_include_path(*ignored, path);
    }
};
}

template <typename T>
void InitConfig(T& config, pcstr name, bool fatal = true,
    bool readOnly = true, bool loadAtStart = true, bool saveAtEnd = true,
    u32 sectCount = 0, const CInifile::allow_include_func_t& allowIncludeFunc = nullptr)
{
    string_path fname;
    FS.update_path(fname, "$game_config$", name);
    config = xr_new<CInifile>(fname, readOnly, loadAtStart, saveAtEnd, sectCount, allowIncludeFunc);

    CHECK_OR_EXIT(config->section_count() || !fatal,
        make_string("Cannot find file %s.\nReinstalling application may fix this problem.", fname));
}

// XXX: make it more fancy
// некрасиво слишком
void set_shoc_mode()
{
    CallOfPripyatMode = false;
    ShadowOfChernobylMode = true;
    ClearSkyMode = false;
}

void set_cs_mode()
{
    CallOfPripyatMode = false;
    ShadowOfChernobylMode = false;
    ClearSkyMode = true;
}

void set_cop_mode()
{
    CallOfPripyatMode = true;
    ShadowOfChernobylMode = false;
    ClearSkyMode = false;
}

void set_free_mode()
{
    CallOfPripyatMode = false;
    ShadowOfChernobylMode = false;
    ClearSkyMode = false;
}

void InitSettings()
{
    ZoneScoped;

    xr_auth_strings_t ignoredPaths, checkedPaths;
    fill_auth_check_params(ignoredPaths, checkedPaths); //TODO port xrNetServer to Linux
    PathIncludePred includePred(&ignoredPaths);
    CInifile::allow_include_func_t includeFilter;
    includeFilter.bind(&includePred, &PathIncludePred::IsIncluded);

    InitConfig(pSettings, "system.ltx");
    InitConfig(pSettingsAuth, "system.ltx", true, true, true, false, 0, includeFilter);
    InitConfig(pSettingsOpenXRay, "openxray.ltx", false, true, true, false);
    InitConfig(pGameIni, "game.ltx");

    if (strstr(Core.Params, "-shoc") || strstr(Core.Params, "-soc"))
        set_shoc_mode();
    else if (strstr(Core.Params, "-cs"))
        set_cs_mode();
    else if (strstr(Core.Params, "-cop"))
        set_cop_mode();
    else if (strstr(Core.Params, "-unlock_game_mode"))
        set_free_mode();
    else
    {
        pcstr gameMode = READ_IF_EXISTS(pSettingsOpenXRay, r_string, "compatibility", "game_mode", "cop");
        if (xr_strcmpi("cop", gameMode) == 0)
            set_cop_mode();
        else if (xr_strcmpi("cs", gameMode) == 0)
            set_cs_mode();
        else if (xr_strcmpi("shoc", gameMode) == 0 || xr_strcmpi("soc", gameMode) == 0)
            set_shoc_mode();
        else if (xr_strcmpi("unlock", gameMode) == 0)
            set_free_mode();
    }
}

void InitConsole()
{
    ZoneScoped;

    if (GEnv.isDedicatedServer)
        Console = xr_new<CTextConsole>();
    else
        Console = xr_new<CConsole>();

    Console->Initialize();
    xr_strcpy(Console->ConfigFile, "user.ltx");
    if (strstr(Core.Params, "-ltx "))
    {
        string64 c_name;
        sscanf(strstr(Core.Params, "-ltx ") + strlen("-ltx "), "%[^ ] ", c_name);
        xr_strcpy(Console->ConfigFile, c_name);
    }
}

void destroySettings()
{
    ZoneScoped;
    auto s = const_cast<CInifile**>(&pSettings);
    xr_delete(*s);

    auto sa = const_cast<CInifile**>(&pSettingsAuth);
    xr_delete(*sa);

    auto so = const_cast<CInifile**>(&pSettingsOpenXRay);
    xr_delete(*so);

    xr_delete(pGameIni);
}

void destroyConsole()
{
    ZoneScoped;
    Console->Execute("cfg_save");
    Console->Destroy();
    xr_delete(Console);
}

void execUserScript()
{
    ZoneScoped;
    Console->Execute("default_controls");
    Console->ExecuteScript(Console->ConfigFile);
}

constexpr pcstr FRAME_MARK_APPLICATION_STARTUP = "Application startup";
constexpr pcstr FRAME_MARK_APPLICATION_SHUTDOWN = "Application shutdown";
constexpr pcstr FRAME_MARK_APPLICATION_RUN = "Application run";

#if defined(XR_PLATFORM_ANDROID)
struct renderer_smoke_state
{
    SDL_Window* window{};
    GLuint program{};
    GLuint vertex_array{};
    GLuint vertex_buffer{};
    bool pixel_readback_done{};
    bool passed{};
};

GLuint compile_renderer_smoke_shader(GLenum type, pcstr source, pcstr label)
{
    const GLuint shader = glCreateShader(type);
    if (!shader)
    {
        Log("! [renderer-smoke] glCreateShader failed");
        return 0;
    }

    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint compiled = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (compiled == GL_TRUE)
        return shader;

    char info_log[2048]{};
    GLsizei info_length = 0;
    glGetShaderInfoLog(shader, sizeof(info_log) - 1, &info_length, info_log);
    glDeleteShader(shader);
    Msg("! [renderer-smoke] %s shader compilation failed: %.*s", label, info_length, info_log);
    return 0;
}

bool initialize_renderer_smoke(renderer_smoke_state& state)
{
    u32 window_flags = SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI;
    xray::render::render_gl::HW.SetPrimaryAttributes(window_flags);

    state.window = SDL_CreateWindow("OpenXRay GLES renderer smoke", SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED, 960, 540, window_flags);
    if (!state.window)
    {
        Msg("! [renderer-smoke] SDL_CreateWindow failed: %s", SDL_GetError());
        return false;
    }

    xray::render::render_gl::HW.CreateDevice(state.window);
    if (!xray::render::render_gl::HW.m_context)
    {
        Log("! [renderer-smoke] OpenXRay CHW could not create an GLES context");
        return false;
    }

    const char* version = reinterpret_cast<const char*>(glGetString(GL_VERSION));
    const char* renderer = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    const char* shading = reinterpret_cast<const char*>(glGetString(GL_SHADING_LANGUAGE_VERSION));
    Msg("[renderer-smoke] GL_VERSION=%s", version ? version : "<null>");
    Msg("[renderer-smoke] GL_RENDERER=%s", renderer ? renderer : "<null>");
    Msg("[renderer-smoke] GLSL=%s", shading ? shading : "<null>");

    constexpr pcstr vertex_source = R"glsl(#version 300 es
precision highp float;
layout(location = 0) in vec2 a_position;
layout(location = 1) in vec3 a_color;
out vec3 v_color;
void main()
{
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_color = a_color;
}
)glsl";

    constexpr pcstr fragment_source = R"glsl(#version 300 es
precision mediump float;
in vec3 v_color;
layout(location = 0) out vec4 out_color;
void main()
{
    out_color = vec4(v_color, 1.0);
}
)glsl";

    const GLuint vertex_shader = compile_renderer_smoke_shader(GL_VERTEX_SHADER, vertex_source, "vertex");
    const GLuint fragment_shader = compile_renderer_smoke_shader(GL_FRAGMENT_SHADER, fragment_source, "fragment");
    if (!vertex_shader || !fragment_shader)
    {
        if (vertex_shader)
            glDeleteShader(vertex_shader);
        if (fragment_shader)
            glDeleteShader(fragment_shader);
        return false;
    }

    state.program = glCreateProgram();
    glAttachShader(state.program, vertex_shader);
    glAttachShader(state.program, fragment_shader);
    glLinkProgram(state.program);
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    GLint linked = GL_FALSE;
    glGetProgramiv(state.program, GL_LINK_STATUS, &linked);
    if (linked != GL_TRUE)
    {
        char info_log[2048]{};
        GLsizei info_length = 0;
        glGetProgramInfoLog(state.program, sizeof(info_log) - 1, &info_length, info_log);
        Msg("! [renderer-smoke] GLES program link failed: %.*s", info_length, info_log);
        return false;
    }

    constexpr float vertices[] =
    {
        -0.80f, -0.75f, 1.0f, 0.15f, 0.10f,
         0.80f, -0.75f, 0.10f, 0.85f, 0.20f,
         0.00f,  0.80f, 0.15f, 0.35f, 1.00f,
    };

    glGenVertexArrays(1, &state.vertex_array);
    glGenBuffers(1, &state.vertex_buffer);
    glBindVertexArray(state.vertex_array);
    glBindBuffer(GL_ARRAY_BUFFER, state.vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
        reinterpret_cast<const void*>(2 * sizeof(float)));
    glBindVertexArray(0);

    Log("[renderer-smoke] OpenXRay CHW and GLES 3.0 shader pipeline initialized");
    return true;
}

void render_renderer_smoke(renderer_smoke_state& state)
{
    int width = 0;
    int height = 0;
    SDL_GL_GetDrawableSize(state.window, &width, &height);
    if (width <= 0 || height <= 0)
        return;

    glViewport(0, 0, width, height);
    glClearColor(0.025f, 0.035f, 0.060f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    glUseProgram(state.program);
    glBindVertexArray(state.vertex_array);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);

    if (!state.pixel_readback_done)
    {
        glFinish();
        std::array<GLubyte, 4> pixel{};
        glReadPixels(width / 2, height / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel.data());
        state.passed = pixel[3] != 0;
        state.pixel_readback_done = true;
        Msg("[renderer-smoke] center pixel RGBA=(%u,%u,%u,%u): %s", pixel[0], pixel[1], pixel[2], pixel[3],
            state.passed ? "PASS" : "FAIL");
    }

    SDL_GL_SwapWindow(state.window);
}

void destroy_renderer_smoke(renderer_smoke_state& state)
{
    if (state.program)
        glDeleteProgram(state.program);
    if (state.vertex_buffer)
        glDeleteBuffers(1, &state.vertex_buffer);
    if (state.vertex_array)
        glDeleteVertexArrays(1, &state.vertex_array);

    if (xray::render::render_gl::HW.m_context)
        xray::render::render_gl::HW.DestroyDevice();
    if (state.window)
        SDL_DestroyWindow(state.window);
}
#endif

CApplication::CApplication(pcstr commandLine, GameModule* game, const std::array<RendererModule*, 2>& modules)
{
    m_headless_smoke = commandLine && strstr(commandLine, "-headless-smoke");
    m_renderer_smoke = commandLine && strstr(commandLine, "-renderer-smoke");

    TracySetProgramName("OpenXRay");
    Threading::SetCurrentThreadName("Primary thread");
    FrameMarkStart(FRAME_MARK_APPLICATION_STARTUP);

    if (strstr(commandLine, "-dedicated"))
        GEnv.isDedicatedServer = true;

    xrDebug::Initialize(commandLine);
    {
        ZoneScopedN("SDL_Init");
        // The smoke path intentionally exercises the native core without
        // requiring a display, EGL context, game data or a sound backend.
        u32 flags = m_headless_smoke ? SDL_INIT_TIMER | SDL_INIT_EVENTS : SDL_INIT_VIDEO;
        if (!m_headless_smoke && !strstr(commandLine, "-no_gamepad"))
            flags |= SDL_INIT_GAMECONTROLLER;
        R_ASSERT3(SDL_Init(flags) == 0, "Unable to initialize SDL", SDL_GetError());
    }

    if (m_headless_smoke)
    {
        // This is a real engine bootstrap: SDL, xrCore, CPU feature probing,
        // task scheduler and the platform filesystem locator are initialized.
        // Renderer, input, sound, scripts and game modules are deliberately
        // skipped so the check is deterministic and requires no proprietary
        // game files.
        m_headless_root = std::filesystem::temp_directory_path() /
            ("openxray-headless-smoke-" + std::to_string(SDL_GetTicks()));
        std::filesystem::create_directories(m_headless_root);
        {
            std::ofstream marker(m_headless_root / "openxray_headless_smoke.marker");
            marker << "OpenXRay headless smoke\n";
        }

        Core.Initialize("OpenXRay", commandLine, false);
        FS._initialize(CLocatorAPI::flTargetFolderOnly, m_headless_root.string().c_str(), nullptr);

        const auto marker = m_headless_root / "openxray_headless_smoke.marker";
        R_ASSERT2(std::filesystem::exists(marker), "headless smoke marker is missing from the native filesystem");
        R_ASSERT2(FS.get_path("$target_folder$") != nullptr, "headless smoke target folder is missing");

        string_path marker_path;
        FS.update_path(marker_path, "$target_folder$", "openxray_headless_smoke.marker", false);
        return;
    }

#if defined(XR_PLATFORM_ANDROID)
    if (m_renderer_smoke)
    {
        auto* state = new renderer_smoke_state;
        m_renderer_smoke_state = state;
        if (!initialize_renderer_smoke(*state))
            Log("! [renderer-smoke] initialization failed");
        return;
    }
#endif

#ifdef XR_PLATFORM_WINDOWS
    AccessibilityShortcuts shortcuts;
    if (!GEnv.isDedicatedServer)
        shortcuts.Disable();
#endif

    if (!strstr(commandLine, "-nosplash"))
    {
        const bool topmost = !strstr(commandLine, "-splashnotop");
        ShowSplash(topmost);
    }

    SDL_StopTextInput(); // It's enabled by default for some reason, we don't want it
    const auto& inputTask = TaskManager::AddTask([]
    {
        const bool captureInput = !strstr(Core.Params, "-i");
        pInput = xr_new<CInput>(captureInput);
    });

    const auto& createSoundDevicesList = TaskManager::AddTask([]
    {
        Engine.Sound.CreateDevicesList();
    });

    pcstr fsltx = "-fsltx ";
    string_path fsgame = "";
    if (strstr(commandLine, fsltx))
    {
        const size_t sz = xr_strlen(fsltx);
        sscanf(strstr(commandLine, fsltx) + sz, "%[^ ] ", fsgame);
    }

    Core.Initialize("OpenXRay", commandLine, true, *fsgame ? fsgame : nullptr);

    InitSettings();
    // Adjust player & computer name for Asian
    if (pSettings->line_exist("string_table", "no_native_input"))
    {
        xr_strcpy(Core.UserName, sizeof(Core.UserName), "Player");
        xr_strcpy(Core.CompName, sizeof(Core.CompName), "Computer");
    }

    Device.InitializeImGui();
    Device.FillVideoModes();
    TaskScheduler->Wait(inputTask);
    InitConsole();

    Engine.Initialize(game, modules);
    Device.Initialize();

    Console->OnDeviceInitialize();

    execUserScript();
    InitializeDiscord();

    TaskScheduler->Wait(createSoundDevicesList);
    Engine.Sound.Create();

    // ...command line for auto start
    pcstr startArgs = strstr(Core.Params, "-start ");
    if (startArgs)
        Console->Execute(startArgs + 1);
    pcstr loadArgs = strstr(Core.Params, "-load ");
    if (loadArgs)
        Console->Execute(loadArgs + 1);

    // Initialize APP
    const auto& createLightAnim = TaskScheduler->AddTask([]
    {
        LALib.OnCreate();
    });

    Device.Create();
    TaskScheduler->Wait(createLightAnim);

    if (game)
    {
        m_game_module = game;
        g_pGamePersistent = game->create_persistent();
        R_ASSERT(g_pGamePersistent);
    }
    if (g_pGamePersistent)
        g_pGamePersistent->OnAppStart();
    else
        Console->Show();

    FrameMarkEnd(FRAME_MARK_APPLICATION_STARTUP);
}

CApplication::~CApplication()
{
    FrameMarkStart(FRAME_MARK_APPLICATION_SHUTDOWN);

    if (m_headless_smoke)
    {
        Core._destroy();
        if (!m_headless_root.empty())
            std::filesystem::remove_all(m_headless_root);
        SDL_Quit();
        xrDebug::Finalize();
        FrameMarkEnd(FRAME_MARK_APPLICATION_SHUTDOWN);
        return;
    }

#if defined(XR_PLATFORM_ANDROID)
    if (m_renderer_smoke)
    {
        auto* state = static_cast<renderer_smoke_state*>(m_renderer_smoke_state);
        if (state)
        {
            destroy_renderer_smoke(*state);
            delete state;
            m_renderer_smoke_state = nullptr;
        }
        SDL_Quit();
        xrDebug::Finalize();
        FrameMarkEnd(FRAME_MARK_APPLICATION_SHUTDOWN);
        return;
    }
#endif

    if (g_pGamePersistent)
        g_pGamePersistent->OnAppEnd();

    if (m_game_module)
        m_game_module->destroy_persistent(g_pGamePersistent);

    Engine.Event.Dump();

    xr_delete(pInput);
    destroySettings();

    LALib.OnDestroy();

    destroyConsole();

    Device.CleanupVideoModes();
    Device.DestroyImGui();
    Engine.Sound.Destroy();

    Device.Destroy();
    Engine.Destroy();

#ifdef USE_DISCORD_INTEGRATION
    discord::Core::Destroy(&m_discord_core);
#endif

    // check for need to execute something external
    if (/*xr_strlen(g_sLaunchOnExit_params) && */ xr_strlen(g_sLaunchOnExit_app))
    {
#if defined(XR_PLATFORM_WINDOWS)
        // CreateProcess need to return results to next two structures
        STARTUPINFO si = {};
        si.cb = sizeof(si);
        PROCESS_INFORMATION pi = {};
        // We use CreateProcess to setup working folder
        pcstr tempDir = xr_strlen(g_sLaunchWorkingFolder) ? g_sLaunchWorkingFolder : nullptr;
        CreateProcess(g_sLaunchOnExit_app, g_sLaunchOnExit_params, nullptr, nullptr, FALSE, 0, nullptr, tempDir, &si, &pi);
#endif
    }

    Core._destroy();
    {
        ZoneScopedN("SDL_Quit");
        SDL_Quit();
    }

    xrDebug::Finalize();
    FrameMarkEnd(FRAME_MARK_APPLICATION_SHUTDOWN);
}

int CApplication::Run()
{
#if defined(XR_PLATFORM_ANDROID)
    if (m_renderer_smoke)
    {
        auto* state = static_cast<renderer_smoke_state*>(m_renderer_smoke_state);
        if (!state || !state->program)
            return EXIT_FAILURE;

        while (!SDL_QuitRequested())
        {
            SDL_Event event;
            while (SDL_PollEvent(&event))
            {
                if (event.type == SDL_QUIT)
                    return state->passed ? EXIT_SUCCESS : EXIT_FAILURE;
            }
            render_renderer_smoke(*state);
        }
        return state->passed ? EXIT_SUCCESS : EXIT_FAILURE;
    }
#endif
    if (m_headless_smoke)
    {
        Log("[headless-smoke] SDL/Core bootstrap completed");
        return 0;
    }

    HideSplash();
    Device.Run();

    while (!SDL_QuitRequested()) // SDL_PumpEvents is here
    {
        FrameMarkStart(FRAME_MARK_APPLICATION_RUN);
        bool canCallActivate = false;
        bool shouldActivate = false;

        SDL_Event events[MAX_WINDOW_EVENTS];
        const int count = SDL_PeepEvents(events, MAX_WINDOW_EVENTS,
            SDL_GETEVENT, SDL_WINDOWEVENT, SDL_WINDOWEVENT);

        for (int i = 0; i < count; ++i)
        {
            const SDL_Event event = events[i];

            switch (event.type)
            {
            case SDL_WINDOWEVENT:
            {
                const auto window = SDL_GetWindowFromID(event.window.windowID);

                switch (event.window.event)
                {
                case SDL_WINDOWEVENT_SHOWN:
                case SDL_WINDOWEVENT_FOCUS_GAINED:
                case SDL_WINDOWEVENT_RESTORED:
                case SDL_WINDOWEVENT_MAXIMIZED:
                    if (window != Device.m_sdlWnd)
                        Device.OnWindowActivate(window, true);
                    else
                    {
                        canCallActivate = true;
                        shouldActivate = true;
                    }
                    continue;

                case SDL_WINDOWEVENT_HIDDEN:
                case SDL_WINDOWEVENT_FOCUS_LOST:
                case SDL_WINDOWEVENT_MINIMIZED:
                    if (window != Device.m_sdlWnd)
                        Device.OnWindowActivate(window, false);
                    else
                    {
                        canCallActivate = true;
                        shouldActivate = false;
                    }
                    continue;
                } // switch (event.window.event)
            }
            } // switch (event.type)

            // Only process event in Device
            // if it wasn't processed in the switch above
            Device.ProcessEvent(event);
        } // for (int i = 0; i < count; ++i)

        // Workaround for screen blinking when there's too much timeouts
        if (canCallActivate)
        {
            Device.OnWindowActivate(Device.m_sdlWnd, shouldActivate);
        }

        Device.ProcessFrame();

        UpdateDiscordStatus();
        FrameMarkEnd(FRAME_MARK_APPLICATION_RUN);
    } // while (!SDL_QuitRequested())

    Device.Shutdown();

    return 0;
}

void CApplication::ShowSplash(bool topmost)
{
    if (m_window)
        return;

    ZoneScoped;

    m_surface = std::move(ExtractSplashScreen());
    if (!m_surface)
    {
        Log("~ Couldn't create surface from image:", SDL_GetError());
        return;
    }

    Uint32 flags = SDL_WINDOW_BORDERLESS | SDL_WINDOW_HIDDEN;

    if (topmost)
        flags |= SDL_WINDOW_ALWAYS_ON_TOP;

    m_window = SDL_CreateWindow("OpenXRay", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, m_surface->w, m_surface->h, flags);
    SDL_ShowWindow(m_window);

    m_splash_thread = Threading::RunThread("Splash Thread", &CApplication::SplashProc, this);
    SDL_PumpEvents();
}

constexpr u32 SPLASH_FRAMERATE = 30;

void CApplication::SplashProc()
{
    {
        ZoneScopedN("Update splash image");
        const auto current = SDL_GetWindowSurface(m_window);
        SDL_BlitSurface(m_surface, nullptr, current, nullptr);
        SDL_UpdateWindowSurface(m_window);
    }
    while (!m_should_exit.load(std::memory_order_acquire))
    {
        UpdateDiscordStatus();
        Sleep(SPLASH_FRAMERATE);
    }
}

void CApplication::HideSplash()
{
    if (!m_window)
        return;

    ZoneScoped;

    m_should_exit.store(true, std::memory_order_release);
    m_splash_thread.join();

    SDL_DestroyWindow(m_window);
    m_window = nullptr;

    SDL_FreeSurface(m_surface);
}

void CApplication::InitializeDiscord()
{
#ifdef USE_DISCORD_INTEGRATION
    ZoneScoped;
    discord::Core* core;
    discord::Core::Create(DISCORD_APP_ID, discord::CreateFlags::NoRequireDiscord, &core);

#   ifndef MASTER_GOLD
    if (core)
    {
        const auto level = xrDebug::DebuggerIsPresent() ? discord::LogLevel::Debug : discord::LogLevel::Info;
        core->SetLogHook(level, [](discord::LogLevel level, pcstr message)
        {
            switch (level)
            {
            case discord::LogLevel::Error: Log("!", message); break;
            case discord::LogLevel::Warn:  Log("~", message); break;
            case discord::LogLevel::Info:  Log("*", message); break;
            case discord::LogLevel::Debug: Log("#", message); break;
            }
        });
    }
#   endif

    if (core)
    {
        const std::locale locale("");

        discord::Activity activity{};
        activity.SetType(discord::ActivityType::Playing);
        activity.SetApplicationId(DISCORD_APP_ID);
        activity.SetState(StringToUTF8(Core.ApplicationTitle, locale).c_str());
        activity.GetAssets().SetLargeImage("logo");
        core->ActivityManager().UpdateActivity(activity, nullptr);

        std::lock_guard guard{ m_discord_lock };
        m_discord_core = core;
    }
#endif
}

void CApplication::UpdateDiscordStatus()
{
#ifdef USE_DISCORD_INTEGRATION
    if (!m_discord_core)
        return;

    ZoneScoped;
    std::lock_guard guard{ m_discord_lock };
    m_discord_core->RunCallbacks();
#endif
}
