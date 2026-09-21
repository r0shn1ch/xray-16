package org.openxray.app;

import android.app.Activity;
import android.app.ActivityManager;
import android.app.AlertDialog;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.graphics.Typeface;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.provider.DocumentsContract;
import android.provider.Settings;
import android.view.Gravity;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

/**
 * Configuration and diagnostics front end for the native SDL engine.
 *
 * The launcher deliberately never rewrites the selected installation,
 * fsgame.ltx or user.ltx. It only stages OpenXRay-owned renderer data in the
 * app's private directory and passes explicit command-line choices to the
 * engine process.
 */
public final class LauncherActivity extends Activity {
    public static final String EXTRA_GAME_PATH = "org.openxray.extra.GAME_PATH";
    public static final String EXTRA_GAME_VARIANT = "org.openxray.extra.GAME_VARIANT";
    public static final String EXTRA_ADDITIONAL_ARGS = "org.openxray.extra.ADDITIONAL_ARGS";
    public static final String EXTRA_RENDERER_SMOKE = "org.openxray.extra.RENDERER_SMOKE";
    public static final String EXTRA_GAMEPAD_ENABLED = "org.openxray.extra.GAMEPAD_ENABLED";
    public static final String EXTRA_SPLASH_ENABLED = "org.openxray.extra.SPLASH_ENABLED";
    public static final String EXTRA_KEEP_SCREEN_ON = "org.openxray.extra.KEEP_SCREEN_ON";
    public static final String EXTRA_IMMERSIVE = "org.openxray.extra.IMMERSIVE";

    private static final String PREFS = "openxray_launcher";
    private static final String PREF_GAME_PATH = "game_path";
    private static final String PREF_GAME_URI = "game_uri";
    private static final String PREF_GAME_PATH_PREFIX = "game_path_profile_";
    private static final String PREF_GAME_URI_PREFIX = "game_uri_profile_";
    private static final String PREF_GAME_VARIANT = "game_variant";
    private static final String PREF_CUSTOM_ARGS = "custom_args";
    private static final String PREF_GAMEPAD = "gamepad";
    private static final String PREF_SPLASH = "splash";
    private static final String PREF_KEEP_SCREEN_ON = "keep_screen_on";
    private static final String PREF_IMMERSIVE = "immersive";
    private static final String PREF_ACTIVE_PAGE = "active_page";

    private static final int PAGE_GAME = 0;
    private static final int PAGE_SETTINGS = 1;
    private static final int PAGE_DIAGNOSTICS = 2;
    private static final int REQUEST_GAME_TREE = 1001;
    private static final int REQUEST_STORAGE_PERMISSIONS = 1002;
    private static final int MAX_LOG_BYTES = 180 * 1024;
    private static final int MAX_SHARED_LOG_CHARS = 600 * 1024;

    private final Handler handler = new Handler(Looper.getMainLooper());
    private EditText gamePath;
    private EditText customArgs;
    private Spinner gameVariant;
    private CheckBox gamepadEnabled;
    private CheckBox splashEnabled;
    private CheckBox keepScreenOn;
    private CheckBox immersiveMode;
    private TextView accessStatus;
    private TextView gameInspection;
    private TextView status;
    private TextView logView;
    private Button launchButton;
    private Button[] tabButtons;
    private View[] pages;
    private SharedPreferences preferences;
    private long engineLaunchTime;
    private boolean engineFailureToastShown;
    private boolean suppressProfileCallbacks = true;
    private int activeGameVariant = 3;
    private int activePage = PAGE_GAME;

    private final Runnable logPoller = new Runnable() {
        @Override
        public void run() {
            refreshLog();
            refreshRunningState();
            handler.postDelayed(this, activePage == PAGE_DIAGNOSTICS ? 700 : 1800);
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        preferences = getSharedPreferences(PREFS, MODE_PRIVATE);
        buildInterface();
        restorePreferences();
        refreshAccessStatus();
        refreshGameInspection();
        refreshLog();
        refreshRunningState();
        handler.post(this::showStorageAccessPromptIfNeeded);
    }

    @Override
    protected void onResume() {
        super.onResume();
        refreshAccessStatus();
        refreshGameInspection();
        refreshLog();
        refreshRunningState();
        handler.removeCallbacks(logPoller);
        handler.post(logPoller);
    }

    @Override
    protected void onPause() {
        savePreferences();
        handler.removeCallbacks(logPoller);
        super.onPause();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != REQUEST_GAME_TREE || resultCode != RESULT_OK || data == null || data.getData() == null)
            return;

        Uri treeUri = data.getData();
        final int takeFlags = data.getFlags()
                & (Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
        try {
            getContentResolver().takePersistableUriPermission(treeUri, takeFlags);
        } catch (SecurityException ignored) {
            // Some OEM document providers do not offer persistable grants.
        }

        String resolvedPath = resolvePrimaryStoragePath(treeUri);
        if (resolvedPath != null) {
            gamePath.setText(resolvedPath);
            preferences.edit().putString(PREF_GAME_PATH, resolvedPath)
                    .putString(profilePreference(PREF_GAME_PATH_PREFIX, activeGameVariant), resolvedPath)
                    .putString(PREF_GAME_URI, treeUri.toString())
                    .putString(profilePreference(PREF_GAME_URI_PREFIX, activeGameVariant), treeUri.toString())
                    .apply();
            setStatus("Выбрана папка: " + resolvedPath);
        } else {
            preferences.edit()
                    .putString(PREF_GAME_URI, treeUri.toString())
                    .putString(profilePreference(PREF_GAME_URI_PREFIX, activeGameVariant), treeUri.toString())
                    .apply();
            setStatus("Папка выбрана через системный проводник, но Android не раскрыл прямой путь. "
                    + "Укажите его вручную: native-движок не может читать content:// URI как обычный каталог.");
        }
        refreshAccessStatus();
        refreshGameInspection();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode != REQUEST_STORAGE_PERMISSIONS)
            return;

        refreshAccessStatus();
        setStatus(hasStorageAccess()
                ? "Доступ к файлам выдан."
                : "Доступ к файлам не выдан. Его можно включить в настройках приложения.");
    }

    private void buildInterface() {
        final int padding = dp(16);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(padding, dp(12), padding, dp(10));

        TextView title = new TextView(this);
        title.setText("OpenXRay Android");
        title.setTextSize(25);
        title.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        root.addView(title, matchWrap());

        TextView version = new TextView(this);
        version.setText("Версия " + BuildConfig.VERSION_NAME + " · ARMv7 · GLES");
        version.setTextSize(13);
        version.setPadding(0, 0, 0, dp(10));
        root.addView(version, matchWrap());

        LinearLayout tabs = new LinearLayout(this);
        tabs.setOrientation(LinearLayout.HORIZONTAL);
        tabButtons = new Button[3];
        tabButtons[PAGE_GAME] = makeTab("Игра", PAGE_GAME);
        tabButtons[PAGE_SETTINGS] = makeTab("Параметры", PAGE_SETTINGS);
        tabButtons[PAGE_DIAGNOSTICS] = makeTab("Диагностика", PAGE_DIAGNOSTICS);
        for (Button button : tabButtons)
            tabs.addView(button, new LinearLayout.LayoutParams(0, dp(48), 1));
        root.addView(tabs, matchWrap());

        pages = new View[] { buildGamePage(), buildSettingsPage(), buildDiagnosticsPage() };
        for (View page : pages)
            root.addView(page, new LinearLayout.LayoutParams(-1, 0, 1));

        setContentView(root);
    }

    private Button makeTab(String label, int page) {
        Button button = new Button(this);
        button.setText(label);
        button.setAllCaps(false);
        button.setOnClickListener(view -> showPage(page));
        return button;
    }

    private View buildGamePage() {
        LinearLayout content = pageContent();
        addSectionTitle(content, "Профиль игры");
        content.addView(bodyText(
                "Для каждой игры сохраняется отдельная папка. Профиль передаётся движку штатным ключом; "
                        + "файлы установки и конфиги не переписываются."), matchWrap());

        gameVariant = new Spinner(this);
        String[] variants = {
                "Автоматически / без ключа",
                "Shadow of Chernobyl (-soc)",
                "Clear Sky (-cs)",
                "Call of Pripyat (-cop)"
        };
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item, variants);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        gameVariant.setAdapter(adapter);
        gameVariant.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                if (!suppressProfileCallbacks && gamePath != null)
                    switchGameProfile(position);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });
        content.addView(gameVariant, matchWrap());

        addSectionTitle(content, "Установка игры");

        content.addView(bodyText(
                "Выберите корневую папку оригинальной ПК-версии. Лаунчер не меняет её содержимое; "
                        + "OpenXRay читает ресурсы напрямую."), matchWrap());

        gamePath = new EditText(this);
        gamePath.setSingleLine(true);
        gamePath.setHint("/storage/emulated/0/STALKER");
        gamePath.setOnFocusChangeListener((view, focused) -> {
            if (!focused)
                refreshGameInspection();
        });
        content.addView(gamePath, matchWrap());

        LinearLayout pathActions = horizontalRow();
        pathActions.addView(actionButton("Выбрать папку", view -> chooseGameFolder()), weightedButton());
        pathActions.addView(actionButton("Проверить", view -> refreshGameInspection()), weightedButton());
        content.addView(pathActions, matchWrap());

        gameInspection = bodyText("");
        gameInspection.setTextIsSelectable(true);
        gameInspection.setPadding(dp(2), dp(6), dp(2), dp(10));
        content.addView(gameInspection, matchWrap());

        accessStatus = bodyText("");
        content.addView(accessStatus, matchWrap());

        content.addView(actionButton("Настроить доступ к памяти", view -> requestAllFilesAccess()), matchWrap());

        addSectionTitle(content, "Запуск");
        launchButton = actionButton("Запустить игру", view -> launchEngine(false));
        launchButton.setTextSize(17);
        content.addView(launchButton, new LinearLayout.LayoutParams(-1, dp(60)));
        content.addView(actionButton("Проверить GLES без игровых файлов", view -> launchEngine(true)),
                new LinearLayout.LayoutParams(-1, dp(52)));

        status = bodyText("Готово к настройке.");
        status.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        status.setPadding(dp(2), dp(10), dp(2), dp(14));
        content.addView(status, matchWrap());
        return scrollPage(content);
    }

    private View buildSettingsPage() {
        LinearLayout content = pageContent();
        addSectionTitle(content, "Управление и экран");
        gamepadEnabled = makeCheckBox("Включить поддержку геймпада",
                "Если выключено, движок получает -no_gamepad.");
        splashEnabled = makeCheckBox("Показывать заставку OpenXRay",
                "Не влияет на оригинальные игровые intro-видео.");
        keepScreenOn = makeCheckBox("Не выключать экран во время игры",
                "Предотвращает системную блокировку при загрузке.");
        immersiveMode = makeCheckBox("Полноэкранный режим Android",
                "Скрывает системные панели; жест от края временно возвращает их.");
        content.addView(gamepadEnabled, matchWrap());
        content.addView(splashEnabled, matchWrap());
        content.addView(keepScreenOn, matchWrap());
        content.addView(immersiveMode, matchWrap());

        addSectionTitle(content, "Дополнительные аргументы");
        content.addView(bodyText(
                "Аргументы разбираются без shell. Кавычки поддерживаются. Путь, профиль игры и smoke-режим "
                        + "задаются полями выше и не могут быть переопределены здесь."), matchWrap());
        customArgs = new EditText(this);
        customArgs.setHint("Например: -novtf");
        customArgs.setMinLines(2);
        customArgs.setGravity(Gravity.TOP | Gravity.START);
        content.addView(customArgs, matchWrap());

        content.addView(actionButton("Сохранить параметры", view -> {
            savePreferences();
            setStatus("Параметры сохранены.");
            showPage(PAGE_GAME);
        }), new LinearLayout.LayoutParams(-1, dp(52)));
        content.addView(actionButton("Сбросить настройки лаунчера", view -> confirmResetPreferences()),
                new LinearLayout.LayoutParams(-1, dp(52)));
        return scrollPage(content);
    }

    private View buildDiagnosticsPage() {
        LinearLayout content = pageContent();
        addSectionTitle(content, "Журнал движка");
        LinearLayout actions = horizontalRow();
        actions.addView(actionButton("Обновить", view -> refreshLog()), weightedButton());
        actions.addView(actionButton("Поделиться", view -> shareLogs()), weightedButton());
        actions.addView(actionButton("Очистить", view -> clearLogs()), weightedButton());
        content.addView(actions, matchWrap());

        TextView help = bodyText(
                "Показываются native-лог OpenXRay и события Android Activity. Полный logcat полезен "
                        + "для ошибок драйвера или системного завершения процесса.");
        help.setPadding(0, dp(8), 0, dp(8));
        content.addView(help, matchWrap());

        logView = new TextView(this);
        logView.setTextSize(11);
        logView.setTypeface(Typeface.MONOSPACE);
        logView.setTextIsSelectable(true);
        logView.setPadding(dp(8), dp(8), dp(8), dp(16));
        content.addView(logView, matchWrap());
        return scrollPage(content);
    }

    private LinearLayout pageContent() {
        LinearLayout content = new LinearLayout(this);
        content.setOrientation(LinearLayout.VERTICAL);
        content.setPadding(0, dp(8), 0, dp(16));
        return content;
    }

    private View scrollPage(LinearLayout content) {
        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        scroll.addView(content, new ScrollView.LayoutParams(-1, -2));
        return scroll;
    }

    private LinearLayout horizontalRow() {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        return row;
    }

    private Button actionButton(String text, View.OnClickListener listener) {
        Button button = new Button(this);
        button.setText(text);
        button.setAllCaps(false);
        button.setOnClickListener(listener);
        return button;
    }

    private CheckBox makeCheckBox(String title, String description) {
        CheckBox checkBox = new CheckBox(this);
        checkBox.setText(title + "\n" + description);
        checkBox.setPadding(0, dp(5), 0, dp(5));
        return checkBox;
    }

    private TextView bodyText(String value) {
        TextView view = new TextView(this);
        view.setText(value);
        view.setTextSize(14);
        view.setLineSpacing(0, 1.08f);
        return view;
    }

    private void addSectionTitle(LinearLayout parent, String value) {
        TextView title = new TextView(this);
        title.setText(value);
        title.setTextSize(18);
        title.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        title.setPadding(0, dp(14), 0, dp(7));
        parent.addView(title, matchWrap());
    }

    private void restorePreferences() {
        int restoredVariant = clampVariant(preferences.getInt(PREF_GAME_VARIANT, 3));
        activeGameVariant = restoredVariant;
        suppressProfileCallbacks = true;
        gameVariant.setSelection(restoredVariant);
        suppressProfileCallbacks = false;
        String legacyPath = preferences.getString(PREF_GAME_PATH, "/storage/emulated/0/STALKER");
        gamePath.setText(preferences.getString(
                profilePreference(PREF_GAME_PATH_PREFIX, restoredVariant), legacyPath));
        customArgs.setText(preferences.getString(PREF_CUSTOM_ARGS, ""));
        gamepadEnabled.setChecked(preferences.getBoolean(PREF_GAMEPAD, false));
        splashEnabled.setChecked(preferences.getBoolean(PREF_SPLASH, false));
        keepScreenOn.setChecked(preferences.getBoolean(PREF_KEEP_SCREEN_ON, true));
        immersiveMode.setChecked(preferences.getBoolean(PREF_IMMERSIVE, true));
        showPage(preferences.getInt(PREF_ACTIVE_PAGE, PAGE_GAME));
    }

    private void savePreferences() {
        if (preferences == null || gamePath == null)
            return;
        preferences.edit()
                .putString(PREF_GAME_PATH, gamePath.getText().toString().trim())
                .putString(profilePreference(PREF_GAME_PATH_PREFIX, activeGameVariant),
                        gamePath.getText().toString().trim())
                .putString(PREF_CUSTOM_ARGS, customArgs.getText().toString())
                .putInt(PREF_GAME_VARIANT, activeGameVariant)
                .putBoolean(PREF_GAMEPAD, gamepadEnabled.isChecked())
                .putBoolean(PREF_SPLASH, splashEnabled.isChecked())
                .putBoolean(PREF_KEEP_SCREEN_ON, keepScreenOn.isChecked())
                .putBoolean(PREF_IMMERSIVE, immersiveMode.isChecked())
                .putInt(PREF_ACTIVE_PAGE, activePage)
                .apply();
    }

    private int clampVariant(int value) {
        return value >= 0 && value <= 3 ? value : 3;
    }

    private String profilePreference(String prefix, int variant) {
        return prefix + clampVariant(variant);
    }

    private String profileName(int variant) {
        switch (clampVariant(variant)) {
        case 1:
            return "Shadow of Chernobyl";
        case 2:
            return "Clear Sky";
        case 3:
            return "Call of Pripyat";
        default:
            return "автоопределение";
        }
    }

    private void switchGameProfile(int requestedVariant) {
        int nextVariant = clampVariant(requestedVariant);
        if (nextVariant == activeGameVariant)
            return;

        preferences.edit()
                .putString(profilePreference(PREF_GAME_PATH_PREFIX, activeGameVariant),
                        gamePath.getText().toString().trim())
                .putInt(PREF_GAME_VARIANT, nextVariant)
                .apply();
        activeGameVariant = nextVariant;
        gamePath.setText(preferences.getString(
                profilePreference(PREF_GAME_PATH_PREFIX, nextVariant), ""));
        refreshGameInspection();
        setStatus("Выбран профиль: " + profileName(nextVariant) + ".");
    }

    private void confirmResetPreferences() {
        new AlertDialog.Builder(this)
                .setTitle("Сбросить настройки лаунчера?")
                .setMessage("Будут забыты пути и параметры только этого лаунчера. Файлы игр, модов, "
                        + "fsgame.ltx и user.ltx останутся без изменений.")
                .setPositiveButton("Сбросить", (dialog, which) -> {
                    preferences.edit().clear().apply();
                    restorePreferences();
                    refreshAccessStatus();
                    refreshGameInspection();
                    setStatus("Настройки лаунчера сброшены. Файлы игры не изменялись.");
                    showPage(PAGE_GAME);
                })
                .setNegativeButton("Отмена", null)
                .show();
    }

    private void showPage(int requestedPage) {
        int page = requestedPage >= PAGE_GAME && requestedPage <= PAGE_DIAGNOSTICS
                ? requestedPage : PAGE_GAME;
        activePage = page;
        if (pages == null)
            return;
        for (int index = 0; index < pages.length; ++index) {
            boolean selected = index == page;
            pages[index].setVisibility(selected ? View.VISIBLE : View.GONE);
            tabButtons[index].setEnabled(!selected);
            tabButtons[index].setTypeface(Typeface.DEFAULT,
                    selected ? Typeface.BOLD : Typeface.NORMAL);
        }
        if (page == PAGE_DIAGNOSTICS)
            refreshLog();
    }

    private void chooseGameFolder() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE);
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION
                | Intent.FLAG_GRANT_WRITE_URI_PERMISSION
                | Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION
                | Intent.FLAG_GRANT_PREFIX_URI_PERMISSION);
        try {
            startActivityForResult(intent, REQUEST_GAME_TREE);
        } catch (ActivityNotFoundException error) {
            setStatus("На устройстве нет системного выбора папки: " + error.getMessage());
        }
    }

    private void requestAllFilesAccess() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M
                && Build.VERSION.SDK_INT < Build.VERSION_CODES.R
                && requestLegacyStoragePermissions()) {
            return;
        }

        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) {
            setStatus("На этой версии Android отдельный All files access не требуется.");
            return;
        }
        if (Environment.isExternalStorageManager()) {
            setStatus("Доступ ко всей памяти уже выдан.");
            return;
        }

        try {
            Intent intent = new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                    Uri.parse("package:" + getPackageName()));
            startActivity(intent);
        } catch (SecurityException | ActivityNotFoundException error) {
            try {
                startActivity(new Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION));
            } catch (SecurityException | ActivityNotFoundException fallbackError) {
                setStatus("Android не смог открыть страницу доступа к файлам: "
                        + fallbackError.getMessage());
            }
        }
    }

    private boolean requestLegacyStoragePermissions() {
        ArrayList<String> missing = new ArrayList<>();
        if (checkSelfPermission(android.Manifest.permission.READ_EXTERNAL_STORAGE)
                != android.content.pm.PackageManager.PERMISSION_GRANTED)
            missing.add(android.Manifest.permission.READ_EXTERNAL_STORAGE);
        if (checkSelfPermission(android.Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != android.content.pm.PackageManager.PERMISSION_GRANTED)
            missing.add(android.Manifest.permission.WRITE_EXTERNAL_STORAGE);
        if (missing.isEmpty())
            return false;
        requestPermissions(missing.toArray(new String[0]), REQUEST_STORAGE_PERMISSIONS);
        return true;
    }

    private void showStorageAccessPromptIfNeeded() {
        if (hasStorageAccess())
            return;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            new AlertDialog.Builder(this)
                    .setTitle("Нужен доступ к файлам")
                    .setMessage("OpenXRay читает оригинальные файлы STALKER непосредственно из выбранной папки. "
                            + "На Android 11 и новее включите «Разрешить управление всеми файлами» для приложения.")
                    .setPositiveButton("Открыть настройки", (dialog, which) -> requestAllFilesAccess())
                    .setNegativeButton("Позже", (dialog, which) ->
                            setStatus("Доступ пока не выдан. Его можно включить на вкладке «Игра»."))
                    .show();
        } else {
            requestAllFilesAccess();
        }
    }

    private boolean hasStorageAccess() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R)
            return Environment.isExternalStorageManager();
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M)
            return checkSelfPermission(android.Manifest.permission.READ_EXTERNAL_STORAGE)
                    == android.content.pm.PackageManager.PERMISSION_GRANTED
                    && checkSelfPermission(android.Manifest.permission.WRITE_EXTERNAL_STORAGE)
                    == android.content.pm.PackageManager.PERMISSION_GRANTED;
        return true;
    }

    private void launchEngine(boolean rendererSmoke) {
        if (!rendererSmoke && isEngineProcessRunning()) {
            Intent resume = new Intent(this, XRayActivity.class);
            resume.addFlags(Intent.FLAG_ACTIVITY_REORDER_TO_FRONT | Intent.FLAG_ACTIVITY_SINGLE_TOP);
            setStatus("Возвращаю уже запущенный движок на экран…");
            startActivity(resume);
            return;
        }
        if (!rendererSmoke && !prepareEngineLaunch())
            return;

        String[] additionalArgs;
        try {
            additionalArgs = parseAdditionalArguments(customArgs.getText().toString());
        } catch (IllegalArgumentException error) {
            setStatus("Ошибка в дополнительных аргументах: " + error.getMessage());
            showPage(PAGE_SETTINGS);
            return;
        }

        clearLogs();
        savePreferences();
        String selectedPath = gamePath.getText().toString().trim();
        writeLauncherLog("[launcher] starting "
                + (rendererSmoke ? "renderer smoke test" : "OpenXRay; game root=" + selectedPath));
        engineLaunchTime = SystemClock.elapsedRealtime();
        engineFailureToastShown = false;

        Intent intent = new Intent(this, XRayActivity.class);
        intent.putExtra(EXTRA_RENDERER_SMOKE, rendererSmoke);
        intent.putExtra(EXTRA_GAMEPAD_ENABLED, gamepadEnabled.isChecked());
        intent.putExtra(EXTRA_SPLASH_ENABLED, splashEnabled.isChecked());
        intent.putExtra(EXTRA_KEEP_SCREEN_ON, keepScreenOn.isChecked());
        intent.putExtra(EXTRA_IMMERSIVE, immersiveMode.isChecked());
        intent.putExtra(EXTRA_ADDITIONAL_ARGS, additionalArgs);
        if (!rendererSmoke) {
            intent.putExtra(EXTRA_GAME_PATH, selectedPath);
            intent.putExtra(EXTRA_GAME_VARIANT, activeGameVariant);
        }
        setStatus(rendererSmoke ? "Запускаю GLES smoke test…" : "Запускаю OpenXRay…");
        Toast.makeText(this, "OpenXRay: запуск движка…", Toast.LENGTH_SHORT).show();
        try {
            startActivity(intent);
        } catch (RuntimeException error) {
            setStatus("Не удалось запустить процесс движка: " + error.getMessage());
            Toast.makeText(this, "OpenXRay: не удалось запустить движок", Toast.LENGTH_LONG).show();
        }
    }

    private String[] parseAdditionalArguments(String commandLine) {
        ArrayList<String> result = new ArrayList<>();
        StringBuilder token = new StringBuilder();
        char quote = 0;
        boolean escaped = false;
        boolean tokenStarted = false;

        for (int index = 0; index < commandLine.length(); ++index) {
            char current = commandLine.charAt(index);
            if (escaped) {
                token.append(current);
                escaped = false;
                tokenStarted = true;
                continue;
            }
            if (current == '\\') {
                escaped = true;
                tokenStarted = true;
                continue;
            }
            if (quote != 0) {
                if (current == quote)
                    quote = 0;
                else
                    token.append(current);
                tokenStarted = true;
                continue;
            }
            if (current == '\'' || current == '"') {
                quote = current;
                tokenStarted = true;
                continue;
            }
            if (Character.isWhitespace(current)) {
                if (tokenStarted) {
                    addCheckedArgument(result, token.toString());
                    token.setLength(0);
                    tokenStarted = false;
                }
                continue;
            }
            token.append(current);
            tokenStarted = true;
        }

        if (escaped)
            throw new IllegalArgumentException("последний символ '\\' не экранирует аргумент");
        if (quote != 0)
            throw new IllegalArgumentException("не закрыта кавычка");
        if (tokenStarted)
            addCheckedArgument(result, token.toString());
        return result.toArray(new String[0]);
    }

    private void addCheckedArgument(List<String> target, String argument) {
        String normalized = argument.toLowerCase(Locale.US);
        if (normalized.equals("-android-game-root-hex")
                || normalized.equals("-renderer-smoke")
                || normalized.equals("-headless-smoke")
                || normalized.equals("-nogame")
                || normalized.equals("-soc")
                || normalized.equals("-shoc")
                || normalized.equals("-cs")
                || normalized.equals("-cop")) {
            throw new IllegalArgumentException("зарезервированный аргумент " + argument);
        }
        target.add(argument);
    }

    private boolean prepareEngineLaunch() {
        if (!hasStorageAccess()) {
            writeLauncherLog("[launcher] storage access is missing; game root was not passed to the engine");
            setStatus("Сначала включите доступ ко всей памяти.");
            return false;
        }

        try {
            String selectedPath = gamePath.getText().toString().trim();
            if (selectedPath.isEmpty()) {
                setStatus("Папка STALKER не выбрана.");
                return false;
            }
            File root = new File(selectedPath);
            if (!root.isDirectory() || !root.canRead()) {
                writeLauncherLog("[launcher] selected game root is not a readable directory: " + selectedPath);
                setStatus("Выбранный путь не является доступной для чтения папкой.");
                refreshGameInspection();
                return false;
            }
            if (!prepareBundledEngineData()) {
                writeLauncherLog("[launcher] bundled OpenXRay engine data is unavailable");
                setStatus("Не удалось подготовить внутренние данные рендера. Смотрите диагностику.");
                return false;
            }
            writeLauncherLog("[launcher] passing game root to engine without modifying it: " + selectedPath);
            return true;
        } catch (IOException | SecurityException error) {
            writeLauncherLog("[launcher] cannot prepare engine data: "
                    + error.getClass().getSimpleName() + ": " + error.getMessage());
            setStatus("Не удалось подготовить внутренние данные движка: " + error.getMessage());
            return false;
        }
    }

    private boolean prepareBundledEngineData() throws IOException {
        File privateRoot = new File(getFilesDir(), "openxray");
        File destination = new File(privateRoot, "engine-gamedata");
        File marker = new File(privateRoot, "engine-data.version");
        String desiredVersion = Integer.toString(BuildConfig.VERSION_CODE);

        if (desiredVersion.equals(readSmallTextFile(marker)) && isCompleteEngineData(destination)) {
            writeLauncherLog("[launcher] OpenXRay engine data is current: " + destination.getAbsolutePath());
            return true;
        }
        if (!privateRoot.exists() && !privateRoot.mkdirs())
            throw new IOException("cannot create " + privateRoot);

        File staging = new File(privateRoot, "engine-gamedata.new");
        File backup = new File(privateRoot, "engine-gamedata.old");
        deleteRecursively(staging);
        deleteRecursively(backup);
        copyBundledAssetTree("gamedata", staging);
        if (!isCompleteEngineData(staging)) {
            deleteRecursively(staging);
            writeLauncherLog("[launcher] bundled engine gamedata is incomplete");
            return false;
        }

        boolean hadDestination = destination.exists();
        if (hadDestination && !destination.renameTo(backup)) {
            deleteRecursively(staging);
            throw new IOException("cannot replace " + destination);
        }
        if (!staging.renameTo(destination)) {
            if (hadDestination)
                backup.renameTo(destination);
            throw new IOException("cannot activate staged engine data");
        }
        deleteRecursively(backup);
        writeSmallTextFileAtomically(marker, desiredVersion);
        writeLauncherLog("[launcher] OpenXRay engine data updated atomically: "
                + destination.getAbsolutePath());
        return true;
    }

    private boolean isCompleteEngineData(File directory) {
        File configs = new File(directory, "configs");
        File shaders = new File(directory, "shaders");
        return configs.isDirectory() && shaders.isDirectory()
                && hasDirectoryEntries(configs) && hasDirectoryEntries(shaders);
    }

    private boolean hasDirectoryEntries(File directory) {
        File[] entries = directory.listFiles();
        return entries != null && entries.length != 0;
    }

    private void copyBundledAssetTree(String assetPath, File destination) throws IOException {
        String[] children = getAssets().list(assetPath);
        if (children == null || children.length == 0) {
            File parent = destination.getParentFile();
            if (parent != null && !parent.exists() && !parent.mkdirs())
                throw new IOException("cannot create " + parent);
            try (InputStream input = getAssets().open(assetPath);
                 OutputStream output = new FileOutputStream(destination)) {
                byte[] buffer = new byte[8192];
                int count;
                while ((count = input.read(buffer)) != -1)
                    output.write(buffer, 0, count);
            }
            return;
        }
        if (!destination.exists() && !destination.mkdirs())
            throw new IOException("cannot create " + destination);
        for (String child : children)
            copyBundledAssetTree(assetPath + "/" + child, new File(destination, child));
    }

    private void deleteRecursively(File target) throws IOException {
        if (!target.exists())
            return;
        if (target.isDirectory()) {
            File[] children = target.listFiles();
            if (children == null)
                throw new IOException("cannot list " + target);
            for (File child : children)
                deleteRecursively(child);
        }
        if (!target.delete())
            throw new IOException("cannot delete " + target);
    }

    private String readSmallTextFile(File file) {
        if (!file.isFile() || file.length() > 128)
            return "";
        try (FileInputStream input = new FileInputStream(file)) {
            byte[] bytes = new byte[(int) file.length()];
            int count = input.read(bytes);
            return count > 0 ? new String(bytes, 0, count, StandardCharsets.UTF_8).trim() : "";
        } catch (IOException | SecurityException ignored) {
            return "";
        }
    }

    private void writeSmallTextFileAtomically(File file, String value) throws IOException {
        File temporary = new File(file.getParentFile(), file.getName() + ".new");
        try (FileOutputStream output = new FileOutputStream(temporary)) {
            output.write(value.getBytes(StandardCharsets.UTF_8));
            output.getFD().sync();
        }
        if (file.exists() && !file.delete())
            throw new IOException("cannot replace " + file);
        if (!temporary.renameTo(file))
            throw new IOException("cannot activate " + file);
    }

    private String resolvePrimaryStoragePath(Uri treeUri) {
        if (!DocumentsContract.isTreeUri(treeUri))
            return null;
        String documentId = DocumentsContract.getTreeDocumentId(treeUri);
        if (documentId == null || !documentId.startsWith("primary:"))
            return null;
        String relative = Uri.decode(documentId.substring("primary:".length()));
        return "/storage/emulated/0" + (relative.isEmpty() ? "" : "/" + relative);
    }

    private void refreshAccessStatus() {
        if (accessStatus == null)
            return;
        boolean granted = hasStorageAccess();
        accessStatus.setText("Доступ ко всей памяти: " + (granted ? "выдан" : "требуется"));
        accessStatus.setTextColor(granted ? Color.rgb(25, 115, 55) : Color.rgb(190, 70, 35));
    }

    private void refreshGameInspection() {
        if (gameInspection == null || gamePath == null)
            return;
        String selectedPath = gamePath.getText().toString().trim();
        if (selectedPath.isEmpty()) {
            gameInspection.setText("Профиль: " + profileName(activeGameVariant)
                    + ". Путь для этого профиля ещё не выбран.");
            gameInspection.setTextColor(Color.rgb(150, 95, 25));
            return;
        }
        File root = new File(selectedPath);
        if (!root.isDirectory()) {
            gameInspection.setText("Профиль: " + profileName(activeGameVariant)
                    + ". Папка не найдена или недоступна.");
            gameInspection.setTextColor(Color.rgb(190, 70, 35));
            return;
        }

        boolean fsgame = new File(root, "fsgame.ltx").isFile();
        boolean gamedata = new File(root, "gamedata").isDirectory();
        boolean resources = new File(root, "resources").isDirectory();
        boolean socExecutable = new File(root, "bin/XR_3DA.exe").isFile()
                || new File(root, "bin/xr_3da.exe").isFile();
        boolean laterExecutable = new File(root, "bin/xrEngine.exe").isFile()
                || new File(root, "bin/xrengine.exe").isFile();
        String hint = socExecutable ? "похоже на Shadow of Chernobyl"
                : laterExecutable ? "обнаружена установка CS/CoP"
                : "точная игра будет определена выбранным профилем";
        String mismatch = socExecutable && activeGameVariant > 1
                ? " · профиль не совпадает с найденным XR_3DA.exe"
                : laterExecutable && activeGameVariant == 1
                ? " · профиль SoC не совпадает с найденным xrEngine.exe" : "";
        gameInspection.setText("Профиль: " + profileName(activeGameVariant)
                + " · папка читается · fsgame.ltx: " + yesNo(fsgame)
                + " · gamedata: " + yesNo(gamedata)
                + " · resources: " + yesNo(resources) + "\n" + hint
                + mismatch + ". Проверка информационная и не изменяет файлы.");
        gameInspection.setTextColor(fsgame && mismatch.isEmpty()
                ? Color.rgb(25, 115, 55) : Color.rgb(150, 95, 25));
    }

    private String yesNo(boolean value) {
        return value ? "есть" : "нет";
    }

    private void refreshRunningState() {
        if (launchButton != null)
            launchButton.setText(isEngineProcessRunning()
                    ? "Вернуться в запущенную игру" : "Запустить игру");
    }

    private void refreshLog() {
        if (logView == null)
            return;
        String log = collectLogs();
        logView.setText(log.isEmpty() ? "Лог пока пуст. Запустите GLES-проверку или игру." : log);
        updateEngineStatus(log);
    }

    private String collectLogs() {
        StringBuilder result = new StringBuilder();
        appendLog(result, new File(Environment.getExternalStorageDirectory(), "openxray/android.log"));
        appendLog(result, new File(Environment.getExternalStorageDirectory(), "openxray/activity.log"));
        File external = getExternalFilesDir("openxray");
        if (external != null) {
            appendLog(result, new File(external, "android.log"));
            appendLog(result, new File(external, "activity.log"));
        }
        File internal = new File(getFilesDir(), "openxray");
        appendLog(result, new File(internal, "android.log"));
        appendLog(result, new File(internal, "activity.log"));
        return result.toString();
    }

    private void updateEngineStatus(String log) {
        if (engineLaunchTime == 0)
            return;
        if (log.contains("[renderer-smoke] center pixel") && log.contains(": PASS")) {
            setStatus("GLES renderer smoke test завершён: PASS.");
            return;
        }
        if (log.contains("[android] engine loaded")) {
            setStatus("Движок загрузил игру. Если Activity была свёрнута, нажмите «Вернуться в запущенную игру».");
            return;
        }
        if (log.contains("[renderer-smoke] initialization failed") || log.contains("engine load failed")) {
            showEngineFailureStatus();
            return;
        }
        long elapsed = SystemClock.elapsedRealtime() - engineLaunchTime;
        if (elapsed > 4000 && !isEngineProcessRunning())
            showEngineFailureStatus();
    }

    private void showEngineFailureStatus() {
        setStatus("Процесс движка завершился до штатной загрузки. Откройте вкладку «Диагностика».");
        if (!engineFailureToastShown) {
            Toast.makeText(this, "OpenXRay: загрузка не удалась; смотрите лог", Toast.LENGTH_LONG).show();
            engineFailureToastShown = true;
        }
    }

    private boolean isEngineProcessRunning() {
        ActivityManager manager = (ActivityManager) getSystemService(ACTIVITY_SERVICE);
        if (manager == null)
            return false;
        String engineProcess = getPackageName() + ":engine";
        List<ActivityManager.RunningAppProcessInfo> processes = manager.getRunningAppProcesses();
        if (processes == null)
            return false;
        for (ActivityManager.RunningAppProcessInfo process : processes) {
            if (engineProcess.equals(process.processName))
                return true;
        }
        return false;
    }

    private void appendLog(StringBuilder result, File file) {
        if (!file.isFile())
            return;
        String text = readTail(file);
        if (!text.isEmpty())
            result.append("\n===== ").append(file.getAbsolutePath()).append(" =====\n").append(text);
    }

    private String readTail(File file) {
        try (FileInputStream input = new FileInputStream(file)) {
            long skip = Math.max(0, file.length() - MAX_LOG_BYTES);
            while (skip > 0) {
                long skipped = input.skip(skip);
                if (skipped <= 0)
                    break;
                skip -= skipped;
            }
            byte[] data = new byte[(int) Math.min(file.length(), MAX_LOG_BYTES)];
            int size = input.read(data);
            return size > 0 ? new String(data, 0, size, StandardCharsets.UTF_8) : "";
        } catch (IOException | SecurityException error) {
            return "Не удалось прочитать " + file + ": " + error.getMessage();
        }
    }

    private void shareLogs() {
        String log = collectLogs();
        if (log.isEmpty()) {
            Toast.makeText(this, "Лог пока пуст", Toast.LENGTH_SHORT).show();
            return;
        }
        if (log.length() > MAX_SHARED_LOG_CHARS)
            log = log.substring(log.length() - MAX_SHARED_LOG_CHARS);
        Intent share = new Intent(Intent.ACTION_SEND);
        share.setType("text/plain");
        share.putExtra(Intent.EXTRA_SUBJECT, "OpenXRay Android " + BuildConfig.VERSION_NAME + " logs");
        share.putExtra(Intent.EXTRA_TEXT, log);
        try {
            startActivity(Intent.createChooser(share, "Поделиться логом OpenXRay"));
        } catch (ActivityNotFoundException error) {
            Toast.makeText(this, "Нет приложения для отправки текста", Toast.LENGTH_LONG).show();
        }
    }

    private void clearLogs() {
        deleteLog(new File(Environment.getExternalStorageDirectory(), "openxray/android.log"));
        deleteLog(new File(Environment.getExternalStorageDirectory(), "openxray/activity.log"));
        File external = getExternalFilesDir("openxray");
        if (external != null) {
            deleteLog(new File(external, "android.log"));
            deleteLog(new File(external, "activity.log"));
        }
        File internal = new File(getFilesDir(), "openxray");
        deleteLog(new File(internal, "android.log"));
        deleteLog(new File(internal, "activity.log"));
        refreshLog();
    }

    private void writeLauncherLog(String message) {
        File externalRoot = getExternalFilesDir("openxray");
        File[] candidates = new File[] {
                new File(Environment.getExternalStorageDirectory(), "openxray/android.log"),
                externalRoot == null ? null : new File(externalRoot, "android.log"),
                new File(getFilesDir(), "openxray/android.log")
        };
        String timestamp = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US)
                .format(new Date());
        for (File file : candidates) {
            if (file == null)
                continue;
            File parent = file.getParentFile();
            if (parent == null || (!parent.exists() && !parent.mkdirs()))
                continue;
            try (PrintWriter writer = new PrintWriter(new FileWriter(file, true))) {
                writer.println(timestamp + " " + message);
                return;
            } catch (IOException | SecurityException ignored) {
                // Try the app-specific fallback when shared storage is unavailable.
            }
        }
    }

    private void deleteLog(File file) {
        if (file.isFile())
            file.delete();
    }

    private void setStatus(String message) {
        if (status != null)
            status.setText(message);
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private LinearLayout.LayoutParams matchWrap() {
        return new LinearLayout.LayoutParams(-1, -2);
    }

    private LinearLayout.LayoutParams weightedButton() {
        return new LinearLayout.LayoutParams(0, dp(52), 1);
    }
}
