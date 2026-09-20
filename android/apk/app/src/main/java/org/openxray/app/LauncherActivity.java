package org.openxray.app;

import android.app.Activity;
import android.app.ActivityManager;
import android.app.AlertDialog;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.content.SharedPreferences;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.provider.DocumentsContract;
import android.provider.Settings;
import android.graphics.Typeface;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
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
import java.util.ArrayList;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/**
 * Android launcher for selecting the STALKER installation and inspecting the
 * native engine log after a run. The engine itself remains in XRayActivity so
 * SDL owns the GLES surface and native main thread.
 */
public final class LauncherActivity extends Activity {
    public static final String EXTRA_GAME_PATH = "org.openxray.extra.GAME_PATH";
    public static final String EXTRA_RENDERER_SMOKE = "org.openxray.extra.RENDERER_SMOKE";

    private static final String PREFS = "openxray_launcher";
    private static final String PREF_GAME_PATH = "game_path";
    private static final String PREF_GAME_URI = "game_uri";
    private static final int REQUEST_GAME_TREE = 1001;
    private static final int REQUEST_STORAGE_PERMISSIONS = 1002;
    private static final int MAX_LOG_BYTES = 180 * 1024;
    private static final String[] COP_RESOURCE_DIRECTORIES = {
            "levels", "localization", "mp", "patches", "resources"
    };

    private final Handler handler = new Handler(Looper.getMainLooper());
    private EditText gamePath;
    private TextView accessStatus;
    private TextView status;
    private TextView logView;
    private SharedPreferences preferences;
    private long engineLaunchTime;
    private boolean engineFailureToastShown;

    private final Runnable logPoller = new Runnable() {
        @Override
        public void run() {
            refreshLog();
            handler.postDelayed(this, 700);
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        preferences = getSharedPreferences(PREFS, MODE_PRIVATE);
        buildInterface();
        gamePath.setText(preferences.getString(PREF_GAME_PATH, "/storage/emulated/0/STALKER"));
        refreshAccessStatus();
        refreshLog();
        handler.post(this::showStorageAccessPromptIfNeeded);
    }

    @Override
    protected void onResume() {
        super.onResume();
        refreshAccessStatus();
        refreshLog();
        handler.removeCallbacks(logPoller);
        handler.post(logPoller);
    }

    @Override
    protected void onPause() {
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
                    .putString(PREF_GAME_URI, treeUri.toString()).apply();
            setStatus("Выбрана папка: " + resolvedPath);
        } else {
            preferences.edit().putString(PREF_GAME_URI, treeUri.toString()).apply();
            setStatus("Папка выбрана через SAF, но прямой путь не определён. "
                    + "Для native-движка укажите путь вручную и включите доступ ко всей памяти.");
        }
        refreshAccessStatus();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode != REQUEST_STORAGE_PERMISSIONS)
            return;

        refreshAccessStatus();
        if (hasStorageAccess())
            setStatus("Доступ к файлам выдан.");
        else
            setStatus("Доступ к файлам не выдан. Его можно включить в настройках приложения.");
    }

    private void buildInterface() {
        final int padding = dp(16);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(padding, padding, padding, padding);

        TextView title = new TextView(this);
        title.setText("OpenXRay Launcher " + BuildConfig.VERSION_NAME);
        title.setTextSize(24);
        title.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        root.addView(title, matchWrap());

        TextView subtitle = new TextView(this);
        subtitle.setText("Выберите папку установки STALKER, затем запустите движок или renderer smoke test.");
        subtitle.setPadding(0, dp(6), 0, dp(12));
        root.addView(subtitle, matchWrap());

        LinearLayout pathRow = new LinearLayout(this);
        pathRow.setOrientation(LinearLayout.HORIZONTAL);
        gamePath = new EditText(this);
        gamePath.setSingleLine(true);
        gamePath.setHint("/storage/emulated/0/STALKER");
        pathRow.addView(gamePath, new LinearLayout.LayoutParams(0, dp(52), 1));
        Button choose = new Button(this);
        choose.setText("Выбрать");
        choose.setOnClickListener(view -> chooseGameFolder());
        pathRow.addView(choose, new LinearLayout.LayoutParams(dp(112), dp(52)));
        root.addView(pathRow, matchWrap());

        accessStatus = new TextView(this);
        accessStatus.setPadding(0, dp(8), 0, dp(8));
        root.addView(accessStatus, matchWrap());

        LinearLayout accessRow = new LinearLayout(this);
        accessRow.setOrientation(LinearLayout.HORIZONTAL);
        Button storage = new Button(this);
        storage.setText("Доступ к памяти");
        storage.setOnClickListener(view -> requestAllFilesAccess());
        accessRow.addView(storage, new LinearLayout.LayoutParams(0, dp(52), 1));
        Button clear = new Button(this);
        clear.setText("Очистить лог");
        clear.setOnClickListener(view -> clearLogs());
        accessRow.addView(clear, new LinearLayout.LayoutParams(0, dp(52), 1));
        root.addView(accessRow, matchWrap());

        LinearLayout launchRow = new LinearLayout(this);
        launchRow.setOrientation(LinearLayout.HORIZONTAL);
        Button smoke = new Button(this);
        smoke.setText("Проверить GLES");
        smoke.setOnClickListener(view -> launchEngine(true));
        launchRow.addView(smoke, new LinearLayout.LayoutParams(0, dp(56), 1));
        Button launch = new Button(this);
        launch.setText("Запустить движок");
        launch.setOnClickListener(view -> launchEngine(false));
        launchRow.addView(launch, new LinearLayout.LayoutParams(0, dp(56), 1));
        root.addView(launchRow, matchWrap());

        status = new TextView(this);
        status.setPadding(0, dp(8), 0, dp(8));
        root.addView(status, matchWrap());

        TextView logTitle = new TextView(this);
        logTitle.setText("Лог запуска движка");
        logTitle.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        root.addView(logTitle, matchWrap());

        logView = new TextView(this);
        logView.setTextSize(11);
        logView.setTypeface(Typeface.MONOSPACE);
        logView.setTextIsSelectable(true);
        logView.setPadding(dp(8), dp(8), dp(8), dp(8));
        ScrollView logScroll = new ScrollView(this);
        logScroll.addView(logView, new ScrollView.LayoutParams(-1, -2));
        root.addView(logScroll, new LinearLayout.LayoutParams(-1, 0, 1));

        setContentView(root);
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
                    .setMessage("OpenXRay должен читать файлы STALKER и писать лог в "
                            + "/storage/emulated/0/openxray/android.log. Android 11 и новее "
                            + "не показывают для этого обычное окно разрешения: включите "
                            + "«Разрешить управление всеми файлами» на системной странице приложения.")
                    .setPositiveButton("Открыть настройки", (dialog, which) -> requestAllFilesAccess())
                    .setNegativeButton("Позже", (dialog, which) ->
                            setStatus("Доступ не выдан. Для запуска игры откройте «Доступ к памяти»."))
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
        clearLogs();

        if (!rendererSmoke && !prepareGameRoot())
            return;

        String selectedPath = gamePath.getText().toString().trim();
        preferences.edit().putString(PREF_GAME_PATH, selectedPath).apply();

        writeLauncherLog("[launcher] starting "
                + (rendererSmoke ? "renderer smoke test" : "OpenXRay; game root=" + selectedPath));
        engineLaunchTime = SystemClock.elapsedRealtime();
        engineFailureToastShown = false;

        Intent intent = new Intent(this, XRayActivity.class);
        intent.putExtra(EXTRA_RENDERER_SMOKE, rendererSmoke);
        if (!rendererSmoke)
            intent.putExtra(EXTRA_GAME_PATH, selectedPath);
        setStatus(rendererSmoke ? "Запускаю GLES smoke test…" : "Запускаю OpenXRay…");
        Toast.makeText(this, "OpenXRay: запуск движка…", Toast.LENGTH_SHORT).show();
        try {
            startActivity(intent);
        } catch (RuntimeException error) {
            setStatus("Не удалось запустить процесс движка: " + error.getMessage());
            Toast.makeText(this, "OpenXRay: не удалось запустить движок", Toast.LENGTH_LONG).show();
        }
    }

    private boolean prepareGameRoot() {
        if (!hasStorageAccess()) {
            writeLauncherLog("[launcher] storage access is missing; STALKER folder check was not allowed");
            setStatus("Сначала нажмите «Доступ к памяти» и включите «Разрешить управление всеми файлами»."
                    + " Обычного разрешения при установке Android не показывает.");
            return false;
        }

        try {
            String selectedPath = gamePath.getText().toString().trim();
            if (selectedPath.isEmpty()) {
                writeLauncherLog("[launcher] STALKER folder not selected");
                setStatus("Папка STALKER не выбрана.");
                return false;
            }

            File root = new File(selectedPath);
            if (!root.isDirectory()) {
                writeLauncherLog("[launcher] STALKER folder not found: " + root.getAbsolutePath());
                setStatus("Папка игры не найдена: " + root);
                return false;
            }

            if (!validateCallOfPripyatResources(root))
                return false;

            File fsgame = new File(root, "fsgame.ltx");
            if (!fsgame.exists() && !copyBundledFsgame(fsgame)) {
                writeLauncherLog("[launcher] fsgame.ltx is missing and could not be created: " + fsgame);
                setStatus("В папке нет fsgame.ltx и не удалось создать его: " + fsgame);
                return false;
            }

            if (!installBundledEngineGamedata(root)) {
                writeLauncherLog("[launcher] OpenXRay engine gamedata could not be installed under "
                        + root.getAbsolutePath());
                setStatus("Не удалось подготовить gamedata OpenXRay. Проверьте доступ к папке игры.");
                return false;
            }
            writeLauncherLog("[launcher] STALKER folder validated: " + root.getAbsolutePath());
            return true;
        } catch (SecurityException error) {
            writeLauncherLog("[launcher] Android denied access to STALKER folder: "
                    + error.getClass().getSimpleName() + ": " + error.getMessage());
            setStatus("Android запретил доступ к папке игры: " + error.getMessage());
            return false;
        }
    }

    private boolean copyBundledFsgame(File destination) {
        try (InputStream input = getAssets().open("fsgame.ltx");
             OutputStream output = new FileOutputStream(destination)) {
            byte[] buffer = new byte[8192];
            int count;
            while ((count = input.read(buffer)) != -1)
                output.write(buffer, 0, count);
            return true;
        } catch (IOException | SecurityException error) {
            return false;
        }
    }

    private boolean validateCallOfPripyatResources(File root) {
        for (String directoryName : COP_RESOURCE_DIRECTORIES) {
            File directory = new File(root, directoryName);
            if (!directory.isDirectory()) {
                writeLauncherLog("[launcher] required CoP resource directory is missing: "
                        + directory.getAbsolutePath());
                setStatus("Не найдена папка ресурсов CoP: " + directoryName
                        + ". Нужны levels, localization, mp, patches и resources.");
                return false;
            }
            if (!hasDirectoryEntries(directory)) {
                writeLauncherLog("[launcher] required CoP resource directory is empty: "
                        + directory.getAbsolutePath());
                setStatus("Папка ресурсов CoP пустая: " + directoryName + ".");
                return false;
            }
        }
        return true;
    }

    private boolean hasDirectoryEntries(File directory) {
        File[] entries = directory.listFiles();
        return entries != null && entries.length != 0;
    }

    private boolean installBundledEngineGamedata(File gameRoot) {
        try {
            copyBundledAssetTree("gamedata", new File(gameRoot, "gamedata"));
            File configs = new File(gameRoot, "gamedata/configs");
            File shaders = new File(gameRoot, "gamedata/shaders");
            if (!configs.isDirectory() || !shaders.isDirectory()
                    || !hasDirectoryEntries(configs) || !hasDirectoryEntries(shaders)) {
                writeLauncherLog("[launcher] engine gamedata is incomplete after installation: "
                        + gameRoot.getAbsolutePath());
                return false;
            }
            writeLauncherLog("[launcher] OpenXRay engine gamedata is ready: "
                    + new File(gameRoot, "gamedata").getAbsolutePath());
            return true;
        } catch (IOException | SecurityException error) {
            writeLauncherLog("[launcher] cannot install engine gamedata: "
                    + error.getClass().getSimpleName() + ": " + error.getMessage());
            return false;
        }
    }

    private void copyBundledAssetTree(String assetPath, File destination) throws IOException {
        String[] children = getAssets().list(assetPath);
        if (children == null || children.length == 0) {
            if (destination.isFile())
                return;
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
        boolean granted = hasStorageAccess();
        accessStatus.setText("Доступ ко всей памяти: " + (granted ? "выдан" : "нужен")
                + "\nЛоги также читаются из app-specific fallback, если root storage закрыт.");
    }

    private void refreshLog() {
        if (logView == null)
            return;
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
        String log = result.toString();
        logView.setText(log.length() == 0 ? "Лог пока пуст. Запустите GLES-проверку или движок." : log);
        updateEngineStatus(log);
    }

    private void updateEngineStatus(String log) {
        if (engineLaunchTime == 0)
            return;

        if (log.contains("[renderer-smoke] center pixel") && log.contains(": PASS")) {
            setStatus("Движок загружен: GLES renderer smoke test PASS.");
            return;
        }
        if (log.contains("[android] engine loaded")) {
            setStatus("Движок загружен успешно.");
            return;
        }
        if (log.contains("[renderer-smoke] initialization failed")
                || log.contains("engine load failed")) {
            showEngineFailureStatus();
            return;
        }

        long elapsed = SystemClock.elapsedRealtime() - engineLaunchTime;
        if (elapsed > 4000 && !isEngineProcessRunning())
            showEngineFailureStatus();
    }

    private void showEngineFailureStatus() {
        setStatus("Движок завершился с ошибкой. Откройте полный лог ниже и logcat.");
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
        java.util.List<ActivityManager.RunningAppProcessInfo> processes = manager.getRunningAppProcesses();
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
        if (text.isEmpty())
            return;
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
}
