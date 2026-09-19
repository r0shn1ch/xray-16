package org.openxray.app;

import android.app.Activity;
import android.app.ActivityManager;
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
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

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
    private static final int MAX_LOG_BYTES = 180 * 1024;

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

    private void buildInterface() {
        final int padding = dp(16);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(padding, padding, padding, padding);

        TextView title = new TextView(this);
        title.setText("OpenXRay Launcher");
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
        startActivityForResult(intent, REQUEST_GAME_TREE);
    }

    private void requestAllFilesAccess() {
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
        } catch (SecurityException error) {
            startActivity(new Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION));
        }
    }

    private void launchEngine(boolean rendererSmoke) {
        if (!rendererSmoke && !prepareGameRoot())
            return;

        String selectedPath = gamePath.getText().toString().trim();
        preferences.edit().putString(PREF_GAME_PATH, selectedPath).apply();

        clearLogs();
        engineLaunchTime = SystemClock.elapsedRealtime();
        engineFailureToastShown = false;

        Intent intent = new Intent(this, XRayActivity.class);
        intent.putExtra(EXTRA_RENDERER_SMOKE, rendererSmoke);
        if (!rendererSmoke)
            intent.putExtra(EXTRA_GAME_PATH, selectedPath);
        setStatus(rendererSmoke ? "Запускаю GLES smoke test…" : "Запускаю OpenXRay…");
        Toast.makeText(this, "OpenXRay: запуск движка…", Toast.LENGTH_SHORT).show();
        startActivity(intent);
    }

    private boolean prepareGameRoot() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R && !Environment.isExternalStorageManager()) {
            setStatus("Сначала нажмите «Доступ к памяти» и включите «Разрешить управление всеми файлами»."
                    + " Обычного разрешения при установке Android не показывает.");
            return false;
        }

        File root = new File(gamePath.getText().toString().trim());
        if (!root.isDirectory()) {
            setStatus("Папка игры не найдена: " + root);
            return false;
        }

        File fsgame = new File(root, "fsgame.ltx");
        if (!fsgame.exists() && !copyBundledFsgame(fsgame)) {
            setStatus("В папке нет fsgame.ltx и не удалось создать его: " + fsgame);
            return false;
        }
        if (!new File(root, "gamedata").isDirectory())
            setStatus("Предупреждение: в папке нет gamedata/. Запуск всё равно продолжится.");
        return true;
    }

    private boolean copyBundledFsgame(File destination) {
        try (InputStream input = getAssets().open("fsgame.ltx");
             OutputStream output = new FileOutputStream(destination)) {
            byte[] buffer = new byte[8192];
            int count;
            while ((count = input.read(buffer)) != -1)
                output.write(buffer, 0, count);
            return true;
        } catch (IOException error) {
            return false;
        }
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
        boolean granted = Build.VERSION.SDK_INT < Build.VERSION_CODES.R
                || Environment.isExternalStorageManager();
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
