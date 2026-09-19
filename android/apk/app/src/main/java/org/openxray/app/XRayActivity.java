package org.openxray.app;

import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.nio.charset.StandardCharsets;

import org.libsdl.app.SDLActivity;

/**
 * SDL entry point for the Android renderer bring-up APK.
 *
 * The default mode creates a real GLES 3.0 context and runs a native shader
 * smoke test without bundling proprietary game data. Java-side lifecycle and
 * uncaught-exception records are written to app-specific external storage so
 * a phone test remains diagnosable without root access.
 */
public final class XRayActivity extends SDLActivity {
    private static final String TAG = "OpenXRay";
    private File diagnosticsFile;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        diagnosticsFile = createDiagnosticsFile();
        installCrashHandler();
        writeDiagnostic("activity onCreate; sdk=" + Build.VERSION.SDK_INT
                + "; abi=" + Build.SUPPORTED_ABIS[0]);
    }

    @Override
    protected void onResume() {
        super.onResume();
        writeDiagnostic("activity onResume");
    }

    @Override
    protected void onPause() {
        writeDiagnostic("activity onPause");
        super.onPause();
    }

    @Override
    protected String[] getLibraries() {
        // SDL2 is linked statically into libmain.so for this APK.
        return new String[] { "main" };
    }

    @Override
    protected String[] getArguments() {
        boolean rendererSmoke = getIntent().getBooleanExtra(LauncherActivity.EXTRA_RENDERER_SMOKE, true);
        String selectedPath = getIntent().getStringExtra(LauncherActivity.EXTRA_GAME_PATH);
        String[] args;
        if (rendererSmoke) {
            args = new String[] { "-renderer-smoke", "-nogame", "-no_gamepad", "-nosplash" };
        } else if (selectedPath != null && !selectedPath.isEmpty()) {
            args = new String[] { "-android-game-root-hex", encodeHex(selectedPath),
                    "-no_gamepad", "-nosplash" };
        } else {
            args = new String[] { "-headless-smoke", "-no_gamepad", "-nosplash" };
        }
        writeDiagnostic("native arguments: " + String.join(" ", args));
        return args;
    }

    private String encodeHex(String value) {
        byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
        StringBuilder result = new StringBuilder(bytes.length * 2);
        for (byte item : bytes)
            result.append(String.format(Locale.US, "%02x", item & 0xff));
        return result.toString();
    }

    private File createDiagnosticsFile() {
        File publicFile = new File(Environment.getExternalStorageDirectory(), "openxray/activity.log");
        if (canAppend(publicFile))
            return publicFile;

        File root = getExternalFilesDir("openxray");
        if (root == null)
            root = new File(getFilesDir(), "openxray");
        if (!root.exists() && !root.mkdirs())
            Log.e(TAG, "Unable to create diagnostics directory: " + root);
        return new File(root, "activity.log");
    }

    private boolean canAppend(File file) {
        File parent = file.getParentFile();
        if (parent == null || (!parent.exists() && !parent.mkdirs()))
            return false;

        try (FileOutputStream stream = new FileOutputStream(file, true)) {
            return true;
        } catch (IOException error) {
            Log.w(TAG, "Shared-storage diagnostics unavailable: " + file, error);
            return false;
        }
    }

    private void installCrashHandler() {
        final Thread.UncaughtExceptionHandler previous = Thread.getDefaultUncaughtExceptionHandler();
        Thread.setDefaultUncaughtExceptionHandler((thread, throwable) -> {
            writeDiagnostic("uncaught Java exception on " + thread.getName());
            writeDiagnostic(Log.getStackTraceString(throwable));
            Log.e(TAG, "uncaught Java exception", throwable);
            if (previous != null) {
                previous.uncaughtException(thread, throwable);
            }
        });
    }

    private synchronized void writeDiagnostic(String message) {
        Log.i(TAG, message);
        if (diagnosticsFile == null) {
            diagnosticsFile = createDiagnosticsFile();
        }
        try (PrintWriter writer = new PrintWriter(new FileWriter(diagnosticsFile, true))) {
            String timestamp = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US)
                    .format(new Date());
            writer.println(timestamp + " " + message);
        } catch (IOException error) {
            Log.e(TAG, "Unable to write diagnostics: " + diagnosticsFile, error);
        }
    }
}
