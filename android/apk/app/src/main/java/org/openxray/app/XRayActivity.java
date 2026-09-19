package org.openxray.app;

import android.os.Build;
import android.os.Bundle;
import android.util.Log;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

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
        String[] args = new String[] { "-renderer-smoke", "-nogame", "-no_gamepad", "-nosplash" };
        writeDiagnostic("native arguments: " + String.join(" ", args));
        return args;
    }

    private File createDiagnosticsFile() {
        File root = getExternalFilesDir("openxray");
        if (root == null) {
            root = new File(getFilesDir(), "openxray");
        }
        if (!root.exists() && !root.mkdirs()) {
            Log.e(TAG, "Unable to create diagnostics directory: " + root);
        }
        return new File(root, "android.log");
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
