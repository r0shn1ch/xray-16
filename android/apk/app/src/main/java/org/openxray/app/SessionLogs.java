package org.openxray.app;

import android.content.Context;
import android.system.Os;
import android.util.AtomicFile;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

final class SessionLogs {
    private SessionLogs() {}

    private static AtomicFile pointer(Context context) {
        return new AtomicFile(new File(context.getFilesDir(), "last-engine-log.txt"));
    }

    static File start(Context context, String gamePath) throws IOException {
        File root = gamePath == null || gamePath.isEmpty()
                ? new File(context.getFilesDir(), "openxray/logs")
                : new File(gamePath, "_appdata_/logs");
        if (!root.isDirectory() && !root.mkdirs())
            throw new IOException("Cannot create log directory: " + root);
        File activity = File.createTempFile("session-" + System.currentTimeMillis() + "-", ".activity.log", root);
        File engine = new File(activity.getPath().replace(".activity.log", ".engine.log"));
        if (!engine.createNewFile())
            throw new IOException("Log session already exists: " + engine);
        try {
            // Set before SDL loads libmain, including its crash-handler constructors.
            Os.setenv("OPENXRAY_ENGINE_LOG", engine.getAbsolutePath(), true);
        } catch (android.system.ErrnoException error) {
            throw new IOException(error);
        }
        AtomicFile pointer = pointer(context);
        FileOutputStream output = null;
        try {
            output = pointer.startWrite();
            output.write(engine.getAbsolutePath().getBytes(StandardCharsets.UTF_8));
            pointer.finishWrite(output);
        } catch (IOException error) {
            if (output != null) pointer.failWrite(output);
            throw error;
        }
        return activity;
    }

    static File[] latest(Context context) {
        try {
            String path = new String(pointer(context).readFully(), StandardCharsets.UTF_8);
            if (!path.endsWith(".engine.log")) return new File[0];
            return new File[] {new File(path), new File(path.replace(".engine.log", ".activity.log"))};
        } catch (IOException error) {
            return new File[0];
        }
    }
}
