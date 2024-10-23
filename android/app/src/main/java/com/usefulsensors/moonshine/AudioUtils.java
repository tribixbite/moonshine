package com.usefulsensors.moonshine;

import android.content.Context;
import android.net.Uri;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class AudioUtils {

    private static final String TAG = "AudioUtils";

    public static File getFileFromUri(Context context, Uri uri) throws IOException {
        InputStream inputStream = context.getContentResolver().openInputStream(uri);
        File tempFile = File.createTempFile("audio", null, context.getCacheDir());
        FileOutputStream outputStream = new FileOutputStream(tempFile);

        byte[] buffer = new byte[1024];
        int length;
        while ((length = inputStream.read(buffer)) > 0) {
            outputStream.write(buffer, 0, length);
        }

        outputStream.close();
        inputStream.close();

        return tempFile;
    }

    public static String transcribeAudio(File audioFile) {
        // Placeholder for actual transcription logic
        Log.i(TAG, "Transcribing audio file: " + audioFile.getAbsolutePath());
        return "Transcription result";
    }
}
