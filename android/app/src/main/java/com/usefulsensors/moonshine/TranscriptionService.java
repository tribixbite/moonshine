package com.usefulsensors.moonshine;

import android.app.Service;
import android.content.Intent;
import android.net.Uri;
import android.os.IBinder;
import android.util.Log;
import androidx.annotation.Nullable;

import java.io.File;
import java.io.IOException;

public class TranscriptionService extends Service {

    private static final String TAG = "TranscriptionService";

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        Uri uri = intent.getData();
        String type = intent.getStringExtra("type");

        if (uri != null && type != null) {
            new Thread(() -> {
                try {
                    if (type.equals("audio")) {
                        handleAudioTranscription(uri);
                    } else if (type.equals("video")) {
                        handleVideoTranscription(uri);
                    }
                } catch (IOException e) {
                    Log.e(TAG, "Transcription failed", e);
                }
            }).start();
        }

        return START_STICKY;
    }

    private void handleAudioTranscription(Uri uri) throws IOException {
        File audioFile = AudioUtils.getFileFromUri(this, uri);
        String transcription = AudioUtils.transcribeAudio(audioFile);
        saveTranscription(transcription, "audio_transcription");
    }

    private void handleVideoTranscription(Uri uri) throws IOException {
        File videoFile = VideoUtils.getFileFromUri(this, uri);
        String transcription = VideoUtils.transcribeVideo(videoFile);
        saveTranscription(transcription, "video_transcription");
    }

    private void saveTranscription(String transcription, String fileName) {
        File file = new File(getExternalFilesDir(null), fileName + ".txt");
        try {
            TranscriptionUtils.saveToFile(transcription, file);
            Log.i(TAG, "Transcription saved: " + file.getAbsolutePath());
        } catch (IOException e) {
            Log.e(TAG, "Failed to save transcription", e);
        }
    }
}
