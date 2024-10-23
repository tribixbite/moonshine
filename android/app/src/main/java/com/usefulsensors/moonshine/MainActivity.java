package com.usefulsensors.moonshine;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_CODE_PICK_AUDIO = 1;
    private static final int REQUEST_CODE_PICK_VIDEO = 2;
    private static final int REQUEST_CODE_PERMISSIONS = 3;

    private TextView statusTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        statusTextView = findViewById(R.id.statusTextView);

        Button pickAudioButton = findViewById(R.id.pickAudioButton);
        pickAudioButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                pickAudioFile();
            }
        });

        Button pickVideoButton = findViewById(R.id.pickVideoButton);
        pickVideoButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                pickVideoFile();
            }
        });

        if (!hasPermissions()) {
            requestPermissions();
        }
    }

    private boolean hasPermissions() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;
    }

    private void requestPermissions() {
        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, REQUEST_CODE_PERMISSIONS);
    }

    private void pickAudioFile() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Audio.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, REQUEST_CODE_PICK_AUDIO);
    }

    private void pickVideoFile() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Video.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, REQUEST_CODE_PICK_VIDEO);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && data != null) {
            Uri uri = data.getData();
            if (requestCode == REQUEST_CODE_PICK_AUDIO) {
                transcribeAudio(uri);
            } else if (requestCode == REQUEST_CODE_PICK_VIDEO) {
                transcribeVideo(uri);
            }
        }
    }

    private void transcribeAudio(Uri uri) {
        Intent intent = new Intent(this, TranscriptionService.class);
        intent.setData(uri);
        intent.putExtra("type", "audio");
        startService(intent);
        statusTextView.setText("Transcribing audio...");
    }

    private void transcribeVideo(Uri uri) {
        Intent intent = new Intent(this, TranscriptionService.class);
        intent.setData(uri);
        intent.putExtra("type", "video");
        startService(intent);
        statusTextView.setText("Transcribing video...");
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // Permissions granted
            } else {
                // Permissions denied
                statusTextView.setText("Permissions denied. Cannot access files.");
            }
        }
    }
}
