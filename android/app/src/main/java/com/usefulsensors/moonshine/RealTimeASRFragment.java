package com.usefulsensors.moonshine;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import java.util.ArrayList;
import java.util.List;

public class RealTimeASRFragment extends Fragment {

    private static final int REQUEST_CODE_PERMISSIONS = 1;
    private Button startASRButton;
    private Button stopASRButton;
    private RecyclerView transcriptionRecyclerView;
    private TranscriptionAdapter transcriptionAdapter;
    private List<TranscriptionItem> transcriptionItems;
    private Handler handler;
    private Runnable asrRunnable;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_real_time_asr, container, false);

        startASRButton = view.findViewById(R.id.startASRButton);
        stopASRButton = view.findViewById(R.id.stopASRButton);
        transcriptionRecyclerView = view.findViewById(R.id.transcriptionRecyclerView);

        transcriptionItems = new ArrayList<>();
        transcriptionAdapter = new TranscriptionAdapter(transcriptionItems);
        transcriptionRecyclerView.setLayoutManager(new LinearLayoutManager(getContext()));
        transcriptionRecyclerView.setAdapter(transcriptionAdapter);

        startASRButton.setOnClickListener(v -> startRealTimeASR());
        stopASRButton.setOnClickListener(v -> stopRealTimeASR());

        handler = new Handler(Looper.getMainLooper());

        if (!hasPermissions()) {
            requestPermissions();
        }

        return view;
    }

    private boolean hasPermissions() {
        return ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED;
    }

    private void requestPermissions() {
        ActivityCompat.requestPermissions(requireActivity(), new String[]{Manifest.permission.RECORD_AUDIO}, REQUEST_CODE_PERMISSIONS);
    }

    private void startRealTimeASR() {
        Toast.makeText(getContext(), "Starting Real-Time ASR...", Toast.LENGTH_SHORT).show();
        asrRunnable = new Runnable() {
            @Override
            public void run() {
                // Placeholder for actual ASR logic
                String transcription = "Real-time transcription result with timestamp";
                long timestamp = System.currentTimeMillis();
                transcriptionItems.add(new TranscriptionItem(transcription, timestamp));
                transcriptionAdapter.notifyDataSetChanged();
                handler.postDelayed(this, 1000); // Simulate real-time ASR every second
            }
        };
        handler.post(asrRunnable);
    }

    private void stopRealTimeASR() {
        Toast.makeText(getContext(), "Stopping Real-Time ASR...", Toast.LENGTH_SHORT).show();
        handler.removeCallbacks(asrRunnable);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // Permissions granted
            } else {
                // Permissions denied
                Toast.makeText(getContext(), "Permissions denied. Cannot start real-time ASR.", Toast.LENGTH_SHORT).show();
            }
        }
    }
}
