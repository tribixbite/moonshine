package com.usefulsensors.moonshine;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

public class FilePickerFragment extends Fragment {

    private static final int REQUEST_CODE_PICK_AUDIO = 1;
    private static final int REQUEST_CODE_PICK_VIDEO = 2;

    private TextView statusTextView;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_file_picker, container, false);

        statusTextView = view.findViewById(R.id.statusTextView);

        Button pickAudioButton = view.findViewById(R.id.pickAudioButton);
        pickAudioButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                pickAudioFile();
            }
        });

        Button pickVideoButton = view.findViewById(R.id.pickVideoButton);
        pickVideoButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                pickVideoFile();
            }
        });

        return view;
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
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == getActivity().RESULT_OK && data != null) {
            Uri uri = data.getData();
            if (requestCode == REQUEST_CODE_PICK_AUDIO) {
                transcribeAudio(uri);
            } else if (requestCode == REQUEST_CODE_PICK_VIDEO) {
                transcribeVideo(uri);
            }
        }
    }

    private void transcribeAudio(Uri uri) {
        Intent intent = new Intent(getActivity(), TranscriptionService.class);
        intent.setData(uri);
        intent.putExtra("type", "audio");
        getActivity().startService(intent);
        statusTextView.setText("Transcribing audio...");
    }

    private void transcribeVideo(Uri uri) {
        Intent intent = new Intent(getActivity(), TranscriptionService.class);
        intent.setData(uri);
        intent.putExtra("type", "video");
        getActivity().startService(intent);
        statusTextView.setText("Transcribing video...");
    }
}
