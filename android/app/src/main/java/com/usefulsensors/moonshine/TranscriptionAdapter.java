package com.usefulsensors.moonshine;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class TranscriptionAdapter extends RecyclerView.Adapter<TranscriptionAdapter.TranscriptionViewHolder> {

    private List<TranscriptionItem> transcriptionItems;

    public TranscriptionAdapter(List<TranscriptionItem> transcriptionItems) {
        this.transcriptionItems = transcriptionItems;
    }

    @NonNull
    @Override
    public TranscriptionViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.item_transcription, parent, false);
        return new TranscriptionViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull TranscriptionViewHolder holder, int position) {
        TranscriptionItem item = transcriptionItems.get(position);
        holder.transcriptionTextView.setText(item.getTranscription());
        holder.timestampTextView.setText(formatTimestamp(item.getTimestamp()));
    }

    @Override
    public int getItemCount() {
        return transcriptionItems.size();
    }

    private String formatTimestamp(long timestamp) {
        SimpleDateFormat sdf = new SimpleDateFormat("HH:mm:ss", Locale.getDefault());
        return sdf.format(new Date(timestamp));
    }

    static class TranscriptionViewHolder extends RecyclerView.ViewHolder {

        TextView transcriptionTextView;
        TextView timestampTextView;

        public TranscriptionViewHolder(@NonNull View itemView) {
            super(itemView);
            transcriptionTextView = itemView.findViewById(R.id.transcriptionTextView);
            timestampTextView = itemView.findViewById(R.id.timestampTextView);
        }
    }
}
