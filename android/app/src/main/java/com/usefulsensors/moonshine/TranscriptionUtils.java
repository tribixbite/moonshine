package com.usefulsensors.moonshine;

import android.util.Log;

import org.json.JSONObject;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class TranscriptionUtils {

    private static final String TAG = "TranscriptionUtils";

    public static void saveToFile(String transcription, File file) throws IOException {
        FileWriter writer = new FileWriter(file);
        writer.write(transcription);
        writer.close();
    }

    public static void saveToMd(String transcription, File file) throws IOException {
        String mdContent = "# Transcription\n\n" + transcription;
        saveToFile(mdContent, file);
    }

    public static void saveToTxt(String transcription, File file) throws IOException {
        saveToFile(transcription, file);
    }

    public static void saveToJson(String transcription, File file) throws IOException {
        JSONObject jsonObject = new JSONObject();
        jsonObject.put("transcription", transcription);
        saveToFile(jsonObject.toString(), file);
    }

    public static String convertToMd(String transcription) {
        return "# Transcription\n\n" + transcription;
    }

    public static String convertToTxt(String transcription) {
        return transcription;
    }

    public static String convertToJson(String transcription) {
        JSONObject jsonObject = new JSONObject();
        jsonObject.put("transcription", transcription);
        return jsonObject.toString();
    }
}
