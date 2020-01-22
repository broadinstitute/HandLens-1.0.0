package com.example.myfirstapp;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.text.Html;
import android.util.Log;
import android.widget.TextView;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.net.HttpURLConnection;
import java.net.URL;

public class ResultsPageActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_results_page);

        // Get the Intent that started this activity and extract the string
        Intent intent = getIntent();
        String message = intent.getStringExtra(SabetiCameraActivity.IMAGE_FILE_NAME);

        // Capture the layout's TextView and set the string as its text
        TextView textView = findViewById(R.id.textView2);
        textView.setText(message);

        String charset = "UTF-8";
        String requestURL = "http://35.232.84.84:3000/upload";

        try {
            // disable online uploading functionality temporarily while front-end work is goign on.
//            String response = uploadFile(message);
            String response = "OK";
            textView.setText(message + " succesfully uploaded\n\n" + response);
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                textView.setText(Html.fromHtml("<h2>Results</h2><br><p>Strip 1: Positive</p><p>Strip 2: Positive</p><p>Strip 3: Control</p>", Html.FROM_HTML_MODE_COMPACT));
            } else {
                textView.setText(Html.fromHtml("<p>Strip 1: Positive</p><p>Strip2: Positive here</p><p>Strip3: Control</p>"));
            }
        } catch (Exception e) {
            //textView.setText(message + " upload failed");
            textView.setText(e.getMessage());
            Log.e("Camera", "exception", e);
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                textView.setText(Html.fromHtml("<h2>Results</h2><br><p>Strip 1: Positive</p><p>Strip2: Positive here</p><p>Strip3: Control</p>", Html.FROM_HTML_MODE_COMPACT));
            } else {
                textView.setText(Html.fromHtml("<p>Strip 1: Positive</p><p>Strip2: Positive here</p><p>Strip3: Control</p>"));
            }
        }


    }

    public String uploadFile(String imagePath) throws Exception {
        String fileName = imagePath;
        HttpURLConnection conn = null;
        DataOutputStream dos = null;
        String lineEnd = "\r\n";
        String twoHyphens = "--";
        String boundary = "*****";
        int bytesRead, bytesAvailable, bufferSize;
        byte[] buffer;
        int maxBufferSize = 1 * 1024 * 1024;
        File sourceFile = new File(imagePath);

        // open a URL connection to the Servlet
        FileInputStream fileInputStream = new FileInputStream(sourceFile);
        URL url = new URL("http://35.232.84.84:3000/upload");

        // Open a HTTP  connection to  the URL
        conn = (HttpURLConnection) url.openConnection();
        conn.setDoInput(true); // Allow Inputs
        conn.setDoOutput(true); // Allow Outputs
        conn.setUseCaches(false); // Don't use a Cached Copy
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Connection", "Keep-Alive");
        conn.setRequestProperty("ENCTYPE", "multipart/form-data");
        conn.setRequestProperty("Content-Type", "multipart/form-data;boundary=" + boundary);
        conn.setRequestProperty("upload", fileName);

        dos = new DataOutputStream(conn.getOutputStream());

        dos.writeBytes(twoHyphens + boundary + lineEnd);
        dos.writeBytes("Content-Disposition: form-data; name=\"upload\";filename=\""
                + fileName + "\"" + lineEnd);

        dos.writeBytes(lineEnd);

        // create a buffer of  maximum size
        bytesAvailable = fileInputStream.available();

        bufferSize = Math.min(bytesAvailable, maxBufferSize);
        buffer = new byte[bufferSize];

        // read file and write it into form...
        bytesRead = fileInputStream.read(buffer, 0, bufferSize);

        while (bytesRead > 0) {

            dos.write(buffer, 0, bufferSize);
            bytesAvailable = fileInputStream.available();
            bufferSize = Math.min(bytesAvailable, maxBufferSize);
            bytesRead = fileInputStream.read(buffer, 0, bufferSize);
        }

        // send multipart form data necesssary after file data...
        dos.writeBytes(lineEnd);
        dos.writeBytes(twoHyphens + boundary + twoHyphens + lineEnd);

        // Responses from the server (code and message)
        String serverResponseMessage = conn.getResponseMessage();

        Log.v("uploadFile", "HTTP Response is : "
                + serverResponseMessage);

        //close the streams
        fileInputStream.close();
        dos.flush();
        dos.close();

        return serverResponseMessage;
    }

    public Boolean saveResultsFile(String imagePath, String serverResponse){
        File image_file = new File(imagePath);
        String image_name = image_file.getName();
        String sample_name = image_name.substring(0, image_name.length() - 4);
        File results_file = new File(image_file.getParentFile().toString(), sample_name + ".txt");

        // TODO: save file.

        return Boolean.TRUE;
    }
}
