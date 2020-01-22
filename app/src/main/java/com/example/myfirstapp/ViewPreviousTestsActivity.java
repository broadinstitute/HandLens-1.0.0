package com.example.myfirstapp;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.FileProvider;

import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.StrictMode;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TableLayout;
import android.widget.TableRow;
import android.widget.TextView;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Random;

import static com.example.myfirstapp.SabetiCameraActivity.RESULTS_DIRECTORY;

public class ViewPreviousTestsActivity extends AppCompatActivity {

    private TableLayout table;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_view_previous_tests);

        File storageDir = getExternalFilesDir(null);
        File outputDirectory = new File(storageDir, RESULTS_DIRECTORY);

        File[] files = outputDirectory.listFiles();

        final TableLayout tableLayout = findViewById(R.id.resultsTable);
        android.widget.TableRow.LayoutParams layoutParams = new TableRow.LayoutParams(TableRow.LayoutParams.WRAP_CONTENT, TableRow.LayoutParams.WRAP_CONTENT);
        layoutParams.setMargins(0, 13, 35, 9);

        // Create header
        Typeface boldTypeface = Typeface.defaultFromStyle(Typeface.BOLD);
        final TableRow headerRow = new TableRow(this);
        headerRow.setLayoutParams(new TableLayout.LayoutParams(TableLayout.LayoutParams.WRAP_CONTENT, TableLayout.LayoutParams.WRAP_CONTENT));
        final TextView dateTextHeader = new TextView(this);
        dateTextHeader.setText("Date");
        dateTextHeader.setTypeface(boldTypeface);
        dateTextHeader.setTextSize(20);
        dateTextHeader.setLayoutParams(new TableRow.LayoutParams(TableRow.LayoutParams.WRAP_CONTENT, TableRow.LayoutParams.WRAP_CONTENT));
        final TextView idTextHeader = new TextView(this);
        idTextHeader.setText("Sample name");
        idTextHeader.setLayoutParams(new TableRow.LayoutParams(TableRow.LayoutParams.WRAP_CONTENT, TableRow.LayoutParams.WRAP_CONTENT));
        idTextHeader.setTypeface(boldTypeface);
        idTextHeader.setTextSize(20);
        final TextView resultTextHeader = new TextView(this);
        resultTextHeader.setText("Results");
        resultTextHeader.setLayoutParams(new TableRow.LayoutParams(TableRow.LayoutParams.WRAP_CONTENT, TableRow.LayoutParams.WRAP_CONTENT));
        resultTextHeader.setTypeface(boldTypeface);
        resultTextHeader.setTextSize(20);
        headerRow.addView(dateTextHeader, layoutParams);
        headerRow.addView(idTextHeader, layoutParams);
        headerRow.addView(resultTextHeader, layoutParams);
        tableLayout.addView(headerRow);


        for (int i = 0; i < files.length; i++) {
            android.content.Context context = this;
            // Get snapshot date
            String picDate = null;
            try {
                picDate = new SimpleDateFormat("MMM dd yyyy").format(
                        new SimpleDateFormat("yyyyMMdd_HHmmss")
                                .parse(Uri.parse(files[i].toString()).getLastPathSegment().substring(4)));
            } catch (Exception e) {
                Log.e("Camera", "exception", e);
            }

            // Find results file and image file:
            final File results_folder = files[i];
            File[] subfolder_files = results_folder.listFiles();
            String image_file = "", result_file = "", sample_name = "", image_fullpath = "";
            final String image_fullpath_final;
            final File image_file_final;
            File image_file_nonFinal = files[i];

            for (int j = 0; j < subfolder_files.length; j++) {
                String fileName = subfolder_files[j].getName();
                if (fileName.charAt(0) == 'I') { // images are saved as IMG_.jpg
                    image_fullpath = subfolder_files[j].toString();
                    image_file_nonFinal = subfolder_files[j];
                    image_file = fileName;
                    sample_name = fileName.substring(4, image_file.length() - 4);
                } else if (fileName.charAt(0) == 't') { // results in *.txt files
                    result_file = fileName;
                    sample_name = result_file.substring(0, result_file.length() - 4);
                }
            }
            image_fullpath_final = image_fullpath;
            image_file_final = image_file_nonFinal;

            // Creation row
            final TableRow tableRow = new TableRow(this);
            tableRow.setLayoutParams(new TableLayout.LayoutParams(TableLayout.LayoutParams.WRAP_CONTENT, TableLayout.LayoutParams.WRAP_CONTENT));

            // Creation textView
            final TextView dateText = new TextView(this);
            dateText.setText(picDate);
            dateText.setLayoutParams(new TableRow.LayoutParams(TableRow.LayoutParams.WRAP_CONTENT, TableRow.LayoutParams.WRAP_CONTENT));

            final TextView idText = new TextView(this);
            idText.setText(sample_name);
            idText.setLayoutParams(new TableRow.LayoutParams(TableRow.LayoutParams.WRAP_CONTENT, TableRow.LayoutParams.WRAP_CONTENT));

            final TextView resultText = new TextView(this);
            Random rand = new Random();
            int numb_samples = rand.nextInt(5) + 1;
            String result = "";
            for (int k = 0; k < numb_samples; k++){
                result = rand.nextBoolean() ? result + "+, " : result + "-, ";
            }
            result = result + "C";
            resultText.setText(result);
            resultText.setLayoutParams(new TableRow.LayoutParams(TableRow.LayoutParams.WRAP_CONTENT, TableRow.LayoutParams.WRAP_CONTENT));

            final TextView viewImageText = new TextView(this);
            viewImageText.setText("View image");
            viewImageText.setLayoutParams(new TableRow.LayoutParams(TableRow.LayoutParams.WRAP_CONTENT, TableRow.LayoutParams.WRAP_CONTENT));
            viewImageText.setTextColor(Color.BLUE);
            viewImageText.setPaintFlags(viewImageText.getPaintFlags() | Paint.UNDERLINE_TEXT_FLAG);
            viewImageText.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    final Intent intent = new Intent(Intent.ACTION_VIEW)//
                            .setDataAndType(Build.VERSION.SDK_INT >= Build.VERSION_CODES.N ?
                                            FileProvider.getUriForFile(context,getPackageName() + ".provider", image_file_final) : Uri.fromFile(image_file_final),
                                    "image/*").addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
                    startActivity(intent);
//                    Intent openImage = new Intent();
//                    openImage.setAction(Intent.ACTION_VIEW);
//                    openImage.setDataAndType(Uri.parse("file://" + image_fullpath_final), "image/*");
//                    openImage.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
//                    startActivity(openImage);
                }
            });

            // Creation  button
            final Button button = new Button(this);
            button.setText("Delete");
            button.setLayoutParams(new TableRow.LayoutParams(TableRow.LayoutParams.WRAP_CONTENT, TableRow.LayoutParams.WRAP_CONTENT));
            button.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    AlertDialog.Builder builder = new AlertDialog.Builder(context);
                    builder.setTitle("Confirm");
                    builder.setMessage("Are you sure?");
                    builder.setPositiveButton("YES", new DialogInterface.OnClickListener() {
                        public void onClick(DialogInterface dialog, int which) {
                            // Do nothing but close the dialog
                            final TableRow parent = (TableRow) v.getParent();
                            tableLayout.removeView(parent);
                            File[] files_in_folder = results_folder.listFiles();
                            for (int j = 0; j < files_in_folder.length; j++ ) {
                                if (files_in_folder[j].delete()) {
                                    Log.v("ViewPreviousTestsAct", "file Deleted :" + files_in_folder[j].toString());
                                } else {
                                    Log.v("ViewPreviousTestsAct", "file not Deleted :" + files_in_folder[j].toString());
                                }
                            }
                            if (results_folder.delete()) {
                                Log.v("ViewPreviousTestsAct", "folder Deleted :" + results_folder.toString());
                            } else {
                                Log.v("ViewPreviousTestsAct", "folder not Deleted :" + results_folder.toString());
                            }
                            dialog.dismiss();
                        }
                    });
                    builder.setNegativeButton("NO", new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {
                            // Do nothing
                            dialog.dismiss();
                        }
                    });

                    AlertDialog alert = builder.create();
                    alert.show();
                }
            });

            tableRow.addView(dateText, layoutParams);
            tableRow.addView(idText, layoutParams);
            tableRow.addView(resultText, layoutParams);
            tableRow.addView(viewImageText, layoutParams);
            tableRow.addView(button, layoutParams);

            tableLayout.addView(tableRow);
        }


    }
}
