package com.example.myfirstapp;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.ImageView;
import android.widget.Toast;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

import static com.example.myfirstapp.MainActivity.EXTRA_MESSAGE;

public class SabetiLaunchCameraAppActivity extends AppCompatActivity {
    private ImageView imageView;
    File photoFile = null;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_sabeti_launch_camera_app);

        imageView = (ImageView) findViewById(R.id.cameraCaptureimageView);

        if (allPermissionsGranted()) {
            // Get the Intent that started this activity and extract the string
            Intent intent = getIntent();
            String message = intent.getStringExtra(EXTRA_MESSAGE);
            dispatchTakePictureIntent(message);
        } else {
            ActivityCompat.requestPermissions(this,
                    MainActivity.REQUIRED_PERMISSIONS,
                    MainActivity.REQUEST_CODE_PERMISSIONS);
        }
    }

    static final int REQUEST_IMAGE_CAPTURE = 1;

    private void dispatchTakePictureIntent(String sampleName) {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        // Ensure that there's a camera activity to handle the intent
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            // Create the File where the photo should go
            try {
                photoFile = createImageFile(sampleName);
            } catch (IOException ex) {
                // Error occurred while creating the File
            }
            // Continue only if the File was successfully created
            if (photoFile != null) {
                Uri photoURI = FileProvider.getUriForFile(this,
                        "com.example.myfirstapp.provider",
                        photoFile);
                takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
                startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
            }
        }

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
//            Bundle extras = data.getExtras();
//            Bitmap imageBitmap = (Bitmap) extras.get("data");
//            imageView.setImageBitmap(imageBitmap);

            Intent intent = new Intent(this, ImageViewBoxSelectActivity.class);
            intent.putExtra(EXTRA_MESSAGE, photoFile.getAbsolutePath());
            startActivity(intent);
//
//            Log.d("SabetiLaunchCameraAp...", "setting ImageView");
//            if (photoFile != null && photoFile.exists()) {
//                Bitmap sourceImage = BitmapFactory.decodeFile(photoFile.getAbsolutePath());
//                Matrix rotationMatrix = new Matrix();
//                rotationMatrix.postRotate(getCameraPhotoOrientation(this,
//                        FileProvider.getUriForFile(this,
//                        "com.example.myfirstapp.provider",
//                        photoFile),
//                        photoFile.getAbsolutePath()));
//                imageView.setImageBitmap(Bitmap.createBitmap(sourceImage, 0, 0,
//                        sourceImage.getWidth(), sourceImage.getHeight(), rotationMatrix, true));
////                imageView.setImageBitmap(BitmapFactory.decodeFile(photoFile.getAbsolutePath()));
////                imageView.setImageURI(Uri.fromFile(photoFile));
////                imageView.setRotation(getCameraPhotoOrientation(this,
////                        FileProvider.getUriForFile(this,
////                        "com.example.myfirstapp.provider",
////                        photoFile),
////                        photoFile.getAbsolutePath()));
//            }

        }
    }

    public static int getCameraPhotoOrientation(Context context, Uri imageUri,
                                         String imagePath) {
        // source: https://stackoverflow.com/a/36995847
        // MIT license (https://meta.stackexchange.com/questions/271080)
        int rotate = 0;
        try {
            context.getContentResolver().notifyChange(imageUri, null);
            File imageFile = new File(imagePath);
            ExifInterface exif = new ExifInterface(imageFile.getAbsolutePath());
            int orientation = exif.getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL);

            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_270:
                    rotate = 270;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_180:
                    rotate = 180;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_90:
                    rotate = 90;
                    break;
            }

            Log.i("RotateImage", "Exif orientation: " + orientation);
            Log.i("RotateImage", "Rotate value: " + rotate);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return rotate;
    }


    private File createImageFile(String sampleName) throws IOException {
        // Create an image file name
        String timeStamp =
                new SimpleDateFormat(MainActivity.CAMERA_DATE_FORMAT,
                        Locale.getDefault()).format(new Date());
        File storageDir =
                getExternalFilesDir(null);
        String imageFileName = "IMG_" + sampleName;
        File outputDirectory = new File(storageDir, MainActivity.RESULTS_DIRECTORY + "/IMG_" + timeStamp);
//                String fileName = storageDir + "/results/" + imageFileName + ".jpg";
        if (!outputDirectory.exists()) {
            if (!outputDirectory.mkdirs()) {
                        Log.e("SabetiLaunchcameraAp...",
                                "Failed to create directory: " + outputDirectory.getAbsolutePath());
                outputDirectory = null;
            }
        }
        return new File(outputDirectory, imageFileName + ".jpg");
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {

        if (requestCode == MainActivity.REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                dispatchTakePictureIntent(getIntent().getStringExtra(EXTRA_MESSAGE));
            } else {
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }

    @Override
    public void onBackPressed() {
        startActivity(new Intent(this, MainActivity.class));
    }


    public boolean allPermissionsGranted() {

        for (String permission : MainActivity.REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

}
