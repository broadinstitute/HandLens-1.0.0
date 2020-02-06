package com.example.myfirstapp;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.camera2.impl.Camera2CaptureRequestBuilder;
import androidx.camera.core.CameraX;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureConfig;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.PreviewConfig;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.Log;
import android.util.Rational;
import android.util.Size;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageButton;
import android.widget.Toast;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Collections;
import java.util.Date;
import java.util.Locale;

import static com.example.myfirstapp.MainActivity.EXTRA_MESSAGE;


public class SabetiCameraActivity extends AppCompatActivity {

    public static int REQUEST_CODE_PERMISSIONS = 101;
    public static final String[] REQUIRED_PERMISSIONS = new String[]{"android.permission.CAMERA",
            "android.permission.WRITE_EXTERNAL_STORAGE",
            "android.permission.READ_EXTERNAL_STORAGE",
            "android.permission.INTERNET",
            "android.permission.ACCESS_NETWORK_STATE",
            "android.permission.ACCESS_WIFI_STATE"};
    public static final String IMAGE_FILE_NAME = "IMAGE_FILE_NAME";
    TextureView textureView;
    public static final String CAMERA_DATE_FORMAT = "yyyyMMdd_HHmmss";
    public static final String RESULTS_DIRECTORY = "/results";
    private long mLastAnalysisResultTime;
    private double exposure_required = 1;

    private int viewHeight;
    private int viewWidth;

    private int rearCameradId;
    String cameraIdFacing;
    int cameraFacing;
    Size previewSize;
    CameraManager manager;
    HandlerThread backgroundThread;
    Handler backgroundHandler;
    CameraDevice.StateCallback stateCallback;
    CameraDevice cameraDevice;
    TextureView.SurfaceTextureListener surfaceTextureListener;
    CaptureRequest captureRequest;
    CameraCaptureSession cameraCaptureSession;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);

        // set instance variables
        textureView = findViewById(R.id.view_finder);
        textureView.measure(View.MeasureSpec.UNSPECIFIED, View.MeasureSpec.UNSPECIFIED);
        viewWidth = textureView.getMeasuredWidth();
        viewHeight = textureView.getMeasuredHeight();
        mLastAnalysisResultTime = SystemClock.elapsedRealtime();
        cameraFacing = CameraCharacteristics.LENS_FACING_BACK;
        manager = (CameraManager) getSystemService(CAMERA_SERVICE);

        // get sample name
        Intent startingIntent = getIntent();
        String sampleName = startingIntent.getStringExtra(EXTRA_MESSAGE);

        Log.d("SabetiCameraActivity/",
                "imageView.getWidth()" + Integer.toString(textureView.getWidth())
                        + " imageView.getHeight()" + Integer.toString(textureView.getHeight()));

        ImageButton imageCaptureView = findViewById(R.id.imgCapture);
        Intent nextIntent = new Intent(this, ImageViewBoxSelectActivity.class);

        imageCaptureView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String timeStamp =
                        new SimpleDateFormat(CAMERA_DATE_FORMAT,
                                Locale.getDefault()).format(new Date());
                File storageDir =
                        getExternalFilesDir(null);
                String imageFileName = "IMG_" + sampleName;
                File outputDirectory = new File(storageDir, RESULTS_DIRECTORY + "/IMG_" + timeStamp);
//                String fileName = storageDir + "/results/" + imageFileName + ".jpg";
                if (!outputDirectory.exists()) {
                    if (!outputDirectory.mkdirs()) {
//                        Log.e(LogHelper.LogTag, "Failed to create directory: " + outputDirectory.getAbsolutePath());
                        outputDirectory = null;
                    }
                }
                File imageFile = new File(outputDirectory, imageFileName + ".jpg");
                String fileName = imageFile.toString();

                FileOutputStream outputPhoto = null;
                try {
                    outputPhoto = new FileOutputStream(imageFile);
                    textureView.getBitmap()
                            .compress(Bitmap.CompressFormat.PNG, 100, outputPhoto);
                    nextIntent.putExtra(EXTRA_MESSAGE, imageFile.getAbsolutePath());
                    startActivity(nextIntent);
                } catch (Exception e) {
                    e.printStackTrace();
                } finally {
                    try {
                        if (outputPhoto != null) {
                            outputPhoto.close();
                        }
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }

            }
        });

        if (allPermissionsGranted()) {
            // Get the Intent that started this activity and extract the string
            Intent intent = getIntent();
            String message = intent.getStringExtra(MainActivity.EXTRA_MESSAGE);

            surfaceTextureListener = new TextureView.SurfaceTextureListener() {

                @Override
                public void onSurfaceTextureUpdated(SurfaceTexture surfaceTexture) {

                }

                @Override
                public void onSurfaceTextureSizeChanged(SurfaceTexture surfaceTexture, int width, int height) {

                }

                @Override
                public void onSurfaceTextureAvailable(SurfaceTexture surfaceTexture, int width, int height) {
                    setUpCamera();
                    openCamera();
                }

                @Override
                public boolean onSurfaceTextureDestroyed(SurfaceTexture surfaceTexture) {
                    return false;
                }
            };

            stateCallback = new CameraDevice.StateCallback() {
                @Override
                public void onOpened(CameraDevice cd) {
                    cameraDevice = cd;
                    createPreviewSession();
                }

                @Override
                public void onDisconnected(CameraDevice cd) {
                    cameraDevice.close();
                    cameraDevice = null;
                }

                @Override
                public void onError(CameraDevice cd, int error) {
                    cameraDevice.close();
                    cameraDevice = null;
                }
            };

        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
        }
    }

    private void createPreviewSession() {
        try {
            SurfaceTexture surfaceTexture = textureView.getSurfaceTexture();
            surfaceTexture.setDefaultBufferSize(previewSize.getWidth(), previewSize.getHeight());
            Surface previewSurface = new Surface(surfaceTexture);
            CaptureRequest.Builder captureRequestBuilder =
                    cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            captureRequestBuilder.addTarget(previewSurface);

            cameraDevice.createCaptureSession(Collections.singletonList(previewSurface),
                    new CameraCaptureSession.StateCallback() {

                        @Override
                        public void onConfigured(CameraCaptureSession ccSession) {
                            if (cameraDevice == null) {
                                return;
                            }

                            try {
                                captureRequest = captureRequestBuilder.build();
                                cameraCaptureSession = ccSession;
                                cameraCaptureSession.setRepeatingRequest(captureRequest,
                                        null, backgroundHandler);
                            } catch (CameraAccessException e) {
                                e.printStackTrace();
                            }
                        }

                        @Override
                        public void onConfigureFailed(CameraCaptureSession ccSession) {

                        }
                    }, backgroundHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        openBackgroundThread();
        if (textureView.isAvailable()) {
            setUpCamera();
            openCamera();
        } else {
            textureView.setSurfaceTextureListener(surfaceTextureListener);
        }
    }

    @Override
    protected void onStop() {
        super.onStop();
        closeCamera();
        closeBackgroundThread();
    }

    private void closeCamera() {
        if (cameraCaptureSession != null) {
            cameraCaptureSession.close();
            cameraCaptureSession = null;
        }

        if (cameraDevice != null) {
            cameraDevice.close();
            cameraDevice = null;
        }
    }

    private void closeBackgroundThread() {
        if (backgroundHandler != null) {
            backgroundThread.quitSafely();
            backgroundThread = null;
            backgroundHandler = null;
        }
    }

    private void setUpCamera() {
        try {
            for (String cameraId : manager.getCameraIdList()) {
                CameraCharacteristics chars = manager.getCameraCharacteristics(cameraId);
                if (chars.get(CameraCharacteristics.LENS_FACING) == cameraFacing) {
                    previewSize = chooseOptimalSize(chars.get(
                            CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP).getOutputSizes(SurfaceTexture.class),
                            viewWidth, viewHeight);
                    cameraIdFacing = cameraId;
                }
            }
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private Size chooseOptimalSize(Size[] outputSizes, int width, int height) {
        double preferredRatio = height / (double) width;
        Size currentOptimalSize = outputSizes[0];
        double currentOptimalRatio = currentOptimalSize.getWidth() / (double) currentOptimalSize.getHeight();
        for (Size currentSize : outputSizes) {
            double currentRatio = currentSize.getWidth() / (double) currentSize.getHeight();
            if (Math.abs(preferredRatio - currentRatio) <
                    Math.abs(preferredRatio - currentOptimalRatio)) {
                currentOptimalSize = currentSize;
                currentOptimalRatio = currentRatio;
            }
        }
        return currentOptimalSize;
    }

    private void openCamera() {
        try {
            if (ActivityCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA)
                    == PackageManager.PERMISSION_GRANTED) {
                manager.openCamera(cameraIdFacing, stateCallback, backgroundHandler);
            }
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private void openBackgroundThread() {
        backgroundThread = new HandlerThread("camera_background_thread");
        backgroundThread.start();
        backgroundHandler = new Handler(backgroundThread.getLooper());
    }

    private void startCamera(String sampleName) {

        Log.d("SabetiCameraActivity/",
                "imageView.getWidth()" + Integer.toString(textureView.getWidth())
                        + " imageView.getHeight()" + Integer.toString(textureView.getHeight()));

        ImageButton imageCaptureView = findViewById(R.id.imgCapture);
        Intent resultsPageIntent = new Intent(this, ResultsPageActivity.class);

        imageCaptureView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String timeStamp =
                        new SimpleDateFormat(CAMERA_DATE_FORMAT,
                                Locale.getDefault()).format(new Date());
                File storageDir =
                        getExternalFilesDir(null);
                String imageFileName = "IMG_" + sampleName;
                File outputDirectory = new File(storageDir, RESULTS_DIRECTORY + "/IMG_" + timeStamp);
//                String fileName = storageDir + "/results/" + imageFileName + ".jpg";
                if (!outputDirectory.exists()) {
                    if (!outputDirectory.mkdirs()) {
//                        Log.e(LogHelper.LogTag, "Failed to create directory: " + outputDirectory.getAbsolutePath());
                        outputDirectory = null;
                    }
                }
                File imageFile = new File(outputDirectory, imageFileName + ".jpg");
                String fileName = imageFile.toString();

                FileOutputStream outputPhoto = null;
                try {
                    outputPhoto = new FileOutputStream(imageFile);
                    textureView.getBitmap()
                            .compress(Bitmap.CompressFormat.PNG, 100, outputPhoto);
                } catch (Exception e) {
                    e.printStackTrace();
                } finally {
                    try {
                        if (outputPhoto != null) {
                            outputPhoto.close();
                        }
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }

            }
        });

    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {

        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                ;//startCamera(getIntent().getStringExtra(MainActivity.EXTRA_MESSAGE));
            } else {
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }

    public boolean allPermissionsGranted() {

        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }
}
