package com.example.myfirstapp;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraX;
import androidx.camera.core.FocusMeteringAction;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageAnalysisConfig;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureConfig;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.MeteringPoint;
import androidx.camera.core.MeteringPointFactory;
import androidx.camera.core.Preview;
import androidx.camera.core.PreviewConfig;
import androidx.camera.core.SensorOrientedMeteringPointFactory;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;


import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.util.Rational;
import android.util.Size;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewTreeObserver;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.Toast;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.concurrent.TimeUnit;

import static android.view.View.LAYOUT_DIRECTION_INHERIT;


public class SabetiCameraActivity extends AppCompatActivity {

    private int REQUEST_CODE_PERMISSIONS = 101;
    private final String[] REQUIRED_PERMISSIONS = new String[]{"android.permission.CAMERA", "android.permission.WRITE_EXTERNAL_STORAGE"};
    public static final String IMAGE_FILE_NAME = "IMAGE_FILE_NAME";
    TextureView textureView;
    public static final String CAMERA_DATE_FORMAT = "yyyyMMdd_HHmmss";
    public static final String RESULTS_DIRECTORY = "/results";
    private long mLastAnalysisResultTime;
    private double exposure_required = 1;

    private int viewHeight;
    private int viewWidth;

    private class Box extends View {
        private Paint paint = new Paint();
        private ScaleGestureDetector mScaleGestureDetector;
        private float mScaleFactor = 1.f;
        private static final int INVALID_POINTER_ID = -1;
        private int mActivePointerId = INVALID_POINTER_ID;
        private float mPrevX;
        private float mPrevY;
        private float mPosX = 0f;
        private float mPosY = 0f;

        private ImageButton mImageButton;

        Box(Context context, ImageButton imageButton) {
            super(context);
            mImageButton = imageButton;
            mScaleGestureDetector = new ScaleGestureDetector(context, new ScaleListener());
//            mGestureDetector = new GestureDetector(context, new GestureDetector.SimpleOnGestureListener());

        }

        @Override
        public boolean onTouchEvent(MotionEvent ev) {
            // Let the ScaleGestureDetector inspect all events.
            boolean retVal = mScaleGestureDetector.onTouchEvent(ev);
            final int eventAction = ev.getAction();
            switch (eventAction & MotionEvent.ACTION_MASK) {
                case MotionEvent.ACTION_DOWN: {
                    mPrevX = ev.getX();
                    mPrevY = ev.getY();
                    mActivePointerId = ev.getPointerId(0);
                    break;
                }

                case MotionEvent.ACTION_MOVE: {
                    int pointerIndex = ev.findPointerIndex(mActivePointerId);
                    float currX = ev.getX(pointerIndex);
                    float currY = ev.getY(pointerIndex);
                    // if ScaleGestureDetector is in progress, don't move
                    if (!mScaleGestureDetector.isInProgress()) {
                        float dx = currX - mPrevX;
                        float dy = currY - mPrevY;

                        mPosX += dx;
                        mPosY += dy;
                        invalidate();
                    }
                    mPrevX = currX;
                    mPrevY = currY;
                    break;
                }

                case MotionEvent.ACTION_POINTER_UP: {
                    final int pointerIndex = (ev.getAction() & MotionEvent.ACTION_POINTER_INDEX_MASK) >> MotionEvent.ACTION_POINTER_INDEX_SHIFT;
                    final int pointerId = ev.getPointerId(pointerIndex);
                    if (pointerId == mActivePointerId) {
                        // pointerId was the active pointer on the up motion. We need to chose a new pointer
                        final int newPointerIndex = pointerIndex == 0 ? 1 : 0;
                        mPrevX = ev.getX(newPointerIndex);
                        mPrevY = ev.getY(newPointerIndex);
                        mActivePointerId = ev.getPointerId(newPointerIndex);
                    }
                    break;
                }
                case MotionEvent.ACTION_UP:
                case MotionEvent.ACTION_CANCEL: {
                    mActivePointerId = INVALID_POINTER_ID;
                    break;
                }

            }
            return true;
        }


        private class ScaleListener
                extends ScaleGestureDetector.SimpleOnScaleGestureListener {
            @Override
            public boolean onScale(ScaleGestureDetector detector) {
                mScaleFactor *= detector.getScaleFactor();

//                pivotPointX = detector.getFocusX();
//                pivotPointY = detector.getFocusY();
                // Don't let the object get too small or too large.
                mScaleFactor = Math.max(0.1f, Math.min(mScaleFactor, 5.0f));

                // tell the View to redraw the Canvas
                invalidate();
                return true;
            }
        }

        @Override
        protected void onDraw(Canvas canvas) { // Override the onDraw() Method
            super.onDraw(canvas);

            canvas.save();
            canvas.scale(mScaleFactor, mScaleFactor);
            canvas.translate(mPosX, mPosY);

            //center
            float strip_ratio = 1372.0f / 82.0f;
            int x0 = getWidth();
            int y0 = getHeight();

            int max_strips = 8;
            int strip_buffer = 1; // how many strip widths between each strip
            int total_width_strips = max_strips + max_strips * strip_buffer + strip_buffer;

            float strip_width = x0 / total_width_strips;
            float strip_height = strip_ratio * strip_width;

            paint.setStyle(Paint.Style.STROKE);
            paint.setColor(Color.GREEN);
            paint.setStrokeWidth(x0 / 100);
            //draw guide box
            canvas.drawRect(x0 - strip_width * 2, (y0 - strip_height) / 2, x0 - strip_width, y0 - (y0 - strip_height) / 2, paint);
            paint.setStrokeWidth(x0 / 200);
            paint.setTextSize(16 * getResources().getDisplayMetrics().density);
            canvas.drawText("Ctrl", x0 - strip_width * 2, (y0 - strip_height) / 2 - (y0 - strip_height) / 20, paint);
            ((ViewGroup) mImageButton.getParent()).removeView(mImageButton);
            ((ViewGroup) this.getParent()).addView(mImageButton);

            canvas.restore();
        }
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);

        textureView = findViewById(R.id.view_finder);

        textureView.measure(View.MeasureSpec.UNSPECIFIED, View.MeasureSpec.UNSPECIFIED);
        viewWidth = textureView.getMeasuredWidth();
        viewHeight = textureView.getMeasuredHeight();

        mLastAnalysisResultTime = SystemClock.elapsedRealtime();

        if (allPermissionsGranted()) {
            // Get the Intent that started this activity and extract the string
            Intent intent = getIntent();
            String message = intent.getStringExtra(MainActivity.EXTRA_MESSAGE);
            startCamera(message);
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
        }
    }

    private void startCamera(String sampleName) {
        // Code inspired by https://github.com/journaldev/journaldev/tree/master/Android/AndroidCameraX
        // (MIT license)
        CameraX.unbindAll();
        // Set up preview screen based
        final int screenWidth = textureView.getWidth();
        final int screenHeight = textureView.getHeight();
        Size screen = new Size(screenWidth, screenHeight);
        Rational aspectRatio = new Rational(textureView.getWidth(), textureView.getHeight());
        Log.d("SabetiCameraActivity/", "textureView.getWidt()" + Integer.toString(textureView.getWidth())
                + " textureView.getHeight()" + Integer.toString(textureView.getHeight()));
        PreviewConfig previewConfig = new PreviewConfig.Builder().setTargetAspectRatio(aspectRatio).setTargetResolution(screen).build();
        Preview preview = new Preview(previewConfig);

        ImageButton imageCaptureView = findViewById(R.id.imgCapture);

        Intent resultsPageIntent = new Intent(this, ResultsPageActivity.class);
        Box box = new Box(this, imageCaptureView);
        addContentView(box, new ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
        preview.setOnPreviewOutputUpdateListener(
                new Preview.OnPreviewOutputUpdateListener() {
                    @Override
                    public void onUpdated(Preview.PreviewOutput previewOutput) {
                        ViewGroup viewParent = (ViewGroup) textureView.getParent();
                        viewParent.removeView(textureView);
                        viewParent.addView(textureView, 0);
                        textureView.setSurfaceTexture(previewOutput.getSurfaceTexture());

                        updateTransform();
                    }
                });

        ImageCaptureConfig imageCaptureConfig = new ImageCaptureConfig.Builder().setCaptureMode(ImageCapture.CaptureMode.MIN_LATENCY)
                .setTargetRotation(getWindowManager().getDefaultDisplay().getRotation()).build();
        final ImageCapture imageCapture = new ImageCapture(imageCaptureConfig);


//        final ImageAnalysisConfig imageAnalysisConfig =
//                new ImageAnalysisConfig.Builder()
//                        .setTargetResolution(screen)
//                        .setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE)
//                        .build();
//
//        final ImageAnalysis imageAnalysis = new ImageAnalysis(imageAnalysisConfig);
//        imageAnalysis.setAnalyzer(
//                (image, rotationDegrees) -> {
//                    if (SystemClock.elapsedRealtime() - mLastAnalysisResultTime < 500) {
//                        return;
//                    }
//
//                    exposure_required = analyzeImage(image, rotationDegrees);
//                    if (exposure_required != -1) {
//                        mLastAnalysisResultTime = SystemClock.elapsedRealtime();
////                        runOnUiThread(() -> applyToUiAnalyzeImageResult(result));
//                    }
//                });

        MeteringPointFactory factory = new SensorOrientedMeteringPointFactory(
                screenWidth, screenHeight);
        MeteringPoint point = factory.createPoint(x, y);
        FocusMeteringAction action = FocusMeteringAction.Builder.from(point,
                FocusMeteringAction.MeteringMode.AF_ONLY)
                .addPoint(point2, FocusMeteringAction.MeteringMode.AE_ONLY) // could have many
                .setAutoFocusCallback(new FocusMeteringAction.OnAutoFocusListener() {
                    public void onFocusCompleted(boolean isSuccess) {
                    }
                })
                // auto calling cancelFocusAndMetering in 5 seconds
                .setAutoCancelDuration(5, TimeUnit.SECONDS)
                .build();


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

                imageCapture.takePicture(imageFile,
                        new ImageCapture.OnImageSavedListener() {
                            @Override
                            public void onImageSaved(@NonNull File file) {
                                String msg = "Pic captured at " + file.getAbsolutePath();
                                Toast.makeText(getBaseContext(), msg, Toast.LENGTH_LONG).show();
                                resultsPageIntent.putExtra(IMAGE_FILE_NAME, fileName);
                                startActivity(resultsPageIntent);
                            }

                            @Override
                            public void onError(ImageCapture.ImageCaptureError imageCaptureError,
                                                @NonNull String message, @Nullable Throwable cause) {
                                String msg = "Pic capture failed : " + message + "\nat " + fileName;
                                Toast.makeText(getBaseContext(), msg, Toast.LENGTH_LONG).show();
                                if (cause != null) {
                                    cause.printStackTrace();
                                }
                            }
                        });
            }
        });


        //bind to lifecycle:
        CameraX.bindToLifecycle((LifecycleOwner) this, preview, imageCapture);
    }

    private void updateTransform() {
        Matrix mx = new Matrix();
        float w = textureView.getMeasuredWidth();
        float h = textureView.getMeasuredHeight();

        float cX = w / 2f;
        float cY = h / 2f;

        int rotationDgr;
        int rotation = (int) textureView.getRotation();

        switch (rotation) {
            case Surface.ROTATION_0:
                rotationDgr = 0;
                break;
            case Surface.ROTATION_90:
                rotationDgr = 90;
                break;
            case Surface.ROTATION_180:
                rotationDgr = 180;
                break;
            case Surface.ROTATION_270:
                rotationDgr = 270;
                break;
            default:
                return;
        }

        mx.postRotate((float) rotationDgr, cX, cY);
        textureView.setTransform(mx);
    }

    public double analyzeImage(ImageProxy image, int rotationDegrees) {
        return 0;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {

        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera(getIntent().getStringExtra(MainActivity.EXTRA_MESSAGE));
            } else {
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }

    private boolean allPermissionsGranted() {

        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }
}
