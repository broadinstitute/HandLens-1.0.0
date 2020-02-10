package com.example.myfirstapp;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;

import java.io.File;

import static com.example.myfirstapp.MainActivity.EXTRA_MESSAGE;
import static com.example.myfirstapp.MainActivity.IMAGE_FILE_NAME;
import static com.example.myfirstapp.SabetiLaunchCameraAppActivity.getCameraPhotoOrientation;

public class ImageViewBoxSelectActivity extends AppCompatActivity {

    public static int REQUEST_CODE_PERMISSIONS = 101;
    public static final String[] REQUIRED_PERMISSIONS = new String[]{"android.permission.CAMERA",
            "android.permission.WRITE_EXTERNAL_STORAGE",
            "android.permission.READ_EXTERNAL_STORAGE",
            "android.permission.INTERNET",
            "android.permission.ACCESS_NETWORK_STATE",
            "android.permission.ACCESS_WIFI_STATE"};
    ImageView imageView;
    LinearLayout mlinearLayout;
    public static final String CAMERA_DATE_FORMAT = "yyyyMMdd_HHmmss";
    public static final String RESULTS_DIRECTORY = "/results";
    private static final String TAG = "ImageViewBoxSelectAct";
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
        private Context mContext;
        private RelativeLayout mOuterContainerLayout;
        private RelativeLayout.LayoutParams mOuterButtonParams;
        private RelativeLayout.LayoutParams mInnerContainerLayoutParams;
        private ImageButton mImageButton;

        Box(Context context, ImageButton imageButton) {
            super(context);
            mContext = context;
            mImageButton = imageButton;
            // mlinearLayout =  linearLayout;
            mOuterContainerLayout = new RelativeLayout(mContext);
            mOuterContainerLayout.setLayoutParams(new RelativeLayout.LayoutParams(
                    RelativeLayout.LayoutParams.MATCH_PARENT,
                    RelativeLayout.LayoutParams.MATCH_PARENT));

            mOuterButtonParams = new RelativeLayout.LayoutParams(
                    RelativeLayout.LayoutParams.WRAP_CONTENT,
                    RelativeLayout.LayoutParams.WRAP_CONTENT);
            mOuterButtonParams.addRule(RelativeLayout.CENTER_IN_PARENT,
                    RelativeLayout.TRUE);

            mInnerContainerLayoutParams = new RelativeLayout.LayoutParams(
                    RelativeLayout.LayoutParams.MATCH_PARENT,
                    RelativeLayout.LayoutParams.MATCH_PARENT);
            mInnerContainerLayoutParams.addRule(RelativeLayout.CENTER_IN_PARENT,
                    RelativeLayout.TRUE);

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
            float strip_ratio = 1340.0f / 82.0f;
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
//            ((ViewGroup)this.getParent()).addView(mImageButton);
            mOuterButtonParams.addRule(RelativeLayout.ALIGN_PARENT_BOTTOM);
            mImageButton.setLayoutParams(mOuterButtonParams);
            mOuterContainerLayout.addView(mImageButton);
            if (((ViewGroup) mOuterContainerLayout.getParent()) != null) {
                ((ViewGroup) mOuterContainerLayout.getParent()).removeView(mOuterContainerLayout);
            }
            addContentView(mOuterContainerLayout, mInnerContainerLayoutParams);
            canvas.restore();

        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image_view_box_select);
        imageView = (ImageView) findViewById(R.id.capturedImage);

        String photoFilePath = getIntent().getStringExtra(EXTRA_MESSAGE);
        Log.d("ImageViewBoxSelectAct", "photoFilePath: " + photoFilePath);
        File imageFile = new File(photoFilePath);
        if (!imageFile.exists()) {
            Log.d("ImageViewBoxSelectAct", "DOES NOT EXIST: " + photoFilePath);

        }
//        BitmapFactory.Options options = new BitmapFactory.Options();
//        options.inSampleSize = 16;
        Bitmap sourceImage = BitmapFactory.decodeFile(imageFile.getAbsolutePath());
//        imageView.setImageBitmap(sourceImage);
        Matrix rotationMatrix = new Matrix();
        imageView.setImageURI(Uri.fromFile(imageFile));
        rotationMatrix.postRotate(getCameraPhotoOrientation(this,
                FileProvider.getUriForFile(this,
                        "com.example.myfirstapp.provider",
                        new File(photoFilePath)),
                photoFilePath));
        imageView.setImageBitmap(Bitmap.createBitmap(sourceImage, 0, 0,
                sourceImage.getWidth(), sourceImage.getHeight(), rotationMatrix, true));

        imageView = findViewById(R.id.capturedImage);
        imageView.measure(View.MeasureSpec.UNSPECIFIED, View.MeasureSpec.UNSPECIFIED);
        viewWidth = imageView.getMeasuredWidth();
        viewHeight = imageView.getMeasuredHeight();

        addBox(photoFilePath);

    }

    @Override
    public void onBackPressed() {
        startActivity(new Intent(this, MainActivity.class));
    }


    private void addBox(String photoFilePath) {
        // Set up preview screen based
        final int screenWidth = imageView.getWidth();
        final int screenHeight = imageView.getHeight();
        Size screen = new Size(screenWidth, screenHeight);

        Intent resultsPageIntent = new Intent(this, ResultsPageActivity.class);
        resultsPageIntent.putExtra(IMAGE_FILE_NAME, photoFilePath);

        ImageButton sendToResultsButton = findViewById(R.id.sendToServerButton);

        Box box = new Box(this, sendToResultsButton);
        ViewGroup.LayoutParams params = new ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT);
        addContentView(box, params);

        sendToResultsButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Log.d(TAG, "button pressed");
                startActivity(resultsPageIntent);
            }
        });

    }

}
