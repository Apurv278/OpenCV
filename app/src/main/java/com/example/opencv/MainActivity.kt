package com.example.opencv

import android.Manifest.permission.*
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.camera2.CameraCharacteristics
import android.os.Bundle
import android.util.Log
import android.view.OrientationEventListener
import android.view.Surface
import android.view.SurfaceView
import android.view.WindowManager.LayoutParams.*
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.android.synthetic.main.activity_main.*
import org.opencv.android.*
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.imgproc.Imgproc.resize
import org.opencv.objdetect.CascadeClassifier
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException

private const val REQUEST_CODE = 111
private val PERMISSION = arrayOf(CAMERA, WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE,
    RECORD_AUDIO, ACCESS_FINE_LOCATION)

class MainActivity : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener2 {

    private val viewFinder by lazy { findViewById<JavaCameraView>(R.id.cameraView) }
    lateinit var cvCallback: BaseLoaderCallback
    lateinit var imgMat: Mat
    lateinit var gMat: Mat
//    private val faceStream = resources.openRawResource(R.raw.haarcascade_frontalface_alt2)
    //
    var faceDetector: CascadeClassifier? = null
    lateinit var faceDir: File
    var imgRatio = 0.0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.clearFlags(FLAG_FORCE_NOT_FULLSCREEN)
        window.setFlags(FLAG_FULLSCREEN, FLAG_FULLSCREEN)
        window.addFlags(FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_main)

        val mOrientationEventListener = object : OrientationEventListener(this) {
            override fun onOrientationChanged(orientation: Int) {
                when (orientation) {
                    in 45..134 -> {
                        rotation_tv.text = getString(R.string.n_270_degree)
                    }
                    in 135..224 -> {
                        rotation_tv.text = getString(R.string.n_180_degree)
                    }
                    in 225..314 -> {
                        rotation_tv.text = getString(R.string.n_90_degree)
                    }
                    else -> {
                        rotation_tv.text = getString(R.string.n_0_degree)
                    }
                }
            }
        }
        if (mOrientationEventListener.canDetectOrientation()) {
            mOrientationEventListener.enable()
        } else {
            mOrientationEventListener.disable()
        }

        //Permission:
        if (permission_Granted()) {
            openCV(this)
        } else {
            ActivityCompat.requestPermissions(this, PERMISSION, REQUEST_CODE)
        }
        viewFinder.visibility = SurfaceView.VISIBLE
        viewFinder.setCameraIndex(CameraCharacteristics.LENS_FACING_FRONT)
        viewFinder.setCvCameraViewListener(this)

        cvCallback = object : BaseLoaderCallback(this) {
            override fun onManagerConnected(status: Int) {
                when (status) {
                    LoaderCallbackInterface.SUCCESS -> {
                        Log.d(TAG, OpenCV_SUCCESSFUL)
                        loadFaceLib()
                        if (faceDetector!!.empty()) {
                            faceDetector = null
                        } else {
                            faceDir.delete()
                        }
                        viewFinder.enableView()
//                        sMsg(this@MainActivity, OpenCV_SUCCESSFUL)
                    } else -> super.onManagerConnected(status)
                }
            }
        }
    }

    companion object {

        val TAG = "Open CV LOG: " + MainActivity::class.java.simpleName
        fun sMsg(context: Context, s: String) =
                Toast.makeText(context, s, Toast.LENGTH_SHORT).show()
        // messages:
        private const val OpenCV_SUCCESSFUL = "OpenCV Loaded Successfully!"
        private const val OpenCV_FAIL = "Could not load OpenCV!"
        private const val PERMISSION_NOT_GRANTED = "Permission not granted by the User!"
        private const val OpenCV_Problem = "Problem in CV!"
        private const val FACE_DIR = "facelib"
        private const val FACE_MODEL = "haarcascade_frontalface_alt2.xml"
        private const val byteSize = 4096 // buffer size

        private fun openCV(context: Context) =
                if (OpenCVLoader.initDebug()) sMsg(context, OpenCV_SUCCESSFUL)
                else sMsg(context, OpenCV_FAIL)
    }

    fun loadFaceLib() {
        try {
            val modelInputStream = resources.openRawResource(R.raw.haarcascade_frontalface_alt2)
            faceDir = getDir(FACE_DIR, Context.MODE_PRIVATE)
            val faceModel = File(faceDir, FACE_MODEL)
            val modelOutputStream = FileOutputStream(faceModel)
            val buffer = ByteArray(byteSize)
            var byteRead = modelInputStream.read(buffer)
            while (byteRead != -1) {
                modelOutputStream.write(buffer, 0, byteRead)
                byteRead = modelInputStream.read(buffer)
            }
            modelInputStream.close()
            modelOutputStream.close()
            faceDetector = CascadeClassifier(faceModel.absolutePath)
        } catch (e: IOException) {
            Log.e(TAG, "Error Loading Face Model...$e")
        }
    }

    override fun onResume() {
        super.onResume()
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, cvCallback)
    }

    override fun onPause() {
        super.onPause()
        viewFinder?.let { viewFinder.disableView() }
    }

    override fun onDestroy() {
        super.onDestroy()
        viewFinder?.let { viewFinder.disableView() }
        if (faceDir.exists()) faceDir.delete()
    }

    private fun permission_Granted() = PERMISSION.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        //super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE) {
            if (permission_Granted()) {
                openCV(this)
            } else {
                sMsg(this, PERMISSION_NOT_GRANTED)
                finish()
            }
        }
    }

    fun drawFaceRect() {
        val faceRects = MatOfRect()
        faceDetector!!.detectMultiScale(gMat, faceRects)

        for (rect in faceRects.toArray()) {
            var x = 0.0
            var y = 0.0
            var w = 0.0
            var h = 0.0

            if (imgRatio.equals(1.0)) {
                x = rect.x.toDouble()
                y = rect.y.toDouble()
                w = x + rect.width
                h = y + rect.height
            } else {
                x = rect.x.toDouble() / imgRatio
                y = rect.y.toDouble() / imgRatio
                w = x + (rect.width / imgRatio)
                h = y + (rect.height / imgRatio)
            }
            Imgproc.rectangle(imgMat, Point(x, y), Point(w, h), Scalar(255.0, 0.0, 0.0))
        }
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        imgMat = Mat(width, height, CvType.CV_8UC4)
        gMat = Mat(width, height, CvType.CV_8UC4)
    }

    override fun onCameraViewStopped() {
        imgMat.release()
        gMat.release()
    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame?): Mat {
        imgMat = inputFrame!!.rgba()
        //gMat = inputFrame.gray()
        gMat = get480Img(inputFrame.gray())
        //imgRatio = 1.0
        drawFaceRect()
        return imgMat
    }

    fun ratioTo480(src: Size): Double {
        val w = src.width
        val h = src.height
        val heightMax = 480
        var ratio: Double = 0.0

        if (w > h) {
            if (w < heightMax) return 1.0
            ratio = heightMax / w
        } else {
            if (h < heightMax) return 1.0
            ratio = heightMax / h
        }

        return ratio
    }

    fun get480Img(src: Mat): Mat {
        val imageSize = Size(src.width().toDouble(), src.height().toDouble())
        imgRatio = ratioTo480(imageSize)

        if (imgRatio.equals(1.0)) return src

        val dstSize = Size(imageSize.width*imgRatio, imageSize.height*imgRatio)
        val dst = Mat()
        resize(src, dst, dstSize)
        return dst
    }
}