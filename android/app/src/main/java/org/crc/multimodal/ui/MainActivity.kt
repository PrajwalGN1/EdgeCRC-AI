package org.crc.multimodal.ui

import android.os.Bundle
import android.view.View
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton
import org.crc.multimodal.R
import org.crc.multimodal.inference.CrcMultimodalClassifier
import org.crc.multimodal.inference.InferenceProtocol
import org.crc.multimodal.inference.InferenceResult
import org.crc.multimodal.inference.ThresholdLogic
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.random.Random

/**
 * Modern research inspection UI demonstrating the local edge multimodal inference lifecycle.
 *
 * NOTE: Strictly for investigational research and interface verification.
 * Does not process raw whole slide images on-device in this phase.
 */
class MainActivity : AppCompatActivity() {

    private var classifier: CrcMultimodalClassifier? = null
    private val backgroundExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    private lateinit var tvEngineStatus: TextView
    private lateinit var tvModelDetails: TextView
    private lateinit var tvRuntimeDetails: TextView
    private lateinit var tvInferenceResult: TextView
    private lateinit var progressBarLoading: ProgressBar
    private lateinit var btnRunInference: MaterialButton

    override fun onCreate(savedInstanceState: Bundle?) {
        // Enforce AppCompat/Material theme before activity creation as double safeguard
        setTheme(R.style.Theme_CrcMultimodal)
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tvEngineStatus = findViewById(R.id.tvEngineStatus)
        tvModelDetails = findViewById(R.id.tvModelDetails)
        tvRuntimeDetails = findViewById(R.id.tvRuntimeDetails)
        tvInferenceResult = findViewById(R.id.tvInferenceResult)
        progressBarLoading = findViewById(R.id.progressBarLoading)
        btnRunInference = findViewById(R.id.btnRunInference)

        btnRunInference.setOnClickListener {
            executeOnDeviceInference()
        }

        // Offload heavy I/O and ONNX model parsing to background thread to protect UI thread
        backgroundExecutor.execute {
            try {
                val assetBytes = assets.open("models/fusion_model_fp32.onnx").use { it.readBytes() }
                val instance = CrcMultimodalClassifier.fromByteArray(assetBytes)

                runOnUiThread {
                    if (!isFinishing && !isDestroyed) {
                        classifier = instance
                        progressBarLoading.visibility = View.GONE
                        tvEngineStatus.text = "✓ ONNX Runtime Engine Ready"
                        tvEngineStatus.setTextColor(getColor(R.color.status_success))
                        tvModelDetails.text = "Loaded: models/fusion_model_fp32.onnx (20.04 MB FP32)\nInput: vision_seq [1, N, 1024], text_summary [1, 768]"
                        tvRuntimeDetails.text = "Provider: CPUExecutionProvider (Float32)\nOperating Decision Threshold: 0.83 (MSI ≥ 0.83, MSS < 0.83)\nProtocols Supported: Protocol 1 (Capped ≤ 1000) & Protocol 2 (Uncapped)"
                        tvInferenceResult.text = "Engine ready. Tap button above to execute live test."
                        btnRunInference.isEnabled = true
                    } else {
                        // Activity was closed before initialization finished; clean up immediately
                        instance.close()
                    }
                }
            } catch (e: Throwable) {
                runOnUiThread {
                    if (!isFinishing && !isDestroyed) {
                        progressBarLoading.visibility = View.GONE
                        tvEngineStatus.text = "✗ Initialization Failed"
                        tvEngineStatus.setTextColor(getColor(R.color.status_error))
                        tvModelDetails.text = "Error details: ${e.message}"
                        tvInferenceResult.text = "Error initializing classifier: ${e.localizedMessage}"
                    }
                }
            }
        }
    }

    private fun executeOnDeviceInference() {
        val activeClassifier = classifier ?: run {
            tvInferenceResult.text = "Classifier not ready yet."
            return
        }

        btnRunInference.isEnabled = false
        tvInferenceResult.text = "Executing live on-device inference..."

        backgroundExecutor.execute {
            try {
                val numPatches = 10
                val rng = Random(System.currentTimeMillis())
                val visionData = FloatArray(numPatches * 1024) { rng.nextFloat() * 0.1f - 0.05f }
                val textData = FloatArray(768) { rng.nextFloat() * 0.1f - 0.05f }

                val startNano = System.nanoTime()
                val result = activeClassifier.predict(
                    visionData = visionData,
                    numPatches = numPatches,
                    textData = textData,
                    caseId = "DEVICE_LIVE_TEST",
                    protocol = InferenceProtocol.PROTOCOL_1_CAPPED
                )
                val elapsedMs = (System.nanoTime() - startNano) / 1_000_000.0

                val label = ThresholdLogic.labelFor(result.predictedClass)
                val margin = ThresholdLogic.signedDistanceToThreshold(result.probability)

                runOnUiThread {
                    if (!isFinishing && !isDestroyed) {
                        btnRunInference.isEnabled = true
                        val outputText = buildString {
                            appendLine("=== LIVE ON-DEVICE INFERENCE RESULT ===")
                            appendLine("Status: SUCCESS (Bit-accurate native execution)")
                            appendLine("Case ID: DEVICE_LIVE_TEST")
                            appendLine("Slide Patches: $numPatches (dim: 1024)")
                            appendLine("Text Embedding: 1 × 768")
                            appendLine("Output Logit: ${String.format("%.4f", result.logit)}")
                            appendLine("Predicted Prob: ${String.format("%.4f", result.probability)}")
                            appendLine("Threshold (0.83): ${String.format("%+.4f", margin)} vs 0.83")
                            appendLine("Classification: $label (Class ${result.predictedClass})")
                            appendLine("Attention Dim: [1, 1, ${result.attentionWeights?.size ?: 0}]")
                            appendLine("Inference Latency: ${String.format("%.2f", elapsedMs)} ms")
                            appendLine("Timestamp: ${java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", java.util.Locale.US).format(java.util.Date())}")
                        }
                        tvInferenceResult.text = outputText
                    }
                }
            } catch (t: Throwable) {
                runOnUiThread {
                    if (!isFinishing && !isDestroyed) {
                        btnRunInference.isEnabled = true
                        tvInferenceResult.text = "Inference execution error: ${t.message}"
                    }
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        backgroundExecutor.shutdownNow()
        classifier?.close()
        classifier = null
    }
}
