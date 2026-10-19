package org.crc.multimodal.ui

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.crc.multimodal.inference.CrcMultimodalClassifier
import org.crc.multimodal.inference.InferenceResult

/**
 * Minimal research inspection UI demonstrating the local edge multimodal inference lifecycle.
 *
 * NOTE: Strictly for investigational research and interface verification.
 * Does not process raw whole slide images on-device in this phase.
 */
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private var classifier: CrcMultimodalClassifier? = null
    private val backgroundExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private lateinit var statusTextView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        statusTextView = TextView(this).apply {
            setPadding(40, 60, 40, 60)
            textSize = 15f
            text = buildDisclaimerText()
        }
        setContentView(statusTextView)

        statusTextView.append("\n\n[Background Task] Initializing ONNX Runtime engine in background...")

        // Offload heavy I/O and ONNX model parsing to background thread to protect UI thread
        backgroundExecutor.execute {
            try {
                val assetBytes = assets.open("models/fusion_model_fp32.onnx").use { it.readBytes() }
                val instance = CrcMultimodalClassifier.fromByteArray(assetBytes)

                runOnUiThread {
                    if (!isFinishing && !isDestroyed) {
                        classifier = instance
                        statusTextView.append("\n✓ Model loaded successfully: models/fusion_model_fp32.onnx (20.04 MB)")
                        statusTextView.append("\n✓ Execution Provider: CPUExecutionProvider (Float32)")
                        statusTextView.append("\n✓ Operating Decision Threshold: 0.83")
                        statusTextView.append("\n\nReady to receive precomputed feature embeddings for evaluation.")
                    } else {
                        // Activity was closed before initialization finished; clean up immediately
                        instance.close()
                    }
                }
            } catch (e: Exception) {
                runOnUiThread {
                    if (!isFinishing && !isDestroyed) {
                        statusTextView.append("\n\n✗ Error initializing classifier: ${e.message}")
                    }
                }
            }
        }
    }

    private fun buildDisclaimerText(): String {
        return """
            ====================================================
            CRC MULTIMODAL FUSION CLASSIFIER (EDGE RESEARCH)
            ====================================================
            
            DISCLAIMER:
            This software is an investigational research demonstration.
            It is NOT cleared, certified, or approved for clinical diagnosis,
            treatment planning, or patient care.
            
            Scope:
            Consumes precomputed feature embeddings (Vision: [1, N, 1024], Text: [1, 768]).
            Foundation vision encoder (Phikon-v2) and clinical text encoder (ClinicalBERT)
            are not bundled in this mobile package.
        """.trimIndent()
    }

    override fun onDestroy() {
        super.onDestroy()
        backgroundExecutor.shutdownNow()
        classifier?.close()
        classifier = null
    }
}
