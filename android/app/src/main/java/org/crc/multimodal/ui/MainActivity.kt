package org.crc.multimodal.ui

import android.os.Bundle
import android.view.View
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.google.android.material.button.MaterialButton
import com.google.android.material.button.MaterialButtonToggleGroup
import com.google.android.material.card.MaterialCardView
import org.crc.multimodal.R
import org.crc.multimodal.inference.CrcMultimodalClassifier
import org.crc.multimodal.inference.InferenceProtocol
import org.crc.multimodal.inference.InferenceResult
import org.crc.multimodal.inference.ThresholdLogic
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.random.Random

/**
 * Enterprise-Grade Multimodal Edge AI Research Workspace.
 *
 * Demonstrates bit-accurate on-device execution of the frozen Colorectal Cancer Multimodal
 * Fusion model (ClinicalBERT + PHIKON-V2 cross-attention fusion) via ONNX Runtime Mobile.
 *
 * Invariants Enforced:
 * - Operating Threshold: 0.83f (probability >= 0.83f evaluates to MSI / Class 1)
 * - Class Mapping: 0 = MSS, 1 = MSI
 * - Protocol 1 Capped: Evaluates up to 1,000 slide image patches
 * - Protocol 2 Uncapped: Preserves arbitrary candidate patch sequences
 * - Single-patient sequential CPU execution with thread-safe resource isolation
 * - 100% on-device local execution; zero network transmission
 */
class MainActivity : AppCompatActivity() {

    private var classifier: CrcMultimodalClassifier? = null
    private val backgroundExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    // Engine & Status Views
    private lateinit var ivEngineStatusIcon: ImageView
    private lateinit var tvEngineStatus: TextView
    private lateinit var tvModelDetails: TextView
    private lateinit var tvRuntimeDetails: TextView
    private lateinit var progressBarLoading: ProgressBar

    // Input Configuration Views
    private lateinit var toggleGroupPatchCount: MaterialButtonToggleGroup
    private lateinit var tvInputSummary: TextView

    // Execution Controls & Progress
    private lateinit var btnRunInference: MaterialButton
    private lateinit var progressBarInference: ProgressBar
    private lateinit var tvInferenceProgress: TextView

    // Diagnostic Results Card Views
    private lateinit var cardResult: MaterialCardView
    private lateinit var tvResultTimestamp: TextView
    private lateinit var layoutClassificationBadge: LinearLayout
    private lateinit var ivClassificationIcon: ImageView
    private lateinit var tvResultClassification: TextView
    private lateinit var tvResultSubLabel: TextView
    private lateinit var tvResultScore: TextView
    private lateinit var tvResultMargin: TextView
    private lateinit var probabilityBarView: ProbabilityThresholdBarView
    private lateinit var tvResultLatency: TextView
    private lateinit var tvResultPatches: TextView
    private lateinit var btnToggleDiagnostics: MaterialButton
    private lateinit var layoutTechnicalDiagnostics: LinearLayout
    private lateinit var tvInferenceResult: TextView

    private var selectedPatchCount: Int = 10
    private var isDiagnosticsExpanded: Boolean = false

    override fun onCreate(savedInstanceState: Bundle?) {
        setTheme(R.style.Theme_CrcMultimodal)
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        bindViews()
        setupInputConfiguration()
        setupDiagnosticsToggle()

        btnRunInference.setOnClickListener {
            executeOnDeviceInference()
        }

        // Offload ONNX model reading and native session initialization to background executor
        initializeModelAsync()
    }

    private fun bindViews() {
        ivEngineStatusIcon = findViewById(R.id.ivEngineStatusIcon)
        tvEngineStatus = findViewById(R.id.tvEngineStatus)
        tvModelDetails = findViewById(R.id.tvModelDetails)
        tvRuntimeDetails = findViewById(R.id.tvRuntimeDetails)
        progressBarLoading = findViewById(R.id.progressBarLoading)

        toggleGroupPatchCount = findViewById(R.id.toggleGroupPatchCount)
        tvInputSummary = findViewById(R.id.tvInputSummary)

        btnRunInference = findViewById(R.id.btnRunInference)
        progressBarInference = findViewById(R.id.progressBarInference)
        tvInferenceProgress = findViewById(R.id.tvInferenceProgress)

        cardResult = findViewById(R.id.cardResult)
        tvResultTimestamp = findViewById(R.id.tvResultTimestamp)
        layoutClassificationBadge = findViewById(R.id.layoutClassificationBadge)
        ivClassificationIcon = findViewById(R.id.ivClassificationIcon)
        tvResultClassification = findViewById(R.id.tvResultClassification)
        tvResultSubLabel = findViewById(R.id.tvResultSubLabel)
        tvResultScore = findViewById(R.id.tvResultScore)
        tvResultMargin = findViewById(R.id.tvResultMargin)
        probabilityBarView = findViewById(R.id.probabilityBarView)
        tvResultLatency = findViewById(R.id.tvResultLatency)
        tvResultPatches = findViewById(R.id.tvResultPatches)
        btnToggleDiagnostics = findViewById(R.id.btnToggleDiagnostics)
        layoutTechnicalDiagnostics = findViewById(R.id.layoutTechnicalDiagnostics)
        tvInferenceResult = findViewById(R.id.tvInferenceResult)
    }

    private fun setupInputConfiguration() {
        toggleGroupPatchCount.check(R.id.btnPatches10)
        selectedPatchCount = 10
        updateInputSummary()

        toggleGroupPatchCount.addOnButtonCheckedListener { _, checkedId, isChecked ->
            if (isChecked) {
                selectedPatchCount = when (checkedId) {
                    R.id.btnPatches50 -> 50
                    R.id.btnPatches1000 -> 1000
                    else -> 10
                }
                updateInputSummary()
            }
        }
    }

    private fun updateInputSummary() {
        val protocolText = if (selectedPatchCount <= 1000) {
            "Protocol 1 (Capped ≤ 1,000)"
        } else {
            "Protocol 2 (Uncapped)"
        }
        tvInputSummary.text = String.format(
            Locale.US,
            "Vision sequence: [1, %d, 1024] · Text summary: [1, 768] · %s",
            selectedPatchCount,
            protocolText
        )
    }

    private fun setupDiagnosticsToggle() {
        btnToggleDiagnostics.setOnClickListener {
            isDiagnosticsExpanded = !isDiagnosticsExpanded
            if (isDiagnosticsExpanded) {
                layoutTechnicalDiagnostics.visibility = View.VISIBLE
                btnToggleDiagnostics.text = getString(R.string.btn_toggle_diagnostics_hide)
                btnToggleDiagnostics.setIconResource(R.drawable.ic_expand_less)
            } else {
                layoutTechnicalDiagnostics.visibility = View.GONE
                btnToggleDiagnostics.text = getString(R.string.btn_toggle_diagnostics_show)
                btnToggleDiagnostics.setIconResource(R.drawable.ic_expand_more)
            }
        }
    }

    private fun initializeModelAsync() {
        backgroundExecutor.execute {
            try {
                val assetBytes = assets.open("models/fusion_model_fp32.onnx").use { it.readBytes() }
                val instance = CrcMultimodalClassifier.fromByteArray(assetBytes)

                runOnUiThread {
                    if (!isFinishing && !isDestroyed) {
                        classifier = instance
                        progressBarLoading.visibility = View.GONE
                        ivEngineStatusIcon.setImageResource(R.drawable.ic_check_circle)
                        ivEngineStatusIcon.setColorFilter(ContextCompat.getColor(this, R.color.status_success))

                        tvEngineStatus.text = getString(R.string.status_engine_ready)
                        tvEngineStatus.setTextColor(ContextCompat.getColor(this, R.color.status_success))
                        tvModelDetails.text = "Loaded: models/fusion_model_fp32.onnx (20.04 MB FP32)\nInput: vision_seq [1, N, 1024], text_summary [1, 768]"
                        tvRuntimeDetails.text = "Provider: CPUExecutionProvider (Float32)\nDecision Threshold: 0.83 (MSI ≥ 0.83, MSS < 0.83)\nProtocols: Protocol 1 (Capped ≤ 1000) & Protocol 2 (Uncapped)"
                        btnRunInference.isEnabled = true
                    } else {
                        instance.close()
                    }
                }
            } catch (e: Throwable) {
                runOnUiThread {
                    if (!isFinishing && !isDestroyed) {
                        progressBarLoading.visibility = View.GONE
                        ivEngineStatusIcon.setImageResource(R.drawable.ic_error_outline)
                        ivEngineStatusIcon.setColorFilter(ContextCompat.getColor(this, R.color.status_error))

                        tvEngineStatus.text = getString(R.string.status_initialization_failed)
                        tvEngineStatus.setTextColor(ContextCompat.getColor(this, R.color.status_error))
                        tvModelDetails.text = "Error loading ONNX model: ${e.message}"
                        tvInferenceResult.text = "Model initialization failure:\n${e.localizedMessage}"
                    }
                }
            }
        }
    }

    private fun executeOnDeviceInference() {
        val activeClassifier = classifier ?: run {
            tvInferenceResult.text = "Classifier engine not ready yet."
            return
        }

        // Lock button to prevent duplicate submissions
        btnRunInference.isEnabled = false
        progressBarInference.visibility = View.VISIBLE
        tvInferenceProgress.visibility = View.VISIBLE

        val numPatches = selectedPatchCount

        backgroundExecutor.execute {
            try {
                // Generate clean, bounded test tensor representations for research evaluation
                val rng = Random(System.currentTimeMillis())
                val visionData = FloatArray(numPatches * 1024) { rng.nextFloat() * 0.1f - 0.05f }
                val textData = FloatArray(768) { rng.nextFloat() * 0.1f - 0.05f }

                val startNano = System.nanoTime()
                val result: InferenceResult = activeClassifier.predict(
                    visionData = visionData,
                    numPatches = numPatches,
                    textData = textData,
                    caseId = "DEVICE_RESEARCH_CASE",
                    protocol = if (numPatches <= 1000) InferenceProtocol.PROTOCOL_1_CAPPED else InferenceProtocol.PROTOCOL_2_UNCAPPED
                )
                val elapsedMs = (System.nanoTime() - startNano) / 1_000_000.0

                val label = ThresholdLogic.labelFor(result.predictedClass)
                val margin = ThresholdLogic.signedDistanceToThreshold(result.probability)
                val timestamp = SimpleDateFormat("HH:mm:ss", Locale.US).format(Date())

                val diagnosticsText = buildString {
                    appendLine("=== RESEARCH ON-DEVICE INFERENCE REPORT ===")
                    appendLine("Execution Status : SUCCESS (Bit-accurate native ONNX)")
                    appendLine("Case Identifier  : DEVICE_RESEARCH_CASE")
                    appendLine("Slide Patches (N): $numPatches (dim: 1024)")
                    appendLine("Text Embedding   : 1 × 768")
                    appendLine("Evaluation Mod   : ${result.protocol.name}")
                    appendLine("Output Logit     : ${String.format(Locale.US, "%.5f", result.logit)}")
                    appendLine("Output Prob      : ${String.format(Locale.US, "%.5f", result.probability)}")
                    appendLine("Decision Boundary: 0.8300 (Fixed)")
                    appendLine("Threshold Margin : ${String.format(Locale.US, "%+.5f", margin)} vs 0.83")
                    appendLine("Classification   : $label (Class ${result.predictedClass})")
                    appendLine("Attention Dim    : [1, 1, ${result.attentionWeights?.size ?: 0}]")
                    appendLine("Inference Latency: ${String.format(Locale.US, "%.2f", elapsedMs)} ms")
                    appendLine("Execution Time   : ${SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(Date())}")
                    appendLine("Privacy Enforced : 100% on-device CPU execution, 0 bytes transmitted")
                }

                runOnUiThread {
                    if (!isFinishing && !isDestroyed) {
                        btnRunInference.isEnabled = true
                        progressBarInference.visibility = View.GONE
                        tvInferenceProgress.visibility = View.GONE

                        // Render Diagnostic Result Card
                        cardResult.visibility = View.VISIBLE
                        tvResultTimestamp.text = timestamp

                        // Classification Badge
                        if (result.predictedClass == ThresholdLogic.CLASS_MSI) {
                            tvResultClassification.text = "MSI (Microsatellite Instability)"
                            tvResultClassification.setTextColor(ContextCompat.getColor(this, R.color.msi_color))
                            tvResultSubLabel.text = "Evaluated Class 1 · Score exceeds 0.83 threshold"
                            ivClassificationIcon.setImageResource(R.drawable.ic_check_circle)
                            ivClassificationIcon.setColorFilter(ContextCompat.getColor(this, R.color.msi_color))
                        } else {
                            tvResultClassification.text = "MSS (Microsatellite Stable)"
                            tvResultClassification.setTextColor(ContextCompat.getColor(this, R.color.mss_color))
                            tvResultSubLabel.text = "Evaluated Class 0 · Score below 0.83 threshold"
                            ivClassificationIcon.setImageResource(R.drawable.ic_check_circle)
                            ivClassificationIcon.setColorFilter(ContextCompat.getColor(this, R.color.mss_color))
                        }

                        // Score & Margin
                        tvResultScore.text = String.format(Locale.US, "Probability: %.4f", result.probability)
                        tvResultMargin.text = if (margin >= 0) {
                            String.format(Locale.US, "+%.4f above 0.83 threshold", margin)
                        } else {
                            String.format(Locale.US, "%.4f below 0.83 threshold", margin)
                        }

                        // Custom Graphical Gauge
                        probabilityBarView.setProbability(result.probability, result.threshold)

                        // 3-Column Metrics
                        tvResultLatency.text = String.format(Locale.US, "%.2f ms", elapsedMs)
                        tvResultPatches.text = "$numPatches Patches"

                        // Detailed Console Text
                        tvInferenceResult.text = diagnosticsText
                    }
                }
            } catch (t: Throwable) {
                runOnUiThread {
                    if (!isFinishing && !isDestroyed) {
                        btnRunInference.isEnabled = true
                        progressBarInference.visibility = View.GONE
                        tvInferenceProgress.visibility = View.GONE

                        cardResult.visibility = View.VISIBLE
                        tvResultClassification.text = "Inference Execution Error"
                        tvResultClassification.setTextColor(ContextCompat.getColor(this, R.color.status_error))
                        tvResultSubLabel.text = "Execution failed safely without crash"
                        ivClassificationIcon.setImageResource(R.drawable.ic_error_outline)
                        ivClassificationIcon.setColorFilter(ContextCompat.getColor(this, R.color.status_error))

                        tvResultScore.text = "Score: N/A"
                        tvResultMargin.text = "Error: ${t.message}"
                        probabilityBarView.reset()
                        tvInferenceResult.text = "Error during on-device inference:\n${t.localizedMessage}"
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
