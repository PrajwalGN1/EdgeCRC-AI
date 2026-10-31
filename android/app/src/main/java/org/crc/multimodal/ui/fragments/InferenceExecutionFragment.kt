package org.crc.multimodal.ui.fragments

import android.content.Context
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import com.google.android.material.button.MaterialButton
import com.google.android.material.button.MaterialButtonToggleGroup
import com.google.android.material.card.MaterialCardView
import org.crc.multimodal.R
import org.crc.multimodal.data.CaseStatus
import org.crc.multimodal.data.InferenceRun
import org.crc.multimodal.data.ResearchCase
import org.crc.multimodal.data.SpecimenInput
import org.crc.multimodal.inference.AuthenticBundleLoader
import org.crc.multimodal.inference.AuthenticFeatureBundle
import org.crc.multimodal.inference.BundleStorageManager
import org.crc.multimodal.inference.CrcLogger
import org.crc.multimodal.inference.InferenceProtocol
import org.crc.multimodal.inference.InferenceResult
import org.crc.multimodal.inference.InputProvenance
import org.crc.multimodal.inference.InputValidator
import org.crc.multimodal.inference.ReferenceCaseData
import org.crc.multimodal.inference.ReferenceTensorLoader
import org.crc.multimodal.inference.ThresholdLogic
import org.crc.multimodal.ui.WorkstationNavigationHost
import java.io.InputStream
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs

/**
 * On-device multimodal fusion inference execution fragment.
 *
 * Strict Invariants:
 * 1. Requires verified authentic feature embeddings (either imported bundle or frozen benchmark reference).
 * 2. Absolutely zero synthetic or fabricated fallback vectors. Fails closed if authentic inputs are missing.
 * 3. Enforces single-patient sequential execution on the frozen FP32 ONNX model.
 * 4. Fixed decision threshold tau = 0.83f (MSI >= 0.83, MSS < 0.83).
 */
class InferenceExecutionFragment : Fragment(R.layout.fragment_inference_execution) {

    private var navHost: WorkstationNavigationHost? = null
    private var caseId: Long = 0L
    private val backgroundExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    private var currentCase: ResearchCase? = null
    private var selectedProtocol: InferenceProtocol = InferenceProtocol.PROTOCOL_1_CAPPED

    // UI references
    private var tvExecCaseHeader: TextView? = null
    private var tvExecInputSummary: TextView? = null
    private var toggleGroupProtocol: MaterialButtonToggleGroup? = null
    private var btnRunInferenceExec: MaterialButton? = null
    private var cardInputProvenance: MaterialCardView? = null
    private var ivProvenanceIcon: ImageView? = null
    private var tvProvenanceBadge: TextView? = null
    private var tvProvenanceMessage: TextView? = null
    private var btnImportBundleExec: MaterialButton? = null
    private var layoutExecProgress: LinearLayout? = null
    private var cardExecError: MaterialCardView? = null
    private var tvExecErrorDetails: TextView? = null

    private val openBundleLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        uri?.let { handleImportBundleUri(it) }
    }

    override fun onAttach(context: Context) {
        super.onAttach(context)
        if (context is WorkstationNavigationHost) {
            navHost = context
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        caseId = arguments?.getLong(ARG_CASE_ID) ?: 0L
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        navHost?.setTopToolbar(
            title = "Inference Configuration",
            subtitle = "On-Device Multimodal Fusion Engine",
            showBackButton = true
        )

        val host = navHost ?: return
        val rCase = host.getRepository().getCaseById(caseId) ?: run {
            host.navigateToWorkspace()
            return
        }
        currentCase = rCase

        tvExecCaseHeader = view.findViewById(R.id.tvExecCaseHeader)
        tvExecInputSummary = view.findViewById(R.id.tvExecInputSummary)
        toggleGroupProtocol = view.findViewById(R.id.toggleGroupProtocol)
        btnRunInferenceExec = view.findViewById(R.id.btnRunInferenceExec)

        cardInputProvenance = view.findViewById(R.id.cardInputProvenance)
        ivProvenanceIcon = view.findViewById(R.id.ivProvenanceIcon)
        tvProvenanceBadge = view.findViewById(R.id.tvProvenanceBadge)
        tvProvenanceMessage = view.findViewById(R.id.tvProvenanceMessage)
        btnImportBundleExec = view.findViewById(R.id.btnImportBundleExec)

        layoutExecProgress = view.findViewById(R.id.layoutExecProgress)
        cardExecError = view.findViewById(R.id.cardExecError)
        tvExecErrorDetails = view.findViewById(R.id.tvExecErrorDetails)

        selectedProtocol = rCase.protocol ?: InferenceProtocol.PROTOCOL_1_CAPPED
        if (selectedProtocol == InferenceProtocol.PROTOCOL_2_UNCAPPED) {
            toggleGroupProtocol?.check(R.id.btnProtocol2Uncapped)
        } else {
            toggleGroupProtocol?.check(R.id.btnProtocol1Capped)
        }

        toggleGroupProtocol?.addOnButtonCheckedListener { _, checkedId, isChecked ->
            if (isChecked) {
                selectedProtocol = if (checkedId == R.id.btnProtocol2Uncapped) {
                    InferenceProtocol.PROTOCOL_2_UNCAPPED
                } else {
                    InferenceProtocol.PROTOCOL_1_CAPPED
                }
            }
        }

        btnImportBundleExec?.setOnClickListener {
            openBundleLauncher.launch(arrayOf("*/*"))
        }

        btnRunInferenceExec?.setOnClickListener {
            executeAuthenticInference()
        }

        refreshCaseAndProvenanceUI()
    }

    private fun refreshCaseAndProvenanceUI() {
        val cCase = currentCase ?: return
        val context = context ?: return

        tvExecCaseHeader?.text = "Case: ${cCase.caseIdentifier}"
        tvExecInputSummary?.text = "Vision sequence: [1, ${cCase.patchCount}, 1024] · Text summary: [1, 768]"

        val isBenchmark = ReferenceTensorLoader.isReferenceCase(cCase.caseIdentifier)
        val hasImportedBundle = BundleStorageManager.hasBundle(context, cCase.caseIdentifier)

        if (hasImportedBundle) {
            val bundle = BundleStorageManager.loadBundle(context, cCase.caseIdentifier)
            val patchCount = bundle?.patchCount ?: cCase.patchCount

            cardInputProvenance?.setCardBackgroundColor(ContextCompat.getColor(context, R.color.status_success_bg))
            cardInputProvenance?.strokeColor = ContextCompat.getColor(context, R.color.status_success)
            ivProvenanceIcon?.setImageResource(R.drawable.ic_check_circle)
            ivProvenanceIcon?.setColorFilter(ContextCompat.getColor(context, R.color.status_success))
            tvProvenanceBadge?.setTextColor(ContextCompat.getColor(context, R.color.status_success))
            tvProvenanceBadge?.text = "AUTHENTIC EMBEDDINGS VERIFIED (IMPORTED)"
            tvProvenanceMessage?.setTextColor(ContextCompat.getColor(context, R.color.status_success))
            tvProvenanceMessage?.text = "Specimen: ${bundle?.specimenId ?: cCase.caseIdentifier} · $patchCount patches · Shapes [1, $patchCount, 1024] & [1, 768] · Verified SHA-256"

            btnImportBundleExec?.visibility = View.GONE
            btnRunInferenceExec?.isEnabled = true
            btnRunInferenceExec?.text = getString(R.string.btn_run_inference)
        } else if (isBenchmark) {
            cardInputProvenance?.setCardBackgroundColor(ContextCompat.getColor(context, R.color.surface_variant_light))
            cardInputProvenance?.strokeColor = ContextCompat.getColor(context, R.color.primary)
            ivProvenanceIcon?.setImageResource(R.drawable.ic_check_circle)
            ivProvenanceIcon?.setColorFilter(ContextCompat.getColor(context, R.color.primary))
            tvProvenanceBadge?.setTextColor(ContextCompat.getColor(context, R.color.primary))
            tvProvenanceBadge?.text = "AUTHENTIC BENCHMARK REFERENCE (HOLDOUT)"
            tvProvenanceMessage?.setTextColor(ContextCompat.getColor(context, R.color.text_secondary_light))
            tvProvenanceMessage?.text = "Frozen validation holdout asset · ${cCase.patchCount} patches · Shapes [1, ${cCase.patchCount}, 1024] & [1, 768] · Exact parity target"

            btnImportBundleExec?.visibility = View.GONE
            btnRunInferenceExec?.isEnabled = true
            btnRunInferenceExec?.text = getString(R.string.btn_run_inference)
        } else {
            // STRICT FAIL-CLOSED STATE: No authentic tensors found
            cardInputProvenance?.setCardBackgroundColor(ContextCompat.getColor(context, R.color.status_error_bg))
            cardInputProvenance?.strokeColor = ContextCompat.getColor(context, R.color.status_error)
            ivProvenanceIcon?.setImageResource(R.drawable.ic_error_outline)
            ivProvenanceIcon?.setColorFilter(ContextCompat.getColor(context, R.color.status_error))
            tvProvenanceBadge?.setTextColor(ContextCompat.getColor(context, R.color.status_error))
            tvProvenanceBadge?.text = "AUTHENTIC FEATURE EMBEDDINGS REQUIRED"
            tvProvenanceMessage?.setTextColor(ContextCompat.getColor(context, R.color.status_error))
            tvProvenanceMessage?.text = "Authentic feature embeddings are required. Import a validated vision and clinical-text tensor pair before running inference."

            btnImportBundleExec?.visibility = View.VISIBLE
            btnRunInferenceExec?.isEnabled = false
            btnRunInferenceExec?.text = "Authentic Inputs Required"
        }
    }

    private fun handleImportBundleUri(uri: Uri) {
        val host = navHost ?: return
        val cCase = currentCase ?: return

        try {
            val stream: InputStream = requireContext().contentResolver.openInputStream(uri)
                ?: throw IllegalArgumentException("Cannot open stream from selected URI: $uri")
            val bundle = stream.use { AuthenticBundleLoader.loadFromStream(it) }

            // Save bundle to local storage
            BundleStorageManager.saveBundle(requireContext(), bundle, overwrite = true)

            // Register specimen input in database
            host.getRepository().insertSpecimenInput(
                SpecimenInput(
                    caseId = cCase.id,
                    inputType = "AUTHENTIC_BUNDLE",
                    localUriOrAsset = BundleStorageManager.getBundleFile(requireContext(), bundle.specimenId)?.absolutePath ?: "",
                    filename = "${bundle.specimenId}.crcbundle",
                    validationStatus = "VERIFIED_AUTHENTIC"
                )
            )

            // Update case patch count and protocol
            val updatedCase = cCase.copy(
                caseIdentifier = bundle.specimenId,
                patchCount = bundle.patchCount,
                protocol = if (bundle.patchCount <= 1000) InferenceProtocol.PROTOCOL_1_CAPPED else InferenceProtocol.PROTOCOL_2_UNCAPPED,
                status = CaseStatus.READY_FOR_ANALYSIS,
                updatedAt = System.currentTimeMillis()
            )
            host.getRepository().updateCase(updatedCase)
            currentCase = updatedCase

            refreshCaseAndProvenanceUI()
            Toast.makeText(requireContext(), "Authentic feature bundle imported for ${bundle.specimenId}", Toast.LENGTH_SHORT).show()
        } catch (t: Throwable) {
            cardExecError?.visibility = View.VISIBLE
            tvExecErrorDetails?.text = "Bundle import rejected: ${t.localizedMessage ?: "Invalid or corrupt file."}"
            Toast.makeText(requireContext(), "Failed to import bundle: ${t.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun executeAuthenticInference() {
        val host = navHost ?: return
        val cCase = currentCase ?: return
        val classifier = host.getClassifier()

        if (classifier == null) {
            cardExecError?.visibility = View.VISIBLE
            tvExecErrorDetails?.text = "ONNX classifier engine is not ready. Please wait for model initialization."
            return
        }

        val context = context ?: return
        val isBenchmarkCase = ReferenceTensorLoader.isReferenceCase(cCase.caseIdentifier)
        val hasImportedBundle = BundleStorageManager.hasBundle(context, cCase.caseIdentifier)

        // FAIL-CLOSED GATE: Refuse to execute if no authentic tensors exist
        if (!isBenchmarkCase && !hasImportedBundle) {
            cardExecError?.visibility = View.VISIBLE
            tvExecErrorDetails?.text = "Authentic feature embeddings are required. Import a validated vision and clinical-text tensor pair before running inference."
            return
        }

        // Prevent concurrent duplicate inference for the same specimen
        if (!runningCaseInferences.add(cCase.id)) {
            Toast.makeText(context, "Inference is already executing for case ${cCase.caseIdentifier}", Toast.LENGTH_SHORT).show()
            return
        }

        // Lock controls to prevent duplicate executions
        btnRunInferenceExec?.isEnabled = false
        toggleGroupProtocol?.isEnabled = false
        layoutExecProgress?.visibility = View.VISIBLE
        cardExecError?.visibility = View.GONE

        val protocol = selectedProtocol

        backgroundExecutor.execute {
            try {
                val visionData: FloatArray
                val textData: FloatArray
                val executionPatches: Int
                val provenance: InputProvenance
                var refCaseData: ReferenceCaseData? = null
                val bundleVisionSha256: String
                val bundleTextSha256: String

                if (isBenchmarkCase) {
                    refCaseData = ReferenceTensorLoader.loadFromAssets(requireContext().assets, cCase.caseIdentifier)
                    visionData = refCaseData.visionData
                    textData = refCaseData.textData
                    executionPatches = refCaseData.numPatches
                    provenance = InputProvenance.BENCHMARK_REFERENCE
                    bundleVisionSha256 = "BENCHMARK_REFERENCE_ASSET"
                    bundleTextSha256 = "BENCHMARK_REFERENCE_ASSET"
                } else {
                    val bundle = BundleStorageManager.loadBundle(requireContext(), cCase.caseIdentifier)
                        ?: throw IllegalStateException("Authentic feature embeddings are required. Import a validated vision and clinical-text tensor pair before running inference.")
                    visionData = bundle.visionData
                    textData = bundle.textData
                    executionPatches = bundle.patchCount
                    provenance = InputProvenance.AUTHENTIC_IMPORTED
                    bundleVisionSha256 = bundle.visionSha256
                    bundleTextSha256 = bundle.textSha256
                }

                // Strict input revalidation: dimensions, patch bounds, finiteness, buffer length
                InputValidator.validateVision(visionData, executionPatches, protocol)
                InputValidator.validateText(textData)

                val startNano = System.nanoTime()
                val result: InferenceResult = classifier.predictAuthentic(
                    visionData = visionData,
                    numPatches = executionPatches,
                    textData = textData,
                    caseId = cCase.caseIdentifier,
                    provenance = provenance,
                    protocol = protocol
                )
                val elapsedMs = (System.nanoTime() - startNano) / 1_000_000.0

                // Structural and finiteness verification of outputs
                if (result.probability.isNaN() || result.probability.isInfinite() ||
                    result.logit.isNaN() || result.logit.isInfinite()) {
                    throw IllegalStateException("Model produced non-finite output: logit=${result.logit}, prob=${result.probability}")
                }

                val label = ThresholdLogic.labelFor(result.predictedClass)
                val margin = ThresholdLogic.signedDistanceToThreshold(result.probability)

                val parityPass: Boolean
                val logitDiff: Float
                val probDiff: Float
                if (refCaseData != null) {
                    logitDiff = abs(result.logit - refCaseData.refLogit)
                    probDiff = abs(result.probability - refCaseData.refProb)
                    parityPass = (probDiff <= 1.0e-4f) && (result.predictedClass == refCaseData.refCall)

                    CrcLogger.i(
                        "[PARITY_GATE] Case: ${cCase.caseIdentifier} | " +
                        "OnDeviceLogit: ${String.format(java.util.Locale.US, "%.5f", result.logit)} | " +
                        "RefLogit: ${String.format(java.util.Locale.US, "%.5f", refCaseData.refLogit)} | " +
                        "DiffLogit: ${String.format(java.util.Locale.US, "%.2e", logitDiff)} | " +
                        "OnDeviceProb: ${String.format(java.util.Locale.US, "%.5f", result.probability)} | " +
                        "RefProb: ${String.format(java.util.Locale.US, "%.5f", refCaseData.refProb)} | " +
                        "DiffProb: ${String.format(java.util.Locale.US, "%.2e", probDiff)} | " +
                        "Tolerance: 1.0e-4 | ParityResult: ${if (parityPass) "PASS" else "FAIL"}"
                    )
                } else {
                    logitDiff = 0.0f
                    probDiff = 0.0f
                    parityPass = true
                }

                val diagnosticsSummary = buildString {
                    appendLine("=== RESEARCH ON-DEVICE INFERENCE REPORT ===")
                    appendLine("Execution Status : SUCCESS (Authentic Multimodal Embeddings)")
                    appendLine("Input Provenance : ${provenance.displayName}")
                    if (refCaseData != null) {
                        appendLine("Tensor Origin    : Frozen Validation Holdout Asset")
                        appendLine("Holdout GroundTr : ${if (refCaseData.gtLabel == 1) "MSI (1)" else "MSS (0)"}")
                        appendLine("Parity Gate 1e-4 : ${if (parityPass) "PASS (Tolerance <= 1.0e-4)" else "FAIL"}")
                        appendLine("Ref Logit Target : ${String.format(java.util.Locale.US, "%.5f", refCaseData.refLogit)} (Diff: ${String.format(java.util.Locale.US, "%.2e", logitDiff)})")
                        appendLine("Ref Prob Target  : ${String.format(java.util.Locale.US, "%.5f", refCaseData.refProb)} (Diff: ${String.format(java.util.Locale.US, "%.2e", probDiff)})")
                    } else {
                        appendLine("Tensor Origin    : Verified Imported External Feature Bundle")
                        appendLine("Vision Digest    : $bundleVisionSha256")
                        appendLine("Text Digest      : $bundleTextSha256")
                        appendLine("Integrity Status : SHA-256 Checksums Verified")
                    }
                    appendLine("Case Identifier  : ${cCase.caseIdentifier}")
                    appendLine("Slide Patches (N): $executionPatches (dim: 1024)")
                    appendLine("Text Embedding   : 1 × 768 (dim: 768)")
                    appendLine("Evaluation Mode  : ${InferenceProtocol.formatProtocol(result.protocol)}")
                    appendLine("Output Logit     : ${String.format(java.util.Locale.US, "%.5f", result.logit)}")
                    appendLine("Output Prob      : ${String.format(java.util.Locale.US, "%.5f", result.probability)}")
                    appendLine("Decision Boundary: 0.8300 (Fixed, Class 1 MSI >= 0.83, Class 0 MSS < 0.83)")
                    appendLine("Threshold Margin : ${String.format(java.util.Locale.US, "%+.5f", margin)} vs 0.83")
                    appendLine("Classification   : $label (Class ${result.predictedClass})")
                    appendLine("Inference Latency: ${String.format(java.util.Locale.US, "%.2f", elapsedMs)} ms")
                    appendLine("Engine Execution : Microsoft ONNX Runtime CPUProvider")
                    appendLine("Model Artifact   : fusion_model_fp32.onnx (21,015,452 bytes FP32)")
                    appendLine("Model SHA-256    : 0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73")
                    appendLine("Privacy Enforced : 100% on-device CPU execution, 0 bytes transmitted")
                    appendLine("Clinical Warning : RESEARCH USE ONLY. Not an approved in vitro diagnostic device.")
                }

                val archString = if (refCaseData != null) {
                    "LOCAL_ONNX_FP32 (BENCHMARK_REFERENCE)"
                } else {
                    "LOCAL_ONNX_FP32 (AUTHENTIC_IMPORTED)"
                }

                // Construct InferenceRun
                val inferenceRun = InferenceRun(
                    caseId = cCase.id,
                    modelIdentifier = "fusion_model_fp32.onnx",
                    inferenceArchitecture = archString,
                    protocol = protocol,
                    predictedClass = result.predictedClass,
                    predictedLabel = label,
                    probability = result.probability,
                    logit = result.logit,
                    threshold = ThresholdLogic.OPERATING_THRESHOLD,
                    margin = margin,
                    latencyMs = elapsedMs,
                    numPatches = executionPatches,
                    diagnosticsSummary = diagnosticsSummary,
                    executionTimestamp = System.currentTimeMillis(),
                    status = "SUCCESS"
                )

                // Transactional persistence: atomically inserts InferenceRun and updates case status to ANALYZED
                val runId = host.getRepository().recordInferenceResult(
                    caseId = cCase.id,
                    run = inferenceRun,
                    newStatus = CaseStatus.ANALYZED
                )

                activity?.runOnUiThread {
                    if (isAdded && !isDetached) {
                        btnRunInferenceExec?.isEnabled = true
                        toggleGroupProtocol?.isEnabled = true
                        layoutExecProgress?.visibility = View.GONE
                        host.navigateToFindings(cCase.id, runId)
                    }
                }
            } catch (t: Throwable) {
                // Transactional failure record: ensures case is not falsely marked complete
                try {
                    host.getRepository().recordInferenceFailure(cCase.id, t.message ?: "Unknown inference failure")
                } catch (persEx: Throwable) {
                    CrcLogger.e("Failed to record inference failure in database", persEx)
                }

                activity?.runOnUiThread {
                    if (isAdded && !isDetached) {
                        btnRunInferenceExec?.isEnabled = true
                        toggleGroupProtocol?.isEnabled = true
                        layoutExecProgress?.visibility = View.GONE
                        cardExecError?.visibility = View.VISIBLE
                        tvExecErrorDetails?.text = "Inference execution failure: ${t.localizedMessage}"
                    }
                }
            } finally {
                runningCaseInferences.remove(cCase.id)
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        backgroundExecutor.shutdownNow()
    }

    companion object {
        private const val ARG_CASE_ID = "arg_case_id"
        private val runningCaseInferences = ConcurrentHashMap.newKeySet<Long>()

        fun newInstance(caseId: Long): InferenceExecutionFragment {
            return InferenceExecutionFragment().apply {
                arguments = Bundle().apply {
                    putLong(ARG_CASE_ID, caseId)
                }
            }
        }
    }
}
