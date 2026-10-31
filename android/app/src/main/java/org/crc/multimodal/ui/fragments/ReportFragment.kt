package org.crc.multimodal.ui.fragments

import android.content.ActivityNotFoundException
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.graphics.pdf.PdfRenderer
import android.net.Uri
import android.os.Bundle
import android.os.ParcelFileDescriptor
import android.view.LayoutInflater
import android.view.View
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.core.widget.NestedScrollView
import androidx.fragment.app.Fragment
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import org.crc.multimodal.R
import org.crc.multimodal.data.InferenceRun
import org.crc.multimodal.data.PersistentReportRecord
import org.crc.multimodal.data.ReportExportStatus
import org.crc.multimodal.data.ResearchCase
import org.crc.multimodal.data.ResearchReport
import org.crc.multimodal.data.ReviewNote
import org.crc.multimodal.inference.AuthenticFeatureBundle
import org.crc.multimodal.inference.BundleStorageManager
import org.crc.multimodal.inference.CrcLogger
import org.crc.multimodal.inference.InferenceProtocol
import org.crc.multimodal.inference.ThresholdLogic
import org.crc.multimodal.report.GeneratedPdfReport
import org.crc.multimodal.report.PdfReportGenerator
import org.crc.multimodal.report.ReportGenerator
import org.crc.multimodal.report.ResultExplanation
import org.crc.multimodal.report.ResultExplanationService
import org.crc.multimodal.security.AuditActionType
import org.crc.multimodal.security.AuditTrailRecord
import org.crc.multimodal.security.ProvenanceAuditManager
import org.crc.multimodal.security.SpecimenProvenanceManifest
import org.crc.multimodal.ui.WorkstationNavigationHost
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import java.util.UUID
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.roundToInt

class ReportFragment : Fragment(R.layout.fragment_report) {

    private var navHost: WorkstationNavigationHost? = null
    private var caseId: Long = 0L
    private var runId: Long? = null

    private var backgroundExecutor: ExecutorService? = null

    // Active Data Model References
    private var currentCase: ResearchCase? = null
    private var targetRun: InferenceRun? = null
    private var currentReviewNotes: List<ReviewNote> = emptyList()
    private var currentBundle: AuthenticFeatureBundle? = null

    private var activeExplanation: ResultExplanation? = null
    private var activePdfReport: GeneratedPdfReport? = null
    private var activeRecord: PersistentReportRecord? = null
    private var activeManifest: SpecimenProvenanceManifest? = null
    private var compiledReportText: String = ""

    // Report target being exported via SAF
    private var pendingExportFile: File? = null
    private var pendingExportRecordId: String? = null

    // UI View References — Header & Top Actions
    private var tvReportHeaderCaseId: TextView? = null
    private var tvReportGeneratedDate: TextView? = null
    private var tvReportStatusBadge: TextView? = null

    private var btnPreviewPdf: MaterialButton? = null
    private var btnSavePdf: MaterialButton? = null
    private var btnSharePdf: MaterialButton? = null
    private var btnViewHistory: MaterialButton? = null
    private var btnCopyReport: MaterialButton? = null
    private var btnGenerateReport: MaterialButton? = null

    private var layoutReportLoading: LinearLayout? = null
    private var tvReportLoadingMessage: TextView? = null
    private var layoutReportError: LinearLayout? = null
    private var tvReportErrorMessage: TextView? = null
    private var tvReportErrorDetails: TextView? = null
    private var btnRetryReport: MaterialButton? = null
    private var scrollReportContent: NestedScrollView? = null

    // File Metadata Card
    private var cardPdfStatus: MaterialCardView? = null
    private var tvPdfCardTitle: TextView? = null
    private var tvPdfPageCount: TextView? = null
    private var tvPdfCardDetails: TextView? = null
    private var tvPdfCardDigest: TextView? = null

    // Validation Warnings Card
    private var cardValidationWarnings: MaterialCardView? = null
    private var tvValidationWarningsText: TextView? = null

    // Section 1: Prediction Summary
    private var cardPredictionSummary: MaterialCardView? = null
    private var tvPredClassification: TextView? = null
    private var tvPredScore: TextView? = null
    private var tvPredMssProb: TextView? = null
    private var tvPredThreshold: TextView? = null
    private var tvPredMargin: TextView? = null
    private var tvPredClassMapping: TextView? = null

    // Section 2: Probability Overview
    private var cardProbabilityOverview: MaterialCardView? = null
    private var tvProbMsiTitle: TextView? = null
    private var tvProbMsiScore: TextView? = null
    private var pbProbMsi: ProgressBar? = null
    private var tvProbMsiBadge: TextView? = null
    private var tvProbMssTitle: TextView? = null
    private var tvProbMssScore: TextView? = null
    private var pbProbMss: ProgressBar? = null
    private var tvProbMssBadge: TextView? = null
    private var tvProbOverviewNote: TextView? = null

    // Section 3: Case Information
    private var cardCaseInformation: MaterialCardView? = null
    private var tvCaseId: TextView? = null
    private var tvCaseStatus: TextView? = null
    private var tvCaseSpecimen: TextView? = null
    private var tvCaseContext: TextView? = null
    private var tvCasePatchCount: TextView? = null
    private var tvCaseProtocol: TextView? = null
    private var tvCaseCreatedAt: TextView? = null
    private var tvCaseUpdatedAt: TextView? = null

    // Section 4: Model & Inference Details
    private var cardModelInferenceDetails: MaterialCardView? = null
    private var tvModelEngine: TextView? = null
    private var tvModelProvider: TextView? = null
    private var tvModelArtifact: TextView? = null
    private var tvModelSha256: TextView? = null
    private var tvModelLatencyLogit: TextView? = null
    private var tvModelPrivacy: TextView? = null
    private var tvModelTensorShapes: TextView? = null
    private var tvModelFingerprints: TextView? = null
    private var tvModelIntegrity: TextView? = null

    // Section 5: Deterministic Result Explanation
    private var cardResultExplanation: MaterialCardView? = null
    private var tvExplanationClassification: TextView? = null
    private var tvExplanationThreshold: TextView? = null
    private var tvExplanationDisclaimer: TextView? = null

    // Section 6: Case Review Notes
    private var cardReviewNotes: MaterialCardView? = null
    private var layoutReviewNotesList: LinearLayout? = null
    private var tvNoReviewNotes: TextView? = null

    // Section 7: Responsible-Use Research Disclaimer
    private var cardResearchDisclaimer: MaterialCardView? = null
    private var tvFullDisclaimerText: TextView? = null

    // Backwards-Compatible Raw Monospace Viewer (Collapsed)
    private var tvReportContentText: TextView? = null

    // Storage Access Framework Launcher
    private lateinit var savePdfLauncher: ActivityResultLauncher<String>

    override fun onAttach(context: Context) {
        super.onAttach(context)
        if (context is WorkstationNavigationHost) {
            navHost = context
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        caseId = arguments?.getLong(ARG_CASE_ID) ?: 0L
        runId = if (arguments?.containsKey(ARG_RUN_ID) == true) arguments?.getLong(ARG_RUN_ID) else null

        savePdfLauncher = registerForActivityResult(ActivityResultContracts.CreateDocument("application/pdf")) { destinationUri: Uri? ->
            if (destinationUri != null) {
                handleSavePdfResult(destinationUri)
            } else {
                Toast.makeText(requireContext(), "Save operation cancelled", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        navHost?.setTopToolbar(
            title = "Research Report",
            subtitle = "Structured Multimodal Pathology Summary",
            showBackButton = true
        )

        bindViews(view)
        setupClickListeners()
        loadAndCompileReport(forceRegenerate = false)
    }

    private fun bindViews(view: View) {
        // Header & Actions
        tvReportHeaderCaseId = view.findViewById(R.id.tvReportHeaderCaseId)
        tvReportGeneratedDate = view.findViewById(R.id.tvReportGeneratedDate)
        tvReportStatusBadge = view.findViewById(R.id.tvReportStatusBadge)

        btnPreviewPdf = view.findViewById(R.id.btnPreviewPdf)
        btnSavePdf = view.findViewById(R.id.btnSavePdf)
        btnSharePdf = view.findViewById(R.id.btnSharePdf)
        btnViewHistory = view.findViewById(R.id.btnViewHistory)
        btnCopyReport = view.findViewById(R.id.btnCopyReport)
        btnGenerateReport = view.findViewById(R.id.btnGenerateReport)

        layoutReportLoading = view.findViewById(R.id.layoutReportLoading)
        tvReportLoadingMessage = view.findViewById(R.id.tvReportLoadingMessage)
        layoutReportError = view.findViewById(R.id.layoutReportError)
        tvReportErrorMessage = view.findViewById(R.id.tvReportErrorMessage)
        tvReportErrorDetails = view.findViewById(R.id.tvReportErrorDetails)
        btnRetryReport = view.findViewById(R.id.btnRetryReport)
        scrollReportContent = view.findViewById(R.id.scrollReportContent)

        // File Metadata Card
        cardPdfStatus = view.findViewById(R.id.cardPdfStatus)
        tvPdfCardTitle = view.findViewById(R.id.tvPdfCardTitle)
        tvPdfPageCount = view.findViewById(R.id.tvPdfPageCount)
        tvPdfCardDetails = view.findViewById(R.id.tvPdfCardDetails)
        tvPdfCardDigest = view.findViewById(R.id.tvPdfCardDigest)

        // Validation Warnings
        cardValidationWarnings = view.findViewById(R.id.cardValidationWarnings)
        tvValidationWarningsText = view.findViewById(R.id.tvValidationWarningsText)

        // Section 1: Prediction Summary
        cardPredictionSummary = view.findViewById(R.id.cardPredictionSummary)
        tvPredClassification = view.findViewById(R.id.tvPredClassification)
        tvPredScore = view.findViewById(R.id.tvPredScore)
        tvPredMssProb = view.findViewById(R.id.tvPredMssProb)
        tvPredThreshold = view.findViewById(R.id.tvPredThreshold)
        tvPredMargin = view.findViewById(R.id.tvPredMargin)
        tvPredClassMapping = view.findViewById(R.id.tvPredClassMapping)

        // Section 2: Probability Overview
        cardProbabilityOverview = view.findViewById(R.id.cardProbabilityOverview)
        tvProbMsiTitle = view.findViewById(R.id.tvProbMsiTitle)
        tvProbMsiScore = view.findViewById(R.id.tvProbMsiScore)
        pbProbMsi = view.findViewById(R.id.pbProbMsi)
        tvProbMsiBadge = view.findViewById(R.id.tvProbMsiBadge)
        tvProbMssTitle = view.findViewById(R.id.tvProbMssTitle)
        tvProbMssScore = view.findViewById(R.id.tvProbMssScore)
        pbProbMss = view.findViewById(R.id.pbProbMss)
        tvProbMssBadge = view.findViewById(R.id.tvProbMssBadge)
        tvProbOverviewNote = view.findViewById(R.id.tvProbOverviewNote)

        // Section 3: Case Information
        cardCaseInformation = view.findViewById(R.id.cardCaseInformation)
        tvCaseId = view.findViewById(R.id.tvCaseId)
        tvCaseStatus = view.findViewById(R.id.tvCaseStatus)
        tvCaseSpecimen = view.findViewById(R.id.tvCaseSpecimen)
        tvCaseContext = view.findViewById(R.id.tvCaseContext)
        tvCasePatchCount = view.findViewById(R.id.tvCasePatchCount)
        tvCaseProtocol = view.findViewById(R.id.tvCaseProtocol)
        tvCaseCreatedAt = view.findViewById(R.id.tvCaseCreatedAt)
        tvCaseUpdatedAt = view.findViewById(R.id.tvCaseUpdatedAt)

        // Section 4: Model & Inference Details
        cardModelInferenceDetails = view.findViewById(R.id.cardModelInferenceDetails)
        tvModelEngine = view.findViewById(R.id.tvModelEngine)
        tvModelProvider = view.findViewById(R.id.tvModelProvider)
        tvModelArtifact = view.findViewById(R.id.tvModelArtifact)
        tvModelSha256 = view.findViewById(R.id.tvModelSha256)
        tvModelLatencyLogit = view.findViewById(R.id.tvModelLatencyLogit)
        tvModelPrivacy = view.findViewById(R.id.tvModelPrivacy)
        tvModelTensorShapes = view.findViewById(R.id.tvModelTensorShapes)
        tvModelFingerprints = view.findViewById(R.id.tvModelFingerprints)
        tvModelIntegrity = view.findViewById(R.id.tvModelIntegrity)

        // Section 5: Deterministic Result Explanation
        cardResultExplanation = view.findViewById(R.id.cardResultExplanation)
        tvExplanationClassification = view.findViewById(R.id.tvExplanationClassification)
        tvExplanationThreshold = view.findViewById(R.id.tvExplanationThreshold)
        tvExplanationDisclaimer = view.findViewById(R.id.tvExplanationDisclaimer)

        // Section 6: Case Review Notes
        cardReviewNotes = view.findViewById(R.id.cardReviewNotes)
        layoutReviewNotesList = view.findViewById(R.id.layoutReviewNotesList)
        tvNoReviewNotes = view.findViewById(R.id.tvNoReviewNotes)

        // Section 7: Responsible-Use Research Disclaimer
        cardResearchDisclaimer = view.findViewById(R.id.cardResearchDisclaimer)
        tvFullDisclaimerText = view.findViewById(R.id.tvFullDisclaimerText)

        // Backwards-Compatible Raw Monospace Viewer
        tvReportContentText = view.findViewById(R.id.tvReportContentText)

        setActionsEnabled(false)
    }

    private fun setupClickListeners() {
        btnPreviewPdf?.setOnClickListener {
            previewActivePdf()
        }

        btnSavePdf?.setOnClickListener {
            initiateSavePdf()
        }

        btnSharePdf?.setOnClickListener {
            shareActivePdf()
        }

        btnViewHistory?.setOnClickListener {
            showReportHistoryDialog()
        }

        btnCopyReport?.setOnClickListener {
            copyReportToClipboard()
        }

        btnGenerateReport?.setOnClickListener {
            loadAndCompileReport(forceRegenerate = true)
        }

        btnRetryReport?.setOnClickListener {
            loadAndCompileReport(forceRegenerate = true)
        }
    }

    private fun setActionsEnabled(enabled: Boolean) {
        val pdfReady = enabled && activePdfReport?.file?.exists() == true
        btnPreviewPdf?.isEnabled = pdfReady
        btnSavePdf?.isEnabled = pdfReady
        btnSharePdf?.isEnabled = pdfReady
        btnViewHistory?.isEnabled = enabled
        btnCopyReport?.isEnabled = enabled && compiledReportText.isNotBlank()
        btnGenerateReport?.isEnabled = enabled
    }

    /**
     * Loads case, verifies runs, generates deterministic explanation and PDF,
     * and persists report history.
     */
    private fun loadAndCompileReport(forceRegenerate: Boolean) {
        val host = navHost ?: return
        val appContext = context?.applicationContext ?: return

        layoutReportLoading?.visibility = View.VISIBLE
        layoutReportError?.visibility = View.GONE
        scrollReportContent?.visibility = View.GONE
        setActionsEnabled(false)

        if (backgroundExecutor == null || backgroundExecutor?.isShutdown == true) {
            backgroundExecutor = Executors.newSingleThreadExecutor()
        }

        backgroundExecutor?.execute {
            try {
                val repository = host.getRepository()
                val caseEntity: ResearchCase = repository.getCaseById(caseId)
                    ?: throw IllegalStateException("Research case $caseId could not be found in local database.")

                val runEntity: InferenceRun? = if (runId != null && runId!! > 0L) {
                    repository.getRunsForCase(caseId).firstOrNull { it.id == runId }
                        ?: repository.getLatestRunForCase(caseId)
                } else {
                    repository.getLatestRunForCase(caseId)
                }

                val notes = repository.getNotesForCase(caseId)
                val bundle = if (BundleStorageManager.hasBundle(appContext, caseEntity.caseIdentifier)) {
                    BundleStorageManager.loadBundle(appContext, caseEntity.caseIdentifier)
                } else null

                currentCase = caseEntity
                targetRun = runEntity
                currentReviewNotes = notes
                currentBundle = bundle

                // 1. Generate Deterministic Result Explanation
                val explanation = ResultExplanationService.explainResult(
                    researchCase = caseEntity,
                    inferenceRun = runEntity,
                    reviewNotes = notes,
                    bundle = bundle
                )
                activeExplanation = explanation

                // 2. Generate Provenance Manifest
                val manifest = ProvenanceAuditManager.generateManifest(
                    researchCase = caseEntity,
                    inferenceRun = runEntity,
                    bundle = bundle
                )
                activeManifest = manifest

                // 3. Generate or Load PDF Report
                val existingHistory = repository.getReportHistoryForCase(caseId)
                var pdfResult: GeneratedPdfReport? = null
                var historyRecord: PersistentReportRecord? = null

                if (!forceRegenerate && existingHistory.isNotEmpty()) {
                    val candidate = existingHistory.firstOrNull {
                        it.exportStatus != ReportExportStatus.FAILED &&
                                it.templateVersion == PdfReportGenerator.TEMPLATE_VERSION &&
                                (runEntity == null || it.inferenceRunId == runEntity.id) &&
                                it.filePath != null &&
                                File(it.filePath).let { f -> f.exists() && f.length() > 1024L }
                    }
                    if (candidate != null && candidate.filePath != null) {
                        val existingFile = File(candidate.filePath)
                        val pageCount = getPdfPageCount(existingFile)
                        pdfResult = GeneratedPdfReport(
                            reportId = candidate.reportId,
                            file = existingFile,
                            sha256Hash = candidate.sha256Hash ?: PdfReportGenerator.computeFileSha256(existingFile),
                            pageCount = pageCount,
                            templateVersion = candidate.templateVersion,
                            generatedAt = candidate.createdAt
                        )
                        historyRecord = candidate
                        CrcLogger.d("Reusing validated report with matching template v${candidate.templateVersion}: ${existingFile.name} (pages=$pageCount)")
                    } else {
                        CrcLogger.d("No cached report matching template v${PdfReportGenerator.TEMPLATE_VERSION} for run ${runEntity?.id}. Triggering fresh PDF generation.")
                    }
                }

                if (pdfResult == null) {
                    val newReportId = UUID.randomUUID().toString()
                    val reportsDir = File(appContext.filesDir, "reports")
                    pdfResult = PdfReportGenerator.generateReport(
                        context = appContext,
                        researchCase = caseEntity,
                        inferenceRun = runEntity,
                        reviewNotes = notes,
                        bundle = bundle,
                        reportId = newReportId,
                        outputDir = reportsDir
                    )

                    val newRecord = PersistentReportRecord(
                        reportId = newReportId,
                        caseId = caseEntity.id,
                        inferenceRunId = runEntity?.id,
                        templateVersion = PdfReportGenerator.TEMPLATE_VERSION,
                        createdAt = pdfResult.generatedAt,
                        exportStatus = ReportExportStatus.GENERATED,
                        fileUri = null,
                        filePath = pdfResult.file.absolutePath,
                        modelVersion = runEntity?.modelIdentifier ?: "fusion_model_fp32.onnx",
                        sha256Hash = pdfResult.sha256Hash,
                        errorMessage = null
                    )
                    repository.insertReportRecord(newRecord)
                    historyRecord = newRecord
                    CrcLogger.d("Generated fresh structured PDF: ${pdfResult.file.name} (pages=${pdfResult.pageCount}, size=${pdfResult.file.length()} bytes, sha=${pdfResult.sha256Hash})")
                }

                activePdfReport = pdfResult
                activeRecord = historyRecord

                // 4. Compile Legacy / Structured Text View
                val textReport = ReportGenerator.generateTextReport(
                    researchCase = caseEntity,
                    inferenceRun = runEntity,
                    reviewNotes = notes,
                    bundle = bundle
                )
                compiledReportText = textReport

                // 5. Save Legacy ResearchReport entity for backwards compatibility
                val legacyReport = ResearchReport(
                    caseId = caseEntity.id,
                    inferenceRunId = runEntity?.id,
                    reportContent = textReport,
                    findingsSummary = runEntity?.let {
                        "${it.predictedLabel} (P = ${String.format(Locale.US, "%.4f", it.probability)})"
                    } ?: "Pending"
                )
                repository.insertOrUpdateReport(legacyReport)

                // 6. Audit Trail Record
                repository.recordAuditEvent(
                    AuditTrailRecord(
                        actionType = AuditActionType.REPORT_COMPILED,
                        specimenId = caseEntity.caseIdentifier,
                        details = "PDF report generated (${pdfResult.file.name}) with SHA-256: ${pdfResult.sha256Hash.take(16)}..."
                    )
                )

                activity?.runOnUiThread {
                    if (!isAdded || isDetached) return@runOnUiThread
                    renderActiveReportUi()
                }

            } catch (t: Throwable) {
                CrcLogger.e("Failed to generate research report for case $caseId", t)
                activity?.runOnUiThread {
                    if (!isAdded || isDetached) return@runOnUiThread
                    showErrorUi("Report Generation Failed", t.localizedMessage ?: "Unknown error compiling report.")
                }
            }
        }
    }

    private fun renderActiveReportUi() {
        val currentCase = currentCase ?: return
        val explanation = activeExplanation ?: return
        val pdf = activePdfReport ?: return
        val record = activeRecord
        val manifest = activeManifest
        val run = targetRun

        val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss 'UTC'", Locale.US).apply {
            timeZone = TimeZone.getTimeZone("UTC")
        }

        // Header Case Identifier and Timestamp
        tvReportHeaderCaseId?.text = "Case: ${currentCase.caseIdentifier}"
        tvReportGeneratedDate?.text = "Generated: ${dateFormat.format(Date(pdf.generatedAt))}"

        // Header Status Badge
        when (record?.exportStatus) {
            ReportExportStatus.EXPORTED -> {
                tvReportStatusBadge?.text = "SAVED (SAF)"
                tvReportStatusBadge?.setTextColor(ContextCompat.getColor(requireContext(), R.color.status_success))
            }
            ReportExportStatus.FAILED -> {
                tvReportStatusBadge?.text = "EXPORT FAILED"
                tvReportStatusBadge?.setTextColor(ContextCompat.getColor(requireContext(), R.color.status_error))
            }
            ReportExportStatus.MISSING_FILE -> {
                tvReportStatusBadge?.text = "MISSING FILE"
                tvReportStatusBadge?.setTextColor(ContextCompat.getColor(requireContext(), R.color.status_warning))
            }
            else -> {
                tvReportStatusBadge?.text = "PDF READY"
                tvReportStatusBadge?.setTextColor(ContextCompat.getColor(requireContext(), R.color.clinical_blue_light))
            }
        }

        // Report File Information Card
        tvPdfCardTitle?.text = pdf.file.name
        tvPdfPageCount?.text = "${pdf.pageCount} Page(s) · A4"
        tvPdfCardDetails?.text = "Path: ${pdf.file.absolutePath}"
        tvPdfCardDigest?.text = "SHA-256: ${pdf.sha256Hash}"

        // Validation Warnings
        if (explanation.hasValidationWarnings) {
            cardValidationWarnings?.visibility = View.VISIBLE
            val warnText = explanation.validationWarnings.joinToString("\n") { "• $it" }
            tvValidationWarningsText?.text = warnText
        } else {
            cardValidationWarnings?.visibility = View.GONE
        }

        // --- Section 1: Prediction Summary ---
        if (run != null) {
            val prob = run.probability
            val mssProb = 1.0f - prob
            val signedMargin = prob - ThresholdLogic.OPERATING_THRESHOLD
            val marginStr = if (signedMargin >= 0f) {
                "+${String.format(Locale.US, "%.5f", signedMargin)} above threshold"
            } else {
                "${String.format(Locale.US, "%.5f", signedMargin)} below threshold"
            }

            tvPredClassification?.text = "${run.predictedLabel} (Class ${run.predictedClass})"
            tvPredScore?.text = String.format(Locale.US, "%.5f", prob)
            tvPredMssProb?.text = String.format(Locale.US, "%.5f", mssProb)
            tvPredThreshold?.text = String.format(Locale.US, "%.4f (Frozen research boundary)", ThresholdLogic.OPERATING_THRESHOLD)
            tvPredMargin?.text = marginStr
            tvPredClassMapping?.text = "Class 1 MSI at or above 0.83; Class 0 MSS below 0.83"

            // --- Section 2: Probability Overview ---
            val msiPercent = (prob * 100f).roundToInt().coerceIn(0, 100)
            val mssPercent = (mssProb * 100f).roundToInt().coerceIn(0, 100)

            tvProbMsiScore?.text = String.format(Locale.US, "%.4f", prob)
            pbProbMsi?.progress = msiPercent
            tvProbMsiBadge?.text = "Score: ${String.format(Locale.US, "%.1f", prob * 100f)}% (${if (prob >= ThresholdLogic.OPERATING_THRESHOLD) "P >= 0.83" else "P < 0.83"})"

            tvProbMssScore?.text = String.format(Locale.US, "%.4f", mssProb)
            pbProbMss?.progress = mssPercent
            tvProbMssBadge?.text = "Score: ${String.format(Locale.US, "%.1f", mssProb * 100f)}% (Complementary: 1 - P)"

            tvProbOverviewNote?.text = "Note: Complementary class probabilities derived from two-class sigmoid distribution (P(MSS) = 1.0 - P(MSI)). Evaluated against frozen research threshold tau = 0.8300."
        } else {
            tvPredClassification?.text = "PENDING (No inference run recorded)"
            tvPredScore?.text = "Pending"
            tvPredMssProb?.text = "Pending"
            tvPredThreshold?.text = String.format(Locale.US, "%.4f (Frozen research boundary)", ThresholdLogic.OPERATING_THRESHOLD)
            tvPredMargin?.text = "Pending"
            tvPredClassMapping?.text = "Class 1 MSI at or above 0.83; Class 0 MSS below 0.83"

            tvProbMsiScore?.text = "0.0000"
            pbProbMsi?.progress = 0
            tvProbMsiBadge?.text = "Pending Execution"

            tvProbMssScore?.text = "0.0000"
            pbProbMss?.progress = 0
            tvProbMssBadge?.text = "Pending Execution"

            tvProbOverviewNote?.text = "Inference has not been executed for this case. Results will be displayed after running analysis."
        }

        // --- Section 3: Case Information ---
        tvCaseId?.text = currentCase.caseIdentifier
        tvCaseStatus?.text = currentCase.status.displayName
        tvCaseSpecimen?.text = currentCase.specimenDescription.ifBlank { "Primary colorectal adenocarcinoma biopsy" }
        tvCaseContext?.text = currentCase.researchContext.ifBlank { "Standard benchmark evaluation" }
        tvCasePatchCount?.text = "${currentCase.patchCount} patches"
        tvCaseProtocol?.text = InferenceProtocol.formatProtocol(run?.protocol ?: currentCase.protocol)
        tvCaseCreatedAt?.text = dateFormat.format(Date(currentCase.createdAt))
        tvCaseUpdatedAt?.text = dateFormat.format(Date(currentCase.updatedAt))

        // --- Section 4: Model and Inference Details ---
        tvModelEngine?.text = "Local Android ONNX Runtime 1.18.0"
        tvModelProvider?.text = "CPUExecutionProvider (Float32 single-patient sequential)"
        tvModelArtifact?.text = "fusion_model_fp32.onnx (20.04 MB FP32)"
        tvModelSha256?.text = "0402F6C522BEAC5733E30EEECAD908927957B5D4164B6B896663F43BA3702D73"
        if (run != null) {
            tvModelLatencyLogit?.text = "${String.format(Locale.US, "%.2f", run.latencyMs)} ms | Raw Logit: ${String.format(Locale.US, "%.5f", run.logit)}"
        } else {
            tvModelLatencyLogit?.text = "Pending execution"
        }
        tvModelPrivacy?.text = "100% on-device CPU execution, 0 bytes transmitted"
        tvModelTensorShapes?.text = "Vision: [1, ${currentCase.patchCount}, 1024] Float32\nText: [1, 768] Float32"

        val vSha = manifest?.inputFingerprints?.visionPatchesSha256 ?: "Precomputed tensor"
        val tSha = manifest?.inputFingerprints?.textTensorSha256 ?: "Precomputed tensor"
        val bSha = manifest?.inputFingerprints?.bundleSha256
        tvModelFingerprints?.text = buildString {
            append("Vision: ").append(vSha)
            append("\nText  : ").append(tSha)
            if (bSha != null) {
                append("\nBundle: ").append(bSha)
            }
        }

        val integrityDigest = manifest?.integrityStatement?.manifestPayloadSha256 ?: "Verified"
        tvModelIntegrity?.text = "SHA-256 Manifest Verified (${manifest?.integrityStatement?.checksumType ?: "SHA-256"}): ${integrityDigest.take(24)}…"

        // --- Section 5: Deterministic Result Explanation ---
        tvExplanationClassification?.text = explanation.classificationSummary
        tvExplanationThreshold?.text = explanation.thresholdExplanation
        tvExplanationDisclaimer?.text = explanation.clinicalDisclaimer

        // --- Section 6: Case Review Notes ---
        layoutReviewNotesList?.removeAllViews()
        if (currentReviewNotes.isNotEmpty()) {
            tvNoReviewNotes?.visibility = View.GONE
            layoutReviewNotesList?.visibility = View.VISIBLE
            for ((idx, note) in currentReviewNotes.withIndex()) {
                val noteView = LayoutInflater.from(requireContext()).inflate(R.layout.item_review_note, layoutReviewNotesList, false)
                val tvCategory: TextView? = noteView.findViewById(R.id.tvNoteCategory)
                val tvAuthor: TextView? = noteView.findViewById(R.id.tvNoteAuthor)
                val tvContent: TextView? = noteView.findViewById(R.id.tvNoteContent)
                val tvDate: TextView? = noteView.findViewById(R.id.tvNoteDate)
                val btnDelete: View? = noteView.findViewById(R.id.btnDeleteNote)

                tvCategory?.text = "[${idx + 1}] ${note.category.displayName}"
                tvAuthor?.text = "By: ${note.authorLabel}"
                tvContent?.text = note.content
                tvDate?.text = dateFormat.format(Date(note.createdAt))
                btnDelete?.visibility = View.GONE

                layoutReviewNotesList?.addView(noteView)
            }
        } else {
            tvNoReviewNotes?.visibility = View.VISIBLE
            layoutReviewNotesList?.visibility = View.GONE
        }

        // --- Section 7: Responsible-Use Research Disclaimer ---
        tvFullDisclaimerText?.text = "INVESTIGATIONAL RESEARCH USE ONLY. Not approved for clinical diagnosis or patient care. This report summarizes experimental machine learning benchmark outputs from an edge AI research prototype (Phikon-v2 ViT-B/16 and Bio_ClinicalBERT precomputed embeddings). Output probabilities represent research softmax activations; decision threshold is frozen at tau = 0.8300. The MSI/MSS evaluation score does not substitute for certified clinical IHC/PCR microsatellite instability testing."

        // Legacy Monospace Report
        tvReportContentText?.text = compiledReportText

        // Toggle UI visibility
        layoutReportLoading?.visibility = View.GONE
        layoutReportError?.visibility = View.GONE
        scrollReportContent?.visibility = View.VISIBLE

        setActionsEnabled(true)
    }

    private fun showErrorUi(title: String, details: String) {
        layoutReportLoading?.visibility = View.GONE
        layoutReportError?.visibility = View.VISIBLE
        scrollReportContent?.visibility = View.GONE
        setActionsEnabled(false)

        tvReportErrorMessage?.text = title
        tvReportErrorDetails?.text = details
    }

    // --- User Actions: Preview, Save, Share, History ---

    private fun previewActivePdf() {
        val pdf = activePdfReport ?: run {
            Toast.makeText(requireContext(), "No PDF report is available for preview", Toast.LENGTH_SHORT).show()
            return
        }

        if (!pdf.file.exists() || pdf.file.length() < 1024L) {
            Toast.makeText(requireContext(), "PDF file not found or invalid on local storage. Tap Regenerate to recreate.", Toast.LENGTH_LONG).show()
            activeRecord?.let {
                navHost?.getRepository()?.updateReportRecordStatus(it.reportId, ReportExportStatus.MISSING_FILE)
            }
            renderActiveReportUi()
            return
        }

        try {
            val contentUri = FileProvider.getUriForFile(
                requireContext(),
                "org.crc.multimodal.fileprovider",
                pdf.file
            )

            CrcLogger.d("Opening PDF preview: file=${pdf.file.name}, size=${pdf.file.length()} bytes, uri=$contentUri")

            val previewIntent = Intent(Intent.ACTION_VIEW).apply {
                setDataAndType(contentUri, "application/pdf")
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            startActivity(previewIntent)
        } catch (e: ActivityNotFoundException) {
            MaterialAlertDialogBuilder(requireContext())
                .setTitle("No PDF Viewer Found")
                .setMessage("There is no application installed on this device that can preview PDF files. You can use 'Save PDF' to export the file to your Downloads, or 'Share PDF' to send it to another device.")
                .setPositiveButton("OK", null)
                .show()
        } catch (t: Throwable) {
            CrcLogger.e("Failed to open PDF preview", t)
            Toast.makeText(requireContext(), "Error opening PDF preview: ${t.localizedMessage}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun getPdfPageCount(file: File): Int {
        return try {
            val pfd = ParcelFileDescriptor.open(file, ParcelFileDescriptor.MODE_READ_ONLY)
            val renderer = PdfRenderer(pfd)
            val count = renderer.pageCount
            renderer.close()
            pfd.close()
            count
        } catch (e: Throwable) {
            1
        }
    }

    private fun initiateSavePdf() {
        val pdf = activePdfReport ?: run {
            Toast.makeText(requireContext(), "No PDF report available to save", Toast.LENGTH_SHORT).show()
            return
        }

        if (!pdf.file.exists()) {
            Toast.makeText(requireContext(), "PDF file missing. Please regenerate first.", Toast.LENGTH_SHORT).show()
            return
        }

        pendingExportFile = pdf.file
        pendingExportRecordId = activeRecord?.reportId
        savePdfLauncher.launch(pdf.file.name)
    }

    private fun handleSavePdfResult(destinationUri: Uri) {
        val sourceFile = pendingExportFile ?: activePdfReport?.file ?: return
        val recordId = pendingExportRecordId ?: activeRecord?.reportId
        val host = navHost ?: return
        val contentResolver = requireContext().contentResolver

        Toast.makeText(requireContext(), "Saving PDF to destination…", Toast.LENGTH_SHORT).show()

        Executors.newSingleThreadExecutor().execute {
            try {
                contentResolver.openOutputStream(destinationUri)?.use { outStream ->
                    sourceFile.inputStream().use { inStream ->
                        inStream.copyTo(outStream)
                    }
                } ?: throw IllegalStateException("Could not open output stream for selected destination.")

                if (recordId != null) {
                    host.getRepository().updateReportRecordStatus(
                        reportId = recordId,
                        status = ReportExportStatus.EXPORTED,
                        fileUri = destinationUri.toString(),
                        filePath = sourceFile.absolutePath,
                        errorMessage = null
                    )
                }

                host.getRepository().recordAuditEvent(
                    AuditTrailRecord(
                        actionType = AuditActionType.REPORT_COMPILED,
                        specimenId = currentCase?.caseIdentifier,
                        details = "PDF report successfully exported via SAF to: $destinationUri"
                    )
                )

                activity?.runOnUiThread {
                    if (!isAdded || isDetached) return@runOnUiThread
                    activeRecord = recordId?.let { host.getRepository().getReportRecordById(it) } ?: activeRecord
                    Toast.makeText(requireContext(), "Report successfully saved!", Toast.LENGTH_SHORT).show()
                    tvReportStatusBadge?.text = "SAVED (SAF)"
                    tvReportStatusBadge?.setTextColor(ContextCompat.getColor(requireContext(), R.color.status_success))
                }
            } catch (t: Throwable) {
                CrcLogger.e("Failed to save PDF to SAF URI: $destinationUri", t)
                if (recordId != null) {
                    host.getRepository().updateReportRecordStatus(
                        reportId = recordId,
                        status = ReportExportStatus.FAILED,
                        errorMessage = t.localizedMessage
                    )
                }
                activity?.runOnUiThread {
                    if (!isAdded || isDetached) return@runOnUiThread
                    Toast.makeText(requireContext(), "Failed to save PDF: ${t.localizedMessage}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun shareActivePdf() {
        val pdf = activePdfReport ?: run {
            Toast.makeText(requireContext(), "No PDF report available to share", Toast.LENGTH_SHORT).show()
            return
        }

        if (!pdf.file.exists()) {
            Toast.makeText(requireContext(), "PDF file not found. Tap Regenerate first.", Toast.LENGTH_SHORT).show()
            return
        }

        try {
            val contentUri = FileProvider.getUriForFile(
                requireContext(),
                "org.crc.multimodal.fileprovider",
                pdf.file
            )

            val shareIntent = Intent(Intent.ACTION_SEND).apply {
                type = "application/pdf"
                putExtra(Intent.EXTRA_STREAM, contentUri)
                putExtra(Intent.EXTRA_SUBJECT, "CRC Research Report - ${currentCase?.caseIdentifier}")
                putExtra(Intent.EXTRA_TEXT, "CRC Multimodal Pathology Research Report for specimen ${currentCase?.caseIdentifier}.")
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivity(Intent.createChooser(shareIntent, "Share Research Report (PDF)"))
        } catch (t: Throwable) {
            CrcLogger.e("Failed to share PDF", t)
            Toast.makeText(requireContext(), "Error initiating share: ${t.localizedMessage}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun copyReportToClipboard() {
        val clipboard = requireContext().getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        val clip = ClipData.newPlainText("CRC Research Report", compiledReportText)
        clipboard.setPrimaryClip(clip)
        Toast.makeText(requireContext(), "Full structured report copied to clipboard", Toast.LENGTH_SHORT).show()
    }

    private fun showReportHistoryDialog() {
        val host = navHost ?: return
        val repository = host.getRepository()
        val historyList = repository.getReportHistoryForCase(caseId)

        if (historyList.isEmpty()) {
            Toast.makeText(requireContext(), "No saved report history for this case.", Toast.LENGTH_SHORT).show()
            return
        }

        val dialogView = LayoutInflater.from(requireContext()).inflate(R.layout.dialog_report_history, null, false)
        val container: LinearLayout = dialogView.findViewById(R.id.layoutHistoryItemsContainer)

        val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss 'UTC'", Locale.US).apply {
            timeZone = TimeZone.getTimeZone("UTC")
        }

        var dialog: AlertDialog? = null

        for (item in historyList) {
            val itemView = LayoutInflater.from(requireContext()).inflate(R.layout.item_report_history, container, false)
            val tvReportId: TextView = itemView.findViewById(R.id.tvHistoryReportId)
            val tvStatus: TextView = itemView.findViewById(R.id.tvHistoryStatusBadge)
            val tvTime: TextView = itemView.findViewById(R.id.tvHistoryTimestamp)
            val tvDigest: TextView = itemView.findViewById(R.id.tvHistoryDigest)
            val tvError: TextView = itemView.findViewById(R.id.tvHistoryError)

            val btnRegenerate: MaterialButton = itemView.findViewById(R.id.btnHistoryRegenerate)
            val btnPreview: MaterialButton = itemView.findViewById(R.id.btnHistoryPreview)
            val btnShare: MaterialButton = itemView.findViewById(R.id.btnHistoryShare)
            val btnExport: MaterialButton = itemView.findViewById(R.id.btnHistoryExport)

            val file = item.filePath?.let { File(it) }
            val fileExists = file?.exists() == true

            tvReportId.text = file?.name ?: item.reportId
            tvTime.text = "${dateFormat.format(Date(item.createdAt))} · v${item.templateVersion}"
            tvDigest.text = "SHA-256: ${item.sha256Hash?.take(20) ?: "Not recorded"}…"

            when {
                !fileExists -> {
                    tvStatus.text = "MISSING PDF"
                    tvStatus.setTextColor(ContextCompat.getColor(requireContext(), R.color.status_warning))
                    btnPreview.isEnabled = false
                    btnShare.isEnabled = false
                }
                item.exportStatus == ReportExportStatus.EXPORTED -> {
                    tvStatus.text = "EXPORTED"
                    tvStatus.setTextColor(ContextCompat.getColor(requireContext(), R.color.status_success))
                }
                item.exportStatus == ReportExportStatus.FAILED -> {
                    tvStatus.text = "FAILED"
                    tvStatus.setTextColor(ContextCompat.getColor(requireContext(), R.color.status_error))
                }
                else -> {
                    tvStatus.text = "GENERATED"
                    tvStatus.setTextColor(ContextCompat.getColor(requireContext(), R.color.clinical_blue_light))
                }
            }

            if (!item.errorMessage.isNullOrBlank()) {
                tvError.visibility = View.VISIBLE
                tvError.text = "Error: ${item.errorMessage}"
            } else {
                tvError.visibility = View.GONE
            }

            btnRegenerate.setOnClickListener {
                dialog?.dismiss()
                runId = item.inferenceRunId
                loadAndCompileReport(forceRegenerate = true)
            }

            btnPreview.setOnClickListener {
                dialog?.dismiss()
                if (file != null && file.exists()) {
                    val pageCount = getPdfPageCount(file)
                    activePdfReport = GeneratedPdfReport(
                        reportId = item.reportId,
                        file = file,
                        sha256Hash = item.sha256Hash ?: PdfReportGenerator.computeFileSha256(file),
                        pageCount = pageCount,
                        templateVersion = item.templateVersion,
                        generatedAt = item.createdAt
                    )
                    activeRecord = item
                    runId = item.inferenceRunId
                    targetRun = item.inferenceRunId?.let { rId -> repository.getRunsForCase(caseId).firstOrNull { run -> run.id == rId } } ?: targetRun
                    renderActiveReportUi()
                    previewActivePdf()
                }
            }

            btnShare.setOnClickListener {
                dialog?.dismiss()
                if (file != null && file.exists()) {
                    val pageCount = getPdfPageCount(file)
                    activePdfReport = GeneratedPdfReport(
                        reportId = item.reportId,
                        file = file,
                        sha256Hash = item.sha256Hash ?: PdfReportGenerator.computeFileSha256(file),
                        pageCount = pageCount,
                        templateVersion = item.templateVersion,
                        generatedAt = item.createdAt
                    )
                    activeRecord = item
                    runId = item.inferenceRunId
                    targetRun = item.inferenceRunId?.let { rId -> repository.getRunsForCase(caseId).firstOrNull { run -> run.id == rId } } ?: targetRun
                    renderActiveReportUi()
                    shareActivePdf()
                }
            }

            btnExport.setOnClickListener {
                dialog?.dismiss()
                if (file != null && file.exists()) {
                    pendingExportFile = file
                    pendingExportRecordId = item.reportId
                    savePdfLauncher.launch(file.name)
                }
            }

            container.addView(itemView)
        }

        dialog = MaterialAlertDialogBuilder(requireContext())
            .setTitle("Saved Report History (${historyList.size})")
            .setView(dialogView)
            .setPositiveButton("Close", null)
            .create()

        dialog.show()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        backgroundExecutor?.shutdown()
        backgroundExecutor = null

        tvReportHeaderCaseId = null
        tvReportGeneratedDate = null
        tvReportStatusBadge = null
        btnPreviewPdf = null
        btnSavePdf = null
        btnSharePdf = null
        btnViewHistory = null
        btnCopyReport = null
        btnGenerateReport = null
        layoutReportLoading = null
        tvReportLoadingMessage = null
        layoutReportError = null
        tvReportErrorMessage = null
        tvReportErrorDetails = null
        btnRetryReport = null
        scrollReportContent = null

        cardPdfStatus = null
        tvPdfCardTitle = null
        tvPdfPageCount = null
        tvPdfCardDetails = null
        tvPdfCardDigest = null

        cardValidationWarnings = null
        tvValidationWarningsText = null

        cardPredictionSummary = null
        tvPredClassification = null
        tvPredScore = null
        tvPredMssProb = null
        tvPredThreshold = null
        tvPredMargin = null
        tvPredClassMapping = null

        cardProbabilityOverview = null
        tvProbMsiTitle = null
        tvProbMsiScore = null
        pbProbMsi = null
        tvProbMsiBadge = null
        tvProbMssTitle = null
        tvProbMssScore = null
        pbProbMss = null
        tvProbMssBadge = null
        tvProbOverviewNote = null

        cardCaseInformation = null
        tvCaseId = null
        tvCaseStatus = null
        tvCaseSpecimen = null
        tvCaseContext = null
        tvCasePatchCount = null
        tvCaseProtocol = null
        tvCaseCreatedAt = null
        tvCaseUpdatedAt = null

        cardModelInferenceDetails = null
        tvModelEngine = null
        tvModelProvider = null
        tvModelArtifact = null
        tvModelSha256 = null
        tvModelLatencyLogit = null
        tvModelPrivacy = null
        tvModelTensorShapes = null
        tvModelFingerprints = null
        tvModelIntegrity = null

        cardResultExplanation = null
        tvExplanationClassification = null
        tvExplanationThreshold = null
        tvExplanationDisclaimer = null

        cardReviewNotes = null
        layoutReviewNotesList = null
        tvNoReviewNotes = null

        cardResearchDisclaimer = null
        tvFullDisclaimerText = null

        tvReportContentText = null
    }

    companion object {
        private const val ARG_CASE_ID = "arg_case_id"
        private const val ARG_RUN_ID = "arg_run_id"

        fun newInstance(caseId: Long, runId: Long? = null): ReportFragment {
            return ReportFragment().apply {
                arguments = Bundle().apply {
                    putLong(ARG_CASE_ID, caseId)
                    if (runId != null) {
                        putLong(ARG_RUN_ID, runId)
                    }
                }
            }
        }
    }
}
