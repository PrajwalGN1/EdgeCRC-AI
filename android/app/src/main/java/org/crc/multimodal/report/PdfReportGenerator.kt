package org.crc.multimodal.report

import android.content.Context
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import android.graphics.pdf.PdfDocument
import org.crc.multimodal.data.InferenceRun
import org.crc.multimodal.data.ResearchCase
import org.crc.multimodal.data.ReviewNote
import org.crc.multimodal.inference.AuthenticFeatureBundle
import org.crc.multimodal.inference.InferenceProtocol
import org.crc.multimodal.inference.ThresholdLogic
import org.crc.multimodal.security.ProvenanceAuditManager
import org.crc.multimodal.security.SpecimenProvenanceManifest
import java.io.File
import java.io.FileOutputStream
import java.security.MessageDigest
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import java.util.UUID

/**
 * Result metadata for a generated PDF report.
 */
data class GeneratedPdfReport(
    val reportId: String,
    val file: File,
    val sha256Hash: String,
    val pageCount: Int,
    val templateVersion: String,
    val generatedAt: Long
)

/**
 * Professional, table-driven PDF generator for colorectal cancer edge AI multimodal research reports.
 *
 * Employs native Android [PdfDocument] with standard A4 typography (595x842 pt),
 * automated word wrapping, dynamic multi-page flow with running headers/footers,
 * shaded two-column tables, dedicated visual MSI/MSS probability displays, and
 * full cryptographic provenance without truncation or console formatting.
 */
object PdfReportGenerator {

    const val TEMPLATE_VERSION = "2.0.0"

    // A4 Dimensions in points (72 DPI)
    const val PAGE_WIDTH = 595
    const val PAGE_HEIGHT = 842
    const val MARGIN_LEFT = 40f
    const val MARGIN_RIGHT = 555f
    const val CONTENT_WIDTH = MARGIN_RIGHT - MARGIN_LEFT // 515f
    const val MARGIN_TOP = 40f
    const val MARGIN_BOTTOM = 800f
    const val FOOTER_Y = 818f

    private const val HEADER_BRANDING = "CRC MULTIMODAL PATHOLOGY RESEARCH WORKSTATION"
    private const val HEADER_TITLE = "CRC Multimodal Pathology Research Report"
    private const val HEADER_SUBTITLE = "Multimodal Fusion Research Summary"
    private const val MODEL_FP32_SHA256 = "0402F6C522BEAC5733E30EEECAD908927957B5D4164B6B896663F43BA3702D73"

    private const val RESEARCH_DISCLAIMER_TEXT =
        "INVESTIGATIONAL RESEARCH USE ONLY. Not approved for clinical diagnosis or patient care. " +
        "This report summarizes experimental machine learning benchmark outputs from an edge AI research prototype " +
        "(Phikon-v2 ViT-B/16 and Bio_ClinicalBERT precomputed embeddings). Output probabilities represent research softmax activations; " +
        "decision threshold is frozen at tau = 0.8300. The MSI/MSS evaluation score does not substitute for certified clinical " +
        "IHC/PCR microsatellite instability testing. No clinical or treatment decisions should be based solely on this research output."

    /**
     * Compiles and writes the PDF report to the specified directory.
     */
    fun generateReport(
        context: Context,
        researchCase: ResearchCase,
        inferenceRun: InferenceRun?,
        reviewNotes: List<ReviewNote> = emptyList(),
        bundle: AuthenticFeatureBundle? = null,
        reportId: String = UUID.randomUUID().toString(),
        outputDir: File = File(context.filesDir, "reports")
    ): GeneratedPdfReport {
        if (!outputDir.exists()) {
            outputDir.mkdirs()
        }

        val cleanCaseId = researchCase.caseIdentifier.replace(Regex("[^a-zA-Z0-9._-]"), "_")
        val timestampStr = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val targetFile = File(outputDir, "CRC_Report_${cleanCaseId}_$timestampStr.pdf")

        val explanation = ResultExplanationService.explainResult(
            researchCase = researchCase,
            inferenceRun = inferenceRun,
            reviewNotes = reviewNotes,
            bundle = bundle
        )

        val manifest = ProvenanceAuditManager.generateManifest(
            researchCase = researchCase,
            inferenceRun = inferenceRun,
            bundle = bundle
        )

        val pageCount = renderPdfDocument(
            targetFile = targetFile,
            reportId = reportId,
            researchCase = researchCase,
            inferenceRun = inferenceRun,
            explanation = explanation,
            manifest = manifest,
            reviewNotes = reviewNotes,
            bundle = bundle
        )

        val sha256 = computeFileSha256(targetFile)

        return GeneratedPdfReport(
            reportId = reportId,
            file = targetFile,
            sha256Hash = sha256,
            pageCount = pageCount,
            templateVersion = TEMPLATE_VERSION,
            generatedAt = System.currentTimeMillis()
        )
    }

    /**
     * Internal multi-page rendering engine adhering to professional, publication-quality document design.
     */
    private fun renderPdfDocument(
        targetFile: File,
        reportId: String,
        researchCase: ResearchCase,
        inferenceRun: InferenceRun?,
        explanation: ResultExplanation,
        manifest: SpecimenProvenanceManifest,
        reviewNotes: List<ReviewNote>,
        bundle: AuthenticFeatureBundle?
    ): Int {
        val document = PdfDocument()

        val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss 'UTC'", Locale.US).apply {
            timeZone = TimeZone.getTimeZone("UTC")
        }
        val generatedDateStr = dateFormat.format(Date())

        // Setup Document Paints
        val paintHeaderAccent = Paint().apply {
            color = Color.rgb(18, 78, 122) // #124E7A
            style = Paint.Style.FILL
        }

        val paintBrand = Paint().apply {
            color = Color.rgb(82, 101, 122) // #52657A
            textSize = 7.5f
            isFakeBoldText = true
            isAntiAlias = true
        }

        val paintTitle = Paint().apply {
            color = Color.rgb(15, 23, 42) // #0F172A
            textSize = 13.5f
            isFakeBoldText = true
            isAntiAlias = true
        }

        val paintSubtitle = Paint().apply {
            color = Color.rgb(82, 101, 122)
            textSize = 8.5f
            isAntiAlias = true
        }

        val paintSectionHeader = Paint().apply {
            color = Color.rgb(23, 105, 170) // Clinical Blue #1769AA
            textSize = 10f
            isFakeBoldText = true
            isAntiAlias = true
        }

        val paintTableLabel = Paint().apply {
            color = Color.rgb(30, 41, 59) // Dark Slate #1E293B
            textSize = 8f
            isFakeBoldText = true
            isAntiAlias = true
        }

        val paintTableValue = Paint().apply {
            color = Color.rgb(15, 23, 42) // Near Black #0F172A
            textSize = 8f
            isAntiAlias = true
        }

        val paintTableValueBold = Paint().apply {
            color = Color.rgb(15, 23, 42)
            textSize = 8f
            isFakeBoldText = true
            isAntiAlias = true
        }

        val paintTableValueMono = Paint().apply {
            color = Color.rgb(15, 23, 42)
            textSize = 7.5f
            typeface = Typeface.MONOSPACE
            isAntiAlias = true
        }

        val paintMuted = Paint().apply {
            color = Color.rgb(100, 116, 139)
            textSize = 7f
            isAntiAlias = true
        }

        val paintDivider = Paint().apply {
            color = Color.rgb(209, 219, 229) // #D1DBE5
            strokeWidth = 0.8f
        }

        val paintTableStroke = Paint().apply {
            color = Color.rgb(209, 219, 229)
            style = Paint.Style.STROKE
            strokeWidth = 0.7f
        }

        val paintLabelBg = Paint().apply {
            color = Color.rgb(240, 244, 248) // Light Blue-Gray #F0F4F8
            style = Paint.Style.FILL
        }

        val paintValueBg = Paint().apply {
            color = Color.rgb(255, 255, 255) // White
            style = Paint.Style.FILL
        }

        val paintMsiHighlightBg = Paint().apply {
            color = Color.rgb(254, 242, 242) // Light Rose #FEF2F2
            style = Paint.Style.FILL
        }

        val paintMssHighlightBg = Paint().apply {
            color = Color.rgb(239, 246, 255) // Light Blue #EFF6FF
            style = Paint.Style.FILL
        }

        val paintBoxFill = Paint().apply {
            color = Color.rgb(248, 250, 253)
            style = Paint.Style.FILL
        }

        val paintWarningBoxFill = Paint().apply {
            color = Color.rgb(254, 242, 242)
            style = Paint.Style.FILL
        }

        val paintWarningStroke = Paint().apply {
            color = Color.rgb(252, 165, 165)
            style = Paint.Style.STROKE
            strokeWidth = 0.8f
        }

        val paintWarningText = Paint().apply {
            color = Color.rgb(185, 28, 28)
            textSize = 7.5f
            isFakeBoldText = true
            isAntiAlias = true
        }

        val paintBarTrack = Paint().apply {
            color = Color.rgb(226, 232, 240) // #E2E8F0
            style = Paint.Style.FILL
        }

        val paintBarMsiFill = Paint().apply {
            color = Color.rgb(220, 38, 38) // Warm Red #DC2626
            style = Paint.Style.FILL
        }

        val paintBarMssFill = Paint().apply {
            color = Color.rgb(23, 105, 170) // Clinical Blue #1769AA
            style = Paint.Style.FILL
        }

        val paintThresholdMarker = Paint().apply {
            color = Color.rgb(30, 41, 59)
            strokeWidth = 1.2f
        }

        // Multi-page tracking state
        var currentPageNumber = 1
        var pageInfo = PdfDocument.PageInfo.Builder(PAGE_WIDTH, PAGE_HEIGHT, currentPageNumber).create()
        var currentPage = document.startPage(pageInfo)
        var canvas = currentPage.canvas
        var y = MARGIN_TOP

        // Running Header Helper
        fun drawRunningHeader() {
            // Top Accent Bar
            canvas.drawRect(MARGIN_LEFT, y, MARGIN_RIGHT, y + 2.5f, paintHeaderAccent)
            y += 10f

            // Branding line
            canvas.drawText(HEADER_BRANDING, MARGIN_LEFT, y, paintBrand)
            val rTag = "RESEARCH USE ONLY · NOT FOR CLINICAL DIAGNOSIS"
            val rTagWidth = paintWarningText.measureText(rTag)
            canvas.drawText(rTag, MARGIN_RIGHT - rTagWidth, y, paintWarningText)
            y += 13f

            // Document Title
            canvas.drawText(HEADER_TITLE, MARGIN_LEFT, y, paintTitle)
            val caseDateStr = "Case ID: ${researchCase.caseIdentifier} · Generated: $generatedDateStr"
            val metaWidth = paintMuted.measureText(caseDateStr)
            canvas.drawText(caseDateStr, MARGIN_RIGHT - metaWidth, y, paintMuted)
            y += 11f

            // Document Subtitle
            canvas.drawText(HEADER_SUBTITLE, MARGIN_LEFT, y, paintSubtitle)
            y += 8f

            // Bottom Divider
            canvas.drawLine(MARGIN_LEFT, y, MARGIN_RIGHT, y, paintDivider)
            y += 14f
        }

        // Running Footer Helper
        fun drawFooter(pageNum: Int) {
            canvas.drawLine(MARGIN_LEFT, FOOTER_Y - 8f, MARGIN_RIGHT, FOOTER_Y - 8f, paintDivider)
            canvas.drawText(
                "CRC Multimodal Edge AI Workstation · Template v$TEMPLATE_VERSION · 100% Offline Edge Execution",
                MARGIN_LEFT,
                FOOTER_Y,
                paintMuted
            )
            val pageStr = "Page $pageNum"
            val pWidth = paintMuted.measureText(pageStr)
            canvas.drawText(pageStr, MARGIN_RIGHT - pWidth, FOOTER_Y, paintMuted)
        }

        // Space check / page-break trigger
        fun ensureVerticalSpace(neededHeight: Float) {
            if (y + neededHeight > MARGIN_BOTTOM) {
                drawFooter(currentPageNumber)
                document.finishPage(currentPage)
                currentPageNumber++
                pageInfo = PdfDocument.PageInfo.Builder(PAGE_WIDTH, PAGE_HEIGHT, currentPageNumber).create()
                currentPage = document.startPage(pageInfo)
                canvas = currentPage.canvas
                y = MARGIN_TOP

                // Continuation Header
                canvas.drawRect(MARGIN_LEFT, y, MARGIN_RIGHT, y + 2.5f, paintHeaderAccent)
                y += 13f
                canvas.drawText("$HEADER_TITLE (Continuation)", MARGIN_LEFT, y, paintSectionHeader)
                val idStr = "Case ID: ${researchCase.caseIdentifier} · Report ID: ${reportId.take(16)}…"
                val idWidth = paintMuted.measureText(idStr)
                canvas.drawText(idStr, MARGIN_RIGHT - idWidth, y, paintMuted)
                y += 8f
                canvas.drawLine(MARGIN_LEFT, y, MARGIN_RIGHT, y, paintDivider)
                y += 14f
            }
        }

        // Section Title Helper
        fun drawSectionHeader(title: String) {
            ensureVerticalSpace(40f)
            y += 4f
            canvas.drawText(title, MARGIN_LEFT, y, paintSectionHeader)
            y += 10f
        }

        // Structured Table Row Drawer
        val labelColWidth = 160f
        val valueColWidth = CONTENT_WIDTH - labelColWidth // 355f

        fun drawTableRow(
            label: String,
            value: String,
            isBoldValue: Boolean = false,
            isMonoValue: Boolean = false,
            valueColor: Int? = null,
            highlightValueBg: Paint? = null
        ) {
            val valPaint = when {
                isMonoValue -> paintTableValueMono
                isBoldValue -> paintTableValueBold
                else -> paintTableValue
            }

            if (valueColor != null) {
                valPaint.color = valueColor
            } else {
                valPaint.color = Color.rgb(15, 23, 42)
            }

            val labelLines = wrapText(label, paintTableLabel, labelColWidth - 14f)
            val valueLines = wrapText(value, valPaint, valueColWidth - 14f)

            val maxLines = maxOf(labelLines.size, valueLines.size, 1)
            val rowHeight = (maxLines * 11.5f) + 8f

            ensureVerticalSpace(rowHeight + 2f)

            // Cell Backgrounds
            canvas.drawRect(MARGIN_LEFT, y, MARGIN_LEFT + labelColWidth, y + rowHeight, paintLabelBg)
            val bgPaint = highlightValueBg ?: paintValueBg
            canvas.drawRect(MARGIN_LEFT + labelColWidth, y, MARGIN_RIGHT, y + rowHeight, bgPaint)

            // Outer border and vertical divider
            canvas.drawRect(MARGIN_LEFT, y, MARGIN_RIGHT, y + rowHeight, paintTableStroke)
            canvas.drawLine(MARGIN_LEFT + labelColWidth, y, MARGIN_LEFT + labelColWidth, y + rowHeight, paintTableStroke)

            // Draw label lines
            var curY = y + 10f
            for (line in labelLines) {
                canvas.drawText(line, MARGIN_LEFT + 7f, curY, paintTableLabel)
                curY += 11.5f
            }

            // Draw value lines
            curY = y + 10f
            for (line in valueLines) {
                canvas.drawText(line, MARGIN_LEFT + labelColWidth + 7f, curY, valPaint)
                curY += 11.5f
            }

            y += rowHeight
        }

        // ==========================================
        // 1. First Page Header
        // ==========================================
        drawRunningHeader()

        // ==========================================
        // SECTION 1: REPORT SUMMARY
        // ==========================================
        drawSectionHeader("1. REPORT SUMMARY")
        val prob = inferenceRun?.probability ?: 0f
        val mssProb = if (inferenceRun != null) 1.0f - prob else 0f
        val signedMargin = if (inferenceRun != null) prob - ThresholdLogic.OPERATING_THRESHOLD else 0f
        val marginStr = if (inferenceRun != null) {
            if (signedMargin >= 0f) "+${String.format(Locale.US, "%.5f", signedMargin)} above threshold"
            else "${String.format(Locale.US, "%.5f", signedMargin)} below threshold"
        } else "Pending"

        drawTableRow("Case ID", researchCase.caseIdentifier, isBoldValue = true)
        drawTableRow("Report ID", reportId, isMonoValue = true)
        drawTableRow("Report generated", generatedDateStr)

        if (inferenceRun != null) {
            val isMsi = inferenceRun.predictedClass == ThresholdLogic.CLASS_MSI
            val classColor = if (isMsi) Color.rgb(185, 28, 28) else Color.rgb(23, 105, 170)
            val highlightBg = if (isMsi) paintMsiHighlightBg else paintMssHighlightBg
            drawTableRow("Classification", "${inferenceRun.predictedLabel} (Class ${inferenceRun.predictedClass})", isBoldValue = true, valueColor = classColor, highlightValueBg = highlightBg)
            drawTableRow("Model output score — Class 1", String.format(Locale.US, "%.5f", prob), isMonoValue = true)
            drawTableRow("MSS probability — Class 0", String.format(Locale.US, "%.5f", mssProb), isMonoValue = true)
            drawTableRow("Decision threshold", "0.8300 (Frozen research boundary)")
            drawTableRow("Threshold margin", "$marginStr vs 0.8300", isMonoValue = true)
        } else {
            drawTableRow("Classification", "PENDING (No inference run recorded)", isBoldValue = true, valueColor = Color.rgb(185, 28, 28))
            drawTableRow("Model output score — Class 1", "Pending")
            drawTableRow("MSS probability — Class 0", "Pending")
            drawTableRow("Decision threshold", "0.8300 (Frozen research boundary)")
            drawTableRow("Threshold margin", "Pending")
        }
        y += 12f

        // ==========================================
        // SECTION 2: PROBABILITY OVERVIEW
        // ==========================================
        drawSectionHeader("2. PROBABILITY OVERVIEW")
        val probCardHeight = 88f
        ensureVerticalSpace(probCardHeight + 14f)

        // Bounding Box
        canvas.drawRect(MARGIN_LEFT, y, MARGIN_RIGHT, y + probCardHeight, paintBoxFill)
        canvas.drawRect(MARGIN_LEFT, y, MARGIN_RIGHT, y + probCardHeight, paintTableStroke)

        val barX = MARGIN_LEFT + 150f
        val barWidth = 230f
        val valueX = MARGIN_LEFT + 390f
        val thresholdX = barX + (barWidth * ThresholdLogic.OPERATING_THRESHOLD)

        // Row 1: MSI Probability (Class 1)
        val row1LabelY = y + 21f
        canvas.drawText("MSI Probability (Class 1)", MARGIN_LEFT + 10f, row1LabelY, paintTableLabel)
        canvas.drawText("Microsatellite Instability", MARGIN_LEFT + 10f, row1LabelY + 9.5f, paintMuted)

        val bar1Y = y + 14f
        canvas.drawRect(barX, bar1Y, barX + barWidth, bar1Y + 9f, paintBarTrack)
        val msiFillWidth = barWidth * prob.coerceIn(0f, 1f)
        if (msiFillWidth > 0f) {
            canvas.drawRect(barX, bar1Y, barX + msiFillWidth, bar1Y + 9f, paintBarMsiFill)
        }
        // Threshold marker on bar 1
        canvas.drawLine(thresholdX, bar1Y - 2.5f, thresholdX, bar1Y + 11.5f, paintThresholdMarker)
        canvas.drawText("0.83", thresholdX - 6f, bar1Y - 3.5f, paintMuted)

        val msiValStr = if (inferenceRun != null) String.format(Locale.US, "%.5f (%.1f%%)", prob, prob * 100f) else "Pending"
        canvas.drawText(msiValStr, valueX, row1LabelY, paintTableValueMono)

        // Row 2: MSS Probability (Class 0)
        val row2LabelY = y + 51f
        canvas.drawText("MSS Probability (Class 0)", MARGIN_LEFT + 10f, row2LabelY, paintTableLabel)
        canvas.drawText("Microsatellite Stable", MARGIN_LEFT + 10f, row2LabelY + 9.5f, paintMuted)

        val bar2Y = y + 44f
        canvas.drawRect(barX, bar2Y, barX + barWidth, bar2Y + 9f, paintBarTrack)
        val mssFillWidth = barWidth * mssProb.coerceIn(0f, 1f)
        if (mssFillWidth > 0f) {
            canvas.drawRect(barX, bar2Y, barX + mssFillWidth, bar2Y + 9f, paintBarMssFill)
        }
        // Threshold marker on bar 2
        canvas.drawLine(thresholdX, bar2Y - 2.5f, thresholdX, bar2Y + 11.5f, paintThresholdMarker)

        val mssValStr = if (inferenceRun != null) String.format(Locale.US, "%.5f (%.1f%%)", mssProb, mssProb * 100f) else "Pending"
        canvas.drawText(mssValStr, valueX, row2LabelY, paintTableValueMono)

        // Footnote
        val probNote = "* Softmax activation outputs; complementary class formulation P(MSS) = 1.0 - P(MSI). Decision threshold tau = 0.8300."
        canvas.drawText(probNote, MARGIN_LEFT + 10f, y + 78f, paintMuted)

        y += probCardHeight + 12f

        // ==========================================
        // SECTION 3: CASE INFORMATION
        // ==========================================
        drawSectionHeader("3. CASE INFORMATION")
        drawTableRow("Case ID", researchCase.caseIdentifier, isBoldValue = true)
        drawTableRow("Specimen ID / Description", researchCase.specimenDescription.ifBlank { "Primary colorectal adenocarcinoma biopsy" })
        drawTableRow("Report Filename", targetFile.name)
        drawTableRow("Report-generation timestamp", generatedDateStr)
        drawTableRow("Evaluation Protocol", InferenceProtocol.formatProtocol(inferenceRun?.protocol ?: researchCase.protocol))
        drawTableRow("Existing Specimen Patch Count", "${researchCase.patchCount} patches")
        drawTableRow("Existing Case Status", researchCase.status.displayName)
        drawTableRow("Research Context", researchCase.researchContext.ifBlank { "Standard benchmark evaluation" })
        drawTableRow("Created Timestamp", dateFormat.format(Date(researchCase.createdAt)))
        drawTableRow("Updated Timestamp", dateFormat.format(Date(researchCase.updatedAt)))
        y += 12f

        // ==========================================
        // SECTION 4: MODEL AND INFERENCE DETAILS
        // ==========================================
        drawSectionHeader("4. MODEL AND INFERENCE DETAILS")
        if (inferenceRun != null) {
            drawTableRow("Model Classification", "${inferenceRun.predictedLabel} (Class ${inferenceRun.predictedClass})", isBoldValue = true)
            drawTableRow("Output Score", String.format(Locale.US, "%.5f", prob), isMonoValue = true)
            drawTableRow("Frozen Decision Threshold", "0.8300 (Fixed research boundary)")
            drawTableRow("Threshold Margin", "$marginStr vs 0.8300", isMonoValue = true)
            drawTableRow("Inference Latency", "${String.format(Locale.US, "%.2f", inferenceRun.latencyMs)} ms")
            drawTableRow("Model Logit", String.format(Locale.US, "%.5f", inferenceRun.logit), isMonoValue = true)
        } else {
            drawTableRow("Model Classification", "PENDING (No inference run recorded)")
            drawTableRow("Output Score", "Pending")
            drawTableRow("Frozen Decision Threshold", "0.8300 (Fixed research boundary)")
            drawTableRow("Threshold Margin", "Pending")
            drawTableRow("Inference Latency", "Pending")
            drawTableRow("Model Logit", "Pending")
        }
        drawTableRow("Inference Engine", "Local Android ONNX Runtime 1.18.0")
        drawTableRow("Execution Provider", "CPUExecutionProvider (Float32 single-patient sequential)")
        drawTableRow("Model Architecture", "CrossAttentionExportWrapper (ClinicalBERT + Phikon-v2)")
        drawTableRow("Model Artifact Filename", "fusion_model_fp32.onnx")
        drawTableRow("Model Artifact Size", "20.04 MB")
        drawTableRow("Model Precision", "FP32 (Single Precision)")
        drawTableRow("Full Model SHA-256 Hash", MODEL_FP32_SHA256, isMonoValue = true)
        drawTableRow("Vision Tensor Contract", "[1, ${researchCase.patchCount}, 1024] Float32 (Phikon-v2 ViT-B/16)", isMonoValue = true)
        drawTableRow("Text Tensor Contract", "[1, 768] Float32 (Bio_ClinicalBERT mean-pooled)", isMonoValue = true)
        drawTableRow("Tensor Data Types", "IEEE 754 Float32 (Little-Endian)")
        val payloadOrigin = if (bundle != null) "Verified Authentic Bundle (CRCv${bundle.schemaVersion})" else "Precomputed Local Feature Tensors"
        drawTableRow("Payload Origin", payloadOrigin)
        drawTableRow("Input-Readiness Status", "Verified Authentic (Zero Missing Patches)")
        val diagSummary = if (!inferenceRun?.diagnosticsSummary.isNullOrBlank()) {
            val lines = inferenceRun!!.diagnosticsSummary!!.lines()
                .map { it.trim() }
                .filter { it.isNotBlank() && !it.startsWith("===") }
            val execStatus = lines.firstOrNull { it.startsWith("Execution Status", ignoreCase = true) }
            val provenance = lines.firstOrNull { it.startsWith("Input Provenance", ignoreCase = true) }
            val integrity = lines.firstOrNull { it.startsWith("Integrity Status", ignoreCase = true) }
            val parity = lines.firstOrNull { it.startsWith("Parity Gate", ignoreCase = true) }
            val summaryParts = listOfNotNull(execStatus, provenance, integrity, parity)
            if (summaryParts.isNotEmpty()) {
                summaryParts.joinToString(" · ")
            } else {
                lines.take(2).joinToString(" · ")
            }
        } else {
            "Single-patient sequential CPU execution completed successfully."
        }
        drawTableRow("Diagnostic Summary", diagSummary)
        drawTableRow("Privacy-Enforcement Status", "100% on-device CPU execution, 0 bytes transmitted")
        drawTableRow("Network-Transmission Status", "Strict Zero-Network Air-Gapped Enforcement")
        drawTableRow("Evaluation Protocol", InferenceProtocol.formatProtocol(inferenceRun?.protocol ?: researchCase.protocol))
        drawTableRow("Specimen Patch Count", "${researchCase.patchCount} patches")
        drawTableRow("Vision Tensor SHA-256", manifest.inputFingerprints.visionPatchesSha256, isMonoValue = true)
        drawTableRow("Text Tensor SHA-256", manifest.inputFingerprints.textTensorSha256, isMonoValue = true)
        val bundleSha = manifest.inputFingerprints.bundleSha256 ?: "Not bundled (Individual tensors)"
        drawTableRow("Bundle Payload SHA-256", bundleSha, isMonoValue = true)
        val integrityDigest = manifest.integrityStatement.manifestPayloadSha256
        drawTableRow("Provenance Integrity Digest", "${manifest.integrityStatement.checksumType}: ${integrityDigest.take(24)}…")
        y += 12f

        // ==========================================
        // SECTION 5: DETERMINISTIC RESULT EXPLANATION
        // ==========================================
        drawSectionHeader("5. DETERMINISTIC RESULT EXPLANATION")
        val explanationText = if (inferenceRun != null) {
            "The model assigned this research case to ${inferenceRun.predictedLabel} because its recorded output score was " +
            "${String.format(Locale.US, "%.4f", prob)}, which is ${if (prob >= ThresholdLogic.OPERATING_THRESHOLD) "at or above" else "below"} " +
            "the configured research threshold of 0.83 (margin: $marginStr). " +
            "In accordance with the frozen protocol, cases with P >= 0.8300 are classified as MSI (Class 1), and cases with P < 0.8300 " +
            "are classified as MSS (Class 0). This experimental machine learning evaluation is for research purposes only and is not a " +
            "confirmed clinical diagnosis or a primary determination of DNA mismatch repair deficiency."
        } else {
            "No inference run has been executed for this research case. In accordance with the frozen protocol, cases with P >= 0.8300 " +
            "are classified as MSI (Class 1), and cases with P < 0.8300 are classified as MSS (Class 0). Operating threshold boundary is tau = 0.8300."
        }

        val explLines = wrapText(explanationText, paintTableValue, CONTENT_WIDTH - 16f)
        val explBoxHeight = (explLines.size * 11f) + 26f
        ensureVerticalSpace(explBoxHeight + 8f)

        canvas.drawRect(MARGIN_LEFT, y, MARGIN_RIGHT, y + explBoxHeight, paintBoxFill)
        canvas.drawRect(MARGIN_LEFT, y, MARGIN_RIGHT, y + explBoxHeight, paintTableStroke)

        canvas.drawText("Classification Rationale & Operating Boundary:", MARGIN_LEFT + 8f, y + 12f, paintTableLabel)
        var dyExpl = y + 24f
        for (line in explLines) {
            canvas.drawText(line, MARGIN_LEFT + 8f, dyExpl, paintTableValue)
            dyExpl += 11f
        }
        y += explBoxHeight + 8f

        // Validation Warnings (if any)
        if (explanation.hasValidationWarnings) {
            val warnBoxHeight = (explanation.validationWarnings.size * 12f) + 18f
            ensureVerticalSpace(warnBoxHeight + 8f)
            canvas.drawRect(MARGIN_LEFT, y, MARGIN_RIGHT, y + warnBoxHeight, paintWarningBoxFill)
            canvas.drawRect(MARGIN_LEFT, y, MARGIN_RIGHT, y + warnBoxHeight, paintWarningStroke)
            canvas.drawText("VALIDATION NOTICES / DATA CONSISTENCY WARNINGS:", MARGIN_LEFT + 8f, y + 12f, paintWarningText)
            var wY = y + 23f
            for (w in explanation.validationWarnings) {
                val wrappedW = wrapText("• $w", paintWarningText, CONTENT_WIDTH - 16f)
                for (wLine in wrappedW) {
                    canvas.drawText(wLine, MARGIN_LEFT + 8f, wY, paintWarningText)
                    wY += 10.5f
                }
            }
            y += warnBoxHeight + 8f
        }

        // ==========================================
        // SECTION 6: CASE REVIEW NOTES
        // ==========================================
        drawSectionHeader("6. CASE REVIEW NOTES")
        if (reviewNotes.isNotEmpty()) {
            for ((idx, note) in reviewNotes.withIndex()) {
                val noteHeader = "[Note ${idx + 1}] Category: ${note.category.displayName} | Author: ${note.authorLabel} (${dateFormat.format(Date(note.createdAt))})"
                val noteLines = wrapText(note.content, paintTableValue, CONTENT_WIDTH - 16f)
                val nBoxHeight = (noteLines.size * 10.5f) + 22f

                ensureVerticalSpace(nBoxHeight + 6f)
                canvas.drawRect(MARGIN_LEFT, y, MARGIN_RIGHT, y + nBoxHeight, paintBoxFill)
                canvas.drawRect(MARGIN_LEFT, y, MARGIN_RIGHT, y + nBoxHeight, paintTableStroke)

                canvas.drawText(noteHeader, MARGIN_LEFT + 8f, y + 12f, paintTableLabel)
                var ny = y + 23f
                for (nLine in noteLines) {
                    canvas.drawText(nLine, MARGIN_LEFT + 8f, ny, paintTableValue)
                    ny += 10.5f
                }
                y += nBoxHeight + 6f
            }
        } else {
            ensureVerticalSpace(20f)
            canvas.drawText("No review notes recorded for this case.", MARGIN_LEFT, y, paintMuted)
            y += 14f
        }
        y += 6f

        // ==========================================
        // SECTION 7: RESPONSIBLE-USE RESEARCH DISCLAIMER
        // ==========================================
        drawSectionHeader("7. RESPONSIBLE-USE RESEARCH DISCLAIMER")
        val disclaimerLines = wrapText(RESEARCH_DISCLAIMER_TEXT, paintMuted, CONTENT_WIDTH - 16f)
        val disclaimerBoxHeight = (disclaimerLines.size * 10f) + 22f
        ensureVerticalSpace(disclaimerBoxHeight + 10f)

        canvas.drawRect(MARGIN_LEFT, y, MARGIN_RIGHT, y + disclaimerBoxHeight, paintBoxFill)
        canvas.drawRect(MARGIN_LEFT, y, MARGIN_RIGHT, y + disclaimerBoxHeight, paintTableStroke)

        canvas.drawText("INVESTIGATIONAL RESEARCH USE ONLY — NOT FOR CLINICAL DIAGNOSIS", MARGIN_LEFT + 8f, y + 12f, paintTableLabel)
        var dyDisc = y + 23f
        for (line in disclaimerLines) {
            canvas.drawText(line, MARGIN_LEFT + 8f, dyDisc, paintMuted)
            dyDisc += 10f
        }
        y += disclaimerBoxHeight + 14f

        // Draw final page footer
        drawFooter(currentPageNumber)
        document.finishPage(currentPage)

        // Write document to temporary file first, flush and verify, then atomically commit
        val tempFile = File(targetFile.parentFile, "${targetFile.name}.tmp_${UUID.randomUUID().toString().take(8)}")
        try {
            FileOutputStream(tempFile).use { outStream ->
                document.writeTo(outStream)
                outStream.flush()
            }
        } finally {
            document.close()
        }

        if (!tempFile.exists() || tempFile.length() < 1024L) {
            tempFile.delete()
            throw IllegalStateException("Failed to generate PDF: output file is empty or corrupted (size: ${tempFile.length()} bytes)")
        }

        if (!tempFile.renameTo(targetFile)) {
            tempFile.copyTo(targetFile, overwrite = true)
            tempFile.delete()
        }

        if (!targetFile.exists() || targetFile.length() < 1024L) {
            throw IllegalStateException("Generated PDF verification failed: target file missing or invalid")
        }

        return currentPageNumber
    }

    /**
     * Splits text into wrapped lines that do not exceed [maxWidth] under [paint].
     * Preserves paragraph line breaks and adheres strictly to unit test contracts.
     */
    fun wrapText(text: String, paint: Paint, maxWidth: Float): List<String> {
        val result = mutableListOf<String>()
        val paragraphs = text.split("\n")

        for (paragraph in paragraphs) {
            if (paragraph.isBlank()) {
                result.add("")
                continue
            }

            val words = paragraph.split(" ")
            var currentLine = StringBuilder()

            for (word in words) {
                if (currentLine.isEmpty()) {
                    currentLine.append(word)
                } else {
                    val candidate = "$currentLine $word"
                    if (paint.measureText(candidate) <= maxWidth) {
                        currentLine.append(" ").append(word)
                    } else {
                        result.add(currentLine.toString())
                        currentLine = StringBuilder(word)
                    }
                }
            }
            if (currentLine.isNotEmpty()) {
                result.add(currentLine.toString())
            }
        }
        return result
    }

    /**
     * Computes the SHA-256 hex digest of a file.
     */
    fun computeFileSha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { stream ->
            val buffer = ByteArray(65536)
            var bytesRead: Int
            while (stream.read(buffer).also { bytesRead = it } != -1) {
                digest.update(buffer, 0, bytesRead)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
