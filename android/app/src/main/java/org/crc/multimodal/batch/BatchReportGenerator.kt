
package org.crc.multimodal.batch

import android.content.Context
import android.graphics.Color
import android.graphics.Paint
import android.graphics.pdf.PdfDocument
import java.io.File
import java.io.FileOutputStream
import java.security.MessageDigest
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Result bundle containing exported report files and cryptographic digests.
 */
data class ExportedReportBundle(
    val batchId: String,
    val pdfFile: File?,
    val jsonFile: File,
    val manifestFile: File,
    val pdfSha256: String?,
    val jsonSha256: String
)

/**
 * Generates audit-grade research reports in PDF and canonical JSON formats
 * with cryptographic SHA-256 integrity verification.
 */
object BatchReportGenerator {

    private const val DISCLAIMER =
        "RESEARCH USE ONLY: Multimodal fusion inference output on precomputed Phikon-v2 and ClinicalBERT embeddings. " +
        "Not approved as a clinical diagnostic device. Predicted MSI/MSS probabilities represent softmax research activations, " +
        "not calibrated clinical risks. Independent molecular validation (IHC/PCR) is required for clinical decision making."

    /**
     * Generates both PDF and JSON reports with SHA-256 integrity manifest.
     */
    fun exportReport(
        context: Context? = null,
        batchRun: BatchRun,
        outputDir: File = context?.let { File(it.filesDir, "reports") } ?: File("build/tmp/reports")
    ): ExportedReportBundle {
        outputDir.mkdirs()
        val batchId = batchRun.batchId

        // 1. Generate Canonical JSON Report
        val jsonFile = File(outputDir, "${batchId}_report.json")
        val jsonContent = generateJsonReport(batchRun)
        jsonFile.writeText(jsonContent, Charsets.UTF_8)
        val jsonSha256 = computeFileSha256(jsonFile)

        // 2. Generate PDF Report (if platform supports android.graphics.pdf)
        val pdfFile = File(outputDir, "${batchId}_report.pdf")
        var pdfSha256: String? = null
        try {
            generatePdfReport(batchRun, pdfFile)
            pdfSha256 = computeFileSha256(pdfFile)
        } catch (e: Throwable) {
            // In unit-test JVM environments where android.graphics.pdf is mocked, log and skip PDF creation
            pdfSha256 = null
        }

        // 3. Generate Cryptographic Manifest
        val manifestFile = File(outputDir, "${batchId}_manifest.sha256")
        val manifestLines = StringBuilder()
        manifestLines.append("# CRC Multimodal Edge AI - Cryptographic Report Manifest\n")
        manifestLines.append("# Batch ID: $batchId\n")
        manifestLines.append("# Generated UTC: ${SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US).format(Date())}\n")
        manifestLines.append("# Model SHA-256: ${batchRun.modelSha256}\n\n")
        manifestLines.append("$jsonSha256  ${jsonFile.name}\n")
        if (pdfSha256 != null) {
            manifestLines.append("$pdfSha256  ${pdfFile.name}\n")
        }
        manifestFile.writeText(manifestLines.toString(), Charsets.UTF_8)

        return ExportedReportBundle(
            batchId = batchId,
            pdfFile = if (pdfFile.exists()) pdfFile else null,
            jsonFile = jsonFile,
            manifestFile = manifestFile,
            pdfSha256 = pdfSha256,
            jsonSha256 = jsonSha256
        )
    }

    /**
     * Verifies cryptographic integrity of a report file against an expected SHA-256 hash.
     */
    fun verifyReportIntegrity(file: File, expectedSha256: String): Boolean {
        if (!file.exists() || !file.isFile) return false
        val actualSha256 = computeFileSha256(file)
        return actualSha256.equals(expectedSha256.trim(), ignoreCase = true)
    }

    /**
     * Constructs canonical JSON research report representation.
     */
    fun generateJsonReport(batchRun: BatchRun): String {
        val stats = batchRun.statistics
        val dateFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US)

        val sb = StringBuilder()
        sb.append("{\n")
        sb.append("  \"report_title\": \"CRC Multimodal Edge AI Batch Research Report\",\n")
        sb.append("  \"research_use_disclaimer\": \"$DISCLAIMER\",\n")
        sb.append("  \"batch_id\": \"${batchRun.batchId}\",\n")
        sb.append("  \"created_at\": \"${dateFormat.format(Date(batchRun.createdAt))}\",\n")
        sb.append("  \"updated_at\": \"${dateFormat.format(Date(batchRun.updatedAt))}\",\n")
        sb.append("  \"status\": \"${batchRun.status.name}\",\n")
        sb.append("  \"model_metadata\": {\n")
        sb.append("    \"architecture\": \"SummaryToPatchCrossAttention\",\n")
        sb.append("    \"frozen_onnx_sha256\": \"${batchRun.modelSha256}\",\n")
        sb.append("    \"execution_provider\": \"${batchRun.executionProvider}\",\n")
        sb.append("    \"operating_threshold\": ${batchRun.decisionThreshold},\n")
        sb.append("    \"class_mapping\": {\"0\": \"MSS\", \"1\": \"MSI\"},\n")
        sb.append("    \"upstream_encoders\": {\n")
        sb.append("      \"vision\": \"OWKIN Phikon-v2 (ViT-B/16, [1, N, 1024])\",\n")
        sb.append("      \"text\": \"Bio_ClinicalBERT (mean-pooled, [1, 768])\"\n")
        sb.append("    }\n")
        sb.append("  },\n")

        // Summary statistics
        sb.append("  \"summary_statistics\": {\n")
        sb.append("    \"total_selected\": ${stats.totalSelected},\n")
        sb.append("    \"total_accepted\": ${stats.totalAccepted},\n")
        sb.append("    \"total_rejected\": ${stats.totalRejected},\n")
        sb.append("    \"total_processed\": ${stats.totalProcessed},\n")
        sb.append("    \"total_successful\": ${stats.totalSuccessful},\n")
        sb.append("    \"total_failed\": ${stats.totalFailed},\n")
        sb.append("    \"total_cancelled\": ${stats.totalCancelled},\n")
        sb.append("    \"msi_predicted_count\": ${stats.msiCount},\n")
        sb.append("    \"mss_predicted_count\": ${stats.mssCount},\n")
        sb.append("    \"msi_proportion\": ${String.format(Locale.US, "%.4f", stats.msiProportion)},\n")
        sb.append("    \"mss_proportion\": ${String.format(Locale.US, "%.4f", stats.mssProportion)},\n")
        sb.append("    \"mean_inference_latency_ms\": ${String.format(Locale.US, "%.2f", stats.meanLatencyMs)},\n")
        sb.append("    \"median_inference_latency_ms\": ${String.format(Locale.US, "%.2f", stats.medianLatencyMs)},\n")
        sb.append("    \"min_inference_latency_ms\": ${stats.minLatencyMs},\n")
        sb.append("    \"max_inference_latency_ms\": ${stats.maxLatencyMs},\n")
        sb.append("    \"total_duration_ms\": ${stats.totalDurationMs},\n")
        sb.append("    \"patch_count_min\": ${stats.patchCountMin},\n")
        sb.append("    \"patch_count_max\": ${stats.patchCountMax},\n")
        sb.append("    \"patch_count_mean\": ${String.format(Locale.US, "%.1f", stats.patchCountMean)},\n")
        sb.append("    \"exclusion_count\": ${stats.exclusionCount}\n")
        sb.append("  },\n")

        // Per-specimen results
        sb.append("  \"specimens\": [\n")
        batchRun.items.forEachIndexed { index, item ->
            sb.append("    {\n")
            sb.append("      \"specimen_id\": \"${item.specimenId}\",\n")
            sb.append("      \"status\": \"${item.status.name}\",\n")
            sb.append("      \"patch_count\": ${item.patchCount},\n")
            sb.append("      \"file_name\": \"${item.fileName}\",\n")
            sb.append("      \"bundle_sha256\": \"${item.bundleSha256}\",\n")
            sb.append("      \"vision_sha256\": \"${item.visionSha256}\",\n")
            sb.append("      \"text_sha256\": \"${item.textSha256}\",\n")
            sb.append("      \"schema_version\": ${item.schemaVersion},\n")
            sb.append("      \"provenance\": \"${item.provenance.name}\",\n")
            sb.append("      \"protocol\": \"${item.protocol}\",\n")
            if (item.protocolReason != null) {
                sb.append("      \"protocol_reason\": \"${escapeJson(item.protocolReason)}\",\n")
            }
            if (item.probability != null) {
                sb.append("      \"msi_probability\": ${String.format(Locale.US, "%.6f", item.probability)},\n")
                sb.append("      \"raw_logit\": ${String.format(Locale.US, "%.6f", item.logit ?: 0f)},\n")
                sb.append("      \"predicted_class\": ${item.predictedClass},\n")
                sb.append("      \"predicted_label\": \"${item.predictedLabel}\",\n")
                sb.append("      \"decision_threshold\": ${item.threshold},\n")
                sb.append("      \"inference_latency_ms\": ${item.inferenceLatencyMs},\n")
            }
            if (item.validationError != null) {
                sb.append("      \"validation_error\": \"${escapeJson(item.validationError)}\",\n")
            }
            if (item.errorDetails != null) {
                sb.append("      \"error_details\": \"${escapeJson(item.errorDetails)}\",\n")
            }
            sb.append("      \"completed_timestamp\": ${item.completedTimestamp ?: 0L}\n")
            sb.append("    }${if (index < batchRun.items.size - 1) "," else ""}\n")
        }
        sb.append("  ]\n")
        sb.append("}\n")
        return sb.toString()
    }

    /**
     * Renders standard research report PDF.
     */
    private fun generatePdfReport(batchRun: BatchRun, targetFile: File) {
        val document = PdfDocument()
        val stats = batchRun.statistics

        // A4 Dimensions at 72 dpi: 595 x 842 points
        val pageWidth = 595
        val pageHeight = 842
        val pageInfo = PdfDocument.PageInfo.Builder(pageWidth, pageHeight, 1).create()
        val page = document.startPage(pageInfo)
        val canvas = page.canvas

        val paintTitle = Paint().apply {
            color = Color.rgb(24, 43, 73) // deep navy
            textSize = 16f
            isFakeBoldText = true
            isAntiAlias = true
        }

        val paintHeader = Paint().apply {
            color = Color.rgb(40, 60, 90)
            textSize = 10f
            isFakeBoldText = true
            isAntiAlias = true
        }

        val paintBody = Paint().apply {
            color = Color.rgb(40, 40, 40)
            textSize = 8.5f
            isAntiAlias = true
        }

        val paintMuted = Paint().apply {
            color = Color.rgb(100, 100, 100)
            textSize = 7.5f
            isAntiAlias = true
        }

        val paintLine = Paint().apply {
            color = Color.rgb(210, 215, 225)
            strokeWidth = 1f
        }

        var y = 40f
        val leftMargin = 36f
        val rightMargin = pageWidth - 36f

        // Title & Header
        canvas.drawText("CRC Multimodal Edge AI — Batch Evaluation Report", leftMargin, y, paintTitle)
        y += 16f
        canvas.drawText("Frozen FP32 Cross-Attention Multimodal Pathology Inference Workstation", leftMargin, y, paintMuted)
        y += 12f
        canvas.drawLine(leftMargin, y, rightMargin, y, paintLine)
        y += 16f

        // Disclaimer Box
        val paintBox = Paint().apply {
            color = Color.rgb(245, 247, 250)
            style = Paint.Style.FILL
        }
        canvas.drawRect(leftMargin, y, rightMargin, y + 42f, paintBox)
        val paintWarn = Paint().apply {
            color = Color.rgb(180, 50, 40)
            textSize = 7f
            isFakeBoldText = true
            isAntiAlias = true
        }
        canvas.drawText("RESEARCH USE ONLY — NOT FOR CLINICAL DIAGNOSIS", leftMargin + 8f, y + 12f, paintWarn)
        val paintDisc = Paint().apply {
            color = Color.rgb(70, 70, 70)
            textSize = 6.8f
            isAntiAlias = true
        }
        canvas.drawText("Model evaluates precomputed Phikon-v2 [1, N, 1024] and ClinicalBERT [1, 768] embeddings on-device.", leftMargin + 8f, y + 24f, paintDisc)
        canvas.drawText("Outputs represent research softmax activations; threshold τ = 0.83. IHC/PCR validation required.", leftMargin + 8f, y + 34f, paintDisc)
        y += 54f

        // Batch Metadata
        canvas.drawText("BATCH METADATA", leftMargin, y, paintHeader)
        y += 12f
        canvas.drawText("Batch Identifier: ${batchRun.batchId}", leftMargin, y, paintBody)
        canvas.drawText("Status: ${batchRun.status.name}", leftMargin + 260f, y, paintBody)
        y += 11f
        canvas.drawText("Execution Provider: ${batchRun.executionProvider} (2 threads)", leftMargin, y, paintBody)
        canvas.drawText("Operating Threshold: τ = ${batchRun.decisionThreshold}", leftMargin + 260f, y, paintBody)
        y += 11f
        canvas.drawText("Model SHA-256: ${batchRun.modelSha256}", leftMargin, y, paintMuted)
        y += 16f

        // Summary Statistics Box
        canvas.drawText("COHORT EVALUATION SUMMARY", leftMargin, y, paintHeader)
        y += 12f
        canvas.drawText("Total Selected: ${stats.totalSelected} | Accepted: ${stats.totalAccepted} | Rejected: ${stats.totalRejected}", leftMargin, y, paintBody)
        y += 11f
        canvas.drawText("Processed: ${stats.totalProcessed} | Successful: ${stats.totalSuccessful} | Failed: ${stats.totalFailed} | Cancelled: ${stats.totalCancelled}", leftMargin, y, paintBody)
        y += 11f
        canvas.drawText("MSI Predictions: ${stats.msiCount} (${String.format(Locale.US, "%.1f", stats.msiProportion * 100)}%) | MSS Predictions: ${stats.mssCount} (${String.format(Locale.US, "%.1f", stats.mssProportion * 100)}%)", leftMargin, y, paintBody)
        y += 11f
        canvas.drawText("Inference Latency: Mean ${String.format(Locale.US, "%.1f", stats.meanLatencyMs)}ms | Median ${String.format(Locale.US, "%.1f", stats.medianLatencyMs)}ms (Min ${stats.minLatencyMs}ms, Max ${stats.maxLatencyMs}ms)", leftMargin, y, paintBody)
        y += 16f
        canvas.drawLine(leftMargin, y, rightMargin, y, paintLine)
        y += 16f

        // Specimen Table Header
        canvas.drawText("SPECIMEN INFERENCE RESULTS", leftMargin, y, paintHeader)
        y += 14f

        canvas.drawRect(leftMargin, y, rightMargin, y + 16f, paintBox)
        val paintTh = Paint().apply {
            color = Color.rgb(20, 30, 50)
            textSize = 7.5f
            isFakeBoldText = true
            isAntiAlias = true
        }
        val yTh = y + 11f
        canvas.drawText("Specimen ID", leftMargin + 4f, yTh, paintTh)
        canvas.drawText("Patches", leftMargin + 110f, yTh, paintTh)
        canvas.drawText("Probability", leftMargin + 160f, yTh, paintTh)
        canvas.drawText("Predicted", leftMargin + 230f, yTh, paintTh)
        canvas.drawText("Status", leftMargin + 290f, yTh, paintTh)
        canvas.drawText("Latency", leftMargin + 360f, yTh, paintTh)
        canvas.drawText("Digest (SHA-256)", leftMargin + 415f, yTh, paintTh)
        y += 20f

        // Specimen Rows
        for (item in batchRun.items.take(28)) { // Fit up to 28 items on first page
            val rowColor = if (item.predictedClass == 1) Color.rgb(180, 40, 30) else Color.rgb(30, 40, 60)
            val paintRow = Paint().apply {
                color = rowColor
                textSize = 7.5f
                isAntiAlias = true
            }

            canvas.drawText(item.specimenId.take(18), leftMargin + 4f, y, paintBody)
            canvas.drawText("${item.patchCount}", leftMargin + 110f, y, paintBody)

            val probStr = if (item.probability != null) String.format(Locale.US, "%.4f", item.probability) else "—"
            canvas.drawText(probStr, leftMargin + 160f, y, paintBody)

            val labelStr = item.predictedLabel ?: "—"
            canvas.drawText(labelStr, leftMargin + 230f, y, paintRow)

            val protTag = if (item.protocol == "PROTOCOL_2_UNCAPPED") "P2" else "P1"
            val statusDisplay = "${item.status.name.take(4)} ($protTag)"
            canvas.drawText(statusDisplay, leftMargin + 290f, y, paintBody)
            canvas.drawText("${item.inferenceLatencyMs}ms", leftMargin + 360f, y, paintBody)
            canvas.drawText(item.bundleSha256.take(12), leftMargin + 415f, y, paintMuted)

            y += 13f
        }

        // Footer
        val footerY = pageHeight - 30f
        canvas.drawLine(leftMargin, footerY, rightMargin, footerY, paintLine)
        canvas.drawText("Generated by CRC Multimodal Edge AI Workstation · 100% Offline Edge Computation", leftMargin, footerY + 12f, paintMuted)
        canvas.drawText("Page 1 of 1", rightMargin - 40f, footerY + 12f, paintMuted)

        document.finishPage(page)

        FileOutputStream(targetFile).use { out ->
            document.writeTo(out)
        }
        document.close()
    }

    /**
     * Generates a combined multi-specimen PDF report containing dedicated report sections
     * for all selected eligible analyzed specimens, along with a cryptographic SHA-256 manifest.
     */
    fun exportCombinedReport(
        context: Context? = null,
        batchId: String,
        selectedItems: List<BatchSpecimenItem>,
        outputDir: File = context?.let { File(it.filesDir, "reports") } ?: File("build/tmp/reports")
    ): ExportedReportBundle {
        outputDir.mkdirs()
        val eligibleItems = selectedItems.filter { it.status == BatchSpecimenStatus.COMPLETED && it.probability != null }
        require(eligibleItems.isNotEmpty()) { "Cannot export combined report: No eligible analyzed specimens selected." }

        val timestampStr = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val cleanBatchId = batchId.replace(Regex("[^a-zA-Z0-9._-]"), "_")
        val pdfFile = File(outputDir, "CRC_Batch_Report_${cleanBatchId}_$timestampStr.pdf")
        val jsonFile = File(outputDir, "CRC_Batch_Report_${cleanBatchId}_$timestampStr.json")

        // 1. Generate Combined JSON Report
        val jsonContent = generateCombinedJsonReport(batchId, eligibleItems)
        jsonFile.writeText(jsonContent, Charsets.UTF_8)
        val jsonSha256 = computeFileSha256(jsonFile)

        // 2. Generate Combined Multi-Page PDF Report
        var pdfSha256: String? = null
        try {
            generateCombinedPdf(batchId, eligibleItems, pdfFile)
            pdfSha256 = computeFileSha256(pdfFile)
        } catch (e: Throwable) {
            pdfSha256 = null
        }

        // 3. Cryptographic Manifest
        val manifestFile = File(outputDir, "CRC_Batch_Report_${cleanBatchId}_${timestampStr}_manifest.sha256")
        val manifestLines = StringBuilder()
        manifestLines.append("# CRC Multimodal Edge AI - Combined Batch Report Manifest\n")
        manifestLines.append("# Batch ID: $batchId\n")
        manifestLines.append("# Total Selected Specimens: ${eligibleItems.size}\n")
        manifestLines.append("# Generated UTC: ${SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US).format(Date())}\n\n")
        manifestLines.append("$jsonSha256  ${jsonFile.name}\n")
        if (pdfSha256 != null) {
            manifestLines.append("$pdfSha256  ${pdfFile.name}\n")
        }
        manifestFile.writeText(manifestLines.toString(), Charsets.UTF_8)

        return ExportedReportBundle(
            batchId = batchId,
            pdfFile = if (pdfFile.exists()) pdfFile else null,
            jsonFile = jsonFile,
            manifestFile = manifestFile,
            pdfSha256 = pdfSha256,
            jsonSha256 = jsonSha256
        )
    }

    private fun generateCombinedJsonReport(batchId: String, items: List<BatchSpecimenItem>): String {
        val dateFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US)
        val msiCount = items.count { it.predictedClass == 1 }
        val mssCount = items.count { it.predictedClass == 0 }

        val sb = StringBuilder()
        sb.append("{\n")
        sb.append("  \"report_title\": \"CRC Multimodal Edge AI Combined Batch Specimen Report\",\n")
        sb.append("  \"research_use_disclaimer\": \"$DISCLAIMER\",\n")
        sb.append("  \"batch_id\": \"$batchId\",\n")
        sb.append("  \"generated_at\": \"${dateFormat.format(Date())}\",\n")
        sb.append("  \"total_selected_specimens\": ${items.size},\n")
        sb.append("  \"msi_predicted_count\": $msiCount,\n")
        sb.append("  \"mss_predicted_count\": $mssCount,\n")
        sb.append("  \"specimens\": [\n")
        items.forEachIndexed { index, item ->
            sb.append("    {\n")
            sb.append("      \"specimen_id\": \"${item.specimenId}\",\n")
            sb.append("      \"status\": \"${item.status.name}\",\n")
            sb.append("      \"patch_count\": ${item.patchCount},\n")
            sb.append("      \"protocol\": \"${item.protocol}\",\n")
            sb.append("      \"msi_probability\": ${String.format(Locale.US, "%.6f", item.probability ?: 0f)},\n")
            sb.append("      \"raw_logit\": ${String.format(Locale.US, "%.6f", item.logit ?: 0f)},\n")
            sb.append("      \"predicted_class\": ${item.predictedClass},\n")
            sb.append("      \"predicted_label\": \"${item.predictedLabel}\",\n")
            sb.append("      \"decision_threshold\": ${item.threshold},\n")
            sb.append("      \"inference_latency_ms\": ${item.inferenceLatencyMs},\n")
            sb.append("      \"bundle_sha256\": \"${item.bundleSha256}\",\n")
            sb.append("      \"completed_timestamp\": ${item.completedTimestamp ?: 0L}\n")
            sb.append("    }${if (index < items.size - 1) "," else ""}\n")
        }
        sb.append("  ]\n")
        sb.append("}\n")
        return sb.toString()
    }

    private fun generateCombinedPdf(batchId: String, items: List<BatchSpecimenItem>, targetFile: File) {
        val document = PdfDocument()

        val pageWidth = 595
        val pageHeight = 842
        val leftMargin = 36f
        val rightMargin = pageWidth - 36f

        val paintTitle = Paint().apply {
            color = Color.rgb(24, 43, 73)
            textSize = 15f
            isFakeBoldText = true
            isAntiAlias = true
        }

        val paintHeader = Paint().apply {
            color = Color.rgb(40, 60, 90)
            textSize = 10f
            isFakeBoldText = true
            isAntiAlias = true
        }

        val paintSubheader = Paint().apply {
            color = Color.rgb(24, 43, 73)
            textSize = 11f
            isFakeBoldText = true
            isAntiAlias = true
        }

        val paintBody = Paint().apply {
            color = Color.rgb(40, 40, 40)
            textSize = 8.5f
            isAntiAlias = true
        }

        val paintMuted = Paint().apply {
            color = Color.rgb(100, 100, 100)
            textSize = 7.5f
            isAntiAlias = true
        }

        val paintLine = Paint().apply {
            color = Color.rgb(210, 215, 225)
            strokeWidth = 1f
        }

        val paintBox = Paint().apply {
            color = Color.rgb(245, 247, 250)
            style = Paint.Style.FILL
        }

        val paintCardStroke = Paint().apply {
            color = Color.rgb(220, 225, 235)
            style = Paint.Style.STROKE
            strokeWidth = 1f
        }

        // Compute total pages: 1 overview page + ceil(items.size / 2.0)
        val specimenPages = if (items.isNotEmpty()) ((items.size + 1) / 2) else 0
        val totalPages = 1 + specimenPages

        var pageNum = 1

        // ================= PAGE 1: Overview & Cohort Summary =================
        val pageInfo1 = PdfDocument.PageInfo.Builder(pageWidth, pageHeight, pageNum).create()
        val page1 = document.startPage(pageInfo1)
        val canvas1 = page1.canvas
        var y = 40f

        canvas1.drawText("CRC Multimodal Edge AI — Bulk Specimen Analysis Report", leftMargin, y, paintTitle)
        y += 16f
        canvas1.drawText("Batch ID: $batchId · Selected Cohort Export (${items.size} Specimens)", leftMargin, y, paintMuted)
        y += 12f
        canvas1.drawLine(leftMargin, y, rightMargin, y, paintLine)
        y += 16f

        // Disclaimer box
        canvas1.drawRect(leftMargin, y, rightMargin, y + 40f, paintBox)
        val paintWarn = Paint().apply {
            color = Color.rgb(180, 50, 40)
            textSize = 7f
            isFakeBoldText = true
            isAntiAlias = true
        }
        canvas1.drawText("RESEARCH USE ONLY — NOT FOR CLINICAL DIAGNOSTIC DECISIONS", leftMargin + 8f, y + 12f, paintWarn)
        val paintDisc = Paint().apply {
            color = Color.rgb(70, 70, 70)
            textSize = 6.8f
            isAntiAlias = true
        }
        canvas1.drawText("Output probabilities represent softmax research activations on precomputed embeddings.", leftMargin + 8f, y + 23f, paintDisc)
        canvas1.drawText("Decision threshold frozen at τ = 0.8300 (Class 0: MSS, Class 1: MSI). Independent IHC/PCR required.", leftMargin + 8f, y + 33f, paintDisc)
        y += 52f

        // Selected Cohort Summary Box
        val msiCount = items.count { it.predictedClass == 1 }
        val mssCount = items.count { it.predictedClass == 0 }
        val msiPct = if (items.isNotEmpty()) (msiCount * 100f) / items.size else 0f
        val mssPct = if (items.isNotEmpty()) (mssCount * 100f) / items.size else 0f
        val latencies = items.map { it.inferenceLatencyMs }
        val meanLat = if (latencies.isNotEmpty()) latencies.average() else 0.0

        canvas1.drawText("SELECTED COHORT EVALUATION METRICS", leftMargin, y, paintHeader)
        y += 12f
        canvas1.drawText("Total Selected Specimens: ${items.size}", leftMargin, y, paintBody)
        canvas1.drawText("Decision Threshold: τ = 0.8300", leftMargin + 260f, y, paintBody)
        y += 11f
        canvas1.drawText("MSI Predicted: $msiCount (${String.format(Locale.US, "%.1f", msiPct)}%)", leftMargin, y, paintBody)
        canvas1.drawText("MSS Predicted: $mssCount (${String.format(Locale.US, "%.1f", mssPct)}%)", leftMargin + 260f, y, paintBody)
        y += 11f
        canvas1.drawText("Mean Latency: ${String.format(Locale.US, "%.1f", meanLat)}ms", leftMargin, y, paintBody)
        canvas1.drawText("Execution Provider: CPUExecutionProvider", leftMargin + 260f, y, paintBody)
        y += 16f
        canvas1.drawLine(leftMargin, y, rightMargin, y, paintLine)
        y += 16f

        // Table Header
        canvas1.drawText("SELECTED SPECIMENS OVERVIEW", leftMargin, y, paintHeader)
        y += 14f

        canvas1.drawRect(leftMargin, y, rightMargin, y + 16f, paintBox)
        val paintTh = Paint().apply {
            color = Color.rgb(20, 30, 50)
            textSize = 7.5f
            isFakeBoldText = true
            isAntiAlias = true
        }
        val yTh = y + 11f
        canvas1.drawText("Specimen ID", leftMargin + 4f, yTh, paintTh)
        canvas1.drawText("Protocol", leftMargin + 130f, yTh, paintTh)
        canvas1.drawText("Patches", leftMargin + 200f, yTh, paintTh)
        canvas1.drawText("MSI Prob", leftMargin + 260f, yTh, paintTh)
        canvas1.drawText("Predicted", leftMargin + 340f, yTh, paintTh)
        canvas1.drawText("Latency", leftMargin + 420f, yTh, paintTh)
        y += 20f

        for (item in items.take(28)) {
            val isMsi = item.predictedClass == 1
            val paintRowLabel = Paint().apply {
                color = if (isMsi) Color.rgb(180, 40, 30) else Color.rgb(46, 125, 50)
                textSize = 7.5f
                isFakeBoldText = true
                isAntiAlias = true
            }

            canvas1.drawText(item.specimenId.take(20), leftMargin + 4f, y, paintBody)
            val protShort = if (item.protocol == "PROTOCOL_2_UNCAPPED") "Protocol 2" else "Protocol 1"
            canvas1.drawText(protShort, leftMargin + 130f, y, paintBody)
            canvas1.drawText("${item.patchCount}", leftMargin + 200f, y, paintBody)

            val probStr = if (item.probability != null) String.format(Locale.US, "%.5f", item.probability) else "—"
            canvas1.drawText(probStr, leftMargin + 260f, y, paintBody)

            val labelStr = item.predictedLabel ?: if (isMsi) "MSI" else "MSS"
            canvas1.drawText(labelStr, leftMargin + 340f, y, paintRowLabel)
            canvas1.drawText("${item.inferenceLatencyMs}ms", leftMargin + 420f, y, paintBody)

            y += 13f
        }

        // Footer Page 1
        val footerY1 = pageHeight - 30f
        canvas1.drawLine(leftMargin, footerY1, rightMargin, footerY1, paintLine)
        canvas1.drawText("CRC Multimodal Edge AI Research Workstation · Batch: $batchId", leftMargin, footerY1 + 12f, paintMuted)
        canvas1.drawText("Page $pageNum of $totalPages", rightMargin - 55f, footerY1 + 12f, paintMuted)

        document.finishPage(page1)

        // ================= SUBSEQUENT PAGES: Detailed Per-Specimen Sections =================
        var itemIdx = 0
        while (itemIdx < items.size) {
            pageNum++
            val pageInfo = PdfDocument.PageInfo.Builder(pageWidth, pageHeight, pageNum).create()
            val page = document.startPage(pageInfo)
            val canvas = page.canvas
            var curY = 40f

            // Running Page Header
            canvas.drawText("CRC MULTIMODAL PATHOLOGY RESEARCH WORKSTATION", leftMargin, curY, paintHeader)
            curY += 12f
            canvas.drawText("Batch ID: $batchId · Specimen Detailed Reports", leftMargin, curY, paintMuted)
            curY += 10f
            canvas.drawLine(leftMargin, curY, rightMargin, curY, paintLine)
            curY += 16f

            // Fit up to 2 specimen sections per page
            val itemsForPage = items.subList(itemIdx, minOf(itemIdx + 2, items.size))
            for (specimen in itemsForPage) {
                val isMsi = specimen.predictedClass == 1
                val sectionHeight = 330f
                canvas.drawRect(leftMargin, curY, rightMargin, curY + sectionHeight, paintBox)
                canvas.drawRect(leftMargin, curY, rightMargin, curY + sectionHeight, paintCardStroke)

                var cardY = curY + 16f
                val cardPad = leftMargin + 12f

                // Specimen ID Header
                canvas.drawText("SPECIMEN: ${specimen.specimenId}", cardPad, cardY, paintSubheader)
                val badgeColor = if (isMsi) Color.rgb(180, 40, 30) else Color.rgb(46, 125, 50)
                val paintBadge = Paint().apply {
                    color = badgeColor
                    textSize = 10f
                    isFakeBoldText = true
                    isAntiAlias = true
                }
                val classLabel = if (isMsi) "MSI (P ≥ 0.8300)" else "MSS (P < 0.8300)"
                canvas.drawText(classLabel, rightMargin - 120f, cardY, paintBadge)
                cardY += 16f

                canvas.drawLine(cardPad, cardY, rightMargin - 12f, cardY, paintLine)
                cardY += 16f

                // Section 1: Prediction Activations
                canvas.drawText("PREDICTION & ACTIVATION METRICS", cardPad, cardY, paintHeader)
                cardY += 13f
                val probFormatted = if (specimen.probability != null) String.format(Locale.US, "%.5f", specimen.probability) else "N/A"
                val logitFormatted = if (specimen.logit != null) String.format(Locale.US, "%.4f", specimen.logit) else "N/A"
                val marginVal = if (specimen.probability != null) specimen.probability - specimen.threshold else 0f
                val marginFormatted = String.format(Locale.US, "%+.5f", marginVal)

                canvas.drawText("MSI Softmax Probability: $probFormatted", cardPad, cardY, paintBody)
                canvas.drawText("Operating Threshold: τ = ${specimen.threshold}", cardPad + 240f, cardY, paintBody)
                cardY += 12f
                canvas.drawText("Raw Output Logit: $logitFormatted", cardPad, cardY, paintBody)
                canvas.drawText("Margin to Threshold: $marginFormatted", cardPad + 240f, cardY, paintBody)
                cardY += 16f

                // Section 2: Input Tensor Specs & Protocol
                canvas.drawText("INPUT TENSOR & PROTOCOL CONFIGURATION", cardPad, cardY, paintHeader)
                cardY += 13f
                val protocolDisplay = org.crc.multimodal.inference.InferenceProtocol.fromStringOrNull(specimen.protocol)?.displayName ?: specimen.protocol
                canvas.drawText("Patch Count: ${specimen.patchCount} patches", cardPad, cardY, paintBody)
                canvas.drawText("Inference Protocol: $protocolDisplay", cardPad + 240f, cardY, paintBody)
                cardY += 12f
                canvas.drawText("Vision Tensor Shape: [1, ${specimen.patchCount}, 1024]", cardPad, cardY, paintBody)
                canvas.drawText("Clinical Text Shape: [1, 768]", cardPad + 240f, cardY, paintBody)
                cardY += 16f

                // Section 3: Execution Runtime & Performance
                canvas.drawText("EDGE RUNTIME & PERFORMANCE TELEMETRY", cardPad, cardY, paintHeader)
                cardY += 13f
                canvas.drawText("Inference Latency: ${specimen.inferenceLatencyMs} ms", cardPad, cardY, paintBody)
                canvas.drawText("Execution Provider: CPUExecutionProvider", cardPad + 240f, cardY, paintBody)
                cardY += 12f
                canvas.drawText("Model Engine: ONNX Runtime FP32", cardPad, cardY, paintBody)
                canvas.drawText("Architecture: SummaryToPatchCrossAttention", cardPad + 240f, cardY, paintBody)
                cardY += 16f

                // Section 4: Provenance & Cryptographic Verification
                canvas.drawText("DATA PROVENANCE & CRYPTOGRAPHIC INTEGRITY", cardPad, cardY, paintHeader)
                cardY += 13f
                val digestShort = if (specimen.bundleSha256.isNotBlank()) specimen.bundleSha256 else "N/A"
                canvas.drawText("Bundle SHA-256: $digestShort", cardPad, cardY, paintMuted)
                cardY += 11f
                canvas.drawText("Provenance Mode: ${specimen.provenance.name} · Schema v${specimen.schemaVersion}", cardPad, cardY, paintMuted)
                cardY += 11f
                val completedDateStr = if (specimen.completedTimestamp != null) SimpleDateFormat("yyyy-MM-dd HH:mm:ss 'UTC'", Locale.US).format(Date(specimen.completedTimestamp)) else "N/A"
                canvas.drawText("Analysis Timestamp: $completedDateStr", cardPad, cardY, paintMuted)

                curY += sectionHeight + 14f
            }

            // Running Page Footer
            val footerY = pageHeight - 30f
            canvas.drawLine(leftMargin, footerY, rightMargin, footerY, paintLine)
            canvas.drawText("INVESTIGATIONAL RESEARCH USE ONLY · Not for clinical diagnostic use", leftMargin, footerY + 12f, paintMuted)
            canvas.drawText("Page $pageNum of $totalPages", rightMargin - 55f, footerY + 12f, paintMuted)

            document.finishPage(page)
            itemIdx += itemsForPage.size
        }

        FileOutputStream(targetFile).use { out ->
            document.writeTo(out)
        }
        document.close()
    }

    private fun computeFileSha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { stream ->
            val buf = ByteArray(65536)
            var bytesRead: Int
            while (stream.read(buf).also { bytesRead = it } != -1) {
                digest.update(buf, 0, bytesRead)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun escapeJson(str: String): String {
        return str.replace("\\", "\\\\")
            .replace("\"", "\\\"")
            .replace("\b", "\\b")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
    }
}
