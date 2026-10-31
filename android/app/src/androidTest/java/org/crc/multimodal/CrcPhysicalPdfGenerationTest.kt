package org.crc.multimodal

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.crc.multimodal.data.PersistentReportRecord
import org.crc.multimodal.data.ReportExportStatus
import org.crc.multimodal.data.SqliteWorkstationRepository
import org.crc.multimodal.inference.CrcLogger
import org.crc.multimodal.report.PdfReportGenerator
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.UUID

@RunWith(AndroidJUnit4::class)
class CrcPhysicalPdfGenerationTest {

    @Test
    fun testGenerateStructuredPdfReportOnDevice() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val repository = SqliteWorkstationRepository.getInstance(context)

        // 1. Locate TCGA-WS-AB45 in device database
        val cases = repository.getAllCases()
        val targetCase = cases.firstOrNull { it.caseIdentifier == "TCGA-WS-AB45" }
            ?: cases.firstOrNull()
            ?: throw IllegalStateException("No research cases found in device database.")

        val runs = repository.getRunsForCase(targetCase.id)
        val targetRun = runs.firstOrNull()

        CrcLogger.i("[PDF_AUDIT_DEVICE] Case: ${targetCase.caseIdentifier} (ID: ${targetCase.id})")
        CrcLogger.i("[PDF_AUDIT_DEVICE] Inference Run: ID=${targetRun?.id}, Label=${targetRun?.predictedLabel}, Prob=${targetRun?.probability}")

        // 2. Generate PDF using current PdfReportGenerator
        val outputDir = File(context.filesDir, "reports")
        val reportId = UUID.randomUUID().toString()
        val pdfResult = PdfReportGenerator.generateReport(
            context = context,
            researchCase = targetCase,
            inferenceRun = targetRun,
            reviewNotes = repository.getNotesForCase(targetCase.id),
            bundle = null,
            reportId = reportId,
            outputDir = outputDir
        )

        // 3. Verify generated PDF properties
        assertNotNull(pdfResult)
        assertTrue("PDF file must exist on device storage", pdfResult.file.exists())
        val fileSize = pdfResult.file.length()
        CrcLogger.i("[PDF_AUDIT_DEVICE] Generated File: ${pdfResult.file.absolutePath}")
        CrcLogger.i("[PDF_AUDIT_DEVICE] File Size: $fileSize bytes | Pages: ${pdfResult.pageCount}")
        CrcLogger.i("[PDF_AUDIT_DEVICE] Template Version: ${pdfResult.templateVersion}")
        CrcLogger.i("[PDF_AUDIT_DEVICE] SHA-256: ${pdfResult.sha256Hash}")

        assertTrue("File size must be plausible (> 50 KB)", fileSize > 50000L)
        assertEquals("Template version must be 2.0.0", "2.0.0", pdfResult.templateVersion)
        assertEquals("Document must span 2 pages for publication layout", 2, pdfResult.pageCount)
        assertEquals(64, pdfResult.sha256Hash.length)

        // 4. Record new report in database
        val newRecord = PersistentReportRecord(
            reportId = reportId,
            caseId = targetCase.id,
            inferenceRunId = targetRun?.id,
            templateVersion = PdfReportGenerator.TEMPLATE_VERSION,
            createdAt = pdfResult.generatedAt,
            exportStatus = ReportExportStatus.GENERATED,
            fileUri = null,
            filePath = pdfResult.file.absolutePath,
            modelVersion = targetRun?.modelIdentifier ?: "fusion_model_fp32.onnx",
            sha256Hash = pdfResult.sha256Hash,
            errorMessage = null
        )
        val insertedRow = repository.insertReportRecord(newRecord)
        assertTrue(insertedRow > 0L)

        // 5. Verify that candidate selection now picks this 2.0.0 report and rejects older 1.0.0 reports
        val history = repository.getReportHistoryForCase(targetCase.id)
        val validCandidate = history.firstOrNull {
            it.exportStatus != ReportExportStatus.FAILED &&
                    it.templateVersion == PdfReportGenerator.TEMPLATE_VERSION &&
                    (targetRun == null || it.inferenceRunId == targetRun.id) &&
                    it.filePath != null &&
                    File(it.filePath).let { f -> f.exists() && f.length() > 1024L }
        }

        assertNotNull("Candidate selection must find the v2.0.0 report", validCandidate)
        assertEquals("Candidate must have template v2.0.0", "2.0.0", validCandidate?.templateVersion)
        assertEquals("Candidate must point to new generated PDF", pdfResult.file.absolutePath, validCandidate?.filePath)
        CrcLogger.i("[PDF_AUDIT_DEVICE] Verification SUCCESS. Active report points to: ${validCandidate?.filePath}")
    }
}
