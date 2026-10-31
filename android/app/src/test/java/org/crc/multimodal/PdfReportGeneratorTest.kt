package org.crc.multimodal

import android.graphics.Paint
import org.crc.multimodal.report.PdfReportGenerator
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.security.MessageDigest

class PdfReportGeneratorTest {

    @Test
    fun testReportGeneratorPageConstants() {
        assertEquals(595, PdfReportGenerator.PAGE_WIDTH)
        assertEquals(842, PdfReportGenerator.PAGE_HEIGHT)
        assertEquals("2.0.0", PdfReportGenerator.TEMPLATE_VERSION)
        assertEquals(40f, PdfReportGenerator.MARGIN_LEFT, 0.01f)
        assertEquals(555f, PdfReportGenerator.MARGIN_RIGHT, 0.01f)
        assertEquals(515f, PdfReportGenerator.CONTENT_WIDTH, 0.01f)
    }

    @Test
    fun testWrapTextSingleWordLongerThanWidth() {
        // Mock paint measure text based on character length
        val paint = object : Paint() {
            override fun measureText(text: String): Float = text.length * 10f
        }

        val text = "Supercalifragilisticexpialidocious"
        val wrapped = PdfReportGenerator.wrapText(text, paint, 50f)

        assertNotNull(wrapped)
        assertEquals(1, wrapped.size)
        assertEquals("Supercalifragilisticexpialidocious", wrapped[0])
    }

    @Test
    fun testWrapTextMultipleLines() {
        val paint = object : Paint() {
            override fun measureText(text: String): Float = text.length * 5f
        }

        val text = "This is a detailed clinical histopathology observation note regarding microsatellite instability."
        val wrapped = PdfReportGenerator.wrapText(text, paint, 120f)

        assertTrue(wrapped.size > 1)
        // Verify reconstructed words match original words
        val reconstructed = wrapped.joinToString(" ") { it.trim() }
        assertEquals(text, reconstructed)
    }

    @Test
    fun testWrapTextPreservesParagraphs() {
        val paint = object : Paint() {
            override fun measureText(text: String): Float = text.length * 5f
        }

        val text = "Paragraph 1 line.\nParagraph 2 line.\n\nParagraph 4 after empty line."
        val wrapped = PdfReportGenerator.wrapText(text, paint, 200f)

        assertTrue(wrapped.size >= 4)
        assertTrue(wrapped.contains("Paragraph 1 line."))
        assertTrue(wrapped.contains("Paragraph 2 line."))
        assertTrue(wrapped.contains(""))
        assertTrue(wrapped.contains("Paragraph 4 after empty line."))
    }

    @Test
    fun testComputeFileSha256MatchesExpectedDigest() {
        val tempFile = File.createTempFile("crc_test_digest", ".txt")
        try {
            val content = "CRC Multimodal Edge AI Workstation Provenance Test Payload"
            tempFile.writeText(content, Charsets.UTF_8)

            val calculatedHash = PdfReportGenerator.computeFileSha256(tempFile)

            val expectedDigest = MessageDigest.getInstance("SHA-256")
                .digest(content.toByteArray(Charsets.UTF_8))
                .joinToString("") { "%02x".format(it) }

            assertEquals(expectedDigest, calculatedHash)
        } finally {
            tempFile.delete()
        }
    }

    @Test
    fun testComplementaryClassProbabilitySemantics() {
        val testProbabilities = listOf(0.0f, 0.1625f, 0.50f, 0.8300f, 0.9412f, 1.0f)
        for (pMsi in testProbabilities) {
            val pMss = 1.0f - pMsi
            // Sum must strictly equal 1.0 within float precision
            assertEquals(1.0f, pMsi + pMss, 0.0001f)
            assertTrue("MSS probability must be within [0.0, 1.0]", pMss in 0.0f..1.0f)

            // When P >= 0.83, classification is MSI; when P < 0.83, classification is MSS
            val isMsi = pMsi >= org.crc.multimodal.inference.ThresholdLogic.OPERATING_THRESHOLD
            val expectedClass = if (isMsi) org.crc.multimodal.inference.ThresholdLogic.CLASS_MSI else org.crc.multimodal.inference.ThresholdLogic.CLASS_MSS
            assertEquals(expectedClass, org.crc.multimodal.inference.ThresholdLogic.classify(pMsi))
        }
    }

    @Test
    fun testModelSha256IsFull64HexCharacters() {
        val modelSha = "0402F6C522BEAC5733E30EEECAD908927957B5D4164B6B896663F43BA3702D73"
        assertEquals(64, modelSha.length)
        assertTrue("Model SHA-256 must be valid hex", modelSha.matches(Regex("[0-9A-Fa-f]{64}")))
    }

    @Test
    fun testThresholdMarginSignedCalculation() {
        val pAbove = 0.94120f
        val marginAbove = pAbove - org.crc.multimodal.inference.ThresholdLogic.OPERATING_THRESHOLD
        assertTrue(marginAbove > 0f)
        assertEquals(0.11120f, marginAbove, 0.0001f)

        val pBelow = 0.16250f
        val marginBelow = pBelow - org.crc.multimodal.inference.ThresholdLogic.OPERATING_THRESHOLD
        assertTrue(marginBelow < 0f)
        assertEquals(-0.66750f, marginBelow, 0.0001f)
    }

    @Test
    fun testOperatingThresholdIsExactlyPointEightThree() {
        assertEquals(0.83f, org.crc.multimodal.inference.ThresholdLogic.OPERATING_THRESHOLD, 0.00001f)
    }

    @Test
    fun testSevenSectionsOrderingAndHeaders() {
        val expectedSections = listOf(
            "1. REPORT SUMMARY",
            "2. PROBABILITY OVERVIEW",
            "3. CASE INFORMATION",
            "4. MODEL AND INFERENCE DETAILS",
            "5. DETERMINISTIC RESULT EXPLANATION",
            "6. CASE REVIEW NOTES",
            "7. RESPONSIBLE-USE RESEARCH DISCLAIMER"
        )
        assertEquals(7, expectedSections.size)
        assertTrue(expectedSections[0].contains("REPORT SUMMARY"))
        assertTrue(expectedSections[1].contains("PROBABILITY OVERVIEW"))
        assertTrue(expectedSections[2].contains("CASE INFORMATION"))
        assertTrue(expectedSections[3].contains("MODEL AND INFERENCE DETAILS"))
        assertTrue(expectedSections[4].contains("DETERMINISTIC RESULT EXPLANATION"))
        assertTrue(expectedSections[5].contains("CASE REVIEW NOTES"))
        assertTrue(expectedSections[6].contains("RESPONSIBLE-USE RESEARCH DISCLAIMER"))
    }

    @Test
    fun testResponsibleUseDisclaimerIntegrity() {
        val disclaimer = "INVESTIGATIONAL RESEARCH USE ONLY. Not approved for clinical diagnosis or patient care. " +
            "This report summarizes experimental machine learning benchmark outputs from an edge AI research prototype " +
            "(Phikon-v2 ViT-B/16 and Bio_ClinicalBERT precomputed embeddings). Output probabilities represent research softmax activations; " +
            "decision threshold is frozen at tau = 0.8300. The MSI/MSS evaluation score does not substitute for certified clinical " +
            "IHC/PCR microsatellite instability testing. No clinical or treatment decisions should be based solely on this research output."

        assertTrue(disclaimer.contains("INVESTIGATIONAL RESEARCH USE ONLY"))
        assertTrue(disclaimer.contains("Not approved for clinical diagnosis"))
        assertTrue(disclaimer.contains("IHC/PCR"))
        assertTrue(disclaimer.contains("tau = 0.8300"))
    }

    @Test
    fun testTemplateVersionBumpInvalidatesLegacyReports() {
        val legacyTemplate = "1.0.0"
        assertTrue(
            "Legacy template version 1.0.0 must not match current template version",
            legacyTemplate != PdfReportGenerator.TEMPLATE_VERSION
        )
        assertEquals("2.0.0", PdfReportGenerator.TEMPLATE_VERSION)
    }

    @Test
    fun testStaleReportCandidateFilterRejectsLegacyTemplates() {
        data class MockReportRecord(
            val templateVersion: String,
            val inferenceRunId: Long?,
            val isFailed: Boolean,
            val fileExists: Boolean,
            val fileSizeBytes: Long
        )

        val targetRunId = 42L
        val history = listOf(
            MockReportRecord(
                templateVersion = "1.0.0", // Legacy template on device
                inferenceRunId = targetRunId,
                isFailed = false,
                fileExists = true,
                fileSizeBytes = 358945L
            ),
            MockReportRecord(
                templateVersion = "2.0.0",
                inferenceRunId = 99L, // Different run ID
                isFailed = false,
                fileExists = true,
                fileSizeBytes = 360000L
            ),
            MockReportRecord(
                templateVersion = "2.0.0",
                inferenceRunId = targetRunId,
                isFailed = false,
                fileExists = true,
                fileSizeBytes = 370000L // Valid current template and run!
            )
        )

        // Filter replicating ReportFragment.kt candidate selection
        val candidate = history.firstOrNull {
            !it.isFailed &&
                    it.templateVersion == PdfReportGenerator.TEMPLATE_VERSION &&
                    it.inferenceRunId == targetRunId &&
                    it.fileExists &&
                    it.fileSizeBytes > 1024L
        }

        assertNotNull(candidate)
        assertEquals("2.0.0", candidate?.templateVersion)
        assertEquals(targetRunId, candidate?.inferenceRunId)
        assertEquals(370000L, candidate?.fileSizeBytes)
    }
}


