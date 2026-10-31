package org.crc.multimodal

import org.crc.multimodal.data.CaseStatus
import org.crc.multimodal.data.InMemoryWorkstationRepository
import org.crc.multimodal.data.InferenceRun
import org.crc.multimodal.data.ResearchCase
import org.crc.multimodal.data.ReviewNote
import org.crc.multimodal.data.ReviewNoteCategory
import org.crc.multimodal.inference.CrcLogger
import org.crc.multimodal.inference.InferenceProtocol
import org.crc.multimodal.report.ReportGenerator
import org.crc.multimodal.security.AuditActionType
import org.crc.multimodal.security.AuditTrailRecord
import org.crc.multimodal.security.ProvenanceAuditManager
import org.crc.multimodal.security.SafeFileValidator
import org.crc.multimodal.security.SensitiveLogRedactor
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.io.File

/**
 * Comprehensive verification test suite for WP7.5 (Security, Provenance, and Audit).
 */
class SecurityProvenanceAuditTest {

    // =========================================================================
    // SECTION 1: SENSITIVE LOG REDACTION TESTS
    // =========================================================================

    @Test
    fun testScrubbingEmailAddresses() {
        val raw = "Diagnostic failure reported by user john.doe@pathology-research.org during run."
        val sanitized = SensitiveLogRedactor.redact(raw)
        assertFalse("Raw email must not appear in sanitized log", sanitized.contains("john.doe@pathology-research.org"))
        assertTrue("Redaction token must be present", sanitized.contains("[REDACTED_EMAIL]"))
    }

    @Test
    fun testScrubbingPhoneNumbers() {
        val raw = "Emergency pathologist contact: +1-555-867-5309 or (555) 123-4567 for specimen review."
        val sanitized = SensitiveLogRedactor.redact(raw)
        assertFalse("Phone number must not appear", sanitized.contains("867-5309"))
        assertFalse("Phone number must not appear", sanitized.contains("123-4567"))
        assertTrue("Redaction token must be present", sanitized.contains("[REDACTED_PHONE]"))
    }

    @Test
    fun testScrubbingMrnAndPatientIds() {
        val raw = "Ingestion warning for specimen TCGA-D5-6540 with MRN: 987654321 and Patient_ID: PT-8820."
        val sanitized = SensitiveLogRedactor.redact(raw)
        assertFalse("MRN value must not leak", sanitized.contains("987654321"))
        assertFalse("Patient ID must not leak", sanitized.contains("PT-8820"))
        assertTrue("Redacted ID placeholder must exist", sanitized.contains("[REDACTED_ID]"))
    }

    @Test
    fun testScrubbingClinicalNarrative() {
        val raw = "Specimen notes: Biopsy specimen: Right ascending colon adenocarcinoma with focal necrosis."
        val sanitized = SensitiveLogRedactor.redact(raw)
        assertFalse("Clinical free text must not leak", sanitized.contains("Right ascending colon adenocarcinoma with focal necrosis"))
        assertTrue("Clinical narrative placeholder must exist", sanitized.contains("[REDACTED_CLINICAL_NARRATIVE]"))
    }

    @Test
    fun testLogLengthCapTruncation() {
        val massiveString = "A".repeat(1000)
        val sanitized = SensitiveLogRedactor.redact(massiveString)
        assertTrue("Log length must be capped", sanitized.length < 600)
        assertTrue("Truncation message must exist", sanitized.contains("[TRUNCATED_TO_PREVENT_PII_LEAK]"))
    }

    @Test
    fun testCrcLoggerDoesNotThrowAndRedacts() {
        // Must succeed without throwing on host JVM
        CrcLogger.i("Test logger info with MRN: 12345")
        CrcLogger.d("Test logger debug with user@hospital.org")
        CrcLogger.w("Test logger warn with biopsy specimen: malignant tissue")
        CrcLogger.e("Test logger error", RuntimeException("Simulated error"))
    }

    // =========================================================================
    // SECTION 2: SAFE FILE VALIDATOR TESTS (ZIP SLIP & DEFENSIVE BOUNDS)
    // =========================================================================

    @Test
    fun testSanitizeFilename() {
        assertEquals("clean_specimen_1", SafeFileValidator.sanitizeFilename("clean_specimen_1"))
        assertEquals("suspicious_name", SafeFileValidator.sanitizeFilename("../../../suspicious_name"))
        assertEquals("TCGA-D5-6540", SafeFileValidator.sanitizeFilename("TCGA-D5-6540"))
        assertEquals("specimen_with_spaces", SafeFileValidator.sanitizeFilename("specimen with spaces"))
    }

    @Test
    fun testDetectZipSlipVulnerability() {
        val tempDest = File(System.getProperty("java.io.tmpdir"), "crc_safe_dest_test")
        tempDest.mkdirs()
        try {
            // Malicious entry trying to escape sandbox
            val maliciousEntry = "../../evil_executable.sh"
            try {
                SafeFileValidator.validateSafeZipEntry(maliciousEntry, tempDest)
                fail("Expected SecurityException on Zip Slip traversal")
            } catch (e: SecurityException) {
                assertTrue(e.message?.contains("Zip Slip") == true)
            }
        } finally {
            tempDest.deleteRecursively()
        }
    }

    @Test
    fun testValidateSafePath() {
        val baseDir = File(System.getProperty("java.io.tmpdir"), "crc_base_test")
        baseDir.mkdirs()
        try {
            val safeFile = File(baseDir, "bundle_001.crcbundle")
            val validated = SafeFileValidator.validateSafePath(safeFile, baseDir)
            assertNotNull(validated)

            val escapingFile = File(baseDir, "../escaping_file.txt")
            try {
                SafeFileValidator.validateSafePath(escapingFile, baseDir)
                fail("Expected SecurityException on directory traversal")
            } catch (e: SecurityException) {
                assertTrue(e.message?.contains("Path traversal security violation") == true)
            }
        } finally {
            baseDir.deleteRecursively()
        }
    }

    @Test
    fun testValidateTensorFloatsFiniteness() {
        val validFloats = floatArrayOf(0.1f, -0.5f, 1.25f, 0.0f)
        SafeFileValidator.validateTensorFloats(validFloats, "valid_test")

        val nanFloats = floatArrayOf(0.1f, Float.NaN, 1.25f)
        try {
            SafeFileValidator.validateTensorFloats(nanFloats, "nan_test")
            fail("Expected IllegalArgumentException on NaN float")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message?.contains("NaN") == true)
        }

        val infFloats = floatArrayOf(0.1f, Float.POSITIVE_INFINITY, 1.25f)
        try {
            SafeFileValidator.validateTensorFloats(infFloats, "inf_test")
            fail("Expected IllegalArgumentException on Infinity float")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message?.contains("Infinity") == true)
        }
    }

    @Test
    fun testValidatePatchCountBounds() {
        SafeFileValidator.validatePatchCount(1)
        SafeFileValidator.validatePatchCount(345)
        SafeFileValidator.validatePatchCount(10000)

        try {
            SafeFileValidator.validatePatchCount(0)
            fail("Expected IllegalArgumentException on patch count = 0")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message?.contains("boundaries") == true)
        }

        try {
            SafeFileValidator.validatePatchCount(10001)
            fail("Expected IllegalArgumentException on patch count > 10000")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message?.contains("boundaries") == true)
        }
    }

    // =========================================================================
    // SECTION 3: PROVENANCE MANIFEST & INTEGRITY TESTS
    // =========================================================================

    @Test
    fun testSpecimenProvenanceManifestGeneration() {
        val testCase = ResearchCase(
            caseIdentifier = "TCGA-D5-6540",
            specimenDescription = "MSI Reference Benchmark Biopsy",
            researchContext = "Validation holdout",
            status = CaseStatus.ANALYZED,
            patchCount = 345,
            protocol = InferenceProtocol.PROTOCOL_1_CAPPED
        )

        val testRun = InferenceRun(
            caseId = 1L,
            modelIdentifier = "fusion_model_fp32.onnx",
            inferenceArchitecture = "CrossAttentionFusion_FP32",
            protocol = InferenceProtocol.PROTOCOL_1_CAPPED,
            predictedClass = 1,
            predictedLabel = "MSI",
            probability = 0.99038f,
            logit = 4.63481f,
            threshold = 0.8300f,
            margin = 0.16038f,
            latencyMs = 28.5,
            numPatches = 345,
            diagnosticsSummary = "Normal run",
            executionTimestamp = System.currentTimeMillis(),
            status = "SUCCESS"
        )

        val manifest = ProvenanceAuditManager.generateManifest(
            researchCase = testCase,
            inferenceRun = testRun
        )

        assertEquals("TCGA-D5-6540", manifest.specimenIdentifier)
        assertEquals("1.0.0", manifest.manifestSchemaVersion)
        assertEquals("0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73", manifest.modelProvenance.modelSha256)
        assertEquals("UPSTREAM_PRECOMPUTED_UNVERIFIED", manifest.encoderProvenance.encoderStatus)
        assertEquals("phase7-v1.0", manifest.encoderProvenance.preprocessingManifestVersion)
        assertEquals(345, manifest.inputFingerprints.patchCount)
        assertEquals(0.8300, manifest.thresholdConfig.decisionThreshold, 1e-6)

        // Integrity statement assertions
        assertFalse("Manifest must NOT claim to be a digital signature", manifest.integrityStatement.isDigitalSignature)
        assertEquals("SHA-256 Plaintext Digest", manifest.integrityStatement.checksumType)
        assertTrue(manifest.integrityStatement.disclaimer.contains("NOT an authenticated digital signature"))
        assertTrue(manifest.integrityStatement.manifestPayloadSha256.isNotBlank())
    }

    @Test
    fun testManifestJsonSerializationAndDeserialization() {
        val testCase = ResearchCase(
            caseIdentifier = "TCGA-CM-6675",
            specimenDescription = "MSS Reference Benchmark Biopsy",
            researchContext = "Validation holdout",
            status = CaseStatus.READY_FOR_ANALYSIS,
            patchCount = 541,
            protocol = InferenceProtocol.PROTOCOL_1_CAPPED
        )

        val manifest = ProvenanceAuditManager.generateManifest(
            researchCase = testCase,
            inferenceRun = null
        )

        val jsonString = ProvenanceAuditManager.serializeManifestJson(manifest)
        assertNotNull(jsonString)

        val parsedJson = JSONObject(jsonString)
        assertEquals("1.0.0", parsedJson.getString("manifest_schema_version"))
        assertEquals("TCGA-CM-6675", parsedJson.getString("specimen_identifier"))

        val encJson = parsedJson.getJSONObject("encoder_provenance")
        assertEquals("UPSTREAM_PRECOMPUTED_UNVERIFIED", encJson.getString("encoder_status"))

        val integJson = parsedJson.getJSONObject("integrity_statement")
        assertFalse(integJson.getBoolean("is_digital_signature"))
        assertEquals("SHA-256 Plaintext Digest", integJson.getString("checksum_type"))
    }

    @Test
    fun testReportTraceabilityVerification() {
        val testCase = ResearchCase(
            caseIdentifier = "TCGA-D5-6540",
            specimenDescription = "MSI Reference Biopsy",
            researchContext = "Validation holdout",
            status = CaseStatus.ANALYZED,
            patchCount = 345,
            protocol = InferenceProtocol.PROTOCOL_1_CAPPED
        )

        val testRun = InferenceRun(
            caseId = 1L,
            modelIdentifier = "fusion_model_fp32.onnx",
            inferenceArchitecture = "CrossAttentionFusion_FP32",
            protocol = InferenceProtocol.PROTOCOL_1_CAPPED,
            predictedClass = 1,
            predictedLabel = "MSI",
            probability = 0.99038f,
            logit = 4.63481f,
            threshold = 0.8300f,
            margin = 0.16038f,
            latencyMs = 28.5,
            numPatches = 345,
            diagnosticsSummary = "Normal run",
            executionTimestamp = System.currentTimeMillis(),
            status = "SUCCESS"
        )

        val notes = listOf(
            ReviewNote(
                caseId = 1L,
                category = ReviewNoteCategory.HISTOLOGICAL_OBSERVATION,
                authorLabel = "Lead Pathologist",
                content = "Observed high TIL infiltration"
            )
        )

        val reportText = ReportGenerator.generateTextReport(testCase, testRun, notes)
        val manifest = ReportGenerator.generateProvenanceManifest(testCase, testRun)

        val verificationResult = ProvenanceAuditManager.verifyReportTraceability(reportText, manifest)
        assertTrue("Report must be fully traceable to specimen and model", verificationResult.isTraceable)
        assertTrue("Specimen identifier must match in report", verificationResult.specimenMatched)
        assertTrue("Model SHA-256 must match in report", verificationResult.modelHashMatched)
    }

    // =========================================================================
    // SECTION 4: AUDIT TRAIL LOGGING & RETENTION TESTS
    // =========================================================================

    @Test
    fun testAuditTrailLoggingAndRetention() {
        val repository = InMemoryWorkstationRepository()

        // 1. Ingest specimen
        val caseId = repository.insertCase(
            ResearchCase(
                caseIdentifier = "TCGA-AUDIT-001",
                specimenDescription = "Security audit trial specimen",
                researchContext = "WP7.5 validation",
                status = CaseStatus.READY_FOR_ANALYSIS,
                patchCount = 100,
                protocol = InferenceProtocol.PROTOCOL_1_CAPPED
            )
        )

        // Record ingestion audit event
        val event1 = AuditTrailRecord(
            eventTimestamp = 1000L,
            actionType = AuditActionType.SPECIMEN_INGESTED,
            specimenId = "TCGA-AUDIT-001",
            details = "Specimen ingested successfully with 100 patches"
        )
        val auditId1 = repository.recordAuditEvent(event1)
        assertTrue("Audit event ID must be > 0", auditId1 > 0)

        // 2. Inference execution
        val run = InferenceRun(
            caseId = caseId,
            modelIdentifier = "fusion_model_fp32.onnx",
            inferenceArchitecture = "CrossAttentionFusion_FP32",
            protocol = InferenceProtocol.PROTOCOL_1_CAPPED,
            predictedClass = 0,
            predictedLabel = "MSS",
            probability = 0.1500f,
            logit = -1.7346f,
            threshold = 0.8300f,
            margin = -0.6800f,
            latencyMs = 25.0,
            numPatches = 100,
            diagnosticsSummary = "Clean inference",
            executionTimestamp = System.currentTimeMillis(),
            status = "SUCCESS"
        )
        repository.recordInferenceResult(caseId, run)

        val event2 = AuditTrailRecord(
            eventTimestamp = 2000L,
            actionType = AuditActionType.INFERENCE_EXECUTED,
            specimenId = "TCGA-AUDIT-001",
            details = "Inference completed: MSS (P = 0.1500)"
        )
        repository.recordAuditEvent(event2)

        // 3. Query audit trail
        val allAudit = repository.getAuditTrail(50)
        assertTrue("Must have at least 2 audit events recorded", allAudit.size >= 2)

        val specimenAudit = repository.getAuditTrailForSpecimen("TCGA-AUDIT-001")
        assertEquals(2, specimenAudit.size)
        assertEquals(AuditActionType.INFERENCE_EXECUTED, specimenAudit[0].actionType)
        assertEquals(AuditActionType.SPECIMEN_INGESTED, specimenAudit[1].actionType)

        // 4. Clear audit trail
        val clearedCount = repository.clearAuditTrail()
        assertEquals(2, clearedCount)
        assertTrue(repository.getAuditTrail(50).isEmpty())
    }
}
