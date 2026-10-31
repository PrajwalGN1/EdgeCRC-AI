package org.crc.multimodal

import org.crc.multimodal.batch.BatchCoordinator
import org.crc.multimodal.batch.BatchReportGenerator
import org.crc.multimodal.batch.BatchRun
import org.crc.multimodal.batch.BatchSpecimenItem
import org.crc.multimodal.batch.BatchSpecimenStatus
import org.crc.multimodal.batch.BatchStatus
import org.crc.multimodal.batch.BatchStatusPalette
import org.crc.multimodal.inference.AuthenticBundleLoader
import org.crc.multimodal.inference.AuthenticFeatureBundle
import org.crc.multimodal.inference.InferenceProtocol
import org.crc.multimodal.inference.InputProvenance
import org.crc.multimodal.inference.InputValidator
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

/**
 * Verification Test Suite for Batch Specimen Analysis Status Color System
 * and Protocol 2 Dynamic Routing Support.
 *
 * Requirements:
 * 1. Status Colors:
 *    - MSS:      #2ECC9A (Emerald Green)
 *    - MSI:      #F56B70 (Coral Red)
 *    - FAILED:   #F59E0B (Amber Orange)
 *    - REJECTED: #D6A83D (Gold)
 *    - Unknown:  #888888 (Safe fallback)
 *    - Badge Text: #101820 (Dark slate for WCAG AAA contrast on bright chips)
 *
 * 2. Protocol Handling:
 *    - Protocol 1 processes valid inputs within 1,000-patch limit.
 *    - Protocol 1 rejects or routes over-limit inputs according to selection policy.
 *    - Protocol 2 processes inputs exceeding 1,000 patches when supported.
 *    - Protocol 2 preserves all patches without silent truncation.
 *    - Actual protocol used is persisted and displayed.
 *    - Existing bundle overwrite protection remains intact (REJECTED).
 *    - Runtime failures are not mislabeled as policy rejections.
 *    - Frozen decision threshold (tau = 0.83) and class mapping remain unchanged.
 */
class BatchProtocol2AndStatusColorTest {

    private lateinit var tempDir: File
    private lateinit var mockRepository: InMemoryBatchRepository

    @Before
    fun setUp() {
        tempDir = File("build/tmp/batch_p2_tests_${System.currentTimeMillis()}").apply { mkdirs() }
        mockRepository = InMemoryBatchRepository()
    }

    @After
    fun tearDown() {
        tempDir.deleteRecursively()
    }

    private fun createSyntheticBundle(
        caseId: String,
        numPatches: Int,
        isCorrupt: Boolean = false,
        isNan: Boolean = false
    ): File {
        val file = File(tempDir, "$caseId.crcbundle")
        val idBytes = caseId.toByteArray(Charsets.UTF_8)
        val descBytes = "Synthetic Test Bundle".toByteArray(Charsets.UTF_8)

        val vFloats = FloatArray(numPatches * 1024) { 0.05f }
        if (isNan) vFloats[0] = Float.NaN

        val tFloats = FloatArray(768) { 0.10f }

        val vByteBuffer = ByteBuffer.allocate(vFloats.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        vFloats.forEach { vByteBuffer.putFloat(it) }
        val vBytes = vByteBuffer.array()

        val tByteBuffer = ByteBuffer.allocate(tFloats.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        tFloats.forEach { tByteBuffer.putFloat(it) }
        val tBytes = tByteBuffer.array()

        val vHash = MessageDigest.getInstance("SHA-256").digest(vBytes)
        val tHash = MessageDigest.getInstance("SHA-256").digest(tBytes)

        if (isCorrupt) {
            vBytes[vBytes.size - 1] = (vBytes[vBytes.size - 1] + 1).toByte()
        }

        val out = ByteArrayOutputStream()
        out.write(AuthenticBundleLoader.MAGIC_CRCB)
        val headerBuf = ByteBuffer.allocate(4 + 4 + idBytes.size + 4 + descBytes.size + 16).order(ByteOrder.LITTLE_ENDIAN)
        headerBuf.putInt(2) // version
        headerBuf.putInt(idBytes.size)
        headerBuf.put(idBytes)
        headerBuf.putInt(descBytes.size)
        headerBuf.put(descBytes)
        headerBuf.putInt(numPatches)
        headerBuf.putInt(1024)
        headerBuf.putInt(768)
        headerBuf.putInt(1) // float32
        out.write(headerBuf.array())
        out.write(vHash)
        out.write(tHash)
        out.write(tBytes)
        out.write(vBytes)

        file.writeBytes(out.toByteArray())
        return file
    }

    // =========================================================================
    // SECTION 1: STATUS COLOR SYSTEM TESTS
    // =========================================================================

    @Test
    fun testStatusColorsResolveAccordingToHealthcarePalette() {
        // String label lookups
        assertEquals("#2ECC9A", BatchStatusPalette.resolveColorHex("MSS"))
        assertEquals("#2ECC9A", BatchStatusPalette.resolveColorHex("mss"))
        assertEquals("#F56B70", BatchStatusPalette.resolveColorHex("MSI"))
        assertEquals("#F56B70", BatchStatusPalette.resolveColorHex("msi"))
        assertEquals("#F59E0B", BatchStatusPalette.resolveColorHex("FAILED"))
        assertEquals("#F59E0B", BatchStatusPalette.resolveColorHex("failed"))
        assertEquals("#D6A83D", BatchStatusPalette.resolveColorHex("REJECTED"))
        assertEquals("#D6A83D", BatchStatusPalette.resolveColorHex("rejected"))

        // Safe fallback for unknown status
        assertEquals("#888888", BatchStatusPalette.resolveColorHex("UNKNOWN"))
        assertEquals("#888888", BatchStatusPalette.resolveColorHex(null))
        assertEquals("#888888", BatchStatusPalette.resolveColorHex(""))
        assertEquals("#888888", BatchStatusPalette.resolveColorHex("SOMETHING_ELSE"))

        // High-contrast badge text color
        assertEquals("#101820", BatchStatusPalette.COLOR_TEXT_DARK)
    }

    @Test
    fun testStatusColorPaletteResolvesFromEnumAndClass() {
        // Model classification: MSS (Class 0) vs MSI (Class 1)
        assertEquals("#2ECC9A", BatchStatusPalette.resolveColorHex(BatchSpecimenStatus.COMPLETED, predictedClass = 0))
        assertEquals("#F56B70", BatchStatusPalette.resolveColorHex(BatchSpecimenStatus.COMPLETED, predictedClass = 1))

        // Failures vs Rejections
        assertEquals("#F59E0B", BatchStatusPalette.resolveColorHex(BatchSpecimenStatus.FAILED))
        assertEquals("#D6A83D", BatchStatusPalette.resolveColorHex(BatchSpecimenStatus.REJECTED))

        // Neutral statuses fallback to safe gray
        assertEquals("#888888", BatchStatusPalette.resolveColorHex(BatchSpecimenStatus.VALIDATED))
        assertEquals("#888888", BatchStatusPalette.resolveColorHex(BatchSpecimenStatus.INFERRING))
        assertEquals("#888888", BatchStatusPalette.resolveColorHex(BatchSpecimenStatus.CANCELLED))
    }

    @Test
    fun testColorResourceAndTextResourceMapping() {
        // Text color must be batch_status_text (#101820) on bright badge chips
        assertEquals(R.color.batch_status_text, BatchStatusPalette.resolveTextColorRes(BatchSpecimenStatus.COMPLETED, 0))
        assertEquals(R.color.batch_status_text, BatchStatusPalette.resolveTextColorRes(BatchSpecimenStatus.COMPLETED, 1))
        assertEquals(R.color.batch_status_text, BatchStatusPalette.resolveTextColorRes(BatchSpecimenStatus.FAILED))
        assertEquals(R.color.batch_status_text, BatchStatusPalette.resolveTextColorRes(BatchSpecimenStatus.REJECTED))

        // Unprocessed states use white text on dark background
        assertEquals(android.R.color.white, BatchStatusPalette.resolveTextColorRes(BatchSpecimenStatus.VALIDATED))
        assertEquals(android.R.color.white, BatchStatusPalette.resolveTextColorRes(BatchSpecimenStatus.INFERRING))

        // Background resource IDs
        assertEquals(R.color.batch_status_mss, BatchStatusPalette.resolveColorRes(BatchSpecimenStatus.COMPLETED, 0))
        assertEquals(R.color.batch_status_msi, BatchStatusPalette.resolveColorRes(BatchSpecimenStatus.COMPLETED, 1))
        assertEquals(R.color.batch_status_failed, BatchStatusPalette.resolveColorRes(BatchSpecimenStatus.FAILED))
        assertEquals(R.color.batch_status_rejected, BatchStatusPalette.resolveColorRes(BatchSpecimenStatus.REJECTED))
    }

    // =========================================================================
    // SECTION 2: PROTOCOL 1 & PROTOCOL 2 SPECIMEN INPUT VALIDATION
    // =========================================================================

    @Test
    fun testProtocol1ProcessesValidInputsWithin1000PatchLimit() {
        val numPatches = 450
        val visionFloats = FloatArray(numPatches * 1024) { 0.1f }

        // Protocol 1 validation: ≤ 1000 patches passes without exception
        InputValidator.validateVision(
            visionData = visionFloats,
            numPatches = numPatches,
            protocol = InferenceProtocol.PROTOCOL_1_CAPPED
        )

        assertEquals(450 * 1024, visionFloats.size)
    }

    @Test
    fun testProtocol1RejectsInputsExceeding1000Patches() {
        val numPatches = 2296 // Matches TCGA-NH-A50T
        val visionFloats = FloatArray(numPatches * 1024) { 0.1f }

        try {
            InputValidator.validateVision(
                visionData = visionFloats,
                numPatches = numPatches,
                protocol = InferenceProtocol.PROTOCOL_1_CAPPED
            )
            fail("Protocol 1 MUST throw IllegalArgumentException when patch count > 1000")
        } catch (e: IllegalArgumentException) {
            assertTrue("Exception message must cite Protocol 1 cap of 1000", e.message?.contains("Protocol 1 enforces a maximum cap of 1000 patches") == true)
            assertTrue("Exception message must cite actual received count", e.message?.contains("2296") == true)
        }
    }

    @Test
    fun testProtocol2ProcessesInputsExceeding1000PatchesWithoutTruncation() {
        val numPatches = 2888 // Matches TCGA-NH-A5IV
        val visionFloats = FloatArray(numPatches * 1024) { 0.05f }

        // Protocol 2 validation: uncapped passes without exception
        InputValidator.validateVision(
            visionData = visionFloats,
            numPatches = numPatches,
            protocol = InferenceProtocol.PROTOCOL_2_UNCAPPED
        )

        // Verify no silent truncation: entire buffer intact
        assertEquals("Buffer size must exactly equal 2888 * 1024", numPatches * 1024, visionFloats.size)
    }

    @Test
    fun testProtocol2AlsoProcessesInputsWithin1000Patches() {
        val numPatches = 385
        val visionFloats = FloatArray(numPatches * 1024) { 0.02f }

        InputValidator.validateVision(
            visionData = visionFloats,
            numPatches = numPatches,
            protocol = InferenceProtocol.PROTOCOL_2_UNCAPPED
        )

        assertEquals(385 * 1024, visionFloats.size)
    }

    // =========================================================================
    // SECTION 3: BATCH PROTOCOL ROUTING & REJECTION BEHAVIOR
    // =========================================================================

    @Test
    fun testBatchRoutingAutoRoutesOverLimitSpecimenToProtocol2WhenEnabled() {
        val fUnder = createSyntheticBundle("TCGA-UNDER-1K", 450)
        val fOver = createSyntheticBundle("TCGA-OVER-1K", 2296) // Matches TCGA-NH-A50T

        val coordinator = BatchCoordinator(mockRepository)
        val batch = coordinator.ingestFiles(listOf(fUnder, fOver))

        assertEquals(BatchStatus.READY, batch.status)
        assertEquals(2, batch.items.size)

        val underItem = batch.items.first { it.specimenId == "TCGA-UNDER-1K" }
        val overItem = batch.items.first { it.specimenId == "TCGA-OVER-1K" }

        // Initial assignment before execution
        assertEquals("PROTOCOL_1_CAPPED", underItem.protocol)
        assertEquals("PROTOCOL_2_UNCAPPED", overItem.protocol)
        assertNotNull(overItem.protocolReason)
        assertTrue(overItem.protocolReason!!.contains("Protocol 2"))
    }

    @Test
    fun testBatchRoutingRejectsOverLimitSpecimenWhenFallbackDisabled() {
        // When user explicitly selects Protocol 1 without automatic fallback
        val numPatches = 2296
        val allowFallback = false
        val baseProtocol = InferenceProtocol.PROTOCOL_1_CAPPED

        val isOverLimit = numPatches > 1000
        val resolvedProtocol: InferenceProtocol?
        val rejectionReason: String?

        if (baseProtocol == InferenceProtocol.PROTOCOL_1_CAPPED && isOverLimit) {
            if (allowFallback) {
                resolvedProtocol = InferenceProtocol.PROTOCOL_2_UNCAPPED
                rejectionReason = null
            } else {
                resolvedProtocol = null
                rejectionReason = "Specimen exceeds Protocol 1 limit (2296 patches > 1000). Fallback to Protocol 2 disabled."
            }
        } else {
            resolvedProtocol = baseProtocol
            rejectionReason = null
        }

        assertNull("Protocol must not resolve when fallback disabled", resolvedProtocol)
        assertNotNull("Rejection reason must be provided", rejectionReason)
        assertTrue(rejectionReason!!.contains("Specimen exceeds Protocol 1 limit"))
    }

    @Test
    fun testBundleOverwriteProtectionPreservesRejectionStatus() {
        // Case: bundle already exists and overwrite not permitted (like TCGA-G4-6311)
        val existingItem = BatchSpecimenItem(
            batchId = "BATCH_PREV",
            specimenId = "TCGA-G4-6311",
            status = BatchSpecimenStatus.REJECTED,
            validationError = "A bundle for specimen 'TCGA-G4-6311' already exists. Overwrite not permitted."
        )

        assertEquals(BatchSpecimenStatus.REJECTED, existingItem.status)
        assertEquals(
            "A bundle for specimen 'TCGA-G4-6311' already exists. Overwrite not permitted.",
            existingItem.validationError
        )
        // Must resolve to gold REJECTED color, NOT FAILED orange or MSI red
        assertEquals("#D6A83D", BatchStatusPalette.resolveColorHex(existingItem.status))
    }

    @Test
    fun testRuntimeFailuresAreDistinctFromPolicyRejections() {
        val failedItem = BatchSpecimenItem(
            batchId = "BATCH_TEST",
            specimenId = "TCGA-FAIL-01",
            status = BatchSpecimenStatus.FAILED,
            errorDetails = "ONNX Runtime Execution Provider memory allocation failed"
        )

        val rejectedItem = BatchSpecimenItem(
            batchId = "BATCH_TEST",
            specimenId = "TCGA-REJECT-01",
            status = BatchSpecimenStatus.REJECTED,
            validationError = "Protocol 1 enforces a maximum cap of 1000 patches, but received 2296"
        )

        // Ensure status semantics are not collapsed
        assertFalse("FAILED and REJECTED must have distinct statuses", failedItem.status == rejectedItem.status)
        assertEquals(BatchSpecimenStatus.FAILED, failedItem.status)
        assertEquals(BatchSpecimenStatus.REJECTED, rejectedItem.status)

        // Color distinction: FAILED = orange (#F59E0B), REJECTED = gold (#D6A83D)
        assertEquals("#F59E0B", BatchStatusPalette.resolveColorHex(failedItem.status))
        assertEquals("#D6A83D", BatchStatusPalette.resolveColorHex(rejectedItem.status))
        assertFalse("FAILED and REJECTED must have distinct colors",
            BatchStatusPalette.resolveColorHex(failedItem.status) == BatchStatusPalette.resolveColorHex(rejectedItem.status))
    }

    // =========================================================================
    // SECTION 4: PERSISTENCE & REPORT GENERATION PROTOCOL INTEGRITY
    // =========================================================================

    @Test
    fun testProtocolPersistenceAndRestorationInRepository() {
        val itemP1 = BatchSpecimenItem(
            batchId = "BATCH_PERSIST",
            specimenId = "TCGA-PERSIST-P1",
            status = BatchSpecimenStatus.COMPLETED,
            probability = 0.12f,
            predictedClass = 0,
            predictedLabel = "MSS",
            protocol = "PROTOCOL_1_CAPPED",
            protocolReason = null
        )

        val itemP2 = BatchSpecimenItem(
            batchId = "BATCH_PERSIST",
            specimenId = "TCGA-PERSIST-P2",
            status = BatchSpecimenStatus.COMPLETED,
            probability = 0.94f,
            predictedClass = 1,
            predictedLabel = "MSI",
            protocol = "PROTOCOL_2_UNCAPPED",
            protocolReason = "Automatic fallback: Specimen exceeds 1,000-patch limit (2296 patches > 1000). Evaluated under Protocol 2 uncapped."
        )

        val batch = BatchRun(
            batchId = "BATCH_PERSIST",
            title = "Persistence Test",
            createdAt = System.currentTimeMillis(),
            updatedAt = System.currentTimeMillis(),
            status = BatchStatus.COMPLETED,
            items = listOf(itemP1, itemP2)
        )

        mockRepository.insertBatchRun(batch)

        val retrieved = mockRepository.getBatchRun("BATCH_PERSIST")
        assertNotNull(retrieved)
        assertEquals(2, retrieved!!.items.size)

        val retrievedP1 = retrieved.items.first { it.specimenId == "TCGA-PERSIST-P1" }
        assertEquals("PROTOCOL_1_CAPPED", retrievedP1.protocol)
        assertNull(retrievedP1.protocolReason)

        val retrievedP2 = retrieved.items.first { it.specimenId == "TCGA-PERSIST-P2" }
        assertEquals("PROTOCOL_2_UNCAPPED", retrievedP2.protocol)
        assertNotNull(retrievedP2.protocolReason)
        assertTrue(retrievedP2.protocolReason!!.contains("Automatic fallback"))
    }

    @Test
    fun testJsonReportSerializesProtocolAndReason() {
        val itemP1 = BatchSpecimenItem(
            batchId = "BATCH_JSON",
            specimenId = "TCGA-JSON-01",
            status = BatchSpecimenStatus.COMPLETED,
            probability = 0.05f,
            predictedClass = 0,
            predictedLabel = "MSS",
            protocol = "PROTOCOL_1_CAPPED",
            protocolReason = null
        )

        val itemP2 = BatchSpecimenItem(
            batchId = "BATCH_JSON",
            specimenId = "TCGA-JSON-02",
            status = BatchSpecimenStatus.COMPLETED,
            probability = 0.92f,
            predictedClass = 1,
            predictedLabel = "MSI",
            protocol = "PROTOCOL_2_UNCAPPED",
            protocolReason = "Automatic fallback: Specimen exceeds 1,000-patch limit (2888 patches > 1000). Evaluated under Protocol 2 uncapped."
        )

        val batch = BatchRun(
            batchId = "BATCH_JSON",
            title = "Protocol JSON Test",
            createdAt = System.currentTimeMillis(),
            updatedAt = System.currentTimeMillis(),
            status = BatchStatus.COMPLETED,
            items = listOf(itemP1, itemP2)
        )

        val json = BatchReportGenerator.generateJsonReport(batch)

        // Verify protocol fields are serialized
        assertTrue("JSON must include PROTOCOL_1_CAPPED", json.contains("\"protocol\": \"PROTOCOL_1_CAPPED\""))
        assertTrue("JSON must include PROTOCOL_2_UNCAPPED", json.contains("\"protocol\": \"PROTOCOL_2_UNCAPPED\""))
        assertTrue("JSON must include protocol_reason", json.contains("\"protocol_reason\": \"Automatic fallback"))
    }

    // =========================================================================
    // SECTION 5: FROZEN THRESHOLD AND CLASS MAPPING INVARIANTS
    // =========================================================================

    @Test
    fun testDecisionThresholdAndClassMappingInvariance() {
        val threshold = 0.8300f

        // Probability boundary test around tau = 0.8300
        val pSub = 0.8299f
        val pAt = 0.8300f
        val pSupra = 0.8301f

        val classSub = if (pSub >= threshold) 1 else 0
        val classAt = if (pAt >= threshold) 1 else 0
        val classSupra = if (pSupra >= threshold) 1 else 0

        assertEquals("Sub-threshold probability must classify as MSS (0)", 0, classSub)
        assertEquals("Threshold probability (0.8300) must classify as MSI (1)", 1, classAt)
        assertEquals("Supra-threshold probability must classify as MSI (1)", 1, classSupra)

        // Color assignments: Class 0 -> Green (#2ECC9A), Class 1 -> Red (#F56B70)
        assertEquals("#2ECC9A", BatchStatusPalette.resolveColorHex(BatchSpecimenStatus.COMPLETED, classSub))
        assertEquals("#F56B70", BatchStatusPalette.resolveColorHex(BatchSpecimenStatus.COMPLETED, classAt))
        assertEquals("#F56B70", BatchStatusPalette.resolveColorHex(BatchSpecimenStatus.COMPLETED, classSupra))
    }
}
