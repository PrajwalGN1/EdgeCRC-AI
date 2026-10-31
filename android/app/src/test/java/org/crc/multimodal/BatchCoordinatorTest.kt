package org.crc.multimodal

import org.crc.multimodal.batch.BatchCoordinator
import org.crc.multimodal.batch.BatchReportGenerator
import org.crc.multimodal.batch.BatchRepository
import org.crc.multimodal.batch.BatchRun
import org.crc.multimodal.batch.BatchSpecimenItem
import org.crc.multimodal.batch.BatchSpecimenStatus
import org.crc.multimodal.batch.BatchStatistics
import org.crc.multimodal.batch.BatchStatus
import org.crc.multimodal.inference.AuthenticBundleLoader
import org.crc.multimodal.inference.AuthenticFeatureBundle
import org.crc.multimodal.inference.InputProvenance
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
 * Comprehensive automated test suite for Phase 6 Batch Ingestion,
 * Controlled Execution, Specimen Traceability, Statistics, and Research Reporting.
 */
class BatchCoordinatorTest {

    private lateinit var tempDir: File
    private lateinit var mockRepository: InMemoryBatchRepository

    @Before
    fun setUp() {
        tempDir = File("build/tmp/batch_tests_${System.currentTimeMillis()}").apply { mkdirs() }
        mockRepository = InMemoryBatchRepository()
    }

    @After
    fun tearDown() {
        tempDir.deleteRecursively()
    }

    private fun createSyntheticBundleFile(
        caseId: String,
        numPatches: Int,
        isCorrupt: Boolean = false,
        isNan: Boolean = false
    ): File {
        val file = File(tempDir, "$caseId.crcbundle")
        val idBytes = caseId.toByteArray(Charsets.UTF_8)
        val descBytes = "Test Bundle".toByteArray(Charsets.UTF_8)

        val vFloats = FloatArray(numPatches * 1024) { 0.1f }
        if (isNan) vFloats[0] = Float.NaN

        val tFloats = FloatArray(768) { 0.2f }

        val vByteBuffer = ByteBuffer.allocate(vFloats.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        vFloats.forEach { vByteBuffer.putFloat(it) }
        val vBytes = vByteBuffer.array()

        val tByteBuffer = ByteBuffer.allocate(tFloats.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        tFloats.forEach { tByteBuffer.putFloat(it) }
        val tBytes = tByteBuffer.array()

        val vHash = MessageDigest.getInstance("SHA-256").digest(vBytes)
        val tHash = MessageDigest.getInstance("SHA-256").digest(tBytes)

        if (isCorrupt) {
            // Flip a byte in the payload to corrupt checksum
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

    @Test
    fun testEmptySelectionThrowsException() {
        val coordinator = BatchCoordinator(mockRepository)
        try {
            coordinator.ingestFiles(emptyList())
            fail("Empty files list should throw IllegalArgumentException")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message?.contains("Cannot ingest an empty list") == true)
        }
    }

    @Test
    fun testMultipleValidBundlesIngestion() {
        val f1 = createSyntheticBundleFile("CASE_001", 10)
        val f2 = createSyntheticBundleFile("CASE_002", 25)
        val f3 = createSyntheticBundleFile("CASE_003", 50)

        val coordinator = BatchCoordinator(mockRepository)
        val batch = coordinator.ingestFiles(listOf(f1, f2, f3))

        assertEquals(BatchStatus.READY, batch.status)
        assertEquals(3, batch.items.size)
        assertEquals(3, batch.statistics.totalAccepted)
        assertEquals(0, batch.statistics.totalRejected)
        assertTrue(batch.items.all { it.status == BatchSpecimenStatus.VALIDATED })
        assertEquals("CASE_001", batch.items[0].specimenId)
        assertEquals(10, batch.items[0].patchCount)
        assertEquals("CASE_002", batch.items[1].specimenId)
        assertEquals(25, batch.items[1].patchCount)
        assertEquals("CASE_003", batch.items[2].specimenId)
        assertEquals(50, batch.items[2].patchCount)
    }

    @Test
    fun testMixedValidAndInvalidBundlesIngestion() {
        val fValid = createSyntheticBundleFile("VALID_CASE", 15)
        val fCorrupt = createSyntheticBundleFile("CORRUPT_CASE", 20, isCorrupt = true)
        val fNan = createSyntheticBundleFile("NAN_CASE", 10, isNan = true)

        val coordinator = BatchCoordinator(mockRepository)
        val batch = coordinator.ingestFiles(listOf(fValid, fCorrupt, fNan))

        assertEquals(BatchStatus.READY, batch.status)
        assertEquals(3, batch.items.size)
        assertEquals(1, batch.statistics.totalAccepted)
        assertEquals(2, batch.statistics.totalRejected)

        val validItem = batch.items.first { it.specimenId == "VALID_CASE" }
        assertEquals(BatchSpecimenStatus.VALIDATED, validItem.status)

        val corruptItem = batch.items.first { it.fileName.contains("CORRUPT_CASE") }
        assertEquals(BatchSpecimenStatus.REJECTED, corruptItem.status)
        assertTrue(corruptItem.validationError?.contains("mismatch") == true)

        val nanItem = batch.items.first { it.fileName.contains("NAN_CASE") }
        assertEquals(BatchSpecimenStatus.REJECTED, nanItem.status)
        assertTrue(nanItem.validationError?.contains("NaN") == true)
    }

    @Test
    fun testDuplicateSpecimenIdentifiersRejectedInBatch() {
        val f1 = createSyntheticBundleFile("DUP_CASE_01", 12)
        val f2 = createSyntheticBundleFile("DUP_CASE_01", 12) // exact same ID

        val coordinator = BatchCoordinator(mockRepository)
        val batch = coordinator.ingestFiles(listOf(f1, f2))

        assertEquals(BatchStatus.READY, batch.status)
        assertEquals(2, batch.items.size)
        assertEquals(1, batch.statistics.totalAccepted)
        assertEquals(1, batch.statistics.totalRejected)

        assertEquals(BatchSpecimenStatus.VALIDATED, batch.items[0].status)
        assertEquals(BatchSpecimenStatus.REJECTED, batch.items[1].status)
        assertEquals("Duplicate specimen identifier within batch", batch.items[1].validationError)
    }

    @Test
    fun testBatchStatisticsComputation() {
        val items = listOf(
            BatchSpecimenItem(
                batchId = "B1", specimenId = "S1", patchCount = 100,
                status = BatchSpecimenStatus.COMPLETED, logit = 2.5f, probability = 0.92f,
                predictedClass = 1, predictedLabel = "MSI", inferenceLatencyMs = 45L
            ),
            BatchSpecimenItem(
                batchId = "B1", specimenId = "S2", patchCount = 200,
                status = BatchSpecimenStatus.COMPLETED, logit = -3.0f, probability = 0.05f,
                predictedClass = 0, predictedLabel = "MSS", inferenceLatencyMs = 65L
            ),
            BatchSpecimenItem(
                batchId = "B1", specimenId = "S3", patchCount = 150,
                status = BatchSpecimenStatus.COMPLETED, logit = -1.2f, probability = 0.23f,
                predictedClass = 0, predictedLabel = "MSS", inferenceLatencyMs = 50L
            ),
            BatchSpecimenItem(
                batchId = "B1", specimenId = "S4", patchCount = 0,
                status = BatchSpecimenStatus.REJECTED, validationError = "Corrupted header"
            ),
            BatchSpecimenItem(
                batchId = "B1", specimenId = "S5", patchCount = 120,
                status = BatchSpecimenStatus.FAILED, errorDetails = "Model runtime exception"
            )
        )

        val stats = BatchStatistics.compute(items, totalDurationMs = 250L)

        assertEquals(5, stats.totalSelected)
        assertEquals(4, stats.totalAccepted)
        assertEquals(1, stats.totalRejected)
        assertEquals(4, stats.totalProcessed)
        assertEquals(3, stats.totalSuccessful)
        assertEquals(1, stats.totalFailed)
        assertEquals(0, stats.totalCancelled)

        // Class breakdown over successful items (3)
        assertEquals(1, stats.msiCount)
        assertEquals(2, stats.mssCount)
        assertEquals(1f / 3f, stats.msiProportion, 1e-4f)
        assertEquals(2f / 3f, stats.mssProportion, 1e-4f)

        // Latencies (45, 50, 65)
        assertEquals(45L, stats.minLatencyMs)
        assertEquals(65L, stats.maxLatencyMs)
        assertEquals(50.0, stats.medianLatencyMs, 1e-4)
        assertEquals(53.333, stats.meanLatencyMs, 0.01)

        // Patch count (100, 150, 200)
        assertEquals(100, stats.patchCountMin)
        assertEquals(200, stats.patchCountMax)
        assertEquals(150.0, stats.patchCountMean, 1e-4)

        // Exclusions
        assertEquals(2, stats.exclusionCount)
        assertTrue(stats.exclusionReasons.containsKey("Corrupted header"))
        assertTrue(stats.exclusionReasons.containsKey("Model runtime exception"))
    }

    @Test
    fun testJsonReportStructureAndCryptographicVerification() {
        val items = listOf(
            BatchSpecimenItem(
                batchId = "BATCH-TEST-001", specimenId = "TCGA-TEST-01", patchCount = 100,
                bundleSha256 = "66ffd8343c0c46b18d37e64bff0ed5998c17f7d0df69b604dc16fe4bd260909a",
                status = BatchSpecimenStatus.COMPLETED, logit = 3.5f, probability = 0.97f,
                predictedClass = 1, predictedLabel = "MSI", inferenceLatencyMs = 42L
            ),
            BatchSpecimenItem(
                batchId = "BATCH-TEST-001", specimenId = "TCGA-TEST-02", patchCount = 200,
                bundleSha256 = "ee5930faed928e1d94285d03d37cd72494ebc5472251d56cf43760f0c055bc94",
                status = BatchSpecimenStatus.COMPLETED, logit = -5.0f, probability = 0.006f,
                predictedClass = 0, predictedLabel = "MSS", inferenceLatencyMs = 55L
            )
        )

        val batch = BatchRun(
            batchId = "BATCH-TEST-001",
            status = BatchStatus.COMPLETED,
            totalDurationMs = 120L,
            items = items
        )

        val jsonStr = BatchReportGenerator.generateJsonReport(batch)

        // Verify required metadata fields in JSON
        assertTrue("Must contain report title", jsonStr.contains("report_title"))
        assertTrue("Must contain research use disclaimer", jsonStr.contains("RESEARCH USE ONLY"))
        assertTrue("Must contain batch ID", jsonStr.contains("BATCH-TEST-001"))
        assertTrue("Must contain frozen model SHA256", jsonStr.contains("0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73"))
        assertTrue("Must contain threshold 0.83", jsonStr.contains("0.83"))
        assertTrue("Must contain TCGA-TEST-01", jsonStr.contains("TCGA-TEST-01"))
        assertTrue("Must contain MSI probability", jsonStr.contains("0.970000"))
        assertTrue("Must contain TCGA-TEST-02", jsonStr.contains("TCGA-TEST-02"))
        assertTrue("Must contain MSS probability", jsonStr.contains("0.006000"))

        // Test File Cryptographic Hash Verification
        val reportFile = File(tempDir, "test_report.json").apply { writeText(jsonStr) }
        val digest = MessageDigest.getInstance("SHA-256").digest(reportFile.readBytes()).joinToString("") { "%02x".format(it) }

        assertTrue("Report integrity must pass on matching hash", BatchReportGenerator.verifyReportIntegrity(reportFile, digest))

        // Modify 1 byte of the report
        reportFile.appendText(" ")
        assertFalse("Report integrity must fail when report is tampered", BatchReportGenerator.verifyReportIntegrity(reportFile, digest))
    }

    @Test
    fun testCrashRecoveryMarksInterruptedItems() {
        val repo = InMemoryBatchRepository()
        val batch = BatchRun(
            batchId = "INTERRUPTED_BATCH",
            status = BatchStatus.PROCESSING,
            items = listOf(
                BatchSpecimenItem(batchId = "INTERRUPTED_BATCH", specimenId = "S1", status = BatchSpecimenStatus.COMPLETED, probability = 0.5f),
                BatchSpecimenItem(batchId = "INTERRUPTED_BATCH", specimenId = "S2", status = BatchSpecimenStatus.INFERRING),
                BatchSpecimenItem(batchId = "INTERRUPTED_BATCH", specimenId = "S3", status = BatchSpecimenStatus.VALIDATED)
            )
        )
        repo.insertBatchRun(batch)

        repo.recoverInterruptedBatches()

        val recovered = repo.getBatchRun("INTERRUPTED_BATCH")
        assertNotNull(recovered)
        assertEquals(BatchStatus.PARTIALLY_COMPLETED, recovered!!.status)
        val s2 = recovered.items.first { it.specimenId == "S2" }
        assertEquals(BatchSpecimenStatus.CANCELLED, s2.status)
        assertEquals("Interrupted by system/app termination", s2.errorDetails)

        val s3 = recovered.items.first { it.specimenId == "S3" }
        assertEquals("Queued validated item must transition to CANCELLED", BatchSpecimenStatus.CANCELLED, s3.status)
        assertEquals("Interrupted before execution commenced", s3.errorDetails)
    }

    @Test
    fun testAF04MixedBatchCrashRecovery() {
        val repo = InMemoryBatchRepository()
        val batch = BatchRun(
            batchId = "BATCH_MIXED",
            status = BatchStatus.PROCESSING,
            items = listOf(
                BatchSpecimenItem(batchId = "BATCH_MIXED", specimenId = "C1", status = BatchSpecimenStatus.COMPLETED, probability = 0.92f, predictedClass = 1),
                BatchSpecimenItem(batchId = "BATCH_MIXED", specimenId = "F1", status = BatchSpecimenStatus.FAILED, errorDetails = "Corrupt tensor payload"),
                BatchSpecimenItem(batchId = "BATCH_MIXED", specimenId = "I1", status = BatchSpecimenStatus.INFERRING),
                BatchSpecimenItem(batchId = "BATCH_MIXED", specimenId = "V1", status = BatchSpecimenStatus.VALIDATED),
                BatchSpecimenItem(batchId = "BATCH_MIXED", specimenId = "V2", status = BatchSpecimenStatus.VALIDATED)
            )
        )
        repo.insertBatchRun(batch)

        repo.recoverInterruptedBatches()

        val recovered = repo.getBatchRun("BATCH_MIXED")
        assertNotNull(recovered)
        assertEquals(BatchStatus.PARTIALLY_COMPLETED, recovered!!.status)

        // 1. Completed result preserved
        val c1 = recovered.items.first { it.specimenId == "C1" }
        assertEquals(BatchSpecimenStatus.COMPLETED, c1.status)
        assertEquals(0.92f, c1.probability ?: 0f, 0.001f)
        assertEquals(1, c1.predictedClass)

        // 2. Failed result preserved
        val f1 = recovered.items.first { it.specimenId == "F1" }
        assertEquals(BatchSpecimenStatus.FAILED, f1.status)
        assertEquals("Corrupt tensor payload", f1.errorDetails)

        // 3. Inferring item cancelled with termination detail
        val i1 = recovered.items.first { it.specimenId == "I1" }
        assertEquals(BatchSpecimenStatus.CANCELLED, i1.status)
        assertEquals("Interrupted by system/app termination", i1.errorDetails)

        // 4. Queued validated items cancelled with commenced detail
        val v1 = recovered.items.first { it.specimenId == "V1" }
        val v2 = recovered.items.first { it.specimenId == "V2" }
        assertEquals(BatchSpecimenStatus.CANCELLED, v1.status)
        assertEquals("Interrupted before execution commenced", v1.errorDetails)
        assertEquals(BatchSpecimenStatus.CANCELLED, v2.status)
        assertEquals("Interrupted before execution commenced", v2.errorDetails)

        // 5. Zero items remain in VALIDATED status under this recovered batch
        assertFalse(recovered.items.any { it.status == BatchSpecimenStatus.VALIDATED })
        assertFalse(recovered.items.any { it.status == BatchSpecimenStatus.INFERRING })
    }

    @Test
    fun testAF04UnaffectedBatchPreservedAndIdempotency() {
        val repo = InMemoryBatchRepository()
        // Batch 1: READY (not yet started) with VALIDATED items - must NOT be touched
        val readyBatch = BatchRun(
            batchId = "BATCH_READY",
            status = BatchStatus.READY,
            items = listOf(
                BatchSpecimenItem(batchId = "BATCH_READY", specimenId = "R1", status = BatchSpecimenStatus.VALIDATED),
                BatchSpecimenItem(batchId = "BATCH_READY", specimenId = "R2", status = BatchSpecimenStatus.VALIDATED)
            )
        )
        // Batch 2: PROCESSING (interrupted)
        val processingBatch = BatchRun(
            batchId = "BATCH_PROCESSING",
            status = BatchStatus.PROCESSING,
            items = listOf(
                BatchSpecimenItem(batchId = "BATCH_PROCESSING", specimenId = "P1", status = BatchSpecimenStatus.VALIDATED)
            )
        )
        repo.insertBatchRun(readyBatch)
        repo.insertBatchRun(processingBatch)

        // First recovery execution
        repo.recoverInterruptedBatches()

        // Verify ready batch remains READY and its items remain VALIDATED
        val after1Ready = repo.getBatchRun("BATCH_READY")
        assertNotNull(after1Ready)
        assertEquals(BatchStatus.READY, after1Ready!!.status)
        assertEquals(BatchSpecimenStatus.VALIDATED, after1Ready.items[0].status)
        assertEquals(BatchSpecimenStatus.VALIDATED, after1Ready.items[1].status)

        // Verify processing batch recovered
        val after1Proc = repo.getBatchRun("BATCH_PROCESSING")
        assertNotNull(after1Proc)
        assertEquals(BatchStatus.PARTIALLY_COMPLETED, after1Proc!!.status)
        assertEquals(BatchSpecimenStatus.CANCELLED, after1Proc.items[0].status)

        // Second recovery execution (Idempotency test)
        repo.recoverInterruptedBatches()

        val after2Ready = repo.getBatchRun("BATCH_READY")
        assertNotNull(after2Ready)
        assertEquals(BatchStatus.READY, after2Ready!!.status)
        assertEquals(BatchSpecimenStatus.VALIDATED, after2Ready.items[0].status)

        val after2Proc = repo.getBatchRun("BATCH_PROCESSING")
        assertNotNull(after2Proc)
        assertEquals(BatchStatus.PARTIALLY_COMPLETED, after2Proc!!.status)
        assertEquals(BatchSpecimenStatus.CANCELLED, after2Proc.items[0].status)
    }

    // =========================================================================
    // MASTER DIRECTIVE PHASE 2: AF-04 EXHAUSTIVE CRASH RECOVERY TESTS (10 SCENARIOS)
    // =========================================================================

    @Test
    fun testAF04_Scenario1_InterruptionBeforeFirstItemStarts() {
        val repo = InMemoryBatchRepository()
        val batch = BatchRun(
            batchId = "BATCH_SCENARIO_1",
            status = BatchStatus.PROCESSING,
            items = listOf(
                BatchSpecimenItem(batchId = "BATCH_SCENARIO_1", specimenId = "SPEC_1", status = BatchSpecimenStatus.VALIDATED),
                BatchSpecimenItem(batchId = "BATCH_SCENARIO_1", specimenId = "SPEC_2", status = BatchSpecimenStatus.VALIDATED),
                BatchSpecimenItem(batchId = "BATCH_SCENARIO_1", specimenId = "SPEC_3", status = BatchSpecimenStatus.VALIDATED)
            )
        )
        repo.insertBatchRun(batch)
        repo.recoverInterruptedBatches()

        val recovered = repo.getBatchRun("BATCH_SCENARIO_1")
        assertNotNull(recovered)
        assertEquals(BatchStatus.PARTIALLY_COMPLETED, recovered!!.status)
        assertEquals(3, recovered.items.size)
        assertTrue(recovered.items.all { it.status == BatchSpecimenStatus.CANCELLED })
        assertTrue(recovered.items.all { it.errorDetails == "Interrupted before execution commenced" })
    }

    @Test
    fun testAF04_Scenario2_InterruptionAfterOneItemCompletes() {
        val repo = InMemoryBatchRepository()
        val batch = BatchRun(
            batchId = "BATCH_SCENARIO_2",
            status = BatchStatus.PROCESSING,
            items = listOf(
                BatchSpecimenItem(batchId = "BATCH_SCENARIO_2", specimenId = "DONE_1", status = BatchSpecimenStatus.COMPLETED, probability = 0.9412f, predictedClass = 1, predictedLabel = "MSI", inferenceLatencyMs = 345L),
                BatchSpecimenItem(batchId = "BATCH_SCENARIO_2", specimenId = "PEND_2", status = BatchSpecimenStatus.VALIDATED),
                BatchSpecimenItem(batchId = "BATCH_SCENARIO_2", specimenId = "PEND_3", status = BatchSpecimenStatus.VALIDATED)
            )
        )
        repo.insertBatchRun(batch)
        repo.recoverInterruptedBatches()

        val recovered = repo.getBatchRun("BATCH_SCENARIO_2")
        assertNotNull(recovered)
        assertEquals(BatchStatus.PARTIALLY_COMPLETED, recovered!!.status)
        val done1 = recovered.items.first { it.specimenId == "DONE_1" }
        assertEquals(BatchSpecimenStatus.COMPLETED, done1.status)
        assertEquals(0.9412f, done1.probability ?: 0f, 0.0001f)
        assertEquals(1, done1.predictedClass)
        assertEquals("MSI", done1.predictedLabel)
        assertEquals(345L, done1.inferenceLatencyMs)

        val pend2 = recovered.items.first { it.specimenId == "PEND_2" }
        val pend3 = recovered.items.first { it.specimenId == "PEND_3" }
        assertEquals(BatchSpecimenStatus.CANCELLED, pend2.status)
        assertEquals(BatchSpecimenStatus.CANCELLED, pend3.status)
        assertEquals("Interrupted before execution commenced", pend2.errorDetails)
        assertEquals("Interrupted before execution commenced", pend3.errorDetails)
    }

    @Test
    fun testAF04_Scenario3_InterruptionWhileItemExecuting() {
        val repo = InMemoryBatchRepository()
        val batch = BatchRun(
            batchId = "BATCH_SCENARIO_3",
            status = BatchStatus.PROCESSING,
            items = listOf(
                BatchSpecimenItem(batchId = "BATCH_SCENARIO_3", specimenId = "COMPLETED_1", status = BatchSpecimenStatus.COMPLETED, probability = 0.012f, predictedClass = 0),
                BatchSpecimenItem(batchId = "BATCH_SCENARIO_3", specimenId = "INFERRING_2", status = BatchSpecimenStatus.INFERRING),
                BatchSpecimenItem(batchId = "BATCH_SCENARIO_3", specimenId = "QUEUED_3", status = BatchSpecimenStatus.VALIDATED)
            )
        )
        repo.insertBatchRun(batch)
        repo.recoverInterruptedBatches()

        val recovered = repo.getBatchRun("BATCH_SCENARIO_3")
        assertNotNull(recovered)
        val inferringItem = recovered!!.items.first { it.specimenId == "INFERRING_2" }
        assertEquals(BatchSpecimenStatus.CANCELLED, inferringItem.status)
        assertEquals("Interrupted by system/app termination", inferringItem.errorDetails)

        val queuedItem = recovered.items.first { it.specimenId == "QUEUED_3" }
        assertEquals(BatchSpecimenStatus.CANCELLED, queuedItem.status)
        assertEquals("Interrupted before execution commenced", queuedItem.errorDetails)
    }

    @Test
    fun testAF04_Scenario4_RemainingItemsStillInValidatedEliminated() {
        val repo = InMemoryBatchRepository()
        val batch = BatchRun(
            batchId = "BATCH_SCENARIO_4",
            status = BatchStatus.PROCESSING,
            items = listOf(
                BatchSpecimenItem(batchId = "BATCH_SCENARIO_4", specimenId = "V1", status = BatchSpecimenStatus.VALIDATED),
                BatchSpecimenItem(batchId = "BATCH_SCENARIO_4", specimenId = "V2", status = BatchSpecimenStatus.VALIDATED),
                BatchSpecimenItem(batchId = "BATCH_SCENARIO_4", specimenId = "V3", status = BatchSpecimenStatus.VALIDATED)
            )
        )
        repo.insertBatchRun(batch)
        repo.recoverInterruptedBatches()

        val recovered = repo.getBatchRun("BATCH_SCENARIO_4")
        assertNotNull(recovered)
        val lingeringValidated = recovered!!.items.count { it.status == BatchSpecimenStatus.VALIDATED }
        assertEquals("Zero items must remain in VALIDATED status under interrupted batch", 0, lingeringValidated)
    }

    @Test
    fun testAF04_Scenario5_DatabaseStateAfterProcessRestart() {
        val repo1 = InMemoryBatchRepository()
        val batch = BatchRun(
            batchId = "BATCH_SCENARIO_5",
            status = BatchStatus.PROCESSING,
            items = listOf(
                BatchSpecimenItem(batchId = "BATCH_SCENARIO_5", specimenId = "I1", status = BatchSpecimenStatus.INFERRING),
                BatchSpecimenItem(batchId = "BATCH_SCENARIO_5", specimenId = "V1", status = BatchSpecimenStatus.VALIDATED)
            )
        )
        repo1.insertBatchRun(batch)
        repo1.recoverInterruptedBatches()

        // Simulate process reload by querying another repository view of the data
        val repo2 = InMemoryBatchRepository()
        repo1.getAllBatchRuns().forEach { repo2.insertBatchRun(it) }

        val restoredBatch = repo2.getBatchRun("BATCH_SCENARIO_5")
        assertNotNull(restoredBatch)
        assertEquals(BatchStatus.PARTIALLY_COMPLETED, restoredBatch!!.status)
        assertTrue(restoredBatch.items.none { it.status == BatchSpecimenStatus.INFERRING || it.status == BatchSpecimenStatus.VALIDATED })
    }

    @Test
    fun testAF04_Scenario6_RecoveryInvokedMultipleTimes_Idempotent() {
        val repo = InMemoryBatchRepository()
        val batch = BatchRun(
            batchId = "BATCH_SCENARIO_6",
            status = BatchStatus.PROCESSING,
            items = listOf(
                BatchSpecimenItem(batchId = "BATCH_SCENARIO_6", specimenId = "C1", status = BatchSpecimenStatus.COMPLETED, probability = 0.88f),
                BatchSpecimenItem(batchId = "BATCH_SCENARIO_6", specimenId = "I1", status = BatchSpecimenStatus.INFERRING),
                BatchSpecimenItem(batchId = "BATCH_SCENARIO_6", specimenId = "V1", status = BatchSpecimenStatus.VALIDATED)
            )
        )
        repo.insertBatchRun(batch)

        // Invoke recovery 5 times consecutively
        repeat(5) {
            repo.recoverInterruptedBatches()
        }

        val recovered = repo.getBatchRun("BATCH_SCENARIO_6")
        assertNotNull(recovered)
        assertEquals(BatchStatus.PARTIALLY_COMPLETED, recovered!!.status)
        assertEquals(BatchSpecimenStatus.COMPLETED, recovered.items[0].status)
        assertEquals(BatchSpecimenStatus.CANCELLED, recovered.items[1].status)
        assertEquals(BatchSpecimenStatus.CANCELLED, recovered.items[2].status)
    }

    @Test
    fun testAF04_Scenario7_PreservationOfCompletedResultsAndAuditRecords() {
        val repo = InMemoryBatchRepository()
        val timestamp = System.currentTimeMillis()
        val batch = BatchRun(
            batchId = "BATCH_SCENARIO_7",
            status = BatchStatus.PROCESSING,
            items = listOf(
                BatchSpecimenItem(
                    batchId = "BATCH_SCENARIO_7",
                    specimenId = "AUDIT_CASE_01",
                    status = BatchSpecimenStatus.COMPLETED,
                    probability = 0.965f,
                    predictedClass = 1,
                    predictedLabel = "MSI",
                    bundleSha256 = "aaa111",
                    visionSha256 = "bbb222",
                    textSha256 = "ccc333",
                    patchCount = 420,
                    inferenceLatencyMs = 312L,
                    provenance = InputProvenance.AUTHENTIC_IMPORTED,
                    completedTimestamp = timestamp
                ),
                BatchSpecimenItem(batchId = "BATCH_SCENARIO_7", specimenId = "QUEUED_02", status = BatchSpecimenStatus.VALIDATED)
            )
        )
        repo.insertBatchRun(batch)
        repo.recoverInterruptedBatches()

        val recovered = repo.getBatchRun("BATCH_SCENARIO_7")
        val auditedItem = recovered!!.items.first { it.specimenId == "AUDIT_CASE_01" }
        assertEquals(BatchSpecimenStatus.COMPLETED, auditedItem.status)
        assertEquals(0.965f, auditedItem.probability ?: 0f, 0.0001f)
        assertEquals(1, auditedItem.predictedClass)
        assertEquals("MSI", auditedItem.predictedLabel)
        assertEquals("aaa111", auditedItem.bundleSha256)
        assertEquals("bbb222", auditedItem.visionSha256)
        assertEquals("ccc333", auditedItem.textSha256)
        assertEquals(420, auditedItem.patchCount)
        assertEquals(312L, auditedItem.inferenceLatencyMs)
        assertEquals(InputProvenance.AUTHENTIC_IMPORTED, auditedItem.provenance)
        assertEquals(timestamp, auditedItem.completedTimestamp)
    }

    @Test
    fun testAF04_Scenario8_NoDuplicateResultsOrCompletionEvents() {
        val repo = InMemoryBatchRepository()
        val batch = BatchRun(
            batchId = "BATCH_SCENARIO_8",
            status = BatchStatus.PROCESSING,
            items = listOf(
                BatchSpecimenItem(batchId = "BATCH_SCENARIO_8", specimenId = "S1", status = BatchSpecimenStatus.COMPLETED, probability = 0.91f, predictedClass = 1),
                BatchSpecimenItem(batchId = "BATCH_SCENARIO_8", specimenId = "S2", status = BatchSpecimenStatus.INFERRING),
                BatchSpecimenItem(batchId = "BATCH_SCENARIO_8", specimenId = "S3", status = BatchSpecimenStatus.VALIDATED)
            )
        )
        repo.insertBatchRun(batch)
        repo.recoverInterruptedBatches()

        val recovered = repo.getBatchRun("BATCH_SCENARIO_8")!!
        val stats = recovered.statistics
        assertEquals("Total selected must equal initial items count", 3, stats.totalSelected)
        assertEquals("Total successful must equal 1", 1, stats.totalSuccessful)
        assertEquals("Total cancelled must equal 2", 2, stats.totalCancelled)
        assertEquals("Total processed must equal successful + failed", 1, stats.totalProcessed)
        assertEquals("Zero duplicate items", 3, recovered.items.size)
    }

    @Test
    fun testAF04_Scenario9_BatchAndItemStateConsistency() {
        val repo = InMemoryBatchRepository()
        val batch = BatchRun(
            batchId = "BATCH_SCENARIO_9",
            status = BatchStatus.PROCESSING,
            items = listOf(
                BatchSpecimenItem(batchId = "BATCH_SCENARIO_9", specimenId = "C1", status = BatchSpecimenStatus.COMPLETED, probability = 0.2f, predictedClass = 0),
                BatchSpecimenItem(batchId = "BATCH_SCENARIO_9", specimenId = "F1", status = BatchSpecimenStatus.FAILED, errorDetails = "Malformed"),
                BatchSpecimenItem(batchId = "BATCH_SCENARIO_9", specimenId = "I1", status = BatchSpecimenStatus.INFERRING),
                BatchSpecimenItem(batchId = "BATCH_SCENARIO_9", specimenId = "V1", status = BatchSpecimenStatus.VALIDATED)
            )
        )
        repo.insertBatchRun(batch)
        repo.recoverInterruptedBatches()

        val recovered = repo.getBatchRun("BATCH_SCENARIO_9")!!
        assertEquals(BatchStatus.PARTIALLY_COMPLETED, recovered.status)
        val validTerminalStatuses = setOf(
            BatchSpecimenStatus.COMPLETED,
            BatchSpecimenStatus.FAILED,
            BatchSpecimenStatus.CANCELLED,
            BatchSpecimenStatus.REJECTED
        )
        assertTrue(
            "Every item in a PARTIALLY_COMPLETED batch must be in a terminal state",
            recovered.items.all { it.status in validTerminalStatuses }
        )
    }

    @Test
    fun testAF04_Scenario10_RecoveryWhenItemMalformedOrPermanentlyFails() {
        val repo = InMemoryBatchRepository()
        val batch = BatchRun(
            batchId = "BATCH_SCENARIO_10",
            status = BatchStatus.PROCESSING,
            items = listOf(
                BatchSpecimenItem(batchId = "BATCH_SCENARIO_10", specimenId = "FAIL_1", status = BatchSpecimenStatus.FAILED, errorDetails = "CRC checksum failure in clinical text tensor"),
                BatchSpecimenItem(batchId = "BATCH_SCENARIO_10", specimenId = "INF_2", status = BatchSpecimenStatus.INFERRING),
                BatchSpecimenItem(batchId = "BATCH_SCENARIO_10", specimenId = "VAL_3", status = BatchSpecimenStatus.VALIDATED)
            )
        )
        repo.insertBatchRun(batch)
        repo.recoverInterruptedBatches()

        val recovered = repo.getBatchRun("BATCH_SCENARIO_10")!!
        val failedItem = recovered.items.first { it.specimenId == "FAIL_1" }
        assertEquals(BatchSpecimenStatus.FAILED, failedItem.status)
        assertEquals("CRC checksum failure in clinical text tensor", failedItem.errorDetails)

        val infItem = recovered.items.first { it.specimenId == "INF_2" }
        assertEquals(BatchSpecimenStatus.CANCELLED, infItem.status)
        assertEquals("Interrupted by system/app termination", infItem.errorDetails)

        val valItem = recovered.items.first { it.specimenId == "VAL_3" }
        assertEquals(BatchSpecimenStatus.CANCELLED, valItem.status)
        assertEquals("Interrupted before execution commenced", valItem.errorDetails)
    }

    // =========================================================================
    // PHASE 6.2 REGRESSION TESTS: BATCH ANALYSIS RESTORATION (FIX B)
    // =========================================================================

    @Test
    fun testResolveBatchForDisplay_explicitBatchIdRestoresRequestedBatch() {
        val batch1 = BatchRun(
            batchId = "BATCH-001",
            title = "First Batch",
            createdAt = 1000L,
            updatedAt = 1000L,
            status = BatchStatus.COMPLETED
        )
        val batch2 = BatchRun(
            batchId = "BATCH-002",
            title = "Second Batch",
            createdAt = 2000L,
            updatedAt = 2000L,
            status = BatchStatus.COMPLETED
        )
        mockRepository.insertBatchRun(batch1)
        mockRepository.insertBatchRun(batch2)

        val resolved = BatchCoordinator.resolveBatchForDisplay(mockRepository, "BATCH-001")
        assertNotNull("Must find requested batch", resolved)
        assertEquals("BATCH-001", resolved!!.batchId)
        assertEquals("First Batch", resolved.title)
    }

    @Test
    fun testResolveBatchForDisplay_missingBatchIdRestoresMostRecentBatch() {
        val olderBatch = BatchRun(
            batchId = "BATCH-OLDER",
            title = "Older Batch",
            createdAt = 1000L,
            updatedAt = 1000L,
            status = BatchStatus.COMPLETED
        )
        val newerBatch = BatchRun(
            batchId = "BATCH-NEWER",
            title = "Newer Batch",
            createdAt = 5000L,
            updatedAt = 5000L,
            status = BatchStatus.COMPLETED
        )
        val middleBatch = BatchRun(
            batchId = "BATCH-MIDDLE",
            title = "Middle Batch",
            createdAt = 3000L,
            updatedAt = 3000L,
            status = BatchStatus.COMPLETED
        )
        mockRepository.insertBatchRun(olderBatch)
        mockRepository.insertBatchRun(newerBatch)
        mockRepository.insertBatchRun(middleBatch)

        // Null requested ID -> restores most recent (highest createdAt)
        val resolvedNull = BatchCoordinator.resolveBatchForDisplay(mockRepository, null)
        assertNotNull("Must restore most recent batch when ID is null", resolvedNull)
        assertEquals("BATCH-NEWER", resolvedNull!!.batchId)

        // Blank requested ID -> also restores most recent
        val resolvedBlank = BatchCoordinator.resolveBatchForDisplay(mockRepository, "   ")
        assertNotNull("Must restore most recent batch when ID is blank", resolvedBlank)
        assertEquals("BATCH-NEWER", resolvedBlank!!.batchId)
    }

    @Test
    fun testResolveBatchForDisplay_emptyRepositoryShowsEmptyState() {
        // Repository is empty
        val resolvedNull = BatchCoordinator.resolveBatchForDisplay(mockRepository, null)
        assertNull("Empty repository must return null for missing ID", resolvedNull)

        val resolvedId = BatchCoordinator.resolveBatchForDisplay(mockRepository, "NON_EXISTENT")
        assertNull("Empty repository must return null for requested ID", resolvedId)
    }

    @Test
    fun testResolveBatchForDisplay_unknownExplicitBatchIdDoesNotFallBack() {
        val existingBatch = BatchRun(
            batchId = "BATCH-VALID",
            title = "Valid Batch",
            createdAt = 2000L,
            updatedAt = 2000L,
            status = BatchStatus.COMPLETED
        )
        mockRepository.insertBatchRun(existingBatch)

        // Request an unknown ID - must return null and NEVER fall back to existingBatch
        val resolved = BatchCoordinator.resolveBatchForDisplay(mockRepository, "BATCH-DOES-NOT-EXIST")
        assertNull("Must never silently fall back to another batch when explicit batch ID is unknown", resolved)
    }

    @Test
    fun testResolveBatchForDisplay_latestBatchOrderingIsDeterministic() {
        // Insert multiple batches in non-chronological order
        val b1 = BatchRun(batchId = "B1", title = "T1", createdAt = 100L, updatedAt = 100L, status = BatchStatus.COMPLETED)
        val b4 = BatchRun(batchId = "B4", title = "T4", createdAt = 400L, updatedAt = 400L, status = BatchStatus.COMPLETED)
        val b2 = BatchRun(batchId = "B2", title = "T2", createdAt = 200L, updatedAt = 200L, status = BatchStatus.COMPLETED)
        val b3 = BatchRun(batchId = "B3", title = "T3", createdAt = 300L, updatedAt = 300L, status = BatchStatus.COMPLETED)

        mockRepository.insertBatchRun(b1)
        mockRepository.insertBatchRun(b4)
        mockRepository.insertBatchRun(b2)
        mockRepository.insertBatchRun(b3)

        val allBatches = mockRepository.getAllBatchRuns()
        assertEquals(4, allBatches.size)
        // Verify deterministic descending order
        assertEquals("B4", allBatches[0].batchId)
        assertEquals("B3", allBatches[1].batchId)
        assertEquals("B2", allBatches[2].batchId)
        assertEquals("B1", allBatches[3].batchId)

        // Verify resolveBatchForDisplay picks the top one
        val resolved = BatchCoordinator.resolveBatchForDisplay(mockRepository, null)
        assertNotNull(resolved)
        assertEquals("B4", resolved!!.batchId)
    }

    @Test
    fun testResolveBatchForDisplay_repositoryRetrievalErrorsHandledAppropriately() {
        val throwingRepo = object : BatchRepository {
            override fun insertBatchRun(batch: BatchRun): Long = 0L
            override fun updateBatchRun(batch: BatchRun): Boolean = false
            override fun getBatchRun(batchId: String): BatchRun? {
                throw RuntimeException("Simulated SQLite Disk I/O Error")
            }
            override fun getAllBatchRuns(): List<BatchRun> {
                throw RuntimeException("Simulated SQLite Database Corruption")
            }
            override fun deleteBatchRun(batchId: String): Boolean = false
            override fun updateBatchItem(item: BatchSpecimenItem): Boolean = false
            override fun recoverInterruptedBatches() {}
        }

        // When explicit ID throws, must catch and return null safely
        val resolvedExplicit = BatchCoordinator.resolveBatchForDisplay(throwingRepo, "ANY_ID")
        assertNull("Repository errors on explicit lookup must safely return null", resolvedExplicit)

        // When getAllBatchRuns throws, must catch and return null safely
        val resolvedLatest = BatchCoordinator.resolveBatchForDisplay(throwingRepo, null)
        assertNull("Repository errors on latest lookup must safely return null", resolvedLatest)
    }
}


/**
 * In-memory test repository implementation.
 */
class InMemoryBatchRepository : BatchRepository {
    private val batches = mutableMapOf<String, BatchRun>()

    override fun insertBatchRun(batch: BatchRun): Long {
        batches[batch.batchId] = batch
        return 1L
    }

    override fun updateBatchRun(batch: BatchRun): Boolean {
        batches[batch.batchId] = batch
        return true
    }

    override fun getBatchRun(batchId: String): BatchRun? = batches[batchId]

    override fun getAllBatchRuns(): List<BatchRun> = batches.values.sortedByDescending { it.createdAt }

    override fun deleteBatchRun(batchId: String): Boolean {
        return batches.remove(batchId) != null
    }

    override fun updateBatchItem(item: BatchSpecimenItem): Boolean {
        val batch = batches[item.batchId] ?: return false
        val newItems = batch.items.map { if (it.specimenId == item.specimenId) item else it }
        batches[item.batchId] = batch.copy(items = newItems)
        return true
    }

    override fun archiveBatchItem(batchId: String, specimenId: String): Boolean {
        val batch = batches[batchId] ?: return false
        val newItems = batch.items.map { if (it.specimenId == specimenId) it.copy(isArchived = true) else it }
        batches[batchId] = batch.copy(items = newItems)
        return true
    }

    override fun unarchiveBatchItem(batchId: String, specimenId: String): Boolean {
        val batch = batches[batchId] ?: return false
        val newItems = batch.items.map { if (it.specimenId == specimenId) it.copy(isArchived = false) else it }
        batches[batchId] = batch.copy(items = newItems)
        return true
    }

    override fun deleteBatchItem(batchId: String, specimenId: String): Boolean {
        val batch = batches[batchId] ?: return false
        val newItems = batch.items.filter { it.specimenId != specimenId }
        batches[batchId] = batch.copy(items = newItems)
        return true
    }

    override fun recoverInterruptedBatches() {
        for ((id, batch) in batches) {
            if (batch.status == BatchStatus.PROCESSING) {
                val newItems = batch.items.map {
                    when (it.status) {
                        BatchSpecimenStatus.INFERRING -> it.copy(
                            status = BatchSpecimenStatus.CANCELLED,
                            errorDetails = "Interrupted by system/app termination"
                        )
                        BatchSpecimenStatus.VALIDATED -> it.copy(
                            status = BatchSpecimenStatus.CANCELLED,
                            errorDetails = "Interrupted before execution commenced"
                        )
                        else -> it
                    }
                }
                batches[id] = batch.copy(status = BatchStatus.PARTIALLY_COMPLETED, items = newItems)
            }
        }
    }
}
