package org.crc.multimodal

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.crc.multimodal.batch.BatchCoordinator
import org.crc.multimodal.batch.BatchReportGenerator
import org.crc.multimodal.batch.BatchRun
import org.crc.multimodal.batch.BatchSpecimenItem
import org.crc.multimodal.batch.BatchSpecimenStatus
import org.crc.multimodal.batch.BatchStatus
import org.crc.multimodal.batch.SqliteBatchRepository
import org.crc.multimodal.inference.CrcLogger
import org.crc.multimodal.inference.CrcMultimodalClassifier
import org.crc.multimodal.inference.InputProvenance
import org.json.JSONObject
import org.junit.AfterClass
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.security.MessageDigest

/**
 * Phase 6 Physical Android Device (Samsung Galaxy S24) Batch Ingestion & Reporting Verification Test.
 *
 * Runs natively on the Android OS runtime via connected AndroidTest.
 * Evaluates:
 * 1. Multi-bundle ingestion and fail-closed validation (corrupt bundles, intra-batch duplicate detection).
 * 2. Sequential, deterministic batch inference on authentic bundles (TCGA-A6-3807, TCGA-A6-6138, TCGA-CM-6675, TCGA-D5-6540).
 * 3. Batch statistics with clean denominators and accurate MSI/MSS distribution.
 * 4. Automated research report export (canonical JSON + native PDF) with SHA-256 cryptographic manifest verification.
 * 5. SQLite persistence and interrupted batch crash recovery.
 */
@RunWith(AndroidJUnit4::class)
class CrcPhysicalBatchInferenceTest {

    companion object {
        private const val EXPECTED_MODEL_SHA256 = "0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73"
        private var classifier: CrcMultimodalClassifier? = null
        private lateinit var repository: SqliteBatchRepository
        private lateinit var coordinator: BatchCoordinator
        private lateinit var testDir: File

        @BeforeClass
        @JvmStatic
        fun setUpDevice() {
            val context = InstrumentationRegistry.getInstrumentation().targetContext
            val assets = context.assets

            // 1. Verify frozen FP32 model integrity on physical device
            val assetPath = "models/fusion_model_fp32.onnx"
            val modelBytes = assets.open(assetPath).use { it.readBytes() }

            val digest = MessageDigest.getInstance("SHA-256")
            val sha256 = digest.digest(modelBytes).joinToString("") { "%02x".format(it) }

            CrcLogger.i("[PHYSICAL_BATCH_SETUP] Device Model: ${android.os.Build.MODEL} | Android: ${android.os.Build.VERSION.RELEASE}")
            CrcLogger.i("[PHYSICAL_BATCH_MODEL] Asset: $assetPath | Bytes: ${modelBytes.size} | SHA-256: $sha256")

            assertEquals(
                "Model SHA-256 on physical device must match frozen FP32 baseline",
                EXPECTED_MODEL_SHA256,
                sha256.lowercase()
            )

            classifier = CrcMultimodalClassifier.fromByteArray(modelBytes, numThreads = 2)
            repository = SqliteBatchRepository.getInstance(context)
            coordinator = BatchCoordinator(repository)

            testDir = File(context.cacheDir, "phase6_test_bundles").apply {
                if (exists()) deleteRecursively()
                mkdirs()
            }
        }

        @AfterClass
        @JvmStatic
        fun tearDownDevice() {
            classifier?.close()
            classifier = null
            testDir.deleteRecursively()
            CrcLogger.i("[PHYSICAL_BATCH_TEARDOWN] Classifier closed and test directories cleaned.")
        }
    }

    /**
     * Copies an authentic bundle from app assets into the local test directory.
     */
    private fun copyAssetBundleToTestDir(bundleName: String): File {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val destFile = File(testDir, bundleName)
        context.assets.open("bundles/$bundleName").use { input ->
            destFile.outputStream().use { output ->
                input.copyTo(output)
            }
        }
        return destFile
    }

    @Test
    fun testPhysicalMultiBundleIngestionAndValidation() {
        // Prepare 4 authentic bundles
        val f1 = copyAssetBundleToTestDir("TCGA-A6-3807.crcbundle")
        val f2 = copyAssetBundleToTestDir("TCGA-A6-6138.crcbundle")
        val f3 = copyAssetBundleToTestDir("TCGA-CM-6675.crcbundle")
        val f4 = copyAssetBundleToTestDir("TCGA-D5-6540.crcbundle")

        // Create an intentional corrupt bundle (flipped magic bytes)
        val corruptFile = File(testDir, "CORRUPT_SAMPLE.crcbundle").apply {
            val validBytes = f1.readBytes()
            validBytes[0] = 0x58.toByte() // Corrupt magic
            writeBytes(validBytes)
        }

        // Create an intentional duplicate file of TCGA-A6-3807 with different filename
        val duplicateFile = File(testDir, "TCGA-A6-3807-duplicate.crcbundle").apply {
            writeBytes(f1.readBytes())
        }

        val allFiles = listOf(f1, f2, f3, f4, corruptFile, duplicateFile)
        val batch = coordinator.ingestFiles(allFiles, title = "Physical Test Ingestion Batch")

        CrcLogger.i(
            "[PHYSICAL_INGESTION_RESULTS] Batch: ${batch.batchId} | Total: ${allFiles.size} | " +
            "Accepted: ${batch.statistics.totalAccepted} | Rejected: ${batch.statistics.totalRejected}"
        )

        assertEquals("Total selected files count must be 6", 6, batch.items.size)
        assertEquals("Accepted files count must be 4", 4, batch.statistics.totalAccepted)
        assertEquals("Rejected files count must be 2", 2, batch.statistics.totalRejected)

        // Verify valid items
        val validItems = batch.items.filter { it.status == BatchSpecimenStatus.VALIDATED }
        assertEquals(4, validItems.size)
        val specimenIds = validItems.map { it.specimenId }.toSet()
        assertTrue(specimenIds.contains("TCGA-A6-3807"))
        assertTrue(specimenIds.contains("TCGA-A6-6138"))
        assertTrue(specimenIds.contains("TCGA-CM-6675"))
        assertTrue(specimenIds.contains("TCGA-D5-6540"))

        // Verify corrupt bundle rejected with informative error
        val corruptItem = batch.items.first { it.fileName == "CORRUPT_SAMPLE.crcbundle" }
        assertEquals(BatchSpecimenStatus.REJECTED, corruptItem.status)
        assertNotNull(corruptItem.validationError)
        assertTrue(
            "Corrupt item error must describe header/magic failure",
            corruptItem.validationError!!.contains("Magic") || corruptItem.validationError!!.contains("magic") || corruptItem.validationError!!.contains("CRCB")
        )

        // Verify duplicate bundle rejected with exact policy message
        val dupItem = batch.items.first { it.fileName == "TCGA-A6-3807-duplicate.crcbundle" }
        assertEquals(BatchSpecimenStatus.REJECTED, dupItem.status)
        assertEquals(
            "Duplicate specimen identifier within batch",
            dupItem.validationError
        )
    }

    @Test
    fun testPhysicalControlledBatchExecutionAndStatistics() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val f1 = copyAssetBundleToTestDir("TCGA-A6-3807.crcbundle")
        val f2 = copyAssetBundleToTestDir("TCGA-A6-6138.crcbundle")
        val f3 = copyAssetBundleToTestDir("TCGA-CM-6675.crcbundle")
        val f4 = copyAssetBundleToTestDir("TCGA-D5-6540.crcbundle")

        val batch = coordinator.ingestFiles(listOf(f1, f2, f3, f4), title = "Physical 4-Specimen Execution")
        assertEquals(4, batch.statistics.totalAccepted)

        val completedBatch = coordinator.executeBatchInference(context, batch, classifier!!)

        CrcLogger.i(
            "[PHYSICAL_BATCH_EXECUTION] Batch: ${completedBatch.batchId} | Status: ${completedBatch.status} | " +
            "Total: ${completedBatch.statistics.totalProcessed} | MSI: ${completedBatch.statistics.msiCount} | " +
            "MSS: ${completedBatch.statistics.mssCount} | MeanLatency: ${completedBatch.statistics.meanLatencyMs}ms"
        )

        assertEquals(BatchStatus.COMPLETED, completedBatch.status)
        assertEquals(4, completedBatch.statistics.totalProcessed)
        assertEquals(0, completedBatch.statistics.totalFailed)
        assertEquals(4, completedBatch.items.size)

        // Verify per-specimen predictions and numerical invariants
        for (item in completedBatch.items) {
            assertEquals(BatchSpecimenStatus.COMPLETED, item.status)
            assertNotNull(item.probability)
            assertNotNull(item.logit)
            assertNotNull(item.predictedClass)
            assertTrue("Inference duration must be > 0", (item.inferenceLatencyMs ?: 0L) > 0L)
        }
        assertEquals("Model hash must match frozen FP32 baseline", EXPECTED_MODEL_SHA256, completedBatch.modelSha256)

        // Verify specific known specimen predictions
        val d5 = completedBatch.items.first { it.specimenId == "TCGA-D5-6540" }
        val cm = completedBatch.items.first { it.specimenId == "TCGA-CM-6675" }

        assertEquals("TCGA-D5-6540 must be predicted MSI (Class 1)", 1, d5.predictedClass)
        assertTrue("TCGA-D5-6540 probability must be >= 0.83", (d5.probability ?: 0f) >= 0.83f)

        assertEquals("TCGA-CM-6675 must be predicted MSS (Class 0)", 0, cm.predictedClass)
        assertTrue("TCGA-CM-6675 probability must be < 0.83", (cm.probability ?: 1f) < 0.83f)

        // Verify summary statistics
        val stats = completedBatch.statistics
        assertEquals(4, stats.totalProcessed)
        assertEquals(4, stats.msiCount + stats.mssCount)
        assertTrue(stats.msiCount >= 1)
        assertTrue(stats.mssCount >= 1)
        assertEquals(1.0, (stats.msiProportion + stats.mssProportion).toDouble(), 0.001)
        assertTrue(stats.meanLatencyMs > 0.0)
        assertTrue(stats.minLatencyMs > 0L)
        assertTrue(stats.maxLatencyMs >= stats.minLatencyMs)
        assertTrue(stats.patchCountMin > 0)
        assertTrue(stats.patchCountMax >= stats.patchCountMin)
    }

    @Test
    fun testPhysicalResearchReportExportAndManifestVerification() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val f1 = copyAssetBundleToTestDir("TCGA-A6-3807.crcbundle")
        val f2 = copyAssetBundleToTestDir("TCGA-A6-6138.crcbundle")
        val f3 = copyAssetBundleToTestDir("TCGA-CM-6675.crcbundle")
        val f4 = copyAssetBundleToTestDir("TCGA-D5-6540.crcbundle")

        val batch = coordinator.ingestFiles(listOf(f1, f2, f3, f4), title = "Physical Report Test Batch")
        val completedBatch = coordinator.executeBatchInference(context, batch, classifier!!)

        // Generate reports bundle
        val exportDir = File("/sdcard/Download/phase6_reports").apply { mkdirs() }
        val reportBundle = BatchReportGenerator.exportReport(context, completedBatch, exportDir)
        val jsonFile = reportBundle.jsonFile
        val pdfFile = reportBundle.pdfFile
        val manifestFile = reportBundle.manifestFile

        CrcLogger.i(
            "[PHYSICAL_REPORT_EXPORT] Batch: ${completedBatch.batchId} | JSON: ${jsonFile.name} (${jsonFile.length()}B) | " +
            "PDF: ${pdfFile?.name} (${pdfFile?.length() ?: 0}B) | Manifest: ${manifestFile.name}"
        )

        assertTrue("JSON report must exist and have non-zero size", jsonFile.exists() && jsonFile.length() > 0)
        assertNotNull("PDF report file should be created on physical Android device", pdfFile)
        assertTrue("PDF report must exist and have non-zero size", pdfFile!!.exists() && pdfFile.length() > 0)
        assertTrue("Manifest file must exist", manifestFile.exists())

        // Verify cryptographic integrity via SHA-256 manifest
        val jsonVerified = BatchReportGenerator.verifyReportIntegrity(jsonFile, reportBundle.jsonSha256)
        val pdfVerified = BatchReportGenerator.verifyReportIntegrity(pdfFile, reportBundle.pdfSha256 ?: "")

        assertTrue("JSON report SHA-256 must match manifest record", jsonVerified)
        assertTrue("PDF report SHA-256 must match manifest record", pdfVerified)

        // Verify tamper detection on modified report
        val tamperedFile = File(jsonFile.parentFile, "tampered_report.json").apply {
            writeText(jsonFile.readText() + " ")
        }
        val tamperedVerified = BatchReportGenerator.verifyReportIntegrity(tamperedFile, reportBundle.jsonSha256)
        assertFalse("Tampered report must fail verification", tamperedVerified)
        tamperedFile.delete()

        // Inspect JSON report contents for disclaimers, schema invariants, and zero data leakage
        val jsonContent = jsonFile.readText()
        val jsonObj = JSONObject(jsonContent)

        assertEquals("CRC Multimodal Edge AI Batch Research Report", jsonObj.getString("report_title"))
        assertTrue("Disclaimer must state research use only", jsonObj.getString("research_use_disclaimer").contains("RESEARCH USE ONLY"))
        assertEquals(completedBatch.batchId, jsonObj.getString("batch_id"))
        assertEquals(EXPECTED_MODEL_SHA256, jsonObj.getJSONObject("model_metadata").getString("frozen_onnx_sha256"))

        // Ensure no raw float tensors or unrequested patient identifiers were exported
        assertFalse("Report must not leak vision data", jsonContent.contains("visionData"))
        assertFalse("Report must not leak text data", jsonContent.contains("textData"))
        assertFalse("Report must not contain patient names", jsonContent.contains("patientName") || jsonContent.contains("patient_name"))
    }

    @Test
    fun testPhysicalPersistenceAndCrashRecovery() {
        val f1 = copyAssetBundleToTestDir("TCGA-A6-3807.crcbundle")
        val f2 = copyAssetBundleToTestDir("TCGA-A6-6138.crcbundle")
        val f3 = copyAssetBundleToTestDir("TCGA-CM-6675.crcbundle")
        val f4 = copyAssetBundleToTestDir("TCGA-D5-6540.crcbundle")

        val batch = coordinator.ingestFiles(listOf(f1, f2, f3, f4), title = "Physical Persistence AF-04 Test")
        val batchId = batch.batchId

        // Check SQLite retrieval
        val retrieved = repository.getBatchRun(batchId)
        assertNotNull("Batch run must be retrieved from SQLite database", retrieved)
        assertEquals(batchId, retrieved!!.batchId)
        assertEquals(4, retrieved.items.size)

        // Simulate crash interruption on a mixed batch:
        // Item 0: COMPLETED
        // Item 1: FAILED
        // Item 2: INFERRING (interrupted mid-execution)
        // Item 3: VALIDATED (queued, waiting in line)
        val interruptedBatch = retrieved.copy(
            status = BatchStatus.PROCESSING,
            items = retrieved.items.mapIndexed { idx, itm ->
                when (idx) {
                    0 -> itm.copy(status = BatchSpecimenStatus.COMPLETED, probability = 0.91f, predictedClass = 1)
                    1 -> itm.copy(status = BatchSpecimenStatus.FAILED, errorDetails = "Simulated tensor corrupt")
                    2 -> itm.copy(status = BatchSpecimenStatus.INFERRING)
                    else -> itm.copy(status = BatchSpecimenStatus.VALIDATED)
                }
            }
        )
        repository.updateBatchRun(interruptedBatch)

        // Execute recovery
        repository.recoverInterruptedBatches()

        // Verify recovery transitions
        val recovered = repository.getBatchRun(batchId)
        assertNotNull(recovered)
        assertEquals("Interrupted batch must transition to PARTIALLY_COMPLETED", BatchStatus.PARTIALLY_COMPLETED, recovered!!.status)

        val recoveredItems = recovered.items
        assertEquals(4, recoveredItems.size)

        // Item 0: COMPLETED preserved
        assertEquals(BatchSpecimenStatus.COMPLETED, recoveredItems[0].status)
        assertEquals(0.91f, recoveredItems[0].probability ?: 0f, 0.001f)

        // Item 1: FAILED preserved
        assertEquals(BatchSpecimenStatus.FAILED, recoveredItems[1].status)
        assertEquals("Simulated tensor corrupt", recoveredItems[1].errorDetails)

        // Item 2: INFERRING transitioned to CANCELLED with termination detail
        assertEquals(BatchSpecimenStatus.CANCELLED, recoveredItems[2].status)
        assertEquals("Interrupted by system/app termination", recoveredItems[2].errorDetails)

        // Item 3: Queued VALIDATED transitioned to CANCELLED with commenced detail (AF-04 fix)
        assertEquals(BatchSpecimenStatus.CANCELLED, recoveredItems[3].status)
        assertEquals("Interrupted before execution commenced", recoveredItems[3].errorDetails)

        // Zero items remain in VALIDATED status
        assertFalse("No item should remain VALIDATED after recovery", recoveredItems.any { it.status == BatchSpecimenStatus.VALIDATED })

        // Execute recovery a second time to verify idempotency on SQLite
        repository.recoverInterruptedBatches()
        val recovered2 = repository.getBatchRun(batchId)
        assertNotNull(recovered2)
        assertEquals(BatchStatus.PARTIALLY_COMPLETED, recovered2!!.status)
        assertEquals(BatchSpecimenStatus.CANCELLED, recovered2.items[3].status)
        assertEquals("Interrupted before execution commenced", recovered2.items[3].errorDetails)
    }
}
