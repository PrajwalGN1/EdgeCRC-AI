package org.crc.multimodal

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.crc.multimodal.batch.BatchCoordinator
import org.crc.multimodal.batch.BatchReportGenerator
import org.crc.multimodal.batch.BatchRun
import org.crc.multimodal.batch.BatchSpecimenItem
import org.crc.multimodal.batch.BatchSpecimenStatus
import org.crc.multimodal.batch.BatchStatus
import org.crc.multimodal.batch.BatchStatusPalette
import org.crc.multimodal.batch.SqliteBatchRepository
import org.crc.multimodal.inference.CrcLogger
import org.crc.multimodal.inference.CrcMultimodalClassifier
import org.crc.multimodal.inference.InferenceProtocol
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
 * On-Device Physical Android Verification Test for:
 * 1. Batch Specimen Status Color System (MSS=#2ECC9A, MSI=#F56B70, FAILED=#F59E0B, REJECTED=#D6A83D).
 * 2. Uncapped Protocol 2 Dynamic Routing for specimens exceeding 1,000 patches (TCGA-NH-A50T, TCGA-NH-A5IV, TCGA-QG-A5Z2).
 * 3. Preservation of Protocol 1 (<= 1,000 patches) without degradation.
 * 4. Reporting, persistence, and cryptographic verification on connected hardware.
 */
@RunWith(AndroidJUnit4::class)
class CrcPhysicalBatchProtocol2AndStatusColorTest {

    companion object {
        private const val EXPECTED_MODEL_SHA256 = "0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73"
        private var classifier: CrcMultimodalClassifier? = null
        private lateinit var repository: SqliteBatchRepository
        private lateinit var coordinator: BatchCoordinator

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

            CrcLogger.i("[PHYSICAL_TEST_SETUP] Device Model: ${android.os.Build.MODEL} | Android: ${android.os.Build.VERSION.RELEASE}")
            CrcLogger.i("[PHYSICAL_TEST_MODEL] Asset: $assetPath | Bytes: ${modelBytes.size} | SHA-256: $sha256")

            assertEquals(
                "Model SHA-256 on physical device must match frozen FP32 baseline",
                EXPECTED_MODEL_SHA256,
                sha256.lowercase()
            )

            classifier = CrcMultimodalClassifier.fromByteArray(modelBytes, numThreads = 2)
            repository = SqliteBatchRepository.getInstance(context)
            coordinator = BatchCoordinator(repository)
        }

        @AfterClass
        @JvmStatic
        fun tearDownDevice() {
            classifier?.close()
            classifier = null
            CrcLogger.i("[PHYSICAL_TEST_TEARDOWN] Classifier closed.")
        }
    }

    @Test
    fun testPhysical20BundleBatchWithProtocol2FallbackAndStatusColors() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val extDir = File(context.getExternalFilesDir(null), "CRCBundle")
        val bundleDir = if (extDir.exists() && (extDir.listFiles()?.size ?: 0) >= 20) {
            extDir
        } else {
            File("/sdcard/Download/CRCBundle")
        }
        assertTrue("Bundle directory must exist on device", bundleDir.exists() && bundleDir.isDirectory)

        val bundleFiles = bundleDir.listFiles { _, name -> name.endsWith(".crcbundle") }?.toList() ?: emptyList()
        assertTrue("Must locate 20 transferred bundles on device, found ${bundleFiles.size}", bundleFiles.size >= 20)

        // Sort deterministically by filename
        val sortedFiles = bundleFiles.sortedBy { it.name }

        // 1. Ingest files
        val batch = coordinator.ingestFiles(
            files = sortedFiles,
            title = "Hardware Verification 20 CRC Bundles"
        )
        assertNotNull(batch)
        assertEquals("All 20 files must be ingested", sortedFiles.size, batch.items.size)

        // 2. Execute Batch Inference with Protocol 1 as base and Protocol 2 fallback enabled
        val completedBatch = coordinator.executeBatchInference(
            context = context,
            batchRun = batch,
            classifier = classifier!!,
            baseProtocol = InferenceProtocol.PROTOCOL_1_CAPPED,
            allowProtocol2Fallback = true
        )

        assertEquals(BatchStatus.COMPLETED, completedBatch.status)
        assertEquals(sortedFiles.size, completedBatch.items.size)

        // 3. Inspect Over-Limit Specimens (> 1,000 patches)
        val overLimitSpecimenIds = listOf("TCGA-NH-A50T", "TCGA-NH-A5IV", "TCGA-QG-A5Z2")
        for (specimenId in overLimitSpecimenIds) {
            val item = completedBatch.items.find { it.specimenId == specimenId }
            assertNotNull("Specimen $specimenId must be present in batch", item)
            assertEquals("Specimen $specimenId must succeed under Protocol 2", BatchSpecimenStatus.COMPLETED, item!!.status)
            assertEquals("Specimen $specimenId must be evaluated under PROTOCOL_2_UNCAPPED", "PROTOCOL_2_UNCAPPED", item.protocol)
            assertNotNull("Protocol reason must be present for $specimenId", item.protocolReason)
            assertTrue("Protocol reason must state routed to Protocol 2", item.protocolReason!!.contains("routed to Protocol 2 Uncapped"))
            assertTrue("Patch count must exceed 1000", item.patchCount > 1000)

            CrcLogger.i("[PHYSICAL_P2_SUCCESS] Specimen: $specimenId | Patches: ${item.patchCount} | Protocol: ${item.protocol} | Prob: ${item.probability} | Label: ${item.predictedLabel}")
        }

        // 4. Inspect Standard Specimens (<= 1,000 patches)
        val standardItems = completedBatch.items.filter { !overLimitSpecimenIds.contains(it.specimenId) }
        for (item in standardItems) {
            assertEquals("Standard specimen ${item.specimenId} must succeed", BatchSpecimenStatus.COMPLETED, item.status)
            assertEquals("Standard specimen ${item.specimenId} must be PROTOCOL_1_CAPPED", "PROTOCOL_1_CAPPED", item.protocol)
            assertTrue("Standard specimen ${item.specimenId} patch count <= 1000", item.patchCount <= 1000)
        }

        // 5. Verify Status Color Resolution across all items
        for (item in completedBatch.items) {
            val colorHex = BatchStatusPalette.resolveColorHex(item.status, item.predictedClass)
            if (item.status == BatchSpecimenStatus.COMPLETED) {
                if (item.predictedClass == 1) {
                    assertEquals("MSI specimen must resolve to coral red (#F56B70)", "#F56B70", colorHex)
                    assertTrue("MSI probability must be >= 0.83", (item.probability ?: 0f) >= 0.83f)
                } else {
                    assertEquals("MSS specimen must resolve to emerald green (#2ECC9A)", "#2ECC9A", colorHex)
                    assertTrue("MSS probability must be < 0.83", (item.probability ?: 1f) < 0.83f)
                }
            } else if (item.status == BatchSpecimenStatus.REJECTED) {
                assertEquals("REJECTED specimen must resolve to gold (#D6A83D)", "#D6A83D", colorHex)
            } else if (item.status == BatchSpecimenStatus.FAILED) {
                assertEquals("FAILED specimen must resolve to amber orange (#F59E0B)", "#F59E0B", colorHex)
            }
        }

        // 6. Verify Summary Statistics
        val stats = completedBatch.statistics
        assertEquals(sortedFiles.size, stats.totalProcessed)
        assertEquals(sortedFiles.size, stats.totalSuccessful)
        assertEquals(0, stats.totalFailed)
        assertEquals(0, stats.totalRejected)
        assertTrue(stats.msiCount > 0)
        assertTrue(stats.mssCount > 0)
        assertEquals(sortedFiles.size, stats.msiCount + stats.mssCount)

        // 7. Export Research Reports
        val exportDir = File(context.getExternalFilesDir(null), "phase6_reports").apply { mkdirs() }
        val reportBundle = BatchReportGenerator.exportReport(context, completedBatch, exportDir)
        assertNotNull(reportBundle)
        assertTrue(reportBundle.jsonFile.exists() && reportBundle.jsonFile.length() > 0)
        assertTrue(reportBundle.pdfFile != null && reportBundle.pdfFile!!.exists() && reportBundle.pdfFile!!.length() > 0)

        // 8. Verify JSON Content Contains Protocol Information
        val jsonText = reportBundle.jsonFile.readText()
        assertTrue("JSON must record PROTOCOL_2_UNCAPPED", jsonText.contains("\"protocol\": \"PROTOCOL_2_UNCAPPED\""))
        assertTrue("JSON must record PROTOCOL_1_CAPPED", jsonText.contains("\"protocol\": \"PROTOCOL_1_CAPPED\""))
        assertTrue("JSON must record protocol_reason", jsonText.contains("routed to Protocol 2 Uncapped"))

        CrcLogger.i("[PHYSICAL_BATCH_COMPLETE] 20/20 specimens processed successfully on vivo V2125 hardware.")
    }
}
