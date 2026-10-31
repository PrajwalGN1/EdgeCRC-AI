package org.crc.multimodal

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.crc.multimodal.batch.BatchCoordinator
import org.crc.multimodal.batch.BatchRun
import org.crc.multimodal.batch.BatchSpecimenItem
import org.crc.multimodal.batch.BatchSpecimenStatus
import org.crc.multimodal.batch.BatchStatus
import org.crc.multimodal.batch.SqliteBatchRepository
import org.crc.multimodal.data.SqliteWorkstationRepository
import org.crc.multimodal.evaluation.CohortEvaluationSummary
import org.crc.multimodal.evaluation.MetricValue
import org.crc.multimodal.inference.CrcLogger
import org.crc.multimodal.inference.CrcMultimodalClassifier
import org.crc.multimodal.inference.InferenceProtocol
import org.json.JSONArray
import org.json.JSONObject
import org.junit.AfterClass
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.security.MessageDigest

/**
 * Authoritative On-Device Physical Android Verification Test for TCGA_HOLDOUT_66.
 *
 * Runs natively on Vivo Y21 (API 31, V2125).
 * Executes:
 * 1. Verification of frozen FP32 ONNX model SHA-256.
 * 2. Ingestion of all 66 authentic .crcbundle files from /sdcard/Download/tcga_holdout_66_crcbundles/.
 * 3. Batch inference under Protocol 1 capped (with Protocol 2 uncapped fallback for >1000 patches).
 * 4. Synchronization to shared Cases registry in crc_workstation.db.
 * 5. Cohort Evaluation execution against TCGA_HOLDOUT_66 reference labels.
 * 6. Detailed per-specimen and summary JSON serialization.
 */
@RunWith(AndroidJUnit4::class)
class CrcPhysicalTcgaHoldout66ExecutionTest {

    companion object {
        private const val EXPECTED_MODEL_SHA256 = "0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73"
        private var classifier: CrcMultimodalClassifier? = null
        private lateinit var batchRepository: SqliteBatchRepository
        private lateinit var workstationRepository: SqliteWorkstationRepository
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

            CrcLogger.i("[TCGA66_SETUP] Device Model: ${android.os.Build.MODEL} | Android: ${android.os.Build.VERSION.RELEASE}")
            CrcLogger.i("[TCGA66_MODEL] Asset: $assetPath | Bytes: ${modelBytes.size} | SHA-256: $sha256")

            assertEquals(
                "Model SHA-256 on physical device must match frozen FP32 baseline",
                EXPECTED_MODEL_SHA256,
                sha256.lowercase()
            )

            classifier = CrcMultimodalClassifier.fromByteArray(modelBytes, numThreads = 2)
            batchRepository = SqliteBatchRepository.getInstance(context)
            workstationRepository = SqliteWorkstationRepository.getInstance(context)
            coordinator = BatchCoordinator(batchRepository, workstationRepository)
        }

        @AfterClass
        @JvmStatic
        fun tearDownDevice() {
            classifier?.close()
            classifier = null
            CrcLogger.i("[TCGA66_TEARDOWN] Classifier closed.")
        }
    }

    @Test
    fun testExecuteTcgaHoldout66BatchAndEvaluateCohort() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val extDir = File(context.getExternalFilesDir(null), "tcga_holdout_66_crcbundles")
        val bundleDir = if (extDir.exists() && (extDir.listFiles()?.size ?: 0) >= 66) {
            extDir
        } else {
            File("/sdcard/Download/tcga_holdout_66_crcbundles")
        }
        assertTrue("Bundle directory must exist", bundleDir.exists() && bundleDir.isDirectory)

        val bundleFiles = bundleDir.listFiles { _, name -> name.endsWith(".crcbundle") }?.toList() ?: emptyList()
        assertEquals("Must locate exactly 66 transferred bundles on device", 66, bundleFiles.size)

        // Sort deterministically by filename
        val sortedFiles = bundleFiles.sortedBy { it.name }

        // 1. Ingest files
        CrcLogger.i("[TCGA66_STAGE] Ingesting 66 bundles into BatchCoordinator...")
        val batch = coordinator.ingestFiles(
            files = sortedFiles,
            title = "TCGA_HOLDOUT_66 Physical Batch Evaluation"
        )
        assertNotNull("BatchRun must not be null", batch)
        assertEquals("All 66 files must be ingested", 66, batch.items.size)
        assertEquals("All 66 files must be validated", 66, batch.statistics.totalAccepted)
        assertEquals("Zero files rejected at ingestion", 0, batch.statistics.totalRejected)

        // 2. Execute Batch Inference
        CrcLogger.i("[TCGA66_STAGE] Executing batch inference across 66 specimens...")
        val completedBatch = coordinator.executeBatchInference(
            context = context,
            batchRun = batch,
            classifier = classifier!!,
            baseProtocol = InferenceProtocol.PROTOCOL_1_CAPPED,
            allowProtocol2Fallback = true
        )

        assertEquals("Batch execution must reach COMPLETED status", BatchStatus.COMPLETED, completedBatch.status)
        assertEquals("All 66 items must be processed", 66, completedBatch.statistics.totalProcessed)
        assertEquals("All 66 items must succeed", 66, completedBatch.statistics.totalSuccessful)
        assertEquals("Zero inference failures", 0, completedBatch.statistics.totalFailed)

        // 3. Synchronize Batch to Cases
        CrcLogger.i("[TCGA66_STAGE] Synchronizing batch items to shared Cases registry...")
        BatchCoordinator.syncBatchToCases(context, completedBatch, workstationRepository)

        // 4. Execute Cohort Evaluation
        CrcLogger.i("[TCGA66_STAGE] Running Cohort Evaluation for TCGA_HOLDOUT_66...")
        val cohortSummary: CohortEvaluationSummary = workstationRepository.evaluateCohort("TCGA_HOLDOUT_66", batchRepository)

        assertNotNull("Cohort evaluation summary must not be null", cohortSummary)
        val cm = cohortSummary.confusionMatrix
        CrcLogger.i("[TCGA66_EVAL] Matched: ${cohortSummary.matchedCount} / ${cohortSummary.totalReferenceLabels} | TP: ${cm.tp}, TN: ${cm.tn}, FP: ${cm.fp}, FN: ${cm.fn}")

        // 5. Serialize complete execution results to JSON
        val rootJson = JSONObject()
        rootJson.put("device_model", android.os.Build.MODEL)
        rootJson.put("device_serial", android.os.Build.SERIAL)
        rootJson.put("android_version", android.os.Build.VERSION.RELEASE)
        rootJson.put("api_level", android.os.Build.VERSION.SDK_INT)
        rootJson.put("model_sha256", EXPECTED_MODEL_SHA256)
        rootJson.put("batch_id", completedBatch.batchId)
        rootJson.put("total_specimens", completedBatch.items.size)
        rootJson.put("total_duration_ms", completedBatch.totalDurationMs)
        rootJson.put("mean_latency_ms", completedBatch.statistics.meanLatencyMs)
        rootJson.put("median_latency_ms", completedBatch.statistics.medianLatencyMs)
        rootJson.put("msi_count", completedBatch.statistics.msiCount)
        rootJson.put("mss_count", completedBatch.statistics.mssCount)

        // Cohort evaluation metrics
        val evalJson = JSONObject()
        evalJson.put("cohort_id", cohortSummary.cohortId)
        evalJson.put("total_reference_labels", cohortSummary.totalReferenceLabels)
        evalJson.put("total_predictions_available", cohortSummary.totalPredictionsAvailable)
        evalJson.put("matched_count", cohortSummary.matchedCount)
        evalJson.put("unmatched_reference_count", cohortSummary.unmatchedReferenceCount)
        evalJson.put("unmatched_prediction_count", cohortSummary.unmatchedPredictionCount)
        evalJson.put("true_positives", cm.tp)
        evalJson.put("true_negatives", cm.tn)
        evalJson.put("false_positives", cm.fp)
        evalJson.put("false_negatives", cm.fn)

        fun metricToJson(m: MetricValue): JSONObject {
            val j = JSONObject()
            j.put("is_defined", m.isDefined)
            when (m) {
                is MetricValue.Defined -> j.put("value", m.value)
                is MetricValue.Undefined -> j.put("reason", m.reason)
            }
            return j
        }

        evalJson.put("accuracy", metricToJson(cohortSummary.accuracy))
        evalJson.put("sensitivity", metricToJson(cohortSummary.recallSensitivity))
        evalJson.put("specificity", metricToJson(cohortSummary.specificity))
        evalJson.put("balanced_accuracy", metricToJson(cohortSummary.balancedAccuracy))
        evalJson.put("precision", metricToJson(cohortSummary.precision))
        evalJson.put("f1_score", metricToJson(cohortSummary.f1Score))
        evalJson.put("roc_auc", metricToJson(cohortSummary.rocAuc))
        evalJson.put("pr_auc", metricToJson(cohortSummary.prAuc))

        rootJson.put("cohort_evaluation", evalJson)

        val itemsArray = JSONArray()
        for (item in completedBatch.items) {
            val itemJson = JSONObject()
            itemJson.put("specimen_id", item.specimenId)
            itemJson.put("file_name", item.fileName)
            itemJson.put("file_size_bytes", item.fileSizeBytes)
            itemJson.put("patch_count", item.patchCount)
            itemJson.put("status", item.status.name)
            itemJson.put("protocol", item.protocol)
            itemJson.put("protocol_reason", item.protocolReason ?: "")
            itemJson.put("probability", item.probability)
            itemJson.put("logit", item.logit)
            itemJson.put("predicted_class", item.predictedClass)
            itemJson.put("predicted_label", item.predictedLabel)
            itemJson.put("threshold", item.threshold)
            itemJson.put("inference_latency_ms", item.inferenceLatencyMs)
            itemJson.put("bundle_sha256", item.bundleSha256)
            itemsArray.put(itemJson)

            CrcLogger.i("[TCGA_HOLDOUT_66_SPECIMEN] ${item.specimenId} | Patches: ${item.patchCount} | Protocol: ${item.protocol} | Prob: ${item.probability} | Class: ${item.predictedClass} (${item.predictedLabel}) | Latency: ${item.inferenceLatencyMs}ms")
        }
        rootJson.put("specimens", itemsArray)

        // Matched cohort specimen details
        val matchedArray = JSONArray()
        for (m in cohortSummary.specimenMatches) {
            val mj = JSONObject()
            mj.put("specimen_id", m.specimenId)
            mj.put("ground_truth_label", m.groundTruthLabel)
            mj.put("ground_truth_class", m.groundTruthClass)
            mj.put("predicted_class", m.predictedClass)
            mj.put("predicted_label", m.predictedLabel)
            mj.put("probability", m.predictedProbability)
            mj.put("outcome", m.outcome.code)
            mj.put("is_correct", m.outcome.isCorrect)
            mj.put("decision_margin", m.decisionMargin)
            matchedArray.put(mj)
        }
        rootJson.put("cohort_specimen_matches", matchedArray)

        // Save output JSON to app external files dir and log output
        val outFile = File(context.getExternalFilesDir(null), "tcga_holdout_66_device_results.json")
        outFile.writeText(rootJson.toString(2))
        CrcLogger.i("[TCGA66_COMPLETE] Results written to ${outFile.absolutePath} (${outFile.length()} bytes)")
        println("###TCGA66_JSON_OUTPUT_START###")
        println(rootJson.toString(2))
        println("###TCGA66_JSON_OUTPUT_END###")
    }
}
