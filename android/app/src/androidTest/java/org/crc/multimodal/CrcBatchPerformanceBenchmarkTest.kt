package org.crc.multimodal

import android.content.Context
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.crc.multimodal.batch.BatchCoordinator
import org.crc.multimodal.batch.BatchReportGenerator
import org.crc.multimodal.batch.BatchRun
import org.crc.multimodal.batch.SqliteBatchRepository
import org.crc.multimodal.inference.CrcLogger
import org.crc.multimodal.inference.CrcMultimodalClassifier
import org.json.JSONArray
import org.json.JSONObject
import org.junit.AfterClass
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Phase 6 Performance Profiling & Scaling Benchmark on Physical Samsung Galaxy S24.
 *
 * Measures:
 * - Import and validation latency
 * - Sequential inference latency per specimen (min, mean, median, max)
 * - End-to-end batch duration
 * - Peak runtime heap memory
 * - Scaling behavior across batch sizes 1, 4, 10, and 25
 */
@RunWith(AndroidJUnit4::class)
class CrcBatchPerformanceBenchmarkTest {

    companion object {
        private var classifier: CrcMultimodalClassifier? = null
        private lateinit var repository: SqliteBatchRepository
        private lateinit var coordinator: BatchCoordinator
        private lateinit var context: Context
        private lateinit var workDir: File
        private val benchmarkResults = JSONObject()

        @BeforeClass
        @JvmStatic
        fun setUp() {
            context = InstrumentationRegistry.getInstrumentation().targetContext
            val modelBytes = context.assets.open("models/fusion_model_fp32.onnx").use { it.readBytes() }
            classifier = CrcMultimodalClassifier.fromByteArray(modelBytes, numThreads = 2)
            repository = SqliteBatchRepository.getInstance(context)
            coordinator = BatchCoordinator(repository)

            workDir = File(context.cacheDir, "benchmark_work_bundles").apply {
                if (exists()) deleteRecursively()
                mkdirs()
            }
        }

        @AfterClass
        @JvmStatic
        fun tearDown() {
            classifier?.close()
            classifier = null
            workDir.deleteRecursively()

            // Save benchmark results to file
            val benchmarkOut = File(context.cacheDir, "batch_benchmark_results.json")
            benchmarkOut.writeText(benchmarkResults.toString(2), Charsets.UTF_8)
            CrcLogger.i("[BENCHMARK_COMPLETE] Exported results to ${benchmarkOut.absolutePath}")
        }
    }

    private fun prepareAuthenticFiles(count: Int): List<File> {
        val bundleNames = context.assets.list("bundles")!!
            .filter { it.endsWith(".crcbundle") }
            .sorted()
            .take(count)

        require(bundleNames.size >= count) {
            "Requested $count bundles but only ${bundleNames.size} authentic bundles available in assets."
        }

        val files = mutableListOf<File>()
        for (name in bundleNames) {
            val destFile = File(workDir, name)
            if (!destFile.exists()) {
                context.assets.open("bundles/$name").use { input ->
                    destFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
            }
            files.add(destFile)
        }
        return files
    }

    private fun runBenchmarkForSize(batchSize: Int): JSONObject {
        val files = prepareAuthenticFiles(batchSize)
        System.gc()
        Thread.sleep(100)

        val memBefore = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
        val ingestStart = System.currentTimeMillis()

        // Phase 6A: Ingestion & Validation
        val batch = coordinator.ingestFiles(files, title = "Benchmark Batch Size $batchSize")
        val ingestDuration = System.currentTimeMillis() - ingestStart

        val validItemsCount = batch.items.count { it.status == org.crc.multimodal.batch.BatchSpecimenStatus.VALIDATED }
        val memAfterIngest = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()

        // Phase 6B: Controlled Sequential Inference
        val inferStart = System.currentTimeMillis()
        val completedBatch = coordinator.executeBatchInference(context, batch, classifier!!)
        val inferDuration = System.currentTimeMillis() - inferStart

        val memPeak = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
        val stats = completedBatch.statistics

        val resultJson = JSONObject().apply {
            put("batch_size", batchSize)
            put("total_selected", files.size)
            put("total_accepted", validItemsCount)
            put("total_processed", stats.totalProcessed)
            put("ingestion_duration_ms", ingestDuration)
            put("inference_duration_ms", inferDuration)
            put("total_duration_ms", ingestDuration + inferDuration)
            put("mean_inference_latency_ms", stats.meanLatencyMs)
            put("median_inference_latency_ms", stats.medianLatencyMs)
            put("min_inference_latency_ms", stats.minLatencyMs)
            put("max_inference_latency_ms", stats.maxLatencyMs)
            put("msi_count", stats.msiCount)
            put("mss_count", stats.mssCount)
            put("heap_used_before_bytes", memBefore)
            put("heap_used_after_ingest_bytes", memAfterIngest)
            put("heap_used_peak_bytes", memPeak)
        }

        CrcLogger.i(
            "[BENCHMARK_BATCH_$batchSize] Ingest: ${ingestDuration}ms | Infer: ${inferDuration}ms | " +
            "MeanLat: ${String.format(java.util.Locale.US, "%.2f", stats.meanLatencyMs)}ms | " +
            "PeakHeap: ${memPeak / (1024 * 1024)}MB"
        )

        return resultJson
    }

    @Test
    fun benchmarkBatchSize1() {
        val result = runBenchmarkForSize(1)
        benchmarkResults.put("batch_1", result)
        assertEquals(1, result.getInt("total_processed"))
    }

    @Test
    fun benchmarkBatchSize5() {
        val result = runBenchmarkForSize(5)
        benchmarkResults.put("batch_5", result)
        assertEquals(5, result.getInt("total_processed"))
    }

    @Test
    fun benchmarkBatchSize10() {
        val result = runBenchmarkForSize(10)
        benchmarkResults.put("batch_10", result)
        assertEquals(10, result.getInt("total_processed"))
    }

    @Test
    fun benchmarkBatchSize25() {
        val result = runBenchmarkForSize(25)
        benchmarkResults.put("batch_25", result)
        assertEquals(25, result.getInt("total_processed"))
    }
}
