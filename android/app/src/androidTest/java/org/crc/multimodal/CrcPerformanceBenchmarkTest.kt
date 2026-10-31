package org.crc.multimodal

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.crc.multimodal.inference.CrcLogger
import org.crc.multimodal.inference.CrcMultimodalClassifier
import org.crc.multimodal.inference.InferenceProtocol
import org.crc.multimodal.inference.InputValidator
import org.crc.multimodal.inference.ReferenceCaseData
import org.crc.multimodal.inference.ReferenceTensorLoader
import org.crc.multimodal.inference.ThresholdLogic
import org.junit.AfterClass
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.security.MessageDigest
import java.util.Locale
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Phase 4 Comprehensive On-Device Performance Profiling and Resource Characterization Harness.
 *
 * Runs natively on physical Samsung Galaxy S24 (SM-S921E, Snapdragon 8 Gen 3).
 * Benchmarks 7 authentic patch counts across 1, 2, and 4 threads.
 * Measures distinct latency boundaries:
 * - Stage B: Session initialization
 * - Stage C: Tensor parsing
 * - Stage D: Buffer allocation and tensor creation
 * - Stage E: Pure session.run() execution
 * - Stage F: Output parsing and threshold decoding
 * - Stage G: Total inference pipeline
 * Performs 10 warm-ups and 50 timed iterations per configuration.
 */
@RunWith(AndroidJUnit4::class)
class CrcPerformanceBenchmarkTest {

    data class LatencyStats(
        val count: Int,
        val minMs: Double,
        val medianP50Ms: Double,
        val p90Ms: Double,
        val p95Ms: Double,
        val p99Ms: Double,
        val meanMs: Double,
        val stdDevMs: Double
    ) {
        companion object {
            fun fromSamples(samplesNs: List<Long>): LatencyStats {
                if (samplesNs.isEmpty()) {
                    return LatencyStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                }
                val sorted = samplesNs.sorted().map { it / 1_000_000.0 }
                val count = sorted.size
                val min = sorted.first()
                val median = percentile(sorted, 50.0)
                val p90 = percentile(sorted, 90.0)
                val p95 = percentile(sorted, 95.0)
                val p99 = percentile(sorted, 99.0)
                val mean = sorted.average()
                val variance = sorted.map { (it - mean) * (it - mean) }.average()
                val stdDev = sqrt(variance)

                return LatencyStats(count, min, median, p90, p95, p99, mean, stdDev)
            }

            private fun percentile(sorted: List<Double>, p: Double): Double {
                if (sorted.isEmpty()) return 0.0
                val index = ((p / 100.0) * (sorted.size - 1)).toInt()
                return sorted[index.coerceIn(0, sorted.size - 1)]
            }
        }
    }

    data class ConfigBenchmarkResult(
        val caseId: String,
        val numPatches: Int,
        val threadCount: Int,
        val sessionInitMs: Double,
        val tensorParseMs: Double,
        val stageDAllocation: LatencyStats,
        val stageEPureRun: LatencyStats,
        val stageFOutputDecode: LatencyStats,
        val stageGPipeline: LatencyStats,
        val onDeviceLogit: Float,
        val onDeviceProb: Float,
        val refLogit: Float,
        val refProb: Float,
        val diffLogit: Float,
        val diffProb: Float,
        val parityPass: Boolean
    )

    companion object {
        private const val BENCHMARK_TAG = "CRC_FP32_BENCHMARK"
        private const val EXPECTED_MODEL_SHA256 = "0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73"
        private const val TOLERANCE = 1.0e-4f
        private const val WARMUP_RUNS = 10
        private const val TIMED_RUNS = 50

        private lateinit var modelBytes: ByteArray
        private val allBenchmarkResults = mutableListOf<ConfigBenchmarkResult>()

        @BeforeClass
        @JvmStatic
        fun setUpBenchmarkEnvironment() {
            val context = InstrumentationRegistry.getInstrumentation().targetContext
            val assetPath = "models/fusion_model_fp32.onnx"
            modelBytes = context.assets.open(assetPath).use { it.readBytes() }

            val digest = MessageDigest.getInstance("SHA-256")
            val sha256 = digest.digest(modelBytes).joinToString("") { "%02x".format(it) }

            assertEquals(
                "Model SHA-256 mismatch before Phase 4 benchmarking",
                EXPECTED_MODEL_SHA256,
                sha256.lowercase()
            )
            CrcLogger.i("[BENCHMARK_ENV] Model verified: ${modelBytes.size} bytes | SHA-256: $sha256", BENCHMARK_TAG)
        }

        @AfterClass
        @JvmStatic
        fun tearDownAndExportResults() {
            val context = InstrumentationRegistry.getInstrumentation().targetContext
            val jsonBuilder = StringBuilder()
            jsonBuilder.append("[\n")
            for (i in allBenchmarkResults.indices) {
                val r = allBenchmarkResults[i]
                jsonBuilder.append("  {\n")
                jsonBuilder.append("    \"caseId\": \"${r.caseId}\",\n")
                jsonBuilder.append("    \"numPatches\": ${r.numPatches},\n")
                jsonBuilder.append("    \"threadCount\": ${r.threadCount},\n")
                jsonBuilder.append("    \"sessionInitMs\": ${String.format(Locale.US, "%.3f", r.sessionInitMs)},\n")
                jsonBuilder.append("    \"tensorParseMs\": ${String.format(Locale.US, "%.3f", r.tensorParseMs)},\n")
                jsonBuilder.append("    \"stageD_allocation\": {\"medianMs\": ${String.format(Locale.US, "%.3f", r.stageDAllocation.medianP50Ms)}, \"meanMs\": ${String.format(Locale.US, "%.3f", r.stageDAllocation.meanMs)}},\n")
                jsonBuilder.append("    \"stageE_pureRun\": {\"minMs\": ${String.format(Locale.US, "%.3f", r.stageEPureRun.minMs)}, \"medianMs\": ${String.format(Locale.US, "%.3f", r.stageEPureRun.medianP50Ms)}, \"p90Ms\": ${String.format(Locale.US, "%.3f", r.stageEPureRun.p90Ms)}, \"p95Ms\": ${String.format(Locale.US, "%.3f", r.stageEPureRun.p95Ms)}, \"p99Ms\": ${String.format(Locale.US, "%.3f", r.stageEPureRun.p99Ms)}, \"meanMs\": ${String.format(Locale.US, "%.3f", r.stageEPureRun.meanMs)}, \"stdDevMs\": ${String.format(Locale.US, "%.3f", r.stageEPureRun.stdDevMs)}},\n")
                jsonBuilder.append("    \"stageF_outputDecode\": {\"medianMs\": ${String.format(Locale.US, "%.3f", r.stageFOutputDecode.medianP50Ms)}, \"meanMs\": ${String.format(Locale.US, "%.3f", r.stageFOutputDecode.meanMs)}},\n")
                jsonBuilder.append("    \"stageG_pipeline\": {\"minMs\": ${String.format(Locale.US, "%.3f", r.stageGPipeline.minMs)}, \"medianMs\": ${String.format(Locale.US, "%.3f", r.stageGPipeline.medianP50Ms)}, \"p90Ms\": ${String.format(Locale.US, "%.3f", r.stageGPipeline.p90Ms)}, \"p95Ms\": ${String.format(Locale.US, "%.3f", r.stageGPipeline.p95Ms)}, \"meanMs\": ${String.format(Locale.US, "%.3f", r.stageGPipeline.meanMs)}},\n")
                jsonBuilder.append("    \"onDeviceLogit\": ${String.format(Locale.US, "%.5f", r.onDeviceLogit)},\n")
                jsonBuilder.append("    \"onDeviceProb\": ${String.format(Locale.US, "%.5f", r.onDeviceProb)},\n")
                jsonBuilder.append("    \"refLogit\": ${String.format(Locale.US, "%.5f", r.refLogit)},\n")
                jsonBuilder.append("    \"refProb\": ${String.format(Locale.US, "%.5f", r.refProb)},\n")
                jsonBuilder.append("    \"diffLogit\": ${String.format(Locale.US, "%.2e", r.diffLogit)},\n")
                jsonBuilder.append("    \"diffProb\": ${String.format(Locale.US, "%.2e", r.diffProb)},\n")
                jsonBuilder.append("    \"parityPass\": ${r.parityPass}\n")
                jsonBuilder.append("  }${if (i < allBenchmarkResults.size - 1) "," else ""}\n")
            }
            jsonBuilder.append("]\n")

            val fullJson = jsonBuilder.toString()

            // Stream each line into logcat with identifiable tag
            fullJson.lines().forEach { line ->
                CrcLogger.i(line, "CRC_BENCHMARK_JSON")
            }

            // Write to app-specific external files dir
            try {
                val outFile = File(context.getExternalFilesDir(null), "phase4_benchmark_results.json")
                outFile.writeText(fullJson)
                CrcLogger.i("[BENCHMARK_EXPORT] Wrote to ${outFile.absolutePath}", BENCHMARK_TAG)
            } catch (e: Exception) {
                CrcLogger.w("[BENCHMARK_EXPORT] Failed writing to externalFilesDir: ${e.message}", BENCHMARK_TAG)
            }

            // Write to shared Download directory
            try {
                val downloadFile = File("/sdcard/Download/phase4_benchmark_results.json")
                downloadFile.writeText(fullJson)
                CrcLogger.i("[BENCHMARK_EXPORT] Wrote to ${downloadFile.absolutePath}", BENCHMARK_TAG)
            } catch (e: Exception) {
                CrcLogger.w("[BENCHMARK_EXPORT] Could not write to /sdcard/Download: ${e.message}", BENCHMARK_TAG)
            }
        }
    }

    private fun benchmarkConfiguration(
        caseKey: String,
        threads: Int
    ): ConfigBenchmarkResult {
        val context = InstrumentationRegistry.getInstrumentation().targetContext

        // Stage C: Tensor Loading & Parsing
        val tParseStart = System.nanoTime()
        val caseData: ReferenceCaseData = ReferenceTensorLoader.loadFromAssets(context.assets, caseKey)
        val tensorParseMs = (System.nanoTime() - tParseStart) / 1_000_000.0

        // Stage B: Session Creation with specific intraOp threads
        val env = OrtEnvironment.getEnvironment("BenchmarkEnv_T$threads")
        val sessionOptions = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(threads)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        val tSessionStart = System.nanoTime()
        val session = env.createSession(modelBytes, sessionOptions)
        val sessionInitMs = (System.nanoTime() - tSessionStart) / 1_000_000.0

        val visionShape = longArrayOf(1L, caseData.numPatches.toLong(), 1024L)
        val textShape = longArrayOf(1L, 768L)

        // Warm-up runs (10 iterations) to prime CPU caches and JIT
        for (w in 0 until WARMUP_RUNS) {
            val vBuf = ByteBuffer.allocateDirect(caseData.visionData.size * Float.SIZE_BYTES).order(ByteOrder.nativeOrder()).asFloatBuffer().put(caseData.visionData)
            vBuf.flip()
            val tBuf = ByteBuffer.allocateDirect(caseData.textData.size * Float.SIZE_BYTES).order(ByteOrder.nativeOrder()).asFloatBuffer().put(caseData.textData)
            tBuf.flip()

            val vTensor = OnnxTensor.createTensor(env, vBuf, visionShape)
            val tTensor = OnnxTensor.createTensor(env, tBuf, textShape)
            val res = session.run(mapOf("vision_seq" to vTensor, "text_summary" to tTensor))
            res.close()
            vTensor.close()
            tTensor.close()
        }

        val stageDSamples = mutableListOf<Long>()
        val stageESamples = mutableListOf<Long>()
        val stageFSamples = mutableListOf<Long>()
        val stageGSamples = mutableListOf<Long>()

        var lastLogit = 0.0f
        var lastProb = 0.0f

        // 50 Timed Iterations
        for (iter in 0 until TIMED_RUNS) {
            val tStartPipeline = System.nanoTime()

            // Stage D: Buffer allocation and OnnxTensor creation
            val tStartD = System.nanoTime()
            val vBuf: FloatBuffer = ByteBuffer.allocateDirect(caseData.visionData.size * Float.SIZE_BYTES)
                .order(ByteOrder.nativeOrder()).asFloatBuffer().put(caseData.visionData)
            vBuf.flip()

            val tBuf: FloatBuffer = ByteBuffer.allocateDirect(caseData.textData.size * Float.SIZE_BYTES)
                .order(ByteOrder.nativeOrder()).asFloatBuffer().put(caseData.textData)
            tBuf.flip()

            val vTensor = OnnxTensor.createTensor(env, vBuf, visionShape)
            val tTensor = OnnxTensor.createTensor(env, tBuf, textShape)
            val inputMap = mapOf("vision_seq" to vTensor, "text_summary" to tTensor)
            val tEndD = System.nanoTime()

            // Stage E: Pure session.run() execution
            val tStartE = System.nanoTime()
            val runResult = session.run(inputMap)
            val tEndE = System.nanoTime()

            // Stage F: Output parsing and threshold decoding
            val tStartF = System.nanoTime()
            val logitTensor = runResult.get("logit").get() as OnnxTensor
            val probTensor = runResult.get("probability").get() as OnnxTensor
            val attTensor = runResult.get("attention_weights").get() as OnnxTensor

            val logitVal = logitTensor.floatBuffer.get(0)
            val probVal = probTensor.floatBuffer.get(0)
            val attArr = FloatArray(caseData.numPatches)
            attTensor.floatBuffer.get(attArr)

            val predictedClass = ThresholdLogic.classify(probVal)
            val predictedLabel = ThresholdLogic.labelFor(predictedClass)
            assertTrue("Predicted label must not be empty", predictedLabel.isNotEmpty())
            val tEndF = System.nanoTime()
            val tEndPipeline = System.nanoTime()

            lastLogit = logitVal
            lastProb = probVal

            stageDSamples.add(tEndD - tStartD)
            stageESamples.add(tEndE - tStartE)
            stageFSamples.add(tEndF - tStartF)
            stageGSamples.add(tEndPipeline - tStartPipeline)

            // Explicit cleanup
            runResult.close()
            vTensor.close()
            tTensor.close()
        }

        session.close()

        val stageDStats = LatencyStats.fromSamples(stageDSamples)
        val stageEStats = LatencyStats.fromSamples(stageESamples)
        val stageFStats = LatencyStats.fromSamples(stageFSamples)
        val stageGStats = LatencyStats.fromSamples(stageGSamples)

        val diffLogit = abs(lastLogit - caseData.refLogit)
        val diffProb = abs(lastProb - caseData.refProb)
        val parityPass = (diffProb <= TOLERANCE)

        val result = ConfigBenchmarkResult(
            caseId = caseKey,
            numPatches = caseData.numPatches,
            threadCount = threads,
            sessionInitMs = sessionInitMs,
            tensorParseMs = tensorParseMs,
            stageDAllocation = stageDStats,
            stageEPureRun = stageEStats,
            stageFOutputDecode = stageFStats,
            stageGPipeline = stageGStats,
            onDeviceLogit = lastLogit,
            onDeviceProb = lastProb,
            refLogit = caseData.refLogit,
            refProb = caseData.refProb,
            diffLogit = diffLogit,
            diffProb = diffProb,
            parityPass = parityPass
        )

        allBenchmarkResults.add(result)

        CrcLogger.i(
            "[CONFIG_RESULT] Case: $caseKey (N=${caseData.numPatches}) | T=$threads | " +
            "Init: ${String.format(Locale.US, "%.1f", sessionInitMs)}ms | " +
            "StageD(Alloc): ${String.format(Locale.US, "%.2f", stageDStats.medianP50Ms)}ms | " +
            "StageE(PureRun) [P50: ${String.format(Locale.US, "%.2f", stageEStats.medianP50Ms)}ms, P90: ${String.format(Locale.US, "%.2f", stageEStats.p90Ms)}ms, P95: ${String.format(Locale.US, "%.2f", stageEStats.p95Ms)}ms, P99: ${String.format(Locale.US, "%.2f", stageEStats.p99Ms)}ms, Mean: ${String.format(Locale.US, "%.2f", stageEStats.meanMs)}ms] | " +
            "StageG(Pipeline) P50: ${String.format(Locale.US, "%.2f", stageGStats.medianP50Ms)}ms | " +
            "Parity: ${if (parityPass) "PASS" else "FAIL"}",
            BENCHMARK_TAG
        )

        assertTrue("Parity failed for $caseKey at T=$threads: diff=$diffProb", parityPass)
        return result
    }

    // =========================================================================
    // 1. Thread Scaling for Reference Cases (TCGA-D5-6540 N=345 and TCGA-CM-6675 N=541)
    // =========================================================================
    @Test
    fun benchmarkMsiReferenceTcgaD56540_Threads1() {
        benchmarkConfiguration("TCGA-D5-6540", 1)
    }

    @Test
    fun benchmarkMsiReferenceTcgaD56540_Threads2() {
        benchmarkConfiguration("TCGA-D5-6540", 2)
    }

    @Test
    fun benchmarkMsiReferenceTcgaD56540_Threads4() {
        benchmarkConfiguration("TCGA-D5-6540", 4)
    }

    @Test
    fun benchmarkMssReferenceTcgaCm6675_Threads1() {
        benchmarkConfiguration("TCGA-CM-6675", 1)
    }

    @Test
    fun benchmarkMssReferenceTcgaCm6675_Threads2() {
        benchmarkConfiguration("TCGA-CM-6675", 2)
    }

    @Test
    fun benchmarkMssReferenceTcgaCm6675_Threads4() {
        benchmarkConfiguration("TCGA-CM-6675", 4)
    }

    // =========================================================================
    // 2. Patch Count Scaling across Valid Authentic Holdout Dataset Cases
    // =========================================================================
    @Test
    fun benchmarkPatchScale_N10_TcgaD56539() {
        benchmarkConfiguration("TCGA-D5-6539", 2)
    }

    @Test
    fun benchmarkPatchScale_N52_TcgaAf2693() {
        benchmarkConfiguration("TCGA-AF-2693", 2)
    }

    @Test
    fun benchmarkPatchScale_N99_TcgaAgA00c() {
        benchmarkConfiguration("TCGA-AG-A00C", 2)
    }

    @Test
    fun benchmarkPatchScale_N502_TcgaAa3845() {
        benchmarkConfiguration("TCGA-AA-3845", 2)
    }

    @Test
    fun benchmarkPatchScale_N1000_TcgaCm6674() {
        benchmarkConfiguration("TCGA-CM-6674", 2)
    }

    // =========================================================================
    // 3. Attention Weights Materialization Overhead Characterization (Phase 4E)
    // =========================================================================
    @Test
    fun benchmarkAttentionMaterializationCost() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val caseData = ReferenceTensorLoader.loadFromAssets(context.assets, "TCGA-CM-6674") // N = 1000 patches

        val env = OrtEnvironment.getEnvironment("AttentionBenchEnv")
        val sessionOptions = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        val session = env.createSession(modelBytes, sessionOptions)

        val visionShape = longArrayOf(1L, 1000L, 1024L)
        val textShape = longArrayOf(1L, 768L)

        val vBuf = ByteBuffer.allocateDirect(1000 * 1024 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer().put(caseData.visionData)
        vBuf.flip()
        val tBuf = ByteBuffer.allocateDirect(768 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer().put(caseData.textData)
        tBuf.flip()

        val vTensor = OnnxTensor.createTensor(env, vBuf, visionShape)
        val tTensor = OnnxTensor.createTensor(env, tBuf, textShape)
        val inputMap = mapOf("vision_seq" to vTensor, "text_summary" to tTensor)

        // Warm-up
        for (w in 0 until 5) {
            val r = session.run(inputMap)
            r.close()
        }

        // Test with full attention extraction
        val withAttSamples = mutableListOf<Long>()
        for (i in 0 until 50) {
            val tStart = System.nanoTime()
            val r = session.run(inputMap)
            val logit = (r.get("logit").get() as OnnxTensor).floatBuffer.get(0)
            val prob = (r.get("probability").get() as OnnxTensor).floatBuffer.get(0)
            val attArr = FloatArray(1000)
            (r.get("attention_weights").get() as OnnxTensor).floatBuffer.get(attArr)
            val tEnd = System.nanoTime()
            withAttSamples.add(tEnd - tStart)
            assertTrue("Logit must be finite", logit.isFinite() && prob.isFinite())
            r.close()
        }

        // Test without attention extraction (only logit + probability)
        val withoutAttSamples = mutableListOf<Long>()
        for (i in 0 until 50) {
            val tStart = System.nanoTime()
            val r = session.run(inputMap)
            val logit = (r.get("logit").get() as OnnxTensor).floatBuffer.get(0)
            val prob = (r.get("probability").get() as OnnxTensor).floatBuffer.get(0)
            val tEnd = System.nanoTime()
            withoutAttSamples.add(tEnd - tStart)
            assertTrue("Logit must be finite", logit.isFinite() && prob.isFinite())
            r.close()
        }

        vTensor.close()
        tTensor.close()
        session.close()

        val withStats = LatencyStats.fromSamples(withAttSamples)
        val withoutStats = LatencyStats.fromSamples(withoutAttSamples)
        val diffMedian = withStats.medianP50Ms - withoutStats.medianP50Ms

        CrcLogger.i(
            "[ATTENTION_AUDIT] N=1000 | WithAttention P50: ${String.format(Locale.US, "%.3f", withStats.medianP50Ms)}ms | " +
            "WithoutAttention P50: ${String.format(Locale.US, "%.3f", withoutStats.medianP50Ms)}ms | " +
            "Extraction Delta: ${String.format(Locale.US, "%.3f", diffMedian)}ms",
            BENCHMARK_TAG
        )
    }
}
