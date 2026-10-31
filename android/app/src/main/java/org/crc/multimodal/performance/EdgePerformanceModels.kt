package org.crc.multimodal.performance

import kotlin.math.sqrt

/**
 * Domain data models for on-device edge performance monitoring, latency profiling,
 * memory tracking, and release gate verification.
 */

data class LatencyMetrics(
    val count: Int,
    val minMs: Double,
    val medianMs: Double,
    val p95Ms: Double,
    val maxMs: Double,
    val meanMs: Double,
    val stdDevMs: Double
) {
    companion object {
        fun fromSamples(samplesNs: List<Long>): LatencyMetrics {
            if (samplesNs.isEmpty()) {
                return LatencyMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            }
            val sorted = samplesNs.sorted().map { it / 1_000_000.0 }
            val count = sorted.size
            val min = sorted.first()
            val max = sorted.last()
            val median = percentile(sorted, 50.0)
            val p95 = percentile(sorted, 95.0)
            val mean = sorted.average()
            val variance = sorted.map { (it - mean) * (it - mean) }.average()
            val stdDev = sqrt(variance)

            return LatencyMetrics(count, min, median, p95, max, mean, stdDev)
        }

        fun fromMillisSamples(samplesMs: List<Double>): LatencyMetrics {
            if (samplesMs.isEmpty()) {
                return LatencyMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            }
            val sorted = samplesMs.sorted()
            val count = sorted.size
            val min = sorted.first()
            val max = sorted.last()
            val median = percentile(sorted, 50.0)
            val p95 = percentile(sorted, 95.0)
            val mean = sorted.average()
            val variance = sorted.map { (it - mean) * (it - mean) }.average()
            val stdDev = sqrt(variance)

            return LatencyMetrics(count, min, median, p95, max, mean, stdDev)
        }

        private fun percentile(sorted: List<Double>, p: Double): Double {
            if (sorted.isEmpty()) return 0.0
            if (sorted.size == 1) return sorted[0]
            val index = ((p / 100.0) * (sorted.size - 1)).toInt()
            return sorted[index.coerceIn(0, sorted.size - 1)]
        }
    }
}

data class StageLatencyBreakdown(
    val importValidationMs: Double,
    val allocationMs: Double,
    val warmInferenceMs: Double,
    val outputDecodingMs: Double,
    val dbCommitMs: Double,
    val reportGenerationMs: Double,
    val totalPipelineMs: Double
)

data class MemoryProfile(
    val javaHeapAllocatedMb: Double,
    val nativeHeapAllocatedMb: Double,
    val totalPssMb: Double,
    val peakMemoryMb: Double
)

data class WorkloadSpecimenBenchmark(
    val specimenId: String,
    val patchCount: Int,
    val warmInference: LatencyMetrics,
    val memoryProfile: MemoryProfile,
    val parityPassed: Boolean
)

data class BatchThroughputBenchmark(
    val batchSize: Int,
    val totalDurationMs: Double,
    val throughputSpecimensPerSec: Double,
    val avgLatencyPerSpecimenMs: Double
)

data class ColdStartBenchmark(
    val coldInitDurationMs: Double,
    val sessionCreationDurationMs: Double
)

data class PerformanceGateVerdict(
    val gateName: String,
    val targetThreshold: String,
    val measuredValue: String,
    val isPassed: Boolean,
    val details: String
)

data class EdgePerformanceReport(
    val deviceModel: String = "Samsung Galaxy S24 (SM-S921E)",
    val osVersion: String = "Android 16 (API 36)",
    val onnxRuntimeVersion: String = "1.18.0",
    val executionProvider: String = "CPUExecutionProvider",
    val targetS24Verified: Boolean = true,
    val coldStart: ColdStartBenchmark,
    val reference385Benchmark: WorkloadSpecimenBenchmark,
    val patchScalingBenchmarks: List<WorkloadSpecimenBenchmark>,
    val batchBenchmarks: List<BatchThroughputBenchmark>,
    val overallPeakMemoryMb: Double,
    val gates: List<PerformanceGateVerdict>,
    val overallPassed: Boolean
)
