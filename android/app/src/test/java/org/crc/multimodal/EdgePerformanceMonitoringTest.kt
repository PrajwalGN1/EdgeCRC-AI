package org.crc.multimodal

import org.crc.multimodal.performance.BatchThroughputBenchmark
import org.crc.multimodal.performance.ColdStartBenchmark
import org.crc.multimodal.performance.EdgePerformanceMonitor
import org.crc.multimodal.performance.LatencyMetrics
import org.crc.multimodal.performance.MemoryProfile
import org.crc.multimodal.performance.StageLatencyBreakdown
import org.crc.multimodal.performance.WorkloadSpecimenBenchmark
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Automated test suite verifying the WP7.6 Edge Performance Monitoring framework.
 */
class EdgePerformanceMonitoringTest {

    @Test
    fun testLatencyStatsCalculation() {
        // Samples in milliseconds: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
        val samplesMs = listOf(10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0)
        val stats = LatencyMetrics.fromMillisSamples(samplesMs)

        assertEquals(10, stats.count)
        assertEquals(10.0, stats.minMs, 1e-6)
        assertEquals(100.0, stats.maxMs, 1e-6)
        assertEquals(55.0, stats.meanMs, 1e-6)
        assertEquals(50.0, stats.medianMs, 10.0) // rank-based percentile
        assertTrue(stats.p95Ms >= 90.0)
        assertTrue(stats.stdDevMs > 0.0)
    }

    @Test
    fun testLatencyStatsEmptyAndSingleSample() {
        val emptyStats = LatencyMetrics.fromMillisSamples(emptyList())
        assertEquals(0, emptyStats.count)
        assertEquals(0.0, emptyStats.medianMs, 1e-6)

        val singleStats = LatencyMetrics.fromMillisSamples(listOf(42.5))
        assertEquals(1, singleStats.count)
        assertEquals(42.5, singleStats.minMs, 1e-6)
        assertEquals(42.5, singleStats.maxMs, 1e-6)
        assertEquals(42.5, singleStats.medianMs, 1e-6)
        assertEquals(42.5, singleStats.meanMs, 1e-6)
        assertEquals(0.0, singleStats.stdDevMs, 1e-6)
    }

    @Test
    fun testNanoSamplesConversion() {
        // 20 ms = 20_000_000 ns
        val samplesNs = listOf(20_000_000L, 25_000_000L, 30_000_000L)
        val stats = LatencyMetrics.fromSamples(samplesNs)

        assertEquals(3, stats.count)
        assertEquals(20.0, stats.minMs, 1e-6)
        assertEquals(30.0, stats.maxMs, 1e-6)
        assertEquals(25.0, stats.meanMs, 1e-6)
    }

    @Test
    fun testMemoryProfileExtraction() {
        val mem = EdgePerformanceMonitor.measureMemory()
        assertNotNull(mem)
        assertTrue(mem.javaHeapAllocatedMb >= 0.0)
        assertTrue(mem.totalPssMb >= mem.javaHeapAllocatedMb)
        assertTrue(mem.peakMemoryMb > 0.0)
    }

    @Test
    fun testBatchThroughputComputation() {
        // 10 specimens processed in 200.0 ms -> 50.0 specimens/sec
        val benchmark = EdgePerformanceMonitor.computeBatchThroughput(
            batchSize = 10,
            totalDurationMs = 200.0
        )

        assertEquals(10, benchmark.batchSize)
        assertEquals(200.0, benchmark.totalDurationMs, 1e-6)
        assertEquals(50.0, benchmark.throughputSpecimensPerSec, 1e-6)
        assertEquals(20.0, benchmark.avgLatencyPerSpecimenMs, 1e-6)
    }

    @Test
    fun testPerformanceGateEvaluationPassed() {
        val gates = EdgePerformanceMonitor.evaluatePerformanceTargets(
            warmLatency385P50Ms = 21.0, // Well below 80.0 ms target
            peakMemoryMb = 145.0,       // Well below 300.0 MB target
            parityRegressions = 0,
            crashesCount = 0
        )

        assertEquals(4, gates.size)
        assertTrue("All gates must pass", gates.all { it.isPassed })
    }

    @Test
    fun testPerformanceGateEvaluationFailsOnExcessiveLatency() {
        val gates = EdgePerformanceMonitor.evaluatePerformanceTargets(
            warmLatency385P50Ms = 95.5, // Exceeds 80.0 ms limit
            peakMemoryMb = 145.0,
            parityRegressions = 0,
            crashesCount = 0
        )

        val latencyGate = gates.first { it.gateName.contains("Warm Fusion Latency") }
        assertFalse("Latency gate must fail when > 80.0 ms", latencyGate.isPassed)
    }

    @Test
    fun testPerformanceGateEvaluationFailsOnExcessiveMemory() {
        val gates = EdgePerformanceMonitor.evaluatePerformanceTargets(
            warmLatency385P50Ms = 21.0,
            peakMemoryMb = 345.0, // Exceeds 300.0 MB limit
            parityRegressions = 0,
            crashesCount = 0
        )

        val memoryGate = gates.first { it.gateName.contains("Peak Process Memory") }
        assertFalse("Memory gate must fail when > 300.0 MB", memoryGate.isPassed)
    }

    @Test
    fun testPerformanceGateEvaluationFailsOnParityRegression() {
        val gates = EdgePerformanceMonitor.evaluatePerformanceTargets(
            warmLatency385P50Ms = 21.0,
            peakMemoryMb = 145.0,
            parityRegressions = 1, // Flips detected
            crashesCount = 0
        )

        val parityGate = gates.first { it.gateName.contains("Parity") }
        assertFalse("Parity gate must fail when flips > 0", parityGate.isPassed)
    }

    @Test
    fun testCompileReportOverallStatus() {
        val coldStart = ColdStartBenchmark(
            coldInitDurationMs = 20.5,
            sessionCreationDurationMs = 19.8
        )

        val ref385Benchmark = WorkloadSpecimenBenchmark(
            specimenId = "TCGA-385-REF",
            patchCount = 385,
            warmInference = LatencyMetrics(
                count = 50,
                minMs = 19.5,
                medianMs = 20.8,
                p95Ms = 21.5,
                maxMs = 22.1,
                meanMs = 20.9,
                stdDevMs = 0.4
            ),
            memoryProfile = MemoryProfile(
                javaHeapAllocatedMb = 35.0,
                nativeHeapAllocatedMb = 60.0,
                totalPssMb = 140.0,
                peakMemoryMb = 145.0
            ),
            parityPassed = true
        )

        val report = EdgePerformanceMonitor.compileReport(
            coldStart = coldStart,
            reference385Benchmark = ref385Benchmark,
            patchScalingBenchmarks = listOf(ref385Benchmark),
            batchBenchmarks = listOf(
                BatchThroughputBenchmark(
                    batchSize = 10,
                    totalDurationMs = 210.0,
                    throughputSpecimensPerSec = 47.6,
                    avgLatencyPerSpecimenMs = 21.0
                )
            ),
            peakMemoryMb = 145.0,
            parityRegressions = 0,
            crashesCount = 0
        )

        assertEquals("Samsung Galaxy S24 (SM-S921E)", report.deviceModel)
        assertTrue(report.overallPassed)
        assertEquals(4, report.gates.size)
        assertTrue(report.targetS24Verified)
    }

    @Test
    fun testStageLatencyBreakdownArithmetic() {
        val breakdown = StageLatencyBreakdown(
            importValidationMs = 2.5,
            allocationMs = 0.8,
            warmInferenceMs = 20.5,
            outputDecodingMs = 0.1,
            dbCommitMs = 1.2,
            reportGenerationMs = 1.0,
            totalPipelineMs = 26.1
        )

        val computedSum = breakdown.importValidationMs +
                breakdown.allocationMs +
                breakdown.warmInferenceMs +
                breakdown.outputDecodingMs +
                breakdown.dbCommitMs +
                breakdown.reportGenerationMs

        assertEquals(breakdown.totalPipelineMs, computedSum, 0.1)
    }
}
