package org.crc.multimodal

import org.crc.multimodal.inference.CrcMultimodalClassifier
import org.crc.multimodal.inference.InferenceProtocol
import org.junit.AfterClass
import org.junit.BeforeClass
import org.junit.Test
import java.io.BufferedInputStream
import java.io.DataInputStream
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * End-to-end Holdout Parity Verification Test in Kotlin.
 *
 * Executes the frozen 66-case holdout dataset through the Kotlin Android CrcMultimodalClassifier
 * for both Protocol 1 (manifest-capped) and Protocol 2 (uncapped).
 * Verifies 100% classification agreement (0 flips) and FP32 numerical tolerance compliance (<= 1e-4).
 */
class CrcMultimodalHoldoutParityTest {

    data class HoldoutCase(
        val patientKey: String,
        val numPatches: Int,
        val groundTruthLabel: Int,
        val refLogit: Float,
        val refProb: Float,
        val refCall: Int,
        val textData: FloatArray,
        val visionData: FloatArray
    )

    companion object {
        private var classifier: CrcMultimodalClassifier? = null
        private lateinit var modelFile: File
        private lateinit var proto1File: File
        private lateinit var proto2File: File

        const val ACCEPTANCE_MAX_LOGIT_ERR = 1.0e-4f
        const val ACCEPTANCE_MAX_PROB_ERR = 1.0e-4f
        const val ACCEPTANCE_MEAN_PROB_ERR = 1.0e-5f

        @BeforeClass
        @JvmStatic
        fun setUp() {
            val modelCandidates = listOf(
                File("../../models/fusion_model_fp32.onnx"),
                File("../models/fusion_model_fp32.onnx"),
                File("src/main/assets/models/fusion_model_fp32.onnx"),
                File("app/src/main/assets/models/fusion_model_fp32.onnx"),
                File("models/fusion_model_fp32.onnx")
            )
            modelFile = modelCandidates.firstOrNull { it.exists() }
                ?: error("Could not find fusion_model_fp32.onnx")

            val p1Candidates = listOf(
                File("build/holdout_data/holdout_protocol1.bin"),
                File("app/build/holdout_data/holdout_protocol1.bin"),
                File("../../android/app/build/holdout_data/holdout_protocol1.bin"),
                File("../android/app/build/holdout_data/holdout_protocol1.bin")
            )
            proto1File = p1Candidates.firstOrNull { it.exists() }
                ?: error("Could not find holdout_protocol1.bin")

            val p2Candidates = listOf(
                File("build/holdout_data/holdout_protocol2.bin"),
                File("app/build/holdout_data/holdout_protocol2.bin"),
                File("../../android/app/build/holdout_data/holdout_protocol2.bin"),
                File("../android/app/build/holdout_data/holdout_protocol2.bin")
            )
            proto2File = p2Candidates.firstOrNull { it.exists() }
                ?: error("Could not find holdout_protocol2.bin")

            classifier = CrcMultimodalClassifier.fromFilePath(modelFile.canonicalPath, numThreads = 2)
        }

        @AfterClass
        @JvmStatic
        fun tearDown() {
            classifier?.close()
            classifier = null
        }

        private fun loadCases(file: File): List<HoldoutCase> {
            val cases = mutableListOf<HoldoutCase>()
            val bytes = file.readBytes()
            val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            val numCases = buf.int

            for (i in 0 until numCases) {
                val keyLen = buf.int
                val keyBytes = ByteArray(keyLen)
                buf.get(keyBytes)
                val patientKey = String(keyBytes, Charsets.UTF_8)
                val numPatches = buf.int
                val gtLabel = buf.int
                val refLogit = buf.float
                val refProb = buf.float
                val refCall = buf.int

                val textFloats = FloatArray(768)
                buf.asFloatBuffer().get(textFloats)
                buf.position(buf.position() + 768 * 4)

                val visionFloats = FloatArray(numPatches * 1024)
                buf.asFloatBuffer().get(visionFloats)
                buf.position(buf.position() + numPatches * 1024 * 4)

                cases.add(
                    HoldoutCase(
                        patientKey = patientKey,
                        numPatches = numPatches,
                        groundTruthLabel = gtLabel,
                        refLogit = refLogit,
                        refProb = refProb,
                        refCall = refCall,
                        textData = textFloats,
                        visionData = visionFloats
                    )
                )
            }
            return cases
        }
    }

    @Test
    fun testProtocol1HoldoutParityAll66Cases() {
        val cases = loadCases(proto1File)
        assertEquals(66, cases.size, "Protocol 1 must contain exactly 66 holdout cases")

        var maxLogitErr = 0.0f
        var maxProbErr = 0.0f
        var sumProbErr = 0.0
        var flips = 0

        println("\n=======================================================")
        println("EVALUATING KOTLIN HOLDOUT PARITY: PROTOCOL 1 (66 CASES)")
        println("=======================================================")

        for ((idx, case) in cases.withIndex()) {
            val result = classifier!!.predict(
                visionData = case.visionData,
                numPatches = case.numPatches,
                textData = case.textData,
                caseId = case.patientKey,
                protocol = InferenceProtocol.PROTOCOL_1_CAPPED
            )

            val logitErr = abs(result.logit - case.refLogit)
            val probErr = abs(result.probability - case.refProb)
            sumProbErr += probErr.toDouble()

            if (logitErr > maxLogitErr) maxLogitErr = logitErr
            if (probErr > maxProbErr) maxProbErr = probErr

            val agreement = (result.predictedClass == case.refCall)
            if (!agreement) {
                flips++
                println("FLIP on ${case.patientKey}: Kotlin=${result.predictedClass} (p=${result.probability}), Ref=${case.refCall} (p=${case.refProb})")
            }

            assertTrue(logitErr <= ACCEPTANCE_MAX_LOGIT_ERR, "Logit error for ${case.patientKey} exceeds limit: $logitErr")
            assertTrue(probErr <= ACCEPTANCE_MAX_PROB_ERR, "Prob error for ${case.patientKey} exceeds limit: $probErr")
        }

        val meanProbErr = (sumProbErr / cases.size).toFloat()
        val agreementPct = ((cases.size - flips).toDouble() / cases.size) * 100.0

        println("  Protocol 1 Max Logit Error: $maxLogitErr (limit: $ACCEPTANCE_MAX_LOGIT_ERR)")
        println("  Protocol 1 Max Prob Error:  $maxProbErr (limit: $ACCEPTANCE_MAX_PROB_ERR)")
        println("  Protocol 1 Mean Prob Error: $meanProbErr (limit: $ACCEPTANCE_MEAN_PROB_ERR)")
        println("  Protocol 1 Agreement:       $agreementPct% ($flips flips)")

        assertEquals(0, flips, "Protocol 1 must have 0 classification flips against golden baseline")
        assertEquals(100.0, agreementPct, 1e-6, "Protocol 1 must achieve 100% classification agreement")
        assertTrue(maxLogitErr <= ACCEPTANCE_MAX_LOGIT_ERR)
        assertTrue(maxProbErr <= ACCEPTANCE_MAX_PROB_ERR)
        assertTrue(meanProbErr <= ACCEPTANCE_MEAN_PROB_ERR)
    }

    @Test
    fun testProtocol2HoldoutParityAll66Cases() {
        val cases = loadCases(proto2File)
        assertEquals(66, cases.size, "Protocol 2 must contain exactly 66 holdout cases")

        var maxLogitErr = 0.0f
        var maxProbErr = 0.0f
        var sumProbErr = 0.0
        var flips = 0
        var case20Verified = false

        println("\n=======================================================")
        println("EVALUATING KOTLIN HOLDOUT PARITY: PROTOCOL 2 (66 CASES)")
        println("=======================================================")

        for ((idx, case) in cases.withIndex()) {
            val result = classifier!!.predict(
                visionData = case.visionData,
                numPatches = case.numPatches,
                textData = case.textData,
                caseId = case.patientKey,
                protocol = InferenceProtocol.PROTOCOL_2_UNCAPPED
            )

            if (case.numPatches == 3914) {
                // CASE_20 has 3914 patches
                assertEquals(3914, result.numPatches)
                assertEquals(3914, result.attentionWeights!!.size)
                case20Verified = true
            }

            val logitErr = abs(result.logit - case.refLogit)
            val probErr = abs(result.probability - case.refProb)
            sumProbErr += probErr.toDouble()

            if (logitErr > maxLogitErr) maxLogitErr = logitErr
            if (probErr > maxProbErr) maxProbErr = probErr

            val agreement = (result.predictedClass == case.refCall)
            if (!agreement) {
                flips++
                println("FLIP on ${case.patientKey}: Kotlin=${result.predictedClass} (p=${result.probability}), Ref=${case.refCall} (p=${case.refProb})")
            }

            assertTrue(logitErr <= ACCEPTANCE_MAX_LOGIT_ERR, "Logit error for ${case.patientKey} exceeds limit: $logitErr")
            assertTrue(probErr <= ACCEPTANCE_MAX_PROB_ERR, "Prob error for ${case.patientKey} exceeds limit: $probErr")
        }

        val meanProbErr = (sumProbErr / cases.size).toFloat()
        val agreementPct = ((cases.size - flips).toDouble() / cases.size) * 100.0

        println("  Protocol 2 Max Logit Error: $maxLogitErr (limit: $ACCEPTANCE_MAX_LOGIT_ERR)")
        println("  Protocol 2 Max Prob Error:  $maxProbErr (limit: $ACCEPTANCE_MAX_PROB_ERR)")
        println("  Protocol 2 Mean Prob Error: $meanProbErr (limit: $ACCEPTANCE_MEAN_PROB_ERR)")
        println("  Protocol 2 Agreement:       $agreementPct% ($flips flips)")

        assertTrue(case20Verified, "CASE_20 with 3914 patches must be verified without truncation")
        assertEquals(0, flips, "Protocol 2 must have 0 classification flips against golden baseline")
        assertEquals(100.0, agreementPct, 1e-6, "Protocol 2 must achieve 100% classification agreement")
        assertTrue(maxLogitErr <= ACCEPTANCE_MAX_LOGIT_ERR)
        assertTrue(maxProbErr <= ACCEPTANCE_MAX_PROB_ERR)
        assertTrue(meanProbErr <= ACCEPTANCE_MEAN_PROB_ERR)
    }
}
