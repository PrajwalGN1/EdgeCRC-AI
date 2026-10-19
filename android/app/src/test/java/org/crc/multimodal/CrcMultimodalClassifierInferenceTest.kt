package org.crc.multimodal

import org.crc.multimodal.inference.CrcMultimodalClassifier
import org.crc.multimodal.inference.InferenceProtocol
import org.crc.multimodal.inference.ThresholdLogic
import org.junit.AfterClass
import org.junit.Assert.fail
import org.junit.BeforeClass
import org.junit.Test
import java.io.File
import kotlin.random.Random
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * End-to-end Kotlin unit test executing real ONNX Runtime inference through
 * CrcMultimodalClassifier against the frozen models/fusion_model_fp32.onnx model.
 */
class CrcMultimodalClassifierInferenceTest {

    companion object {
        private var classifier: CrcMultimodalClassifier? = null
        private lateinit var modelFile: File

        @BeforeClass
        @JvmStatic
        fun setUp() {
            // Locate model file relative to project root or android module
            val candidates = listOf(
                File("../../models/fusion_model_fp32.onnx"),
                File("../models/fusion_model_fp32.onnx"),
                File("src/main/assets/models/fusion_model_fp32.onnx"),
                File("app/src/main/assets/models/fusion_model_fp32.onnx"),
                File("models/fusion_model_fp32.onnx")
            )
            modelFile = candidates.firstOrNull { it.exists() }
                ?: error("Could not find fusion_model_fp32.onnx in candidate paths: $candidates")
            
            classifier = CrcMultimodalClassifier.fromFilePath(modelFile.canonicalPath, numThreads = 2)
        }

        @AfterClass
        @JvmStatic
        fun tearDown() {
            classifier?.close()
            classifier = null
        }
    }

    @Test
    fun testModelInputOutputNamesDataTypesAndShapes() {
        val session = classifier!!.session
        
        // Input names
        val inputNames = session.inputNames
        assertEquals(2, inputNames.size)
        assertTrue(inputNames.contains("vision_seq"))
        assertTrue(inputNames.contains("text_summary"))

        // Input types and shapes
        val inputInfo = session.inputInfo
        val visionNode = inputInfo["vision_seq"]
        assertNotNull(visionNode)
        val visionTensorInfo = visionNode.info as ai.onnxruntime.TensorInfo
        assertEquals(ai.onnxruntime.OnnxJavaType.FLOAT, visionTensorInfo.type)
        assertEquals(3, visionTensorInfo.shape.size)
        assertEquals(1024L, visionTensorInfo.shape[2])

        val textNode = inputInfo["text_summary"]
        assertNotNull(textNode)
        val textTensorInfo = textNode.info as ai.onnxruntime.TensorInfo
        assertEquals(ai.onnxruntime.OnnxJavaType.FLOAT, textTensorInfo.type)
        assertEquals(2, textTensorInfo.shape.size)
        assertEquals(768L, textTensorInfo.shape[1])

        // Output names
        val outputNames = session.outputNames
        assertEquals(3, outputNames.size)
        assertTrue(outputNames.contains("logit"))
        assertTrue(outputNames.contains("probability"))
        assertTrue(outputNames.contains("attention_weights"))

        // Output types and shapes
        val outputInfo = session.outputInfo
        val logitNode = outputInfo["logit"]
        assertNotNull(logitNode)
        val logitTensorInfo = logitNode.info as ai.onnxruntime.TensorInfo
        assertEquals(ai.onnxruntime.OnnxJavaType.FLOAT, logitTensorInfo.type)

        val probNode = outputInfo["probability"]
        assertNotNull(probNode)
        val probTensorInfo = probNode.info as ai.onnxruntime.TensorInfo
        assertEquals(ai.onnxruntime.OnnxJavaType.FLOAT, probTensorInfo.type)

        val attnNode = outputInfo["attention_weights"]
        assertNotNull(attnNode)
        val attnTensorInfo = attnNode.info as ai.onnxruntime.TensorInfo
        assertEquals(ai.onnxruntime.OnnxJavaType.FLOAT, attnTensorInfo.type)
        assertEquals(3, attnTensorInfo.shape.size)
    }

    @Test
    fun testSyntheticInferenceDiversePatchCounts() {
        val testPatchCounts = intArrayOf(1, 10, 80, 500, 1000)
        val rng = Random(42)

        for (n in testPatchCounts) {
            val visionData = FloatArray(n * 1024) { rng.nextFloat() * 2f - 1f }
            val textData = FloatArray(768) { rng.nextFloat() * 2f - 1f }

            val result = classifier!!.predict(
                visionData = visionData,
                textData = textData,
                numPatches = n,
                protocol = InferenceProtocol.PROTOCOL_1_CAPPED,
                caseId = "SYNTHETIC_N_$n"
            )

            // 1. Validate result non-null and scalar bounds
            assertEquals("SYNTHETIC_N_$n", result.caseId)
            assertEquals(n, result.numPatches)
            assertEquals(InferenceProtocol.PROTOCOL_1_CAPPED, result.protocol)
            assertTrue(result.logit.isFinite(), "Logit must be finite for N=$n")
            assertTrue(result.probability.isFinite(), "Probability must be finite for N=$n")
            assertTrue(result.probability in 0.0f..1.0f, "Probability must be in [0, 1] for N=$n")

            // 2. Validate attention weights
            assertNotNull(result.attentionWeights)
            assertEquals(n, result.attentionWeights!!.size, "Attention weights vector must have size N=$n")
            for (w in result.attentionWeights!!) {
                assertTrue(w.isFinite() && w >= 0.0f, "Attention weights must be non-negative and finite")
            }

            // 3. Validate threshold classification rule
            val expectedClass = if (result.probability >= 0.83f) 1 else 0
            val expectedLabel = if (expectedClass == 1) "MSI" else "MSS"
            assertEquals(expectedClass, result.predictedClass)
            assertEquals(expectedLabel, result.predictedLabel)
        }
    }

    @Test
    fun testThresholdBoundaryDecisionLogic() {
        // Test helper function ThresholdLogic
        val tau = 0.83f
        assertEquals(0, ThresholdLogic.classify(0.829999f, tau))
        assertEquals(1, ThresholdLogic.classify(0.830000f, tau))
        assertEquals(1, ThresholdLogic.classify(0.830001f, tau))
    }

    @Test
    fun testSequentialExecutionAndSessionStability() {
        val rng = Random(12345)
        val numIterations = 10
        val n = 50

        for (i in 1..numIterations) {
            val visionData = FloatArray(n * 1024) { rng.nextFloat() * 0.1f }
            val textData = FloatArray(768) { rng.nextFloat() * 0.1f }

            val result = classifier!!.predict(
                visionData = visionData,
                textData = textData,
                numPatches = n,
                protocol = InferenceProtocol.PROTOCOL_1_CAPPED,
                caseId = "ITER_$i"
            )

            assertTrue(result.probability in 0.0f..1.0f)
            assertTrue(result.latencyMs >= 0)
        }
    }

    @Test
    fun testProtocol1RejectsAbove1000Patches() {
        val n = 1001
        val visionData = FloatArray(n * 1024)
        val textData = FloatArray(768)

        try {
            classifier!!.predict(
                visionData = visionData,
                textData = textData,
                numPatches = n,
                protocol = InferenceProtocol.PROTOCOL_1_CAPPED
            )
            fail("Expected IllegalArgumentException for Protocol 1 exceeding 1000 patches")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("Protocol 1 enforces a maximum cap of 1000 patches"))
        }
    }

    @Test
    fun testProtocol2UncappedInference1500Patches() {
        val n = 1500
        val rng = Random(999)
        val visionData = FloatArray(n * 1024) { rng.nextFloat() * 0.05f }
        val textData = FloatArray(768) { rng.nextFloat() * 0.05f }

        val result = classifier!!.predict(
            visionData = visionData,
            textData = textData,
            numPatches = n,
            protocol = InferenceProtocol.PROTOCOL_2_UNCAPPED,
            caseId = "UNCAPPED_1500"
        )

        assertEquals(n, result.numPatches)
        assertEquals(InferenceProtocol.PROTOCOL_2_UNCAPPED, result.protocol)
        assertTrue(result.probability in 0.0f..1.0f)
        assertEquals(n, result.attentionWeights!!.size)
    }

    @Test
    fun testProtocol2UncappedInferenceCase20PatchCount() {
        // CASE_20 has exactly 3,914 patches in holdout cohort
        val n = 3914
        val rng = Random(2026)
        val visionData = FloatArray(n * 1024) { rng.nextFloat() * 0.02f }
        val textData = FloatArray(768) { rng.nextFloat() * 0.02f }

        val result = classifier!!.predict(
            visionData = visionData,
            textData = textData,
            numPatches = n,
            protocol = InferenceProtocol.PROTOCOL_2_UNCAPPED,
            caseId = "CASE_20_SYNTHETIC"
        )

        assertEquals(3914, result.numPatches)
        assertEquals(InferenceProtocol.PROTOCOL_2_UNCAPPED, result.protocol)
        assertTrue(result.probability in 0.0f..1.0f)
        assertEquals(3914, result.attentionWeights!!.size)
    }
}
