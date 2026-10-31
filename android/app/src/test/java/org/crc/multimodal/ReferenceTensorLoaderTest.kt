package org.crc.multimodal

import org.crc.multimodal.inference.CrcMultimodalClassifier
import org.crc.multimodal.inference.InferenceProtocol
import org.crc.multimodal.inference.ReferenceTensorLoader
import org.junit.AfterClass
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import java.io.File
import java.io.FileInputStream
import kotlin.math.abs

/**
 * Validates authentic reference benchmark tensors (TCGA-D5-6540 and TCGA-CM-6675)
 * loaded via ReferenceTensorLoader against the frozen FP32 ONNX model.
 */
class ReferenceTensorLoaderTest {

    companion object {
        private var classifier: CrcMultimodalClassifier? = null
        private const val TOLERANCE = 1.0e-4f

        @BeforeClass
        @JvmStatic
        fun setUp() {
            val modelCandidates = listOf(
                File("src/main/assets/models/fusion_model_fp32.onnx"),
                File("app/src/main/assets/models/fusion_model_fp32.onnx"),
                File("../../models/fusion_model_fp32.onnx"),
                File("models/fusion_model_fp32.onnx")
            )
            val modelFile = modelCandidates.firstOrNull { it.exists() }
                ?: error("Could not locate fusion_model_fp32.onnx")
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
    fun testAuthenticMsiBenchmarkCaseTcgaD56540() {
        val fileCandidates = listOf(
            File("src/main/assets/reference_tensors/tcga_d5_6540.bin"),
            File("app/src/main/assets/reference_tensors/tcga_d5_6540.bin")
        )
        val tensorFile = fileCandidates.firstOrNull { it.exists() }
            ?: error("Could not find tcga_d5_6540.bin")

        val caseData = FileInputStream(tensorFile).use {
            ReferenceTensorLoader.loadFromStream(it, "TCGA-D5-6540")
        }

        assertEquals("TCGA-D5-6540", caseData.caseId)
        assertEquals(345, caseData.numPatches)
        assertEquals(1, caseData.gtLabel)
        assertEquals(1, caseData.refCall)
        assertEquals(768, caseData.textData.size)
        assertEquals(345 * 1024, caseData.visionData.size)

        val result = classifier!!.predict(
            visionData = caseData.visionData,
            numPatches = caseData.numPatches,
            textData = caseData.textData,
            caseId = caseData.caseId,
            protocol = InferenceProtocol.PROTOCOL_1_CAPPED
        )

        val logitDiff = abs(result.logit - caseData.refLogit)
        val probDiff = abs(result.probability - caseData.refProb)

        assertEquals("Predicted class must match reference MSI (1)", 1, result.predictedClass)
        assertEquals("Predicted label must be MSI", "MSI", result.predictedLabel)
        assertTrue("Logit error must be <= $TOLERANCE, got $logitDiff", logitDiff <= TOLERANCE)
        assertTrue("Probability error must be <= $TOLERANCE, got $probDiff", probDiff <= TOLERANCE)
    }

    @Test
    fun testAuthenticMssBenchmarkCaseTcgaCm6675() {
        val fileCandidates = listOf(
            File("src/main/assets/reference_tensors/tcga_cm_6675.bin"),
            File("app/src/main/assets/reference_tensors/tcga_cm_6675.bin")
        )
        val tensorFile = fileCandidates.firstOrNull { it.exists() }
            ?: error("Could not find tcga_cm_6675.bin")

        val caseData = FileInputStream(tensorFile).use {
            ReferenceTensorLoader.loadFromStream(it, "TCGA-CM-6675")
        }

        assertEquals("TCGA-CM-6675", caseData.caseId)
        assertEquals(541, caseData.numPatches)
        assertEquals(0, caseData.gtLabel)
        assertEquals(0, caseData.refCall)
        assertEquals(768, caseData.textData.size)
        assertEquals(541 * 1024, caseData.visionData.size)

        val result = classifier!!.predict(
            visionData = caseData.visionData,
            numPatches = caseData.numPatches,
            textData = caseData.textData,
            caseId = caseData.caseId,
            protocol = InferenceProtocol.PROTOCOL_1_CAPPED
        )

        val logitDiff = abs(result.logit - caseData.refLogit)
        val probDiff = abs(result.probability - caseData.refProb)

        assertEquals("Predicted class must match reference MSS (0)", 0, result.predictedClass)
        assertEquals("Predicted label must be MSS", "MSS", result.predictedLabel)
        assertTrue("Logit error must be <= $TOLERANCE, got $logitDiff", logitDiff <= TOLERANCE)
        assertTrue("Probability error must be <= $TOLERANCE, got $probDiff", probDiff <= TOLERANCE)
    }
}
