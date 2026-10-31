package org.crc.multimodal

import org.crc.multimodal.inference.AuthenticBundleLoader
import org.crc.multimodal.inference.AuthenticFeatureBundle
import org.crc.multimodal.inference.CrcMultimodalClassifier
import org.crc.multimodal.inference.InferenceProtocol
import org.crc.multimodal.inference.InputProvenance
import org.crc.multimodal.inference.InputValidator
import org.crc.multimodal.inference.ThresholdLogic
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Base64

/**
 * Stage F: Comprehensive automated test suite for authentic feature bundle ingestion,
 * schema validation, tamper detection, and fail-closed security.
 */
class AuthenticBundleLoaderTest {

    private fun createSampleBundle(
        specimenId: String = "TCGA-TEST-001",
        patchCount: Int = 10,
        visionDim: Int = 1024,
        textDim: Int = 768,
        corruptVisionNan: Boolean = false,
        corruptTextInf: Boolean = false
    ): AuthenticFeatureBundle {
        val visionFloats = FloatArray(patchCount * visionDim) { (it % 100) * 0.01f }
        val textFloats = FloatArray(textDim) { (it % 100) * -0.01f }

        if (corruptVisionNan) {
            visionFloats[10] = Float.NaN
        }
        if (corruptTextInf) {
            textFloats[5] = Float.POSITIVE_INFINITY
        }

        return AuthenticFeatureBundle(
            specimenId = specimenId,
            description = "Test sample specimen bundle",
            patchCount = patchCount,
            visionDim = visionDim,
            textDim = textDim,
            visionSha256 = "",
            textSha256 = "",
            visionData = visionFloats,
            textData = textFloats,
            schemaVersion = 2
        )
    }

    @Test
    fun testValidAuthenticBinaryBundleCRCB() {
        val original = createSampleBundle("TCGA-A6-3807", patchCount = 25)
        val baos = ByteArrayOutputStream()
        AuthenticBundleLoader.writeToStream(original, baos)
        val bytes = baos.toByteArray()

        assertTrue(bytes.size > 25 * 1024 * 4)

        val loaded = AuthenticBundleLoader.loadFromBytes(bytes, expectedSpecimenId = "TCGA-A6-3807")
        assertEquals("TCGA-A6-3807", loaded.specimenId)
        assertEquals(25, loaded.patchCount)
        assertEquals(1024, loaded.visionDim)
        assertEquals(768, loaded.textDim)
        assertEquals(2, loaded.schemaVersion)
        assertTrue(loaded.visionSha256.isNotBlank())
        assertTrue(loaded.textSha256.isNotBlank())

        // Bit-accurate float comparison
        for (i in original.visionData.indices) {
            assertEquals("Vision float at $i mismatch", original.visionData[i], loaded.visionData[i], 0.0f)
        }
        for (i in original.textData.indices) {
            assertEquals("Text float at $i mismatch", original.textData[i], loaded.textData[i], 0.0f)
        }
    }

    @Test
    fun testValidAuthenticJsonBundle() {
        val specimenId = "TCGA-D5-6540"
        val patchCount = 12
        val visionFloats = FloatArray(patchCount * 1024) { (it % 50) * 0.02f }
        val textFloats = FloatArray(768) { (it % 50) * -0.02f }

        val vBytes = ByteArray(visionFloats.size * 4)
        ByteBuffer.wrap(vBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().put(visionFloats)
        val tBytes = ByteArray(textFloats.size * 4)
        ByteBuffer.wrap(tBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().put(textFloats)

        val jsonString = """
        {
          "schema_version": 2,
          "specimen_id": "$specimenId",
          "description": "Validated JSON feature manifest",
          "patch_count": $patchCount,
          "vision_tensor": {
            "shape": [1, $patchCount, 1024],
            "dtype": "float32",
            "data_base64": "${Base64.getEncoder().encodeToString(vBytes)}"
          },
          "text_tensor": {
            "shape": [1, 768],
            "dtype": "float32",
            "data_base64": "${Base64.getEncoder().encodeToString(tBytes)}"
          }
        }
        """.trimIndent()

        val loaded = AuthenticBundleLoader.loadFromBytes(jsonString.toByteArray(Charsets.UTF_8), expectedSpecimenId = specimenId)
        assertEquals(specimenId, loaded.specimenId)
        assertEquals(patchCount, loaded.patchCount)
        assertEquals(1024, loaded.visionDim)
        assertEquals(768, loaded.textDim)
        assertEquals(patchCount * 1024, loaded.visionData.size)
        assertEquals(768, loaded.textData.size)
        assertEquals(visionFloats[0], loaded.visionData[0], 0.0f)
        assertEquals(textFloats[0], loaded.textData[0], 0.0f)
    }

    @Test
    fun testMissingVisionTensorRejected() {
        val jsonString = """
        {
          "schema_version": 2,
          "specimen_id": "SPECIMEN-001",
          "text_tensor": {
            "shape": [1, 768],
            "data": []
          }
        }
        """.trimIndent()

        try {
            AuthenticBundleLoader.loadFromBytes(jsonString.toByteArray(Charsets.UTF_8))
            fail("Expected exception for missing vision_tensor")
        } catch (e: Exception) {
            assertTrue(e.message!!.contains("vision_tensor") || e.message!!.contains("missing"))
        }
    }

    @Test
    fun testMissingTextTensorRejected() {
        val jsonString = """
        {
          "schema_version": 2,
          "specimen_id": "SPECIMEN-001",
          "vision_tensor": {
            "shape": [1, 10, 1024],
            "data": []
          }
        }
        """.trimIndent()

        try {
            AuthenticBundleLoader.loadFromBytes(jsonString.toByteArray(Charsets.UTF_8))
            fail("Expected exception for missing text_tensor")
        } catch (e: Exception) {
            assertTrue(e.message!!.contains("text_tensor") || e.message!!.contains("missing"))
        }
    }

    @Test
    fun testMismatchedSpecimenIdentifierRejected() {
        val bundle = createSampleBundle("TCGA-AAA-111", patchCount = 5)
        val baos = ByteArrayOutputStream()
        AuthenticBundleLoader.writeToStream(bundle, baos)

        try {
            AuthenticBundleLoader.loadFromBytes(baos.toByteArray(), expectedSpecimenId = "TCGA-BBB-222")
            fail("Expected exception for mismatched specimen identifier")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("Specimen ID mismatch"))
        }
    }

    @Test
    fun testInvalidVisionDimensionRejected() {
        val baos = ByteArrayOutputStream()
        val bundle = createSampleBundle("TEST-CASE", patchCount = 5)
        AuthenticBundleLoader.writeToStream(bundle, baos)
        val bytes = baos.toByteArray()

        // Modify visionDim in CRCB header:
        // Header: Magic(4) + Ver(4) + IdLen(4) + 'TEST-CASE'(9) + DescLen(4) + descBytes(25) + patchCount(4) + visionDim(4)
        val visionDimOffset = 4 + 4 + 4 + 9 + 4 + "Test sample specimen bundle".length + 4
        // Corrupt visionDim to 512
        bytes[visionDimOffset] = 0x00
        bytes[visionDimOffset + 1] = 0x02 // 512 in Little-Endian (0x0200)

        try {
            AuthenticBundleLoader.loadFromBytes(bytes)
            fail("Expected exception for invalid vision dimension")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("Vision dimension mismatch"))
        }
    }

    @Test
    fun testWrongDataTypeOrSchemaVersionRejected() {
        val baos = ByteArrayOutputStream()
        val bundle = createSampleBundle("TEST-CASE", patchCount = 5)
        AuthenticBundleLoader.writeToStream(bundle, baos)
        val bytes = baos.toByteArray()

        // Corrupt version field at offset 4 to 99
        bytes[4] = 99
        bytes[5] = 0

        try {
            AuthenticBundleLoader.loadFromBytes(bytes)
            fail("Expected exception for unsupported version")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("Unsupported CRCB version"))
        }
    }

    @Test
    fun testNaNFloatRejected() {
        val bundle = createSampleBundle("TCGA-NAN", patchCount = 10, corruptVisionNan = true)
        val baos = ByteArrayOutputStream()
        AuthenticBundleLoader.writeToStream(bundle, baos)

        try {
            AuthenticBundleLoader.loadFromBytes(baos.toByteArray())
            fail("Expected exception for NaN float in vision tensor")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("NaN"))
        }
    }

    @Test
    fun testInfinityFloatRejected() {
        val bundle = createSampleBundle("TCGA-INF", patchCount = 10, corruptTextInf = true)
        val baos = ByteArrayOutputStream()
        AuthenticBundleLoader.writeToStream(bundle, baos)

        try {
            AuthenticBundleLoader.loadFromBytes(baos.toByteArray())
            fail("Expected exception for Infinite float in text tensor")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("Infinite") || e.message!!.contains("Infinity"))
        }
    }

    @Test
    fun testCorruptedOrTruncatedPayloadRejected() {
        val bundle = createSampleBundle("TCGA-TRUNC", patchCount = 20)
        val baos = ByteArrayOutputStream()
        AuthenticBundleLoader.writeToStream(bundle, baos)
        val bytes = baos.toByteArray()

        // Truncate last 100 bytes
        val truncated = bytes.copyOf(bytes.size - 100)
        try {
            AuthenticBundleLoader.loadFromBytes(truncated)
            fail("Expected exception for truncated payload")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("Truncated") || e.message!!.contains("remaining"))
        }
    }

    @Test
    fun testCorruptedChecksumDigestRejected() {
        val bundle = createSampleBundle("TCGA-TAMPER", patchCount = 10)
        val baos = ByteArrayOutputStream()
        AuthenticBundleLoader.writeToStream(bundle, baos)
        val bytes = baos.toByteArray()

        // Flip a byte in the vision payload (at the end)
        bytes[bytes.size - 10] = (bytes[bytes.size - 10].toInt() xor 0xFF).toByte()

        try {
            AuthenticBundleLoader.loadFromBytes(bytes)
            fail("Expected exception for SHA-256 checksum mismatch")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("SHA-256 mismatch"))
        }
    }

    @Test
    fun testLargePatchCountMemoryBehavior() {
        val patchCount = 1000
        val bundle = createSampleBundle("TCGA-LARGE-N", patchCount = patchCount)
        val baos = ByteArrayOutputStream()
        AuthenticBundleLoader.writeToStream(bundle, baos)
        val bytes = baos.toByteArray()

        val loaded = AuthenticBundleLoader.loadFromBytes(bytes)
        assertEquals(patchCount, loaded.patchCount)
        assertEquals(patchCount * 1024, loaded.visionData.size)
        assertEquals(768, loaded.textData.size)
    }

    @Test
    fun testPreventionOfSyntheticFallbackInAuthenticMode() {
        // Load model bytes from assets
        val modelFile = File("src/main/assets/models/fusion_model_fp32.onnx")
        val altModelFile = File("app/src/main/assets/models/fusion_model_fp32.onnx")
        val finalFile = if (modelFile.exists()) modelFile else altModelFile

        if (!finalFile.exists()) return

        val classifier = CrcMultimodalClassifier.fromByteArray(finalFile.readBytes(), numThreads = 1)
        classifier.use { engine ->
            val numPatches = 10
            val dummyVision = FloatArray(numPatches * 1024) { 0.01f }
            val dummyText = FloatArray(768) { 0.01f }

            try {
                // Must FAIL CLOSED if caller attempts to pass DEMONSTRATION_SIMULATED to predictAuthentic
                engine.predictAuthentic(
                    visionData = dummyVision,
                    numPatches = numPatches,
                    textData = dummyText,
                    caseId = "TEST-SIMULATED",
                    provenance = InputProvenance.DEMONSTRATION_SIMULATED,
                    protocol = InferenceProtocol.PROTOCOL_1_CAPPED
                )
                fail("Expected IllegalArgumentException when passing DEMONSTRATION_SIMULATED to predictAuthentic")
            } catch (e: IllegalArgumentException) {
                assertEquals(
                    "Authentic feature embeddings are required. Import a validated vision and clinical-text tensor pair before running inference.",
                    e.message
                )
            }
        }
    }

    @Test
    fun testThresholdBoundaryAtPoint83() {
        // 0.83000f -> Class 1 (MSI)
        assertEquals(ThresholdLogic.CLASS_MSI, ThresholdLogic.classify(0.83000f))
        // 0.83001f -> Class 1 (MSI)
        assertEquals(ThresholdLogic.CLASS_MSI, ThresholdLogic.classify(0.83001f))
        // 0.82999f -> Class 0 (MSS)
        assertEquals(ThresholdLogic.CLASS_MSS, ThresholdLogic.classify(0.82999f))
        // 0.0f -> Class 0 (MSS)
        assertEquals(ThresholdLogic.CLASS_MSS, ThresholdLogic.classify(0.0f))
        // 1.0f -> Class 1 (MSI)
        assertEquals(ThresholdLogic.CLASS_MSI, ThresholdLogic.classify(1.0f))

        assertEquals("MSI", ThresholdLogic.labelFor(ThresholdLogic.CLASS_MSI))
        assertEquals("MSS", ThresholdLogic.labelFor(ThresholdLogic.CLASS_MSS))
    }
}
