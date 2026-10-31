package org.crc.multimodal

import org.crc.multimodal.inference.AuthenticBundleLoader
import org.crc.multimodal.inference.AuthenticFeatureBundle
import org.crc.multimodal.inference.InputValidator
import org.crc.multimodal.inference.NpyTensorLoader
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Stage WP7.1: Comprehensive automated test suite for NumPy (.npy) tensor parsing,
 * shape verification, finite-value assertions, corrupt input rejection, and separate-tensor pairing.
 */
class NpyTensorLoaderTest {

    private fun createNpyBytes(
        shape: IntArray,
        data: FloatArray,
        majorVersion: Int = 1,
        descr: String = "<f4",
        fortranOrder: Boolean = false,
        padHeaderTo16: Boolean = true
    ): ByteArray {
        val baos = ByteArrayOutputStream()
        // Magic (6 bytes)
        baos.write(byteArrayOf(0x93.toByte(), 'N'.code.toByte(), 'U'.code.toByte(), 'M'.code.toByte(), 'P'.code.toByte(), 'Y'.code.toByte()))
        // Version (2 bytes)
        baos.write(byteArrayOf(majorVersion.toByte(), 0.toByte()))

        val shapeStr = if (shape.size == 1) "(${shape[0]},)" else "(${shape.joinToString(", ")})"
        val headerStr = "{'descr': '$descr', 'fortran_order': ${if (fortranOrder) "True" else "False"}, 'shape': $shapeStr, }"

        if (majorVersion == 1) {
            val unpaddedLen = headerStr.length + 1
            val totalBeforePad = 10 + unpaddedLen
            val padNeeded = if (padHeaderTo16) (16 - (totalBeforePad % 16)) % 16 else 0
            val headerBytes = (headerStr + " ".repeat(padNeeded) + "\n").toByteArray(Charsets.US_ASCII)
            val hLen = headerBytes.size
            baos.write(hLen and 0xFF)
            baos.write((hLen shr 8) and 0xFF)
            baos.write(headerBytes)
        } else {
            val unpaddedLen = headerStr.length + 1
            val totalBeforePad = 12 + unpaddedLen
            val padNeeded = if (padHeaderTo16) (16 - (totalBeforePad % 16)) % 16 else 0
            val headerBytes = (headerStr + " ".repeat(padNeeded) + "\n").toByteArray(Charsets.US_ASCII)
            val hLen = headerBytes.size
            val lenBuf = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(hLen).array()
            baos.write(lenBuf)
            baos.write(headerBytes)
        }

        val dataBuf = ByteBuffer.allocate(data.size * Float.SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN)
        dataBuf.asFloatBuffer().put(data)
        baos.write(dataBuf.array())

        return baos.toByteArray()
    }

    @Test
    fun testValidVisionNpy2D_v1() {
        val numPatches = 15
        val dim = 1024
        val rawData = FloatArray(numPatches * dim) { (it % 50) * 0.02f }
        val npyBytes = createNpyBytes(intArrayOf(numPatches, dim), rawData, majorVersion = 1)

        val array = NpyTensorLoader.loadFromBytes(npyBytes)
        assertEquals(2, array.shape.size)
        assertEquals(numPatches, array.shape[0])
        assertEquals(dim, array.shape[1])
        assertEquals(numPatches * dim, array.data.size)
        assertEquals(rawData[0], array.data[0], 1e-6f)
        assertEquals(rawData[100], array.data[100], 1e-6f)
    }

    @Test
    fun testValidVisionNpy3D_v1() {
        val numPatches = 20
        val dim = 1024
        val rawData = FloatArray(numPatches * dim) { (it % 100) * 0.01f }
        val npyBytes = createNpyBytes(intArrayOf(1, numPatches, dim), rawData, majorVersion = 1)

        val array = NpyTensorLoader.loadFromBytes(npyBytes)
        assertEquals(3, array.shape.size)
        assertEquals(1, array.shape[0])
        assertEquals(numPatches, array.shape[1])
        assertEquals(dim, array.shape[2])
        assertEquals(numPatches * dim, array.data.size)
    }

    @Test
    fun testValidTextNpy1D_v1() {
        val dim = 768
        val rawData = FloatArray(dim) { (it % 20) * -0.05f }
        val npyBytes = createNpyBytes(intArrayOf(dim), rawData, majorVersion = 1)

        val array = NpyTensorLoader.loadFromBytes(npyBytes)
        assertEquals(1, array.shape.size)
        assertEquals(dim, array.shape[0])
        assertEquals(dim, array.data.size)
        assertEquals(rawData[5], array.data[5], 1e-6f)
    }

    @Test
    fun testValidTextNpy2D_v1() {
        val dim = 768
        val rawData = FloatArray(dim) { (it % 20) * -0.05f }
        val npyBytes = createNpyBytes(intArrayOf(1, dim), rawData, majorVersion = 1)

        val array = NpyTensorLoader.loadFromBytes(npyBytes)
        assertEquals(2, array.shape.size)
        assertEquals(1, array.shape[0])
        assertEquals(dim, array.shape[1])
        assertEquals(dim, array.data.size)
    }

    @Test
    fun testValidNpy_v2() {
        val numPatches = 5
        val dim = 1024
        val rawData = FloatArray(numPatches * dim) { 0.123f }
        val npyBytes = createNpyBytes(intArrayOf(numPatches, dim), rawData, majorVersion = 2)

        val array = NpyTensorLoader.loadFromBytes(npyBytes)
        assertEquals(numPatches, array.shape[0])
        assertEquals(dim, array.shape[1])
        assertEquals(numPatches * dim, array.data.size)
    }

    @Test
    fun testRejectInvalidMagicHeader() {
        val rawData = FloatArray(10) { 1.0f }
        val npyBytes = createNpyBytes(intArrayOf(10), rawData)
        npyBytes[0] = 'B'.code.toByte()

        try {
            NpyTensorLoader.loadFromBytes(npyBytes)
            fail("Expected IllegalArgumentException for corrupt magic")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("magic", ignoreCase = true))
        }
    }

    @Test
    fun testRejectUnsupportedMajorVersion() {
        val rawData = FloatArray(10) { 1.0f }
        val npyBytes = createNpyBytes(intArrayOf(10), rawData, majorVersion = 3)

        try {
            NpyTensorLoader.loadFromBytes(npyBytes)
            fail("Expected IllegalArgumentException for major version 3")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("version", ignoreCase = true))
        }
    }

    @Test
    fun testRejectUnsupportedDtype() {
        val rawData = FloatArray(10) { 1.0f }
        val npyBytes = createNpyBytes(intArrayOf(10), rawData, descr = "<f8") // double float64

        try {
            NpyTensorLoader.loadFromBytes(npyBytes)
            fail("Expected IllegalArgumentException for unsupported dtype <f8")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("dtype", ignoreCase = true))
        }
    }

    @Test
    fun testRejectFortranOrder() {
        val rawData = FloatArray(10) { 1.0f }
        val npyBytes = createNpyBytes(intArrayOf(10), rawData, fortranOrder = true)

        try {
            NpyTensorLoader.loadFromBytes(npyBytes)
            fail("Expected IllegalArgumentException for Fortran-order")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("Fortran", ignoreCase = true))
        }
    }

    @Test
    fun testRejectTruncatedPayload() {
        val rawData = FloatArray(100) { 1.0f }
        val npyBytes = createNpyBytes(intArrayOf(100), rawData)
        val truncated = npyBytes.copyOf(npyBytes.size - 20)

        try {
            NpyTensorLoader.loadFromBytes(truncated)
            fail("Expected IllegalArgumentException for truncated payload")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("Truncated", ignoreCase = true))
        }
    }

    @Test
    fun testRejectNaNValueInTensor() {
        val rawData = FloatArray(100) { 1.0f }
        rawData[42] = Float.NaN
        val npyBytes = createNpyBytes(intArrayOf(100), rawData)

        try {
            NpyTensorLoader.loadFromBytes(npyBytes)
            fail("Expected IllegalArgumentException for NaN value in tensor")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("NaN", ignoreCase = true))
        }
    }

    @Test
    fun testRejectInfinityValueInTensor() {
        val rawData = FloatArray(100) { 1.0f }
        rawData[15] = Float.POSITIVE_INFINITY
        val npyBytes = createNpyBytes(intArrayOf(100), rawData)

        try {
            NpyTensorLoader.loadFromBytes(npyBytes)
            fail("Expected IllegalArgumentException for Infinity value in tensor")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("Infinity", ignoreCase = true))
        }
    }

    @Test
    fun testCreateBundleFromNpyPair_Valid() {
        val numPatches = 30
        val vData = FloatArray(numPatches * InputValidator.VISION_FEATURE_DIM) { 0.05f }
        val tData = FloatArray(InputValidator.TEXT_FEATURE_DIM) { -0.02f }

        val visionNpy = NpyTensorLoader.NpyArray(
            shape = intArrayOf(numPatches, InputValidator.VISION_FEATURE_DIM),
            data = vData
        )
        val textNpy = NpyTensorLoader.NpyArray(
            shape = intArrayOf(1, InputValidator.TEXT_FEATURE_DIM),
            data = tData
        )

        val bundle = AuthenticBundleLoader.createBundleFromNpyPair(
            specimenId = "TCGA-PAIR-001",
            description = "Paired NPY specimen",
            visionNpy = visionNpy,
            textNpy = textNpy
        )

        assertEquals("TCGA-PAIR-001", bundle.specimenId)
        assertEquals(numPatches, bundle.patchCount)
        assertEquals(InputValidator.VISION_FEATURE_DIM, bundle.visionDim)
        assertEquals(InputValidator.TEXT_FEATURE_DIM, bundle.textDim)
        assertTrue(bundle.visionSha256.isNotBlank())
        assertTrue(bundle.textSha256.isNotBlank())
        assertEquals(numPatches * 1024, bundle.visionData.size)
        assertEquals(768, bundle.textData.size)
    }

    @Test
    fun testCreateBundleFromNpyPair_IncompatibleVisionDim() {
        val visionNpy = NpyTensorLoader.NpyArray(
            shape = intArrayOf(10, 512), // invalid dim 512 != 1024
            data = FloatArray(10 * 512) { 0.1f }
        )
        val textNpy = NpyTensorLoader.NpyArray(
            shape = intArrayOf(InputValidator.TEXT_FEATURE_DIM),
            data = FloatArray(InputValidator.TEXT_FEATURE_DIM) { 0.1f }
        )

        try {
            AuthenticBundleLoader.createBundleFromNpyPair(
                specimenId = "TCGA-ERR-001",
                visionNpy = visionNpy,
                textNpy = textNpy
            )
            fail("Expected IllegalArgumentException for invalid vision dim")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("shape", ignoreCase = true))
        }
    }

    @Test
    fun testCreateBundleFromNpyPair_IncompatibleTextDim() {
        val visionNpy = NpyTensorLoader.NpyArray(
            shape = intArrayOf(10, InputValidator.VISION_FEATURE_DIM),
            data = FloatArray(10 * InputValidator.VISION_FEATURE_DIM) { 0.1f }
        )
        val textNpy = NpyTensorLoader.NpyArray(
            shape = intArrayOf(512), // invalid dim 512 != 768
            data = FloatArray(512) { 0.1f }
        )

        try {
            AuthenticBundleLoader.createBundleFromNpyPair(
                specimenId = "TCGA-ERR-002",
                visionNpy = visionNpy,
                textNpy = textNpy
            )
            fail("Expected IllegalArgumentException for invalid text dim")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("shape", ignoreCase = true))
        }
    }

    @Test
    fun testCreateBundleFromNpyPair_BlankSpecimenId() {
        val visionNpy = NpyTensorLoader.NpyArray(
            shape = intArrayOf(5, InputValidator.VISION_FEATURE_DIM),
            data = FloatArray(5 * InputValidator.VISION_FEATURE_DIM) { 0.1f }
        )
        val textNpy = NpyTensorLoader.NpyArray(
            shape = intArrayOf(InputValidator.TEXT_FEATURE_DIM),
            data = FloatArray(InputValidator.TEXT_FEATURE_DIM) { 0.1f }
        )

        try {
            AuthenticBundleLoader.createBundleFromNpyPair(
                specimenId = "   ",
                visionNpy = visionNpy,
                textNpy = textNpy
            )
            fail("Expected IllegalArgumentException for blank specimen ID")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("blank", ignoreCase = true))
        }
    }
}
