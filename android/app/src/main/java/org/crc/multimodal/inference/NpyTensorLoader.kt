package org.crc.multimodal.inference

import java.io.File
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.regex.Pattern

/**
 * High-performance, zero-dependency, memory-safe parser for NumPy (.npy) single-array binary files.
 *
 * Implements NPY Format Specification (Versions 1.0 and 2.0):
 * - Magic Header: 0x93 'N' 'U' 'M' 'P' 'Y' (6 bytes)
 * - Major Version: 1 or 2
 * - Minor Version: 0
 * - Header Length: 2 bytes unsigned short (v1.0) or 4 bytes int (v2.0)
 * - ASCII Header: Python dictionary with 'descr', 'fortran_order', 'shape'
 *
 * Validations:
 * - Little-Endian Float32 ('<f4' or '|f4' or 'float32')
 * - C-contiguous ('fortran_order': False)
 * - Exact dimension matching
 * - Strict rejection of non-finite values (NaN, +Infinity, -Infinity)
 * - Overflow protection on array allocation
 */
object NpyTensorLoader {

    val NPY_MAGIC = byteArrayOf(0x93.toByte(), 'N'.code.toByte(), 'U'.code.toByte(), 'M'.code.toByte(), 'P'.code.toByte(), 'Y'.code.toByte())

    data class NpyArray(
        val shape: IntArray,
        val data: FloatArray
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false
            other as NpyArray
            if (!shape.contentEquals(other.shape)) return false
            if (!data.contentEquals(other.data)) return false
            return true
        }

        override fun hashCode(): Int {
            var result = shape.contentHashCode()
            result = 31 * result + data.contentHashCode()
            return result
        }
    }

    fun loadFromFile(file: File): NpyArray {
        require(file.exists() && file.isFile) { "NPY file does not exist: ${file.absolutePath}" }
        require(file.length() > 0) { "NPY file is empty: ${file.absolutePath}" }
        return file.inputStream().use { loadFromStream(it) }
    }

    fun loadFromBytes(bytes: ByteArray): NpyArray {
        require(bytes.size >= 10) { "File too small to be a valid NPY array (${bytes.size} bytes)." }
        return parseNpy(bytes)
    }

    fun loadFromStream(stream: InputStream): NpyArray {
        val bytes = stream.readBytes()
        return loadFromBytes(bytes)
    }

    private fun parseNpy(bytes: ByteArray): NpyArray {
        // 1. Validate Magic (6 bytes)
        for (i in 0 until 6) {
            require(bytes[i] == NPY_MAGIC[i]) { "Invalid NPY magic header." }
        }

        // 2. Version (2 bytes)
        val majorVersion = bytes[6].toInt() and 0xFF
        val minorVersion = bytes[7].toInt() and 0xFF
        require(majorVersion == 1 || majorVersion == 2) {
            "Unsupported NPY major version: $majorVersion (supported: 1, 2)."
        }
        require(minorVersion == 0) {
            "Unsupported NPY minor version: $minorVersion (expected: 0)."
        }

        // 3. Header Length
        val headerLen: Int
        val dataOffset: Int
        if (majorVersion == 1) {
            val b0 = bytes[8].toInt() and 0xFF
            val b1 = bytes[9].toInt() and 0xFF
            headerLen = b0 or (b1 shl 8)
            dataOffset = 10 + headerLen
        } else {
            val buf = ByteBuffer.wrap(bytes, 8, 4).order(ByteOrder.LITTLE_ENDIAN)
            headerLen = buf.int
            require(headerLen > 0) { "Invalid NPY v2 header length: $headerLen" }
            dataOffset = 12 + headerLen
        }

        require(bytes.size >= dataOffset) {
            "Truncated NPY file: header extends beyond file length."
        }

        val headerStr = String(bytes, if (majorVersion == 1) 10 else 12, headerLen, Charsets.US_ASCII)

        // 4. Parse Header Dictionary
        // descr
        val descrMatcher = Pattern.compile("'descr'\\s*:\\s*'([^']+)'").matcher(headerStr)
        require(descrMatcher.find()) { "Malformed NPY header: missing 'descr' field." }
        val descr = descrMatcher.group(1) ?: ""
        require(descr == "<f4" || descr == "|f4" || descr == "float32") {
            "Unsupported NPY dtype: '$descr'. Expected Little-Endian Float32 ('<f4')."
        }

        // fortran_order
        val fortranMatcher = Pattern.compile("'fortran_order'\\s*:\\s*(True|False)").matcher(headerStr)
        require(fortranMatcher.find()) { "Malformed NPY header: missing 'fortran_order' field." }
        val fortranOrder = fortranMatcher.group(1).equals("True", ignoreCase = true)
        require(!fortranOrder) { "Fortran-order (column-major) NPY arrays are not supported. Expected C-order." }

        // shape: e.g. (385, 1024), or (1, 385, 1024), or (768,) or (1, 768)
        val shapeMatcher = Pattern.compile("'shape'\\s*:\\s*\\(([^)]*)\\)").matcher(headerStr)
        require(shapeMatcher.find()) { "Malformed NPY header: missing 'shape' field." }
        val shapeContent = shapeMatcher.group(1)?.trim() ?: ""
        val shapeDims = if (shapeContent.isEmpty()) {
            intArrayOf(1)
        } else {
            shapeContent.split(",")
                .map { it.trim() }
                .filter { it.isNotEmpty() }
                .map { it.toInt() }
                .toIntArray()
        }

        require(shapeDims.isNotEmpty() && shapeDims.all { it > 0 }) {
            "Invalid NPY shape: (${shapeDims.joinToString(", ")})."
        }

        var totalElementsLong = 1L
        for (dim in shapeDims) {
            totalElementsLong *= dim.toLong()
            require(totalElementsLong <= Int.MAX_VALUE.toLong()) {
                "NPY tensor elements ($totalElementsLong) exceed 32-bit array capacity."
            }
        }
        val totalElements = totalElementsLong.toInt()
        val expectedDataBytes = totalElements * 4

        require(bytes.size - dataOffset >= expectedDataBytes) {
            "Truncated NPY payload: expected $expectedDataBytes bytes, available ${bytes.size - dataOffset}."
        }

        // 5. Read Float32 Payload
        val floatData = FloatArray(totalElements)
        val buf = ByteBuffer.wrap(bytes, dataOffset, expectedDataBytes).order(ByteOrder.LITTLE_ENDIAN)
        buf.asFloatBuffer().get(floatData)

        // 6. Strict Finite Value Sanity Check
        for (i in floatData.indices) {
            val v = floatData[i]
            require(!v.isNaN()) { "Non-finite NaN value detected at index $i in NPY tensor." }
            require(!v.isInfinite()) { "Non-finite Infinity value detected at index $i in NPY tensor." }
        }

        return NpyArray(shape = shapeDims, data = floatData)
    }
}
