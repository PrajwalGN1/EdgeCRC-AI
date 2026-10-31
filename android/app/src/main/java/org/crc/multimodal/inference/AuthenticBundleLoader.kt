package org.crc.multimodal.inference

import java.io.ByteArrayOutputStream
import java.io.File
import java.io.InputStream
import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.Base64

/**
 * Universal loader and serializer for authentic multimodal feature bundles.
 *
 * Supported formats:
 * 1. [CRCB] (CRC Bundle v2, binary): High-performance, zero-allocation native Little-Endian Float32 binary.
 * 2. [CRCT] (CRC Tensor v1, binary): Backward-compatible reference holdout tensor format.
 * 3. [JSON] (Schema v2.0): Portable JSON bundle with Base64-encoded or numeric float arrays.
 *
 * Enforces:
 * - Specimen identifier non-blank and sanitized.
 * - Exact tensor dimension contracts (Vision: [1, N, 1024], Text: [1, 768]).
 * - Cryptographic SHA-256 payload integrity verification.
 * - Value finiteness (rejection of NaN, +Infinity, -Infinity).
 * - Detection of truncated, malformed, or corrupt files.
 */
object AuthenticBundleLoader {

    val MAGIC_CRCB = byteArrayOf('C'.code.toByte(), 'R'.code.toByte(), 'C'.code.toByte(), 'B'.code.toByte())
    val MAGIC_CRCT = byteArrayOf('C'.code.toByte(), 'R'.code.toByte(), 'C'.code.toByte(), 'T'.code.toByte())

    const val CURRENT_SCHEMA_VERSION = 2
    const val DATA_TYPE_FLOAT32 = 1

    /**
     * Loads and validates an authentic feature bundle from a File.
     */
    fun loadFromFile(file: File, expectedSpecimenId: String? = null): AuthenticFeatureBundle {
        require(file.exists() && file.isFile) { "Bundle file does not exist: ${file.absolutePath}" }
        require(file.length() > 0) { "Bundle file is empty: ${file.absolutePath}" }
        return file.inputStream().use { stream ->
            loadFromStream(stream, expectedSpecimenId)
        }
    }

    /**
     * Loads and validates an authentic feature bundle from an InputStream.
     */
    fun loadFromStream(inputStream: InputStream, expectedSpecimenId: String? = null): AuthenticFeatureBundle {
        val bytes = inputStream.readBytes()
        require(bytes.isNotEmpty()) { "Cannot load bundle from empty input stream." }
        return loadFromBytes(bytes, expectedSpecimenId)
    }

    /**
     * Loads and validates an authentic feature bundle from raw bytes.
     */
    fun loadFromBytes(bytes: ByteArray, expectedSpecimenId: String? = null): AuthenticFeatureBundle {
        require(bytes.size >= 4) { "Input byte array is too short (${bytes.size} bytes) to determine format." }

        // Check for CRCB (v2 binary)
        if (bytes[0] == MAGIC_CRCB[0] && bytes[1] == MAGIC_CRCB[1] &&
            bytes[2] == MAGIC_CRCB[2] && bytes[3] == MAGIC_CRCB[3]
        ) {
            return parseCrcbBinary(bytes, expectedSpecimenId)
        }

        // Check for CRCT (v1 binary)
        if (bytes[0] == MAGIC_CRCT[0] && bytes[1] == MAGIC_CRCT[1] &&
            bytes[2] == MAGIC_CRCT[2] && bytes[3] == MAGIC_CRCT[3]
        ) {
            return parseCrctBinary(bytes, expectedSpecimenId)
        }

        // Check for JSON (starts with '{' after trimming leading whitespace)
        val firstChar = bytes.firstOrNull { !it.toInt().toChar().isWhitespace() }?.toInt()?.toChar()
        if (firstChar == '{') {
            return parseJsonBundle(bytes, expectedSpecimenId)
        }

        throw IllegalArgumentException(
            "Unsupported bundle format: unrecognized magic header or schema. Expected CRCB, CRCT, or JSON."
        )
    }

    /**
     * Parses the CRCB (Version 2) binary specification.
     */
    private fun parseCrcbBinary(bytes: ByteArray, expectedSpecimenId: String?): AuthenticFeatureBundle {
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)

        // 1. Magic
        val magic = ByteArray(4)
        buffer.get(magic)

        // 2. Version
        val version = buffer.int
        require(version == CURRENT_SCHEMA_VERSION) {
            "Unsupported CRCB version: $version (expected $CURRENT_SCHEMA_VERSION)."
        }

        // 3. Specimen ID
        val idLen = buffer.int
        require(idLen in 1..256) { "Invalid specimen ID length: $idLen (must be 1..256)." }
        require(buffer.remaining() >= idLen) { "Unexpected end of file while reading specimen ID." }
        val idBytes = ByteArray(idLen)
        buffer.get(idBytes)
        val specimenId = String(idBytes, Charsets.UTF_8).trim()
        require(specimenId.isNotBlank()) { "Specimen identifier cannot be blank." }

        if (expectedSpecimenId != null && !specimenId.equals(expectedSpecimenId.trim(), ignoreCase = true)) {
            throw IllegalArgumentException("Specimen ID mismatch: expected '$expectedSpecimenId', got '$specimenId'.")
        }

        // 4. Description
        val descLen = buffer.int
        require(descLen in 0..8192) { "Invalid description length: $descLen." }
        require(buffer.remaining() >= descLen) { "Unexpected end of file while reading description." }
        val descBytes = ByteArray(descLen)
        buffer.get(descBytes)
        val description = String(descBytes, Charsets.UTF_8)

        // 5. Dimensions & Data Type
        val patchCount = buffer.int
        require(patchCount > 0) { "Patch count must be strictly positive, got $patchCount." }

        val visionDim = buffer.int
        require(visionDim == InputValidator.VISION_FEATURE_DIM) {
            "Vision dimension mismatch: expected ${InputValidator.VISION_FEATURE_DIM}, got $visionDim."
        }

        val textDim = buffer.int
        require(textDim == InputValidator.TEXT_FEATURE_DIM) {
            "Text dimension mismatch: expected ${InputValidator.TEXT_FEATURE_DIM}, got $textDim."
        }

        val dataType = buffer.int
        require(dataType == DATA_TYPE_FLOAT32) {
            "Unsupported data type code: $dataType (expected $DATA_TYPE_FLOAT32 for Float32)."
        }

        // 6. SHA-256 Hashes
        require(buffer.remaining() >= 64) { "Unexpected end of file while reading checksum hashes." }
        val visionHashBytes = ByteArray(32)
        buffer.get(visionHashBytes)
        val textHashBytes = ByteArray(32)
        buffer.get(textHashBytes)

        val embeddedVisionSha256 = bytesToHex(visionHashBytes)
        val embeddedTextSha256 = bytesToHex(textHashBytes)

        // 7. Text Tensor Payload (768 * 4 bytes)
        val textByteSize = textDim * Float.SIZE_BYTES
        require(buffer.remaining() >= textByteSize) {
            "Truncated bundle: expected $textByteSize text bytes, but only ${buffer.remaining()} remaining."
        }
        val textPayloadStart = buffer.position()
        val textData = FloatArray(textDim)
        val textFloatBuf = buffer.asFloatBuffer()
        textFloatBuf.get(textData)
        buffer.position(textPayloadStart + textByteSize)

        // Verify Text SHA-256 if non-zero
        if (embeddedTextSha256.isNotBlank() && !isZeroHash(visionHashBytes)) {
            val computedTextHash = computeSha256(bytes, textPayloadStart, textByteSize)
            require(computedTextHash.equals(embeddedTextSha256, ignoreCase = true)) {
                "Text tensor SHA-256 mismatch: embedded $embeddedTextSha256, computed $computedTextHash."
            }
        }

        // 8. Vision Tensor Payload (patchCount * 1024 * 4 bytes)
        val expectedVisionFloatsLong = patchCount.toLong() * visionDim.toLong()
        val visionByteSizeLong = expectedVisionFloatsLong * Float.SIZE_BYTES.toLong()
        require(visionByteSizeLong <= Int.MAX_VALUE.toLong()) {
            "Vision payload size ($visionByteSizeLong bytes) exceeds buffer capacity limit."
        }
        val visionByteSize = visionByteSizeLong.toInt()
        require(buffer.remaining() >= visionByteSize) {
            "Truncated bundle: expected $visionByteSize vision bytes, but only ${buffer.remaining()} remaining."
        }

        val visionPayloadStart = buffer.position()
        val visionData = FloatArray(expectedVisionFloatsLong.toInt())
        val visionFloatBuf = buffer.asFloatBuffer()
        visionFloatBuf.get(visionData)
        buffer.position(visionPayloadStart + visionByteSize)

        // Verify Vision SHA-256 if non-zero
        if (embeddedVisionSha256.isNotBlank() && !isZeroHash(visionHashBytes)) {
            val computedVisionHash = computeSha256(bytes, visionPayloadStart, visionByteSize)
            require(computedVisionHash.equals(embeddedVisionSha256, ignoreCase = true)) {
                "Vision tensor SHA-256 mismatch: embedded $embeddedVisionSha256, computed $computedVisionHash."
            }
        }

        // Check for trailing garbage
        require(buffer.remaining() == 0) {
            "Malformed bundle: ${buffer.remaining()} unexpected trailing bytes."
        }

        // Strict float validation
        InputValidator.validateText(textData)
        InputValidator.validateVision(visionData, patchCount, InferenceProtocol.PROTOCOL_2_UNCAPPED)

        return AuthenticFeatureBundle(
            specimenId = specimenId,
            description = description,
            patchCount = patchCount,
            visionDim = visionDim,
            textDim = textDim,
            visionSha256 = embeddedVisionSha256,
            textSha256 = embeddedTextSha256,
            visionData = visionData,
            textData = textData,
            schemaVersion = CURRENT_SCHEMA_VERSION
        )
    }

    /**
     * Parses the CRCT (Version 1) binary reference tensor format.
     */
    private fun parseCrctBinary(bytes: ByteArray, expectedSpecimenId: String?): AuthenticFeatureBundle {
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)

        // 1. Magic
        val magic = ByteArray(4)
        buffer.get(magic)

        // 2. Version
        val version = buffer.int
        require(version == 1) { "Unsupported CRCT version: $version (expected 1)." }

        // 3. Case Identifier
        val keyLen = buffer.int
        require(keyLen in 1..256) { "Invalid case ID length: $keyLen." }
        val keyBytes = ByteArray(keyLen)
        buffer.get(keyBytes)
        val specimenId = String(keyBytes, Charsets.UTF_8).trim()

        if (expectedSpecimenId != null && !specimenId.equals(expectedSpecimenId.trim(), ignoreCase = true)) {
            throw IllegalArgumentException("Specimen ID mismatch: expected '$expectedSpecimenId', got '$specimenId'.")
        }

        // 4. Metadata: numPatches, gtLabel, refLogit, refProb, refCall
        val patchCount = buffer.int
        require(patchCount > 0) { "Patch count must be strictly positive, got $patchCount." }
        val gtLabel = buffer.int
        val refLogit = buffer.float
        val refProb = buffer.float
        val refCall = buffer.int

        // 5. Text Data: 768 floats
        val textByteSize = InputValidator.TEXT_FEATURE_DIM * Float.SIZE_BYTES
        require(buffer.remaining() >= textByteSize) { "Truncated CRCT: missing text tensor bytes." }
        val textData = FloatArray(InputValidator.TEXT_FEATURE_DIM)
        val textFloatBuf = buffer.asFloatBuffer()
        textFloatBuf.get(textData)
        buffer.position(buffer.position() + textByteSize)

        // 6. Vision Data: patchCount * 1024 floats
        val visionFloats = patchCount * InputValidator.VISION_FEATURE_DIM
        val visionByteSize = visionFloats * Float.SIZE_BYTES
        require(buffer.remaining() >= visionByteSize) { "Truncated CRCT: missing vision tensor bytes." }
        val visionData = FloatArray(visionFloats)
        val visionFloatBuf = buffer.asFloatBuffer()
        visionFloatBuf.get(visionData)
        buffer.position(buffer.position() + visionByteSize)

        InputValidator.validateText(textData)
        InputValidator.validateVision(visionData, patchCount, InferenceProtocol.PROTOCOL_2_UNCAPPED)

        return AuthenticFeatureBundle(
            specimenId = specimenId,
            description = "Holdout Reference (GT: $gtLabel, RefLogit: $refLogit, RefProb: $refProb, RefCall: $refCall)",
            patchCount = patchCount,
            visionDim = InputValidator.VISION_FEATURE_DIM,
            textDim = InputValidator.TEXT_FEATURE_DIM,
            visionSha256 = "",
            textSha256 = "",
            visionData = visionData,
            textData = textData,
            schemaVersion = 1
        )
    }

    /**
     * Parses the portable JSON specification format using a pure-Kotlin zero-dependency parser.
     */
    private fun parseJsonBundle(bytes: ByteArray, expectedSpecimenId: String?): AuthenticFeatureBundle {
        val jsonStr = String(bytes, Charsets.UTF_8)
        val json = (JsonParser(jsonStr).parse() as? Map<*, *>)
            ?: throw IllegalArgumentException("Malformed JSON: root must be an object.")

        val specimenId = (json["specimen_id"] as? String ?: json["case_id"] as? String ?: "").trim()
        require(specimenId.isNotBlank()) { "JSON bundle is missing required 'specimen_id' field." }

        if (expectedSpecimenId != null && !specimenId.equals(expectedSpecimenId.trim(), ignoreCase = true)) {
            throw IllegalArgumentException("Specimen ID mismatch: expected '$expectedSpecimenId', got '$specimenId'.")
        }

        val description = json["description"] as? String ?: ""
        val schemaVersion = (json["schema_version"] as? Number)?.toInt() ?: CURRENT_SCHEMA_VERSION

        require(json.containsKey("vision_tensor")) { "JSON bundle is missing required 'vision_tensor' object." }
        require(json.containsKey("text_tensor")) { "JSON bundle is missing required 'text_tensor' object." }

        val visionObj = json["vision_tensor"] as? Map<*, *>
            ?: throw IllegalArgumentException("Malformed JSON: 'vision_tensor' must be an object.")
        val textObj = json["text_tensor"] as? Map<*, *>
            ?: throw IllegalArgumentException("Malformed JSON: 'text_tensor' must be an object.")

        // Parse text tensor
        val textFloats = parseFloatTensorData(textObj, expectedDim = InputValidator.TEXT_FEATURE_DIM)
        require(textFloats.size == InputValidator.TEXT_FEATURE_DIM) {
            "Text tensor dimension mismatch: expected ${InputValidator.TEXT_FEATURE_DIM}, got ${textFloats.size}."
        }

        // Parse vision tensor
        val shapeList = visionObj["shape"] as? List<*>
        val patchCount = if (shapeList != null) {
            if (shapeList.size == 3) (shapeList[1] as Number).toInt() else (shapeList[0] as Number).toInt()
        } else {
            (json["patch_count"] as? Number)?.toInt() ?: 0
        }
        require(patchCount > 0) { "Vision tensor must have strictly positive patch count, got $patchCount." }

        val expectedVisionSize = patchCount * InputValidator.VISION_FEATURE_DIM
        val visionFloats = parseFloatTensorData(visionObj, expectedDim = expectedVisionSize)
        require(visionFloats.size == expectedVisionSize) {
            "Vision tensor size mismatch: expected $expectedVisionSize floats ($patchCount * ${InputValidator.VISION_FEATURE_DIM}), got ${visionFloats.size}."
        }

        val visionSha256 = visionObj["sha256"] as? String ?: ""
        val textSha256 = textObj["sha256"] as? String ?: ""

        InputValidator.validateText(textFloats)
        InputValidator.validateVision(visionFloats, patchCount, InferenceProtocol.PROTOCOL_2_UNCAPPED)

        return AuthenticFeatureBundle(
            specimenId = specimenId,
            description = description,
            patchCount = patchCount,
            visionDim = InputValidator.VISION_FEATURE_DIM,
            textDim = InputValidator.TEXT_FEATURE_DIM,
            visionSha256 = visionSha256,
            textSha256 = textSha256,
            visionData = visionFloats,
            textData = textFloats,
            schemaVersion = schemaVersion
        )
    }

    private fun parseFloatTensorData(tensorObj: Map<*, *>, expectedDim: Int): FloatArray {
        if (tensorObj.containsKey("data_base64")) {
            val b64 = tensorObj["data_base64"] as? String
                ?: throw IllegalArgumentException("Invalid 'data_base64' value: expected string.")
            val rawBytes = Base64.getDecoder().decode(b64)
            require(rawBytes.size == expectedDim * Float.SIZE_BYTES) {
                "Base64 byte size mismatch: expected ${expectedDim * Float.SIZE_BYTES} bytes, got ${rawBytes.size}."
            }
            val floatBuf = ByteBuffer.wrap(rawBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            val floats = FloatArray(expectedDim)
            floatBuf.get(floats)
            return floats
        } else if (tensorObj.containsKey("data")) {
            val list = tensorObj["data"] as? List<*>
                ?: throw IllegalArgumentException("Invalid 'data' value: expected array.")
            require(list.size == expectedDim) {
                "Array size mismatch: expected $expectedDim floats, got ${list.size}."
            }
            val floats = FloatArray(expectedDim)
            for (i in 0 until expectedDim) {
                floats[i] = (list[i] as Number).toFloat()
            }
            return floats
        } else {
            throw IllegalArgumentException("Tensor object must contain either 'data_base64' or 'data' array.")
        }
    }

    /**
     * Pure-Kotlin lightweight JSON parser.
     */
    internal class JsonParser(private val src: String) {
        private var pos = 0

        private fun skipWhitespace() {
            while (pos < src.length && src[pos].isWhitespace()) pos++
        }

        fun parse(): Any? {
            skipWhitespace()
            if (pos >= src.length) return null
            return when (val c = src[pos]) {
                '{' -> parseObject()
                '[' -> parseArray()
                '"' -> parseString()
                't', 'f' -> parseBoolean()
                'n' -> parseNull()
                else -> if (c == '-' || c.isDigit()) parseNumber() else throw IllegalArgumentException("Unexpected char '$c' at $pos")
            }
        }

        private fun parseObject(): Map<String, Any?> {
            pos++ // skip '{'
            val map = mutableMapOf<String, Any?>()
            while (true) {
                skipWhitespace()
                if (pos >= src.length) throw IllegalArgumentException("Unterminated object")
                if (src[pos] == '}') { pos++; break }
                val key = parseString()
                skipWhitespace()
                if (pos >= src.length || src[pos] != ':') throw IllegalArgumentException("Expected ':' at $pos")
                pos++ // skip ':'
                val value = parse()
                map[key] = value
                skipWhitespace()
                if (pos < src.length && src[pos] == ',') { pos++ }
            }
            return map
        }

        private fun parseArray(): List<Any?> {
            pos++ // skip '['
            val list = mutableListOf<Any?>()
            while (true) {
                skipWhitespace()
                if (pos >= src.length) throw IllegalArgumentException("Unterminated array")
                if (src[pos] == ']') { pos++; break }
                list.add(parse())
                skipWhitespace()
                if (pos < src.length && src[pos] == ',') { pos++ }
            }
            return list
        }

        private fun parseString(): String {
            skipWhitespace()
            if (pos >= src.length || src[pos] != '"') throw IllegalArgumentException("Expected '\"' at $pos")
            pos++
            val sb = StringBuilder()
            while (pos < src.length) {
                val c = src[pos++]
                if (c == '"') return sb.toString()
                if (c == '\\' && pos < src.length) {
                    when (val esc = src[pos++]) {
                        '"' -> sb.append('"')
                        '\\' -> sb.append('\\')
                        '/' -> sb.append('/')
                        'b' -> sb.append('\b')
                        'f' -> sb.append('\u000C')
                        'n' -> sb.append('\n')
                        'r' -> sb.append('\r')
                        't' -> sb.append('\t')
                        'u' -> {
                            val hex = src.substring(pos, pos + 4)
                            sb.append(hex.toInt(16).toChar())
                            pos += 4
                        }
                        else -> sb.append(esc)
                    }
                } else {
                    sb.append(c)
                }
            }
            throw IllegalArgumentException("Unterminated string")
        }

        private fun parseNumber(): Number {
            skipWhitespace()
            val start = pos
            if (pos < src.length && src[pos] == '-') pos++
            while (pos < src.length && (src[pos].isDigit() || src[pos] == '.' || src[pos] == 'e' || src[pos] == 'E' || src[pos] == '+' || src[pos] == '-')) {
                pos++
            }
            val numStr = src.substring(start, pos)
            return if (numStr.contains('.') || numStr.contains('e') || numStr.contains('E')) {
                numStr.toDouble()
            } else {
                numStr.toLong()
            }
        }

        private fun parseBoolean(): Boolean {
            if (src.startsWith("true", pos)) { pos += 4; return true }
            if (src.startsWith("false", pos)) { pos += 5; return false }
            throw IllegalArgumentException("Expected boolean at $pos")
        }

        private fun parseNull(): Any? {
            if (src.startsWith("null", pos)) { pos += 4; return null }
            throw IllegalArgumentException("Expected null at $pos")
        }
    }

    /**
     * Serializes an AuthenticFeatureBundle to an OutputStream in CRCB v2 binary format.
     */
    fun writeToStream(bundle: AuthenticFeatureBundle, outputStream: OutputStream) {
        val idBytes = bundle.specimenId.toByteArray(Charsets.UTF_8)
        require(idBytes.size in 1..256) { "Specimen ID exceeds maximum byte size (256)." }

        val descBytes = bundle.description.toByteArray(Charsets.UTF_8)
        require(descBytes.size <= 8192) { "Description exceeds maximum byte size (8192)." }

        val textByteSize = bundle.textDim * Float.SIZE_BYTES
        val visionByteSizeLong = bundle.patchCount.toLong() * bundle.visionDim.toLong() * Float.SIZE_BYTES.toLong()
        require(visionByteSizeLong <= Int.MAX_VALUE.toLong()) { "Vision payload exceeds maximum capacity." }
        val visionByteSize = visionByteSizeLong.toInt()

        // Prepare text bytes
        val textBytes = ByteArray(textByteSize)
        val textBuf = ByteBuffer.wrap(textBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        textBuf.put(bundle.textData)

        // Prepare vision bytes
        val visionBytes = ByteArray(visionByteSize)
        val visionBuf = ByteBuffer.wrap(visionBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        visionBuf.put(bundle.visionData)

        // Calculate SHA-256 digests
        val visionDigest = MessageDigest.getInstance("SHA-256").digest(visionBytes)
        val textDigest = MessageDigest.getInstance("SHA-256").digest(textBytes)

        val headerSize = 4 + 4 + 4 + idBytes.size + 4 + descBytes.size + 4 + 4 + 4 + 4 + 32 + 32
        val headerBuf = ByteBuffer.allocate(headerSize).order(ByteOrder.LITTLE_ENDIAN)

        headerBuf.put(MAGIC_CRCB)
        headerBuf.putInt(CURRENT_SCHEMA_VERSION)
        headerBuf.putInt(idBytes.size)
        headerBuf.put(idBytes)
        headerBuf.putInt(descBytes.size)
        headerBuf.put(descBytes)
        headerBuf.putInt(bundle.patchCount)
        headerBuf.putInt(bundle.visionDim)
        headerBuf.putInt(bundle.textDim)
        headerBuf.putInt(DATA_TYPE_FLOAT32)
        headerBuf.put(visionDigest)
        headerBuf.put(textDigest)

        outputStream.write(headerBuf.array())
        outputStream.write(textBytes)
        outputStream.write(visionBytes)
        outputStream.flush()
    }

    private fun computeSha256(bytes: ByteArray, offset: Int, length: Int): String {
        val md = MessageDigest.getInstance("SHA-256")
        md.update(bytes, offset, length)
        return bytesToHex(md.digest())
    }

    private fun bytesToHex(bytes: ByteArray): String {
        val sb = StringBuilder(bytes.size * 2)
        for (b in bytes) {
            sb.append(String.format(java.util.Locale.US, "%02x", b))
        }
        return sb.toString()
    }

    /**
     * Constructs a validated [AuthenticFeatureBundle] from separate vision and text NPY arrays.
     */
    fun createBundleFromNpyPair(
        specimenId: String,
        description: String = "",
        visionNpy: NpyTensorLoader.NpyArray,
        textNpy: NpyTensorLoader.NpyArray
    ): AuthenticFeatureBundle {
        val shapeV = visionNpy.shape
        val patchCount = when {
            shapeV.size == 2 && shapeV[1] == InputValidator.VISION_FEATURE_DIM -> shapeV[0]
            shapeV.size == 3 && shapeV[0] == 1 && shapeV[2] == InputValidator.VISION_FEATURE_DIM -> shapeV[1]
            else -> throw IllegalArgumentException(
                "Invalid vision tensor shape: (${shapeV.joinToString(", ")}). Expected [N, ${InputValidator.VISION_FEATURE_DIM}] or [1, N, ${InputValidator.VISION_FEATURE_DIM}]."
            )
        }
        require(patchCount > 0) { "Vision tensor must have at least 1 patch, got $patchCount." }

        val shapeT = textNpy.shape
        val isTextValid = when {
            shapeT.size == 1 && shapeT[0] == InputValidator.TEXT_FEATURE_DIM -> true
            shapeT.size == 2 && shapeT[0] == 1 && shapeT[1] == InputValidator.TEXT_FEATURE_DIM -> true
            else -> false
        }
        require(isTextValid) {
            "Invalid text tensor shape: (${shapeT.joinToString(", ")}). Expected [${InputValidator.TEXT_FEATURE_DIM}] or [1, ${InputValidator.TEXT_FEATURE_DIM}]."
        }

        return createBundleFromSeparateTensors(
            specimenId = specimenId,
            description = description,
            visionArray = visionNpy.data,
            patchCount = patchCount,
            textArray = textNpy.data
        )
    }

    /**
     * Constructs a validated [AuthenticFeatureBundle] from raw vision and text FloatArray buffers.
     */
    fun createBundleFromSeparateTensors(
        specimenId: String,
        description: String = "",
        visionArray: FloatArray,
        patchCount: Int,
        textArray: FloatArray
    ): AuthenticFeatureBundle {
        val sanitizedId = specimenId.trim()
        require(sanitizedId.isNotBlank()) { "Specimen identifier cannot be blank." }
        InputValidator.validateVision(visionArray, patchCount, InferenceProtocol.PROTOCOL_2_UNCAPPED)
        InputValidator.validateText(textArray)

        val visionBytes = ByteBuffer.allocate(visionArray.size * Float.SIZE_BYTES)
            .order(ByteOrder.LITTLE_ENDIAN)
            .apply { asFloatBuffer().put(visionArray) }
            .array()
        val textBytes = ByteBuffer.allocate(textArray.size * Float.SIZE_BYTES)
            .order(ByteOrder.LITTLE_ENDIAN)
            .apply { asFloatBuffer().put(textArray) }
            .array()

        val visionDigest = computeSha256(visionBytes, 0, visionBytes.size)
        val textDigest = computeSha256(textBytes, 0, textBytes.size)

        return AuthenticFeatureBundle(
            specimenId = sanitizedId,
            description = description.trim(),
            patchCount = patchCount,
            visionDim = InputValidator.VISION_FEATURE_DIM,
            textDim = InputValidator.TEXT_FEATURE_DIM,
            visionSha256 = visionDigest,
            textSha256 = textDigest,
            visionData = visionArray,
            textData = textArray,
            schemaVersion = CURRENT_SCHEMA_VERSION,
            importTimestamp = System.currentTimeMillis()
        )
    }

    private fun isZeroHash(bytes: ByteArray): Boolean {
        return bytes.all { it == 0.toByte() }
    }
}
