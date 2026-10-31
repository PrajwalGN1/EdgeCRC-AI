package org.crc.multimodal.security

import java.io.File

/**
 * Defensive file, path, and tensor input validation utilities preventing Zip Slip vulnerabilities,
 * directory traversal attacks, buffer overruns, and non-finite tensor ingestion.
 */
object SafeFileValidator {

    const val MAX_ALLOWED_PATCH_COUNT = 10000
    const val MAX_FILE_SIZE_BYTES = 100L * 1024L * 1024L // 100 MB max bundle size

    /**
     * Sanitizes a filename or specimen identifier, stripping path separators and non-standard tokens.
     */
    fun sanitizeFilename(rawName: String): String {
        require(rawName.isNotBlank()) { "Filename or specimen identifier cannot be blank." }
        val cleaned = rawName.trim()
            .replace("\\", "/")
            .replace(Regex("/+"), "/")
            .split("/")
            .lastOrNull { it.isNotBlank() } ?: "specimen"

        val sanitized = cleaned.replace(Regex("[^a-zA-Z0-9._-]"), "_")
        // Disallow path traversal references
        if (sanitized == ".." || sanitized == ".") {
            return "sanitized_specimen"
        }
        return sanitized
    }

    /**
     * Asserts that a target file resolves strictly inside an authorized base directory.
     * Prevents canonical path traversal attacks.
     */
    fun validateSafePath(targetFile: File, baseDirectory: File): File {
        val canonicalTarget = targetFile.canonicalFile.toPath()
        val canonicalBase = baseDirectory.canonicalFile.toPath()

        if (!canonicalTarget.startsWith(canonicalBase)) {
            throw SecurityException(
                "Path traversal security violation: target path '${targetFile.path}' escapes base directory '${baseDirectory.path}'."
            )
        }
        return targetFile.canonicalFile
    }

    /**
     * Validates an entry name from an archive or zip stream to prevent Zip Slip vulnerabilities.
     */
    fun validateSafeZipEntry(entryName: String, destinationDir: File): File {
        require(entryName.isNotBlank()) { "Archive entry name cannot be blank." }

        // Reject obvious path traversal patterns immediately
        if (entryName.contains("..") || entryName.startsWith("/") || entryName.startsWith("\\")) {
            val resolvedTarget = File(destinationDir, entryName).canonicalFile.toPath()
            val canonicalDest = destinationDir.canonicalFile.toPath()
            if (!resolvedTarget.startsWith(canonicalDest)) {
                throw SecurityException("Zip Slip vulnerability detected: malicious archive entry '$entryName'.")
            }
        }

        val targetFile = File(destinationDir, entryName)
        val canonicalTarget = targetFile.canonicalFile.toPath()
        val canonicalDest = destinationDir.canonicalFile.toPath()

        if (!canonicalTarget.startsWith(canonicalDest)) {
            throw SecurityException("Zip Slip vulnerability detected: malicious archive entry '$entryName'.")
        }
        return targetFile.canonicalFile
    }

    /**
     * Verifies that all float elements in a tensor array are finite (non-NaN and finite).
     */
    fun validateTensorFloats(floats: FloatArray, tensorName: String = "tensor") {
        for (i in floats.indices) {
            val v = floats[i]
            if (v.isNaN() || v.isInfinite()) {
                throw IllegalArgumentException(
                    "Security violation: Tensor '$tensorName' contains non-finite float value ($v) at index $i."
                )
            }
        }
    }

    /**
     * Enforces patch count boundary limits to prevent out-of-memory or denormal attacks.
     */
    fun validatePatchCount(patchCount: Int, maxPatches: Int = MAX_ALLOWED_PATCH_COUNT) {
        require(patchCount in 1..maxPatches) {
            "Specimen patch count ($patchCount) is outside safe operating boundaries [1, $maxPatches]."
        }
    }

    /**
     * Enforces file size limits to prevent device storage or memory exhaustion.
     */
    fun validateFileSize(file: File, maxSizeBytes: Long = MAX_FILE_SIZE_BYTES) {
        require(file.exists() && file.isFile) {
            "Target file does not exist or is not a regular file: ${file.absolutePath}"
        }
        val size = file.length()
        require(size > 0L) {
            "Target file is empty (0 bytes): ${file.absolutePath}"
        }
        require(size <= maxSizeBytes) {
            "Target file size ($size bytes) exceeds maximum safe size threshold ($maxSizeBytes bytes)."
        }
    }
}
