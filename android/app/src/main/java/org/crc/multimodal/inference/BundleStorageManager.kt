package org.crc.multimodal.inference

import android.content.Context
import org.crc.multimodal.security.SafeFileValidator
import java.io.File
import java.io.FileOutputStream

/**
 * Manages local persistent storage of authentic multimodal feature bundles
 * in the application's private file directory.
 *
 * All files are stored in `filesDir/specimen_bundles/` and are isolated from other apps.
 * Enforces SafeFileValidator boundary checks against directory traversal.
 * No broad external storage permissions are required.
 */
object BundleStorageManager {

    private const val BUNDLES_DIR_NAME = "specimen_bundles"
    private const val BUNDLE_EXTENSION = ".crcbundle"

    fun getStorageDir(context: Context): File {
        val dir = File(context.filesDir, BUNDLES_DIR_NAME)
        if (!dir.exists()) {
            dir.mkdirs()
        }
        return dir
    }

    private fun sanitizeFilename(specimenId: String): String {
        return SafeFileValidator.sanitizeFilename(specimenId)
    }

    fun getBundleFile(context: Context, specimenId: String): File? {
        val storageDir = getStorageDir(context)
        val cleanName = sanitizeFilename(specimenId)
        val file = File(storageDir, "$cleanName$BUNDLE_EXTENSION")
        SafeFileValidator.validateSafePath(file, storageDir)
        return if (file.exists() && file.length() > 0) file else null
    }

    fun hasBundle(context: Context, specimenId: String): Boolean {
        return getBundleFile(context, specimenId) != null
    }

    fun saveBundle(context: Context, bundle: AuthenticFeatureBundle, overwrite: Boolean = false): File {
        val storageDir = getStorageDir(context)
        val cleanName = sanitizeFilename(bundle.specimenId)
        val file = File(storageDir, "$cleanName$BUNDLE_EXTENSION")
        SafeFileValidator.validateSafePath(file, storageDir)
        if (file.exists() && !overwrite) {
            throw IllegalArgumentException("A bundle for specimen '${bundle.specimenId}' already exists. Overwrite not permitted.")
        }

        // Write atomically using a temporary file
        val tempFile = File(storageDir, "$cleanName.tmp")
        SafeFileValidator.validateSafePath(tempFile, storageDir)
        try {
            FileOutputStream(tempFile).use { fos ->
                AuthenticBundleLoader.writeToStream(bundle, fos)
            }
            if (file.exists()) {
                file.delete()
            }
            if (!tempFile.renameTo(file)) {
                throw IllegalStateException("Failed to commit bundle file for specimen '${bundle.specimenId}'.")
            }
        } finally {
            if (tempFile.exists()) {
                tempFile.delete()
            }
        }

        CrcLogger.i("[BUNDLE_STORAGE] Saved authentic bundle for specimen '${bundle.specimenId}' (${file.length()} bytes)")
        return file
    }

    fun loadBundle(context: Context, specimenId: String): AuthenticFeatureBundle? {
        val file = getBundleFile(context, specimenId) ?: return null
        return try {
            AuthenticBundleLoader.loadFromFile(file, expectedSpecimenId = specimenId)
        } catch (e: Exception) {
            CrcLogger.e("[BUNDLE_STORAGE] Error loading bundle for specimen '$specimenId': ${e.message}", e)
            null
        }
    }

    fun deleteBundle(context: Context, specimenId: String): Boolean {
        val file = getBundleFile(context, specimenId) ?: return false
        val deleted = file.delete()
        if (deleted) {
            CrcLogger.i("[BUNDLE_STORAGE] Deleted bundle for specimen '$specimenId'")
        }
        return deleted
    }

    fun listBundleFiles(context: Context): List<File> {
        val dir = getStorageDir(context)
        return dir.listFiles { file -> file.isFile && file.name.endsWith(BUNDLE_EXTENSION) }
            ?.toList() ?: emptyList()
    }
}
