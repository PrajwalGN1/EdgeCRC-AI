package org.crc.multimodal.batch

import android.content.ContentValues
import android.content.Context
import android.database.Cursor
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import org.crc.multimodal.inference.InputProvenance

/**
 * Interface for persisting and recovering batch runs and specimen items.
 */
interface BatchRepository {
    fun insertBatchRun(batch: BatchRun): Long
    fun updateBatchRun(batch: BatchRun): Boolean
    fun getBatchRun(batchId: String): BatchRun?
    fun getAllBatchRuns(): List<BatchRun>
    fun deleteBatchRun(batchId: String): Boolean
    fun updateBatchItem(item: BatchSpecimenItem): Boolean
    fun recoverInterruptedBatches()
    fun archiveBatchItem(batchId: String, specimenId: String): Boolean = false
    fun unarchiveBatchItem(batchId: String, specimenId: String): Boolean = false
    fun deleteBatchItem(batchId: String, specimenId: String): Boolean = false
}

/**
 * SQLite-backed implementation of BatchRepository.
 */
class SqliteBatchRepository private constructor(context: Context) : BatchRepository {

    private val dbHelper = BatchDbHelper(context.applicationContext)
    private val lock = Any()

    companion object {
        @Volatile
        private var instance: SqliteBatchRepository? = null

        fun getInstance(context: Context): SqliteBatchRepository {
            return instance ?: synchronized(this) {
                instance ?: SqliteBatchRepository(context).also { instance = it }
            }
        }
    }

    override fun insertBatchRun(batch: BatchRun): Long = synchronized(lock) {
        val db = dbHelper.writableDatabase
        db.beginTransaction()
        try {
            val cv = ContentValues().apply {
                put("batch_id", batch.batchId)
                put("title", batch.title)
                put("created_at", batch.createdAt)
                put("updated_at", batch.updatedAt)
                put("status", batch.status.name)
                put("model_sha256", batch.modelSha256)
                put("app_version", batch.appVersion)
                put("execution_provider", batch.executionProvider)
                put("decision_threshold", batch.decisionThreshold)
                put("total_duration_ms", batch.totalDurationMs)
            }
            val rowId = db.insertWithOnConflict("batch_runs", null, cv, SQLiteDatabase.CONFLICT_REPLACE)

            for (item in batch.items) {
                val itemCv = ContentValues().apply {
                    put("batch_id", batch.batchId)
                    put("specimen_id", item.specimenId)
                    put("source_uri", item.sourceUri)
                    put("file_name", item.fileName)
                    put("file_size_bytes", item.fileSizeBytes)
                    put("bundle_sha256", item.bundleSha256)
                    put("vision_sha256", item.visionSha256)
                    put("text_sha256", item.textSha256)
                    put("schema_version", item.schemaVersion)
                    put("patch_count", item.patchCount)
                    put("status", item.status.name)
                    put("validation_error", item.validationError)
                    put("logit", item.logit)
                    put("probability", item.probability)
                    put("predicted_class", item.predictedClass)
                    put("predicted_label", item.predictedLabel)
                    put("threshold", item.threshold)
                    put("parse_time_ms", item.parseTimeMs)
                    put("validation_time_ms", item.validationTimeMs)
                    put("inference_latency_ms", item.inferenceLatencyMs)
                    put("provenance", item.provenance.name)
                    put("error_details", item.errorDetails)
                    put("completed_timestamp", item.completedTimestamp)
                    put("protocol", item.protocol)
                    put("protocol_reason", item.protocolReason)
                    put("is_archived", if (item.isArchived) 1 else 0)
                }
                db.insertWithOnConflict("batch_items", null, itemCv, SQLiteDatabase.CONFLICT_REPLACE)
            }
            db.setTransactionSuccessful()
            rowId
        } finally {
            db.endTransaction()
        }
    }

    override fun updateBatchRun(batch: BatchRun): Boolean = synchronized(lock) {
        val db = dbHelper.writableDatabase
        db.beginTransaction()
        try {
            val cv = ContentValues().apply {
                put("title", batch.title)
                put("updated_at", System.currentTimeMillis())
                put("status", batch.status.name)
                put("total_duration_ms", batch.totalDurationMs)
            }
            val rows = db.update("batch_runs", cv, "batch_id = ?", arrayOf(batch.batchId))

            for (item in batch.items) {
                updateBatchItem(item)
            }
            db.setTransactionSuccessful()
            rows > 0
        } finally {
            db.endTransaction()
        }
    }

    override fun updateBatchItem(item: BatchSpecimenItem): Boolean = synchronized(lock) {
        val db = dbHelper.writableDatabase
        val cv = ContentValues().apply {
            put("status", item.status.name)
            put("validation_error", item.validationError)
            put("logit", item.logit)
            put("probability", item.probability)
            put("predicted_class", item.predictedClass)
            put("predicted_label", item.predictedLabel)
            put("parse_time_ms", item.parseTimeMs)
            put("validation_time_ms", item.validationTimeMs)
            put("inference_latency_ms", item.inferenceLatencyMs)
            put("provenance", item.provenance.name)
            put("error_details", item.errorDetails)
            put("completed_timestamp", item.completedTimestamp)
            put("protocol", item.protocol)
            put("protocol_reason", item.protocolReason)
            put("is_archived", if (item.isArchived) 1 else 0)
        }
        val rows = db.update("batch_items", cv, "batch_id = ? AND specimen_id = ?", arrayOf(item.batchId, item.specimenId))
        rows > 0
    }

    override fun archiveBatchItem(batchId: String, specimenId: String): Boolean = synchronized(lock) {
        val db = dbHelper.writableDatabase
        val cv = ContentValues().apply {
            put("is_archived", 1)
        }
        val rows = db.update("batch_items", cv, "batch_id = ? AND specimen_id = ?", arrayOf(batchId, specimenId))
        rows > 0
    }

    override fun unarchiveBatchItem(batchId: String, specimenId: String): Boolean = synchronized(lock) {
        val db = dbHelper.writableDatabase
        val cv = ContentValues().apply {
            put("is_archived", 0)
        }
        val rows = db.update("batch_items", cv, "batch_id = ? AND specimen_id = ?", arrayOf(batchId, specimenId))
        rows > 0
    }

    override fun deleteBatchItem(batchId: String, specimenId: String): Boolean = synchronized(lock) {
        val db = dbHelper.writableDatabase
        val rows = db.delete("batch_items", "batch_id = ? AND specimen_id = ?", arrayOf(batchId, specimenId))
        rows > 0
    }

    override fun getBatchRun(batchId: String): BatchRun? = synchronized(lock) {
        val db = dbHelper.readableDatabase
        val cursor = db.query(
            "batch_runs", null,
            "batch_id = ?", arrayOf(batchId),
            null, null, null
        )
        return cursor.use {
            if (it.moveToFirst()) {
                val items = getItemsForBatch(batchId)
                parseBatchRun(it, items)
            } else null
        }
    }

    override fun getAllBatchRuns(): List<BatchRun> = synchronized(lock) {
        val db = dbHelper.readableDatabase
        val cursor = db.query("batch_runs", null, null, null, null, null, "created_at DESC")
        val batches = mutableListOf<BatchRun>()
        cursor.use {
            while (it.moveToNext()) {
                val batchId = it.getString(it.getColumnIndexOrThrow("batch_id"))
                val items = getItemsForBatch(batchId)
                batches.add(parseBatchRun(it, items))
            }
        }
        batches
    }

    override fun deleteBatchRun(batchId: String): Boolean = synchronized(lock) {
        val db = dbHelper.writableDatabase
        db.beginTransaction()
        try {
            db.delete("batch_items", "batch_id = ?", arrayOf(batchId))
            val rows = db.delete("batch_runs", "batch_id = ?", arrayOf(batchId))
            db.setTransactionSuccessful()
            rows > 0
        } finally {
            db.endTransaction()
        }
    }

    override fun recoverInterruptedBatches(): Unit = synchronized(lock) {
        val db = dbHelper.writableDatabase
        db.beginTransaction()
        try {
            // 1. Identify all interrupted PROCESSING batches
            val interruptedBatchIds = mutableListOf<String>()
            db.query(
                "batch_runs", arrayOf("batch_id"),
                "status = ?", arrayOf(BatchStatus.PROCESSING.name),
                null, null, null
            ).use { cursor ->
                while (cursor.moveToNext()) {
                    interruptedBatchIds.add(cursor.getString(0))
                }
            }

            if (interruptedBatchIds.isNotEmpty()) {
                // 2. Transition PROCESSING batch headers to PARTIALLY_COMPLETED
                val cv = ContentValues().apply {
                    put("status", BatchStatus.PARTIALLY_COMPLETED.name)
                    put("updated_at", System.currentTimeMillis())
                }
                db.update("batch_runs", cv, "status = ?", arrayOf(BatchStatus.PROCESSING.name))

                // 3. Transition active INFERRING items to CANCELLED
                val inferringCv = ContentValues().apply {
                    put("status", BatchSpecimenStatus.CANCELLED.name)
                    put("error_details", "Interrupted by system/app termination")
                }
                for (batchId in interruptedBatchIds) {
                    db.update(
                        "batch_items",
                        inferringCv,
                        "batch_id = ? AND status = ?",
                        arrayOf(batchId, BatchSpecimenStatus.INFERRING.name)
                    )
                }

                // 4. Transition queued VALIDATED items to CANCELLED
                val validatedCv = ContentValues().apply {
                    put("status", BatchSpecimenStatus.CANCELLED.name)
                    put("error_details", "Interrupted before execution commenced")
                }
                for (batchId in interruptedBatchIds) {
                    db.update(
                        "batch_items",
                        validatedCv,
                        "batch_id = ? AND status = ?",
                        arrayOf(batchId, BatchSpecimenStatus.VALIDATED.name)
                    )
                }
            }
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
    }

    private fun getItemsForBatch(batchId: String): List<BatchSpecimenItem> {
        val db = dbHelper.readableDatabase
        val cursor = db.query(
            "batch_items", null,
            "batch_id = ?", arrayOf(batchId),
            null, null, "id ASC"
        )
        val items = mutableListOf<BatchSpecimenItem>()
        cursor.use {
            while (it.moveToNext()) {
                items.add(parseBatchItem(it))
            }
        }
        return items
    }

    private fun parseBatchRun(cursor: Cursor, items: List<BatchSpecimenItem>): BatchRun {
        return BatchRun(
            batchId = cursor.getString(cursor.getColumnIndexOrThrow("batch_id")),
            title = cursor.getString(cursor.getColumnIndexOrThrow("title")),
            createdAt = cursor.getLong(cursor.getColumnIndexOrThrow("created_at")),
            updatedAt = cursor.getLong(cursor.getColumnIndexOrThrow("updated_at")),
            status = try {
                BatchStatus.valueOf(cursor.getString(cursor.getColumnIndexOrThrow("status")))
            } catch (e: Exception) { BatchStatus.DRAFT },
            modelSha256 = cursor.getString(cursor.getColumnIndexOrThrow("model_sha256")),
            appVersion = cursor.getString(cursor.getColumnIndexOrThrow("app_version")),
            executionProvider = cursor.getString(cursor.getColumnIndexOrThrow("execution_provider")),
            decisionThreshold = cursor.getFloat(cursor.getColumnIndexOrThrow("decision_threshold")),
            totalDurationMs = cursor.getLong(cursor.getColumnIndexOrThrow("total_duration_ms")),
            items = items
        )
    }

    private fun parseBatchItem(cursor: Cursor): BatchSpecimenItem {
        return BatchSpecimenItem(
            id = cursor.getLong(cursor.getColumnIndexOrThrow("id")),
            batchId = cursor.getString(cursor.getColumnIndexOrThrow("batch_id")),
            specimenId = cursor.getString(cursor.getColumnIndexOrThrow("specimen_id")),
            sourceUri = cursor.getString(cursor.getColumnIndexOrThrow("source_uri")),
            fileName = cursor.getString(cursor.getColumnIndexOrThrow("file_name")),
            fileSizeBytes = cursor.getLong(cursor.getColumnIndexOrThrow("file_size_bytes")),
            bundleSha256 = cursor.getString(cursor.getColumnIndexOrThrow("bundle_sha256")),
            visionSha256 = cursor.getString(cursor.getColumnIndexOrThrow("vision_sha256")),
            textSha256 = cursor.getString(cursor.getColumnIndexOrThrow("text_sha256")),
            schemaVersion = cursor.getInt(cursor.getColumnIndexOrThrow("schema_version")),
            patchCount = cursor.getInt(cursor.getColumnIndexOrThrow("patch_count")),
            status = try {
                BatchSpecimenStatus.valueOf(cursor.getString(cursor.getColumnIndexOrThrow("status")))
            } catch (e: Exception) { BatchSpecimenStatus.PENDING },
            validationError = cursor.getString(cursor.getColumnIndexOrThrow("validation_error")),
            logit = if (cursor.isNull(cursor.getColumnIndexOrThrow("logit"))) null else cursor.getFloat(cursor.getColumnIndexOrThrow("logit")),
            probability = if (cursor.isNull(cursor.getColumnIndexOrThrow("probability"))) null else cursor.getFloat(cursor.getColumnIndexOrThrow("probability")),
            predictedClass = if (cursor.isNull(cursor.getColumnIndexOrThrow("predicted_class"))) null else cursor.getInt(cursor.getColumnIndexOrThrow("predicted_class")),
            predictedLabel = cursor.getString(cursor.getColumnIndexOrThrow("predicted_label")),
            threshold = cursor.getFloat(cursor.getColumnIndexOrThrow("threshold")),
            parseTimeMs = cursor.getLong(cursor.getColumnIndexOrThrow("parse_time_ms")),
            validationTimeMs = cursor.getLong(cursor.getColumnIndexOrThrow("validation_time_ms")),
            inferenceLatencyMs = cursor.getLong(cursor.getColumnIndexOrThrow("inference_latency_ms")),
            provenance = try {
                InputProvenance.valueOf(cursor.getString(cursor.getColumnIndexOrThrow("provenance")))
            } catch (e: Exception) { InputProvenance.AUTHENTIC_IMPORTED },
            errorDetails = cursor.getString(cursor.getColumnIndexOrThrow("error_details")),
            completedTimestamp = if (cursor.isNull(cursor.getColumnIndexOrThrow("completed_timestamp"))) null else cursor.getLong(cursor.getColumnIndexOrThrow("completed_timestamp")),
            protocol = run {
                val idx = cursor.getColumnIndex("protocol")
                if (idx != -1 && !cursor.isNull(idx)) cursor.getString(idx) else "PROTOCOL_1_CAPPED"
            },
            protocolReason = run {
                val idx = cursor.getColumnIndex("protocol_reason")
                if (idx != -1 && !cursor.isNull(idx)) cursor.getString(idx) else null
            },
            isArchived = run {
                val idx = cursor.getColumnIndex("is_archived")
                if (idx != -1 && !cursor.isNull(idx)) cursor.getInt(idx) == 1 else false
            }
        )
    }

    private class BatchDbHelper(context: Context) :
        SQLiteOpenHelper(context, "crc_batch_runs.db", null, 3) {

        override fun onCreate(db: SQLiteDatabase) {
            db.execSQL(
                """
                CREATE TABLE batch_runs (
                    batch_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    model_sha256 TEXT NOT NULL,
                    app_version TEXT NOT NULL,
                    execution_provider TEXT NOT NULL,
                    decision_threshold REAL NOT NULL,
                    total_duration_ms INTEGER NOT NULL
                )
                """.trimIndent()
            )
            db.execSQL("CREATE INDEX idx_batch_created ON batch_runs(created_at DESC)")

            db.execSQL(
                """
                CREATE TABLE batch_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT NOT NULL,
                    specimen_id TEXT NOT NULL,
                    source_uri TEXT,
                    file_name TEXT,
                    file_size_bytes INTEGER,
                    bundle_sha256 TEXT,
                    vision_sha256 TEXT,
                    text_sha256 TEXT,
                    schema_version INTEGER,
                    patch_count INTEGER,
                    status TEXT NOT NULL,
                    validation_error TEXT,
                    logit REAL,
                    probability REAL,
                    predicted_class INTEGER,
                    predicted_label TEXT,
                    threshold REAL,
                    parse_time_ms INTEGER,
                    validation_time_ms INTEGER,
                    inference_latency_ms INTEGER,
                    provenance TEXT NOT NULL,
                    error_details TEXT,
                    completed_timestamp INTEGER,
                    protocol TEXT DEFAULT 'PROTOCOL_1_CAPPED',
                    protocol_reason TEXT,
                    is_archived INTEGER DEFAULT 0,
                    FOREIGN KEY(batch_id) REFERENCES batch_runs(batch_id) ON DELETE CASCADE
                )
                """.trimIndent()
            )
            db.execSQL("CREATE INDEX idx_items_batch ON batch_items(batch_id)")
            db.execSQL("CREATE INDEX idx_items_specimen ON batch_items(specimen_id)")
        }

        override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
            if (oldVersion < 2) {
                db.execSQL("ALTER TABLE batch_items ADD COLUMN protocol TEXT DEFAULT 'PROTOCOL_1_CAPPED'")
                db.execSQL("ALTER TABLE batch_items ADD COLUMN protocol_reason TEXT")
            }
            if (oldVersion < 3) {
                db.execSQL("ALTER TABLE batch_items ADD COLUMN is_archived INTEGER DEFAULT 0")
            }
        }
    }
}
