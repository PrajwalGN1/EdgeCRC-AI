package org.crc.multimodal.data

import android.content.ContentValues
import android.content.Context
import android.database.Cursor
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import org.crc.multimodal.inference.InferenceProtocol

/**
 * SQLite-backed implementation of WorkstationRepository ensuring thread-safe local persistence
 * on physical Android devices.
 */
class SqliteWorkstationRepository private constructor(context: Context) : WorkstationRepository {

    private val dbHelper = WorkstationDbHelper(context.applicationContext)
    private val lock = Any()

    companion object {
        @Volatile
        private var instance: SqliteWorkstationRepository? = null

        fun getInstance(context: Context): SqliteWorkstationRepository {
            return instance ?: synchronized(this) {
                instance ?: SqliteWorkstationRepository(context).also { instance = it }
            }
        }
    }

    override fun insertCase(researchCase: ResearchCase): Long = synchronized(lock) {
        val db = dbHelper.writableDatabase
        val existing = getCaseByIdentifier(researchCase.caseIdentifier)
        if (existing != null) {
            throw IllegalArgumentException("Case with identifier '${researchCase.caseIdentifier}' already exists.")
        }
        val cv = ContentValues().apply {
            put("case_identifier", researchCase.caseIdentifier.trim())
            put("specimen_description", researchCase.specimenDescription)
            put("research_context", researchCase.researchContext)
            put("status", researchCase.status.name)
            put("patch_count", researchCase.patchCount)
            put("protocol", researchCase.protocol?.name)
            put("created_at", researchCase.createdAt)
            put("updated_at", researchCase.updatedAt)
        }
        val caseId = db.insertOrThrow("cases", null, cv)
        val cvAudit = ContentValues().apply {
            put("event_timestamp", System.currentTimeMillis())
            put("action_type", org.crc.multimodal.security.AuditActionType.SPECIMEN_INGESTED.name)
            put("specimen_id", researchCase.caseIdentifier.trim())
            put("details", "Specimen ingested. Patches: ${researchCase.patchCount}, Protocol: ${InferenceProtocol.formatProtocol(researchCase.protocol)}")
            put("performed_by", "Local Research User")
        }
        db.insert("audit_trail", null, cvAudit)
        caseId
    }

    override fun updateCase(researchCase: ResearchCase): Boolean = synchronized(lock) {
        val db = dbHelper.writableDatabase
        val existingWithSameId = getCaseByIdentifier(researchCase.caseIdentifier)
        if (existingWithSameId != null && existingWithSameId.id != researchCase.id) {
            throw IllegalArgumentException("Another case with identifier '${researchCase.caseIdentifier}' already exists.")
        }
        val cv = ContentValues().apply {
            put("case_identifier", researchCase.caseIdentifier.trim())
            put("specimen_description", researchCase.specimenDescription)
            put("research_context", researchCase.researchContext)
            put("status", researchCase.status.name)
            put("patch_count", researchCase.patchCount)
            put("protocol", researchCase.protocol?.name)
            put("updated_at", System.currentTimeMillis())
        }
        val rows = db.update("cases", cv, "id = ?", arrayOf(researchCase.id.toString()))
        rows > 0
    }

    override fun deleteCase(caseId: Long): Boolean = synchronized(lock) {
        val targetCase = getCaseById(caseId)
        val db = dbHelper.writableDatabase
        db.beginTransaction()
        try {
            db.delete("specimen_inputs", "case_id = ?", arrayOf(caseId.toString()))
            db.delete("inference_runs", "case_id = ?", arrayOf(caseId.toString()))
            db.delete("review_notes", "case_id = ?", arrayOf(caseId.toString()))
            db.delete("research_reports", "case_id = ?", arrayOf(caseId.toString()))
            db.delete("report_history", "case_id = ?", arrayOf(caseId.toString()))
            val rows = db.delete("cases", "id = ?", arrayOf(caseId.toString()))
            if (rows > 0 && targetCase != null) {
                val cvAudit = ContentValues().apply {
                    put("event_timestamp", System.currentTimeMillis())
                    put("action_type", org.crc.multimodal.security.AuditActionType.SPECIMEN_DELETED.name)
                    put("specimen_id", targetCase.caseIdentifier)
                    put("details", "Specimen '${targetCase.caseIdentifier}' and associated inputs/runs deleted by user.")
                    put("performed_by", "Local Research User")
                }
                db.insert("audit_trail", null, cvAudit)
            }
            db.setTransactionSuccessful()
            rows > 0
        } finally {
            db.endTransaction()
        }
    }

    override fun getCaseById(id: Long): ResearchCase? = synchronized(lock) {
        val db = dbHelper.readableDatabase
        val cursor = db.query(
            "cases", null, "id = ?", arrayOf(id.toString()),
            null, null, null
        )
        cursor.use {
            if (it.moveToFirst()) parseCase(it) else null
        }
    }

    override fun getCaseByIdentifier(identifier: String): ResearchCase? = synchronized(lock) {
        val db = dbHelper.readableDatabase
        val cursor = db.query(
            "cases", null, "case_identifier = ? COLLATE NOCASE", arrayOf(identifier.trim()),
            null, null, null
        )
        cursor.use {
            if (it.moveToFirst()) parseCase(it) else null
        }
    }

    override fun getAllCases(): List<ResearchCase> = synchronized(lock) {
        val db = dbHelper.readableDatabase
        val cursor = db.query("cases", null, null, null, null, null, "updated_at DESC")
        cursor.use {
            val list = mutableListOf<ResearchCase>()
            while (it.moveToNext()) {
                list.add(parseCase(it))
            }
            list
        }
    }

    override fun getCasesFiltered(
        status: CaseStatus?,
        searchQuery: String?,
        sortBy: CaseSortOption
    ): List<ResearchCase> = synchronized(lock) {
        val db = dbHelper.readableDatabase
        val whereClauses = mutableListOf<String>()
        val whereArgs = mutableListOf<String>()

        if (status != null) {
            whereClauses.add("status = ?")
            whereArgs.add(status.name)
        }

        if (!searchQuery.isNullOrBlank()) {
            val q = "%${searchQuery.trim()}%"
            whereClauses.add("(case_identifier LIKE ? OR specimen_description LIKE ? OR research_context LIKE ?)")
            whereArgs.add(q)
            whereArgs.add(q)
            whereArgs.add(q)
        }

        val selection = if (whereClauses.isEmpty()) null else whereClauses.joinToString(" AND ")
        val selectionArgs = if (whereArgs.isEmpty()) null else whereArgs.toTypedArray()

        val orderBy = when (sortBy) {
            CaseSortOption.DATE_NEWEST -> "updated_at DESC"
            CaseSortOption.DATE_OLDEST -> "updated_at ASC"
            CaseSortOption.CASE_IDENTIFIER -> "case_identifier ASC"
        }

        val cursor = db.query("cases", null, selection, selectionArgs, null, null, orderBy)
        cursor.use {
            val list = mutableListOf<ResearchCase>()
            while (it.moveToNext()) {
                list.add(parseCase(it))
            }
            list
        }
    }

    override fun getRecentCases(limit: Int): List<ResearchCase> = synchronized(lock) {
        val db = dbHelper.readableDatabase
        val cursor = db.query(
            "cases", null, null, null,
            null, null, "updated_at DESC", limit.toString()
        )
        cursor.use {
            val list = mutableListOf<ResearchCase>()
            while (it.moveToNext()) {
                list.add(parseCase(it))
            }
            list
        }
    }

    override fun getDashboardCounts(): DashboardCounts = synchronized(lock) {
        val db = dbHelper.readableDatabase
        val total = getCount(db, "SELECT COUNT(*) FROM cases")
        val drafts = getCount(db, "SELECT COUNT(*) FROM cases WHERE status = 'DRAFT'")
        val ready = getCount(db, "SELECT COUNT(*) FROM cases WHERE status = 'READY_FOR_ANALYSIS'")
        val analyzed = getCount(db, "SELECT COUNT(*) FROM cases WHERE status = 'ANALYZED'")
        val underReview = getCount(db, "SELECT COUNT(*) FROM cases WHERE status = 'UNDER_REVIEW'")
        val reviewed = getCount(db, "SELECT COUNT(*) FROM cases WHERE status = 'REVIEWED'")

        DashboardCounts(
            totalCases = total,
            draftCount = drafts,
            readyCount = ready,
            analyzedCount = analyzed,
            underReviewCount = underReview,
            reviewedCount = reviewed
        )
    }

    private fun getCount(db: SQLiteDatabase, sql: String): Int {
        val cursor = db.rawQuery(sql, null)
        return cursor.use {
            if (it.moveToFirst()) it.getInt(0) else 0
        }
    }

    override fun insertSpecimenInput(input: SpecimenInput): Long = synchronized(lock) {
        val db = dbHelper.writableDatabase
        val cv = ContentValues().apply {
            put("case_id", input.caseId)
            put("input_type", input.inputType)
            put("local_uri", input.localUriOrAsset)
            put("filename", input.filename)
            put("validation_status", input.validationStatus)
            put("created_at", input.createdAt)
        }
        db.insertOrThrow("specimen_inputs", null, cv)
    }

    override fun getInputsForCase(caseId: Long): List<SpecimenInput> = synchronized(lock) {
        val db = dbHelper.readableDatabase
        val cursor = db.query(
            "specimen_inputs", null, "case_id = ?", arrayOf(caseId.toString()),
            null, null, "created_at ASC"
        )
        cursor.use {
            val list = mutableListOf<SpecimenInput>()
            while (it.moveToNext()) {
                list.add(
                    SpecimenInput(
                        id = it.getLong(it.getColumnIndexOrThrow("id")),
                        caseId = it.getLong(it.getColumnIndexOrThrow("case_id")),
                        inputType = it.getString(it.getColumnIndexOrThrow("input_type")),
                        localUriOrAsset = it.getString(it.getColumnIndexOrThrow("local_uri")),
                        filename = it.getString(it.getColumnIndexOrThrow("filename")),
                        validationStatus = it.getString(it.getColumnIndexOrThrow("validation_status")),
                        createdAt = it.getLong(it.getColumnIndexOrThrow("created_at"))
                    )
                )
            }
            list
        }
    }

    override fun deleteInput(inputId: Long): Boolean = synchronized(lock) {
        val db = dbHelper.writableDatabase
        val rows = db.delete("specimen_inputs", "id = ?", arrayOf(inputId.toString()))
        rows > 0
    }

    override fun insertInferenceRun(run: InferenceRun): Long = synchronized(lock) {
        val db = dbHelper.writableDatabase
        val cv = ContentValues().apply {
            put("case_id", run.caseId)
            put("model_identifier", run.modelIdentifier)
            put("inference_architecture", run.inferenceArchitecture)
            put("protocol", run.protocol?.name)
            put("predicted_class", run.predictedClass)
            put("predicted_label", run.predictedLabel)
            put("probability", run.probability)
            put("logit", run.logit)
            put("threshold", run.threshold)
            put("margin", run.margin)
            put("latency_ms", run.latencyMs)
            put("num_patches", run.numPatches)
            put("diagnostics_summary", run.diagnosticsSummary)
            put("execution_timestamp", run.executionTimestamp)
            put("status", run.status)
        }
        db.insertOrThrow("inference_runs", null, cv)
    }

    override fun getRunsForCase(caseId: Long): List<InferenceRun> = synchronized(lock) {
        val db = dbHelper.readableDatabase
        val cursor = db.query(
            "inference_runs", null, "case_id = ?", arrayOf(caseId.toString()),
            null, null, "execution_timestamp DESC"
        )
        cursor.use {
            val list = mutableListOf<InferenceRun>()
            while (it.moveToNext()) {
                list.add(parseRun(it))
            }
            list
        }
    }

    override fun getLatestRunForCase(caseId: Long): InferenceRun? = synchronized(lock) {
        val db = dbHelper.readableDatabase
        val cursor = db.query(
            "inference_runs", null, "case_id = ?", arrayOf(caseId.toString()),
            null, null, "execution_timestamp DESC", "1"
        )
        cursor.use {
            if (it.moveToFirst()) parseRun(it) else null
        }
    }

    override fun getAllInferenceRuns(): List<InferenceRun> = synchronized(lock) {
        val db = dbHelper.readableDatabase
        val cursor = db.query(
            "inference_runs", null, null, null,
            null, null, "execution_timestamp DESC"
        )
        cursor.use {
            val list = mutableListOf<InferenceRun>()
            while (it.moveToNext()) {
                list.add(parseRun(it))
            }
            list
        }
    }

    override fun getResearchAnalyticsSummary(
        batchRepo: org.crc.multimodal.batch.BatchRepository?
    ): org.crc.multimodal.analytics.ResearchAnalyticsSummary {
        val unifiedRecords = getUnifiedPredictionRecords(batchRepo)
        val batchRuns = batchRepo?.getAllBatchRuns() ?: emptyList()
        return org.crc.multimodal.analytics.ResearchAnalyticsEngine.computeSummary(unifiedRecords, batchRuns)
    }

    override fun getUnifiedPredictionRecords(
        batchRepo: org.crc.multimodal.batch.BatchRepository?
    ): List<org.crc.multimodal.analytics.SpecimenPredictionRecord> {
        val allCases = getAllCases()
        val allRuns = getAllInferenceRuns()
        val batchRuns = batchRepo?.getAllBatchRuns() ?: emptyList()
        return org.crc.multimodal.analytics.ResearchAnalyticsEngine.buildUnifiedRecords(allCases, allRuns, batchRuns)
    }

    override fun recordInferenceResult(
        caseId: Long,
        run: InferenceRun,
        newStatus: CaseStatus
    ): Long = synchronized(lock) {
        val db = dbHelper.writableDatabase
        db.beginTransaction()
        try {
            checkNotNull(getCaseById(caseId)) {
                "Cannot persist inference result: Case $caseId not found in database."
            }

            val cvRun = ContentValues().apply {
                put("case_id", run.caseId)
                put("model_identifier", run.modelIdentifier)
                put("inference_architecture", run.inferenceArchitecture)
                put("protocol", run.protocol?.name)
                put("predicted_class", run.predictedClass)
                put("predicted_label", run.predictedLabel)
                put("probability", run.probability)
                put("logit", run.logit)
                put("threshold", run.threshold)
                put("margin", run.margin)
                put("latency_ms", run.latencyMs)
                put("num_patches", run.numPatches)
                put("diagnostics_summary", run.diagnosticsSummary)
                put("execution_timestamp", run.executionTimestamp)
                put("status", run.status)
            }
            val runId = db.insertOrThrow("inference_runs", null, cvRun)

            val cvCase = ContentValues().apply {
                put("status", newStatus.name)
                put("patch_count", run.numPatches)
                put("protocol", run.protocol?.name)
                put("updated_at", System.currentTimeMillis())
            }
            val rows = db.update("cases", cvCase, "id = ?", arrayOf(caseId.toString()))
            if (rows <= 0) {
                throw IllegalStateException("Failed to update status for case $caseId.")
            }

            val targetCase = getCaseById(caseId)
            val cvAudit = ContentValues().apply {
                put("event_timestamp", System.currentTimeMillis())
                put("action_type", org.crc.multimodal.security.AuditActionType.INFERENCE_EXECUTED.name)
                put("specimen_id", targetCase?.caseIdentifier)
                put("details", "Inference result: ${run.predictedLabel} (P = ${run.probability}). Margin: ${run.margin}. Latency: ${run.latencyMs} ms")
                put("sha256_fingerprint", run.modelIdentifier)
                put("performed_by", "Local Research User")
            }
            db.insert("audit_trail", null, cvAudit)

            db.setTransactionSuccessful()
            runId
        } finally {
            db.endTransaction()
        }
    }

    override fun recordInferenceFailure(caseId: Long, errorMessage: String): Unit = synchronized(lock) {
        val db = dbHelper.writableDatabase
        db.beginTransaction()
        try {
            val existingCase = getCaseById(caseId)
            if (existingCase != null) {
                val cvCase = ContentValues().apply {
                    put("status", CaseStatus.READY_FOR_ANALYSIS.name)
                    put("updated_at", System.currentTimeMillis())
                }
                db.update("cases", cvCase, "id = ?", arrayOf(caseId.toString()))
            }
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
    }

    override fun recoverInterruptedCases(): Unit = synchronized(lock) {
        val db = dbHelper.writableDatabase
        db.beginTransaction()
        try {
            val cursor = db.rawQuery(
                """
                SELECT c.id FROM cases c
                LEFT JOIN inference_runs r ON c.id = r.case_id AND r.status = 'SUCCESS'
                WHERE c.status = 'ANALYZED' AND r.id IS NULL
                """.trimIndent(), null
            )
            val orphanCaseIds = mutableListOf<Long>()
            cursor.use {
                while (it.moveToNext()) {
                    orphanCaseIds.add(it.getLong(0))
                }
            }
            for (orphanId in orphanCaseIds) {
                val cv = ContentValues().apply {
                    put("status", CaseStatus.READY_FOR_ANALYSIS.name)
                    put("updated_at", System.currentTimeMillis())
                }
                db.update("cases", cv, "id = ?", arrayOf(orphanId.toString()))
            }
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
    }

    override fun insertReviewNote(note: ReviewNote): Long = synchronized(lock) {
        val db = dbHelper.writableDatabase
        val cv = ContentValues().apply {
            put("case_id", note.caseId)
            put("category", note.category.name)
            put("author_label", note.authorLabel)
            put("content", note.content)
            put("created_at", note.createdAt)
            put("updated_at", note.updatedAt)
        }
        db.insertOrThrow("review_notes", null, cv)
    }

    override fun updateReviewNote(note: ReviewNote): Boolean = synchronized(lock) {
        val db = dbHelper.writableDatabase
        val cv = ContentValues().apply {
            put("category", note.category.name)
            put("author_label", note.authorLabel)
            put("content", note.content)
            put("updated_at", System.currentTimeMillis())
        }
        val rows = db.update("review_notes", cv, "id = ?", arrayOf(note.id.toString()))
        rows > 0
    }

    override fun deleteReviewNote(noteId: Long): Boolean = synchronized(lock) {
        val db = dbHelper.writableDatabase
        val rows = db.delete("review_notes", "id = ?", arrayOf(noteId.toString()))
        rows > 0
    }

    override fun getNotesForCase(caseId: Long): List<ReviewNote> = synchronized(lock) {
        val db = dbHelper.readableDatabase
        val cursor = db.query(
            "review_notes", null, "case_id = ?", arrayOf(caseId.toString()),
            null, null, "created_at DESC"
        )
        cursor.use {
            val list = mutableListOf<ReviewNote>()
            while (it.moveToNext()) {
                list.add(
                    ReviewNote(
                        id = it.getLong(it.getColumnIndexOrThrow("id")),
                        caseId = it.getLong(it.getColumnIndexOrThrow("case_id")),
                        category = ReviewNoteCategory.fromString(it.getString(it.getColumnIndexOrThrow("category"))),
                        authorLabel = it.getString(it.getColumnIndexOrThrow("author_label")),
                        content = it.getString(it.getColumnIndexOrThrow("content")),
                        createdAt = it.getLong(it.getColumnIndexOrThrow("created_at")),
                        updatedAt = it.getLong(it.getColumnIndexOrThrow("updated_at"))
                    )
                )
            }
            list
        }
    }

    override fun insertOrUpdateReport(report: ResearchReport): Long = synchronized(lock) {
        val db = dbHelper.writableDatabase
        val cv = ContentValues().apply {
            put("case_id", report.caseId)
            put("inference_run_id", report.inferenceRunId)
            put("generated_at", System.currentTimeMillis())
            put("report_content", report.reportContent)
            put("findings_summary", report.findingsSummary)
        }
        db.insertWithOnConflict("research_reports", null, cv, SQLiteDatabase.CONFLICT_REPLACE)
    }

    override fun getReportForCase(caseId: Long): ResearchReport? = synchronized(lock) {
        val db = dbHelper.readableDatabase
        val cursor = db.query(
            "research_reports", null, "case_id = ?", arrayOf(caseId.toString()),
            null, null, null
        )
        cursor.use {
            if (it.moveToFirst()) {
                ResearchReport(
                    id = it.getLong(it.getColumnIndexOrThrow("id")),
                    caseId = it.getLong(it.getColumnIndexOrThrow("case_id")),
                    inferenceRunId = if (it.isNull(it.getColumnIndexOrThrow("inference_run_id"))) null
                    else it.getLong(it.getColumnIndexOrThrow("inference_run_id")),
                    generatedAt = it.getLong(it.getColumnIndexOrThrow("generated_at")),
                    reportContent = it.getString(it.getColumnIndexOrThrow("report_content")),
                    findingsSummary = it.getString(it.getColumnIndexOrThrow("findings_summary"))
                )
            } else null
        }
    }

    override fun insertReportRecord(record: PersistentReportRecord): Long = synchronized(lock) {
        val db = dbHelper.writableDatabase
        val cv = ContentValues().apply {
            put("report_id", record.reportId)
            put("case_id", record.caseId)
            put("inference_run_id", record.inferenceRunId)
            put("template_version", record.templateVersion)
            put("created_at", record.createdAt)
            put("export_status", record.exportStatus.name)
            put("file_uri", record.fileUri)
            put("file_path", record.filePath)
            put("model_version", record.modelVersion)
            put("sha256_hash", record.sha256Hash)
            put("error_message", record.errorMessage)
        }
        db.insertWithOnConflict("report_history", null, cv, SQLiteDatabase.CONFLICT_REPLACE)
    }

    override fun updateReportRecordStatus(
        reportId: String,
        status: ReportExportStatus,
        fileUri: String?,
        filePath: String?,
        errorMessage: String?
    ): Boolean = synchronized(lock) {
        val db = dbHelper.writableDatabase
        val cv = ContentValues().apply {
            put("export_status", status.name)
            if (fileUri != null) put("file_uri", fileUri)
            if (filePath != null) put("file_path", filePath)
            if (errorMessage != null) put("error_message", errorMessage)
        }
        val rows = db.update("report_history", cv, "report_id = ?", arrayOf(reportId))
        rows > 0
    }

    override fun getReportHistoryForCase(caseId: Long): List<PersistentReportRecord> = synchronized(lock) {
        val db = dbHelper.readableDatabase
        val cursor = db.query(
            "report_history", null, "case_id = ?", arrayOf(caseId.toString()),
            null, null, "created_at DESC"
        )
        cursor.use {
            val list = mutableListOf<PersistentReportRecord>()
            while (it.moveToNext()) {
                list.add(parseReportRecord(it))
            }
            list
        }
    }

    override fun getReportRecordById(reportId: String): PersistentReportRecord? = synchronized(lock) {
        val db = dbHelper.readableDatabase
        val cursor = db.query(
            "report_history", null, "report_id = ?", arrayOf(reportId),
            null, null, null
        )
        cursor.use {
            if (it.moveToFirst()) parseReportRecord(it) else null
        }
    }

    override fun deleteReportRecord(reportId: String): Boolean = synchronized(lock) {
        val db = dbHelper.writableDatabase
        val rows = db.delete("report_history", "report_id = ?", arrayOf(reportId))
        rows > 0
    }

    private fun parseReportRecord(cursor: Cursor): PersistentReportRecord {
        val rawStatus = cursor.getString(cursor.getColumnIndexOrThrow("export_status"))
        var status = ReportExportStatus.fromString(rawStatus)
        val filePath = cursor.getString(cursor.getColumnIndexOrThrow("file_path"))

        if (filePath != null && (status == ReportExportStatus.GENERATED || status == ReportExportStatus.EXPORTED)) {
            val file = java.io.File(filePath)
            if (!file.exists()) {
                status = ReportExportStatus.MISSING_FILE
            }
        }

        return PersistentReportRecord(
            id = cursor.getLong(cursor.getColumnIndexOrThrow("id")),
            reportId = cursor.getString(cursor.getColumnIndexOrThrow("report_id")),
            caseId = cursor.getLong(cursor.getColumnIndexOrThrow("case_id")),
            inferenceRunId = if (cursor.isNull(cursor.getColumnIndexOrThrow("inference_run_id"))) null
            else cursor.getLong(cursor.getColumnIndexOrThrow("inference_run_id")),
            templateVersion = cursor.getString(cursor.getColumnIndexOrThrow("template_version")),
            createdAt = cursor.getLong(cursor.getColumnIndexOrThrow("created_at")),
            exportStatus = status,
            fileUri = cursor.getString(cursor.getColumnIndexOrThrow("file_uri")),
            filePath = filePath,
            modelVersion = cursor.getString(cursor.getColumnIndexOrThrow("model_version")),
            sha256Hash = cursor.getString(cursor.getColumnIndexOrThrow("sha256_hash")),
            errorMessage = cursor.getString(cursor.getColumnIndexOrThrow("error_message"))
        )
    }

    override fun clearAllData(): Unit = synchronized(lock) {
        val db = dbHelper.writableDatabase
        db.beginTransaction()
        try {
            db.delete("report_history", null, null)
            db.delete("research_reports", null, null)
            db.delete("review_notes", null, null)
            db.delete("inference_runs", null, null)
            db.delete("specimen_inputs", null, null)
            db.delete("cases", null, null)
            val cvAudit = ContentValues().apply {
                put("event_timestamp", System.currentTimeMillis())
                put("action_type", org.crc.multimodal.security.AuditActionType.ALL_DATA_CLEARED.name)
                put("details", "User initiated clear-all operation; all research workstation cases, runs, and notes wiped.")
                put("performed_by", "Local Research User")
            }
            db.insert("audit_trail", null, cvAudit)
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
    }

    override fun seedReferenceCases(): Unit = synchronized(lock) {
        // 1. Reference MSI Benchmark Case (TCGA-D5-6540)
        if (getCaseByIdentifier("TCGA-D5-6540") == null) {
            val msiCaseId = insertCase(
                ResearchCase(
                    caseIdentifier = "TCGA-D5-6540",
                    specimenDescription = "Right ascending colon adenocarcinoma, biopsy specimen (FFPE)",
                    researchContext = "Authentic holdout validation MSI benchmark case (Protocol 1, N=345 patches). Ground Truth: Class 1 (MSI). Reference Logit: 4.63481, Reference Probability: 0.99038, Reference Call: MSI.",
                    status = CaseStatus.READY_FOR_ANALYSIS,
                    patchCount = 345,
                    protocol = InferenceProtocol.PROTOCOL_1_CAPPED
                )
            )
            insertReviewNote(
                ReviewNote(
                    caseId = msiCaseId,
                    category = ReviewNoteCategory.HISTOLOGICAL_OBSERVATION,
                    authorLabel = "Lead Research Pathologist",
                    content = "Frozen benchmark case TCGA-D5-6540. Verified 345 H&E patches with paired Clinical-BERT text embedding. Ready for genuine on-device FP32 inference execution."
                )
            )
        }

        // 2. Reference MSS Benchmark Case (TCGA-CM-6675)
        if (getCaseByIdentifier("TCGA-CM-6675") == null) {
            val mssCaseId = insertCase(
                ResearchCase(
                    caseIdentifier = "TCGA-CM-6675",
                    specimenDescription = "Sigmoid colon tubular adenocarcinoma, resection specimen (FFPE)",
                    researchContext = "Authentic holdout validation MSS benchmark case (Protocol 1, N=541 patches). Ground Truth: Class 0 (MSS). Reference Logit: -1.64158, Reference Probability: 0.16225, Reference Call: MSS.",
                    status = CaseStatus.READY_FOR_ANALYSIS,
                    patchCount = 541,
                    protocol = InferenceProtocol.PROTOCOL_1_CAPPED
                )
            )
            insertReviewNote(
                ReviewNote(
                    caseId = mssCaseId,
                    category = ReviewNoteCategory.HISTOLOGICAL_OBSERVATION,
                    authorLabel = "Lead Research Pathologist",
                    content = "Frozen benchmark case TCGA-CM-6675. Verified 541 H&E patches with paired Clinical-BERT text embedding. Ready for genuine on-device FP32 inference execution."
                )
            )
        }
    }

    private fun parseCase(cursor: Cursor): ResearchCase {
        return ResearchCase(
            id = cursor.getLong(cursor.getColumnIndexOrThrow("id")),
            caseIdentifier = cursor.getString(cursor.getColumnIndexOrThrow("case_identifier")),
            specimenDescription = cursor.getString(cursor.getColumnIndexOrThrow("specimen_description")),
            researchContext = cursor.getString(cursor.getColumnIndexOrThrow("research_context")),
            status = CaseStatus.fromString(cursor.getString(cursor.getColumnIndexOrThrow("status"))),
            patchCount = cursor.getInt(cursor.getColumnIndexOrThrow("patch_count")),
            protocol = {
                val idx = cursor.getColumnIndex("protocol")
                val str = if (idx >= 0 && !cursor.isNull(idx)) cursor.getString(idx) else null
                InferenceProtocol.fromStringOrNull(str)
            }(),
            createdAt = cursor.getLong(cursor.getColumnIndexOrThrow("created_at")),
            updatedAt = cursor.getLong(cursor.getColumnIndexOrThrow("updated_at"))
        )
    }

    private fun parseRun(cursor: Cursor): InferenceRun {
        return InferenceRun(
            id = cursor.getLong(cursor.getColumnIndexOrThrow("id")),
            caseId = cursor.getLong(cursor.getColumnIndexOrThrow("case_id")),
            modelIdentifier = cursor.getString(cursor.getColumnIndexOrThrow("model_identifier")),
            inferenceArchitecture = cursor.getString(cursor.getColumnIndexOrThrow("inference_architecture")),
            protocol = {
                val idx = cursor.getColumnIndex("protocol")
                val str = if (idx >= 0 && !cursor.isNull(idx)) cursor.getString(idx) else null
                InferenceProtocol.fromStringOrNull(str)
            }(),
            predictedClass = cursor.getInt(cursor.getColumnIndexOrThrow("predicted_class")),
            predictedLabel = cursor.getString(cursor.getColumnIndexOrThrow("predicted_label")),
            probability = cursor.getFloat(cursor.getColumnIndexOrThrow("probability")),
            logit = cursor.getFloat(cursor.getColumnIndexOrThrow("logit")),
            threshold = cursor.getFloat(cursor.getColumnIndexOrThrow("threshold")),
            margin = cursor.getFloat(cursor.getColumnIndexOrThrow("margin")),
            latencyMs = cursor.getDouble(cursor.getColumnIndexOrThrow("latency_ms")),
            numPatches = cursor.getInt(cursor.getColumnIndexOrThrow("num_patches")),
            diagnosticsSummary = cursor.getString(cursor.getColumnIndexOrThrow("diagnostics_summary")),
            executionTimestamp = cursor.getLong(cursor.getColumnIndexOrThrow("execution_timestamp")),
            status = cursor.getString(cursor.getColumnIndexOrThrow("status"))
        )
    }

    // Reference Labels & Cohort Evaluation (WP7.4)
    override fun insertReferenceLabel(label: org.crc.multimodal.evaluation.ReferenceLabel): Long = synchronized(lock) {
        val db = dbHelper.writableDatabase
        val cv = ContentValues().apply {
            put("specimen_id", label.specimenId.trim())
            put("ground_truth_label", label.groundTruthLabel)
            put("ground_truth_class", label.groundTruthClass)
            put("cohort_id", label.cohortId)
            put("source", label.source)
            put("imported_at", label.importedAt)
        }
        db.insertWithOnConflict("reference_labels", null, cv, SQLiteDatabase.CONFLICT_REPLACE)
    }

    override fun insertReferenceLabels(labels: List<org.crc.multimodal.evaluation.ReferenceLabel>): Int = synchronized(lock) {
        val db = dbHelper.writableDatabase
        db.beginTransaction()
        try {
            var count = 0
            for (label in labels) {
                val cv = ContentValues().apply {
                    put("specimen_id", label.specimenId.trim())
                    put("ground_truth_label", label.groundTruthLabel)
                    put("ground_truth_class", label.groundTruthClass)
                    put("cohort_id", label.cohortId)
                    put("source", label.source)
                    put("imported_at", label.importedAt)
                }
                val rowId = db.insertWithOnConflict("reference_labels", null, cv, SQLiteDatabase.CONFLICT_REPLACE)
                if (rowId != -1L) count++
            }
            db.setTransactionSuccessful()
            count
        } finally {
            db.endTransaction()
        }
    }

    override fun getReferenceLabel(specimenId: String): org.crc.multimodal.evaluation.ReferenceLabel? = synchronized(lock) {
        val db = dbHelper.readableDatabase
        val cursor = db.query(
            "reference_labels", null, "specimen_id = ? COLLATE NOCASE", arrayOf(specimenId.trim()),
            null, null, null
        )
        cursor.use {
            if (it.moveToFirst()) parseReferenceLabel(it) else null
        }
    }

    override fun getAllReferenceLabels(): List<org.crc.multimodal.evaluation.ReferenceLabel> = synchronized(lock) {
        val db = dbHelper.readableDatabase
        val cursor = db.query("reference_labels", null, null, null, null, null, "specimen_id ASC")
        cursor.use {
            val list = mutableListOf<org.crc.multimodal.evaluation.ReferenceLabel>()
            while (it.moveToNext()) {
                list.add(parseReferenceLabel(it))
            }
            list
        }
    }

    override fun deleteReferenceLabel(specimenId: String): Boolean = synchronized(lock) {
        val db = dbHelper.writableDatabase
        val rows = db.delete("reference_labels", "specimen_id = ? COLLATE NOCASE", arrayOf(specimenId.trim()))
        rows > 0
    }

    override fun clearReferenceLabels(): Int = synchronized(lock) {
        val db = dbHelper.writableDatabase
        db.delete("reference_labels", null, null)
    }

    override fun evaluateCohort(
        cohortId: String?,
        batchRepo: org.crc.multimodal.batch.BatchRepository?
    ): org.crc.multimodal.evaluation.CohortEvaluationSummary = synchronized(lock) {
        val allLabels = getAllReferenceLabels()
        val labels = if (!cohortId.isNullOrBlank()) {
            allLabels.filter { it.cohortId.equals(cohortId, ignoreCase = true) }
        } else {
            allLabels
        }
        val predictions = getUnifiedPredictionRecords(batchRepo)
        org.crc.multimodal.evaluation.CohortEvaluationEngine.evaluateCohort(
            referenceLabels = labels,
            predictions = predictions,
            cohortId = cohortId ?: "LOCAL_COHORT"
        )
    }

    override fun recordAuditEvent(event: org.crc.multimodal.security.AuditTrailRecord): Long = synchronized(lock) {
        val db = dbHelper.writableDatabase
        val cv = ContentValues().apply {
            put("event_timestamp", event.eventTimestamp)
            put("action_type", event.actionType.name)
            put("specimen_id", event.specimenId)
            put("details", event.details)
            put("sha256_fingerprint", event.sha256Fingerprint)
            put("performed_by", event.performedBy)
        }
        db.insert("audit_trail", null, cv)
    }

    override fun getAuditTrail(limit: Int): List<org.crc.multimodal.security.AuditTrailRecord> = synchronized(lock) {
        val db = dbHelper.readableDatabase
        val cursor = db.query(
            "audit_trail", null, null, null,
            null, null, "event_timestamp DESC, id DESC", limit.toString()
        )
        cursor.use {
            val list = mutableListOf<org.crc.multimodal.security.AuditTrailRecord>()
            while (it.moveToNext()) {
                list.add(parseAuditRecord(it))
            }
            list
        }
    }

    override fun getAuditTrailForSpecimen(specimenId: String): List<org.crc.multimodal.security.AuditTrailRecord> = synchronized(lock) {
        val db = dbHelper.readableDatabase
        val cursor = db.query(
            "audit_trail", null, "specimen_id = ? COLLATE NOCASE", arrayOf(specimenId.trim()),
            null, null, "event_timestamp DESC, id DESC"
        )
        cursor.use {
            val list = mutableListOf<org.crc.multimodal.security.AuditTrailRecord>()
            while (it.moveToNext()) {
                list.add(parseAuditRecord(it))
            }
            list
        }
    }

    override fun clearAuditTrail(): Int = synchronized(lock) {
        val db = dbHelper.writableDatabase
        db.delete("audit_trail", null, null)
    }

    private fun parseAuditRecord(cursor: Cursor): org.crc.multimodal.security.AuditTrailRecord {
        return org.crc.multimodal.security.AuditTrailRecord(
            id = cursor.getLong(cursor.getColumnIndexOrThrow("id")),
            eventTimestamp = cursor.getLong(cursor.getColumnIndexOrThrow("event_timestamp")),
            actionType = try {
                org.crc.multimodal.security.AuditActionType.valueOf(
                    cursor.getString(cursor.getColumnIndexOrThrow("action_type"))
                )
            } catch (e: Exception) {
                org.crc.multimodal.security.AuditActionType.SPECIMEN_INGESTED
            },
            specimenId = cursor.getString(cursor.getColumnIndexOrThrow("specimen_id")),
            details = cursor.getString(cursor.getColumnIndexOrThrow("details")),
            sha256Fingerprint = cursor.getString(cursor.getColumnIndexOrThrow("sha256_fingerprint")),
            performedBy = cursor.getString(cursor.getColumnIndexOrThrow("performed_by"))
        )
    }

    private fun parseReferenceLabel(cursor: Cursor): org.crc.multimodal.evaluation.ReferenceLabel {
        return org.crc.multimodal.evaluation.ReferenceLabel(
            specimenId = cursor.getString(cursor.getColumnIndexOrThrow("specimen_id")),
            groundTruthLabel = cursor.getInt(cursor.getColumnIndexOrThrow("ground_truth_label")),
            groundTruthClass = cursor.getString(cursor.getColumnIndexOrThrow("ground_truth_class")),
            cohortId = cursor.getString(cursor.getColumnIndexOrThrow("cohort_id")),
            source = cursor.getString(cursor.getColumnIndexOrThrow("source")),
            importedAt = cursor.getLong(cursor.getColumnIndexOrThrow("imported_at"))
        )
    }

    private class WorkstationDbHelper(context: Context) :
        SQLiteOpenHelper(context, "crc_workstation.db", null, 4) {

        override fun onCreate(db: SQLiteDatabase) {
            db.execSQL(
                """
                CREATE TABLE cases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_identifier TEXT UNIQUE NOT NULL,
                    specimen_description TEXT,
                    research_context TEXT,
                    status TEXT NOT NULL,
                    patch_count INTEGER NOT NULL,
                    protocol TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """.trimIndent()
            )
            db.execSQL("CREATE INDEX idx_cases_updated ON cases(updated_at DESC)")
            db.execSQL("CREATE INDEX idx_cases_status ON cases(status)")

            db.execSQL(
                """
                CREATE TABLE specimen_inputs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id INTEGER NOT NULL,
                    input_type TEXT,
                    local_uri TEXT,
                    filename TEXT,
                    validation_status TEXT,
                    created_at INTEGER NOT NULL,
                    FOREIGN KEY(case_id) REFERENCES cases(id) ON DELETE CASCADE
                )
                """.trimIndent()
            )
            db.execSQL("CREATE INDEX idx_inputs_case ON specimen_inputs(case_id)")

            db.execSQL(
                """
                CREATE TABLE inference_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id INTEGER NOT NULL,
                    model_identifier TEXT,
                    inference_architecture TEXT,
                    protocol TEXT NOT NULL,
                    predicted_class INTEGER NOT NULL,
                    predicted_label TEXT NOT NULL,
                    probability REAL NOT NULL,
                    logit REAL NOT NULL,
                    threshold REAL NOT NULL,
                    margin REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    num_patches INTEGER NOT NULL,
                    diagnostics_summary TEXT,
                    execution_timestamp INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    FOREIGN KEY(case_id) REFERENCES cases(id) ON DELETE CASCADE
                )
                """.trimIndent()
            )
            db.execSQL("CREATE INDEX idx_runs_case ON inference_runs(case_id)")

            db.execSQL(
                """
                CREATE TABLE review_notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id INTEGER NOT NULL,
                    category TEXT NOT NULL,
                    author_label TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    FOREIGN KEY(case_id) REFERENCES cases(id) ON DELETE CASCADE
                )
                """.trimIndent()
            )
            db.execSQL("CREATE INDEX idx_notes_case ON review_notes(case_id)")

            db.execSQL(
                """
                CREATE TABLE research_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id INTEGER UNIQUE NOT NULL,
                    inference_run_id INTEGER,
                    generated_at INTEGER NOT NULL,
                    report_content TEXT NOT NULL,
                    findings_summary TEXT NOT NULL,
                    FOREIGN KEY(case_id) REFERENCES cases(id) ON DELETE CASCADE
                )
                """.trimIndent()
            )

            db.execSQL(
                """
                CREATE TABLE IF NOT EXISTS report_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id TEXT UNIQUE NOT NULL,
                    case_id INTEGER NOT NULL,
                    inference_run_id INTEGER,
                    template_version TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    export_status TEXT NOT NULL,
                    file_uri TEXT,
                    file_path TEXT,
                    model_version TEXT,
                    sha256_hash TEXT,
                    error_message TEXT,
                    FOREIGN KEY(case_id) REFERENCES cases(id) ON DELETE CASCADE
                )
                """.trimIndent()
            )
            db.execSQL("CREATE INDEX IF NOT EXISTS idx_reports_case ON report_history(case_id, created_at DESC)")
            db.execSQL("CREATE UNIQUE INDEX IF NOT EXISTS idx_reports_uid ON report_history(report_id)")

            db.execSQL(
                """
                CREATE TABLE IF NOT EXISTS reference_labels (
                    specimen_id TEXT PRIMARY KEY,
                    ground_truth_label INTEGER NOT NULL,
                    ground_truth_class TEXT NOT NULL,
                    cohort_id TEXT NOT NULL,
                    source TEXT,
                    imported_at INTEGER NOT NULL
                )
                """.trimIndent()
            )
            db.execSQL("CREATE INDEX IF NOT EXISTS idx_ref_cohort ON reference_labels(cohort_id)")

            db.execSQL(
                """
                CREATE TABLE IF NOT EXISTS audit_trail (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_timestamp INTEGER NOT NULL,
                    action_type TEXT NOT NULL,
                    specimen_id TEXT,
                    details TEXT,
                    sha256_fingerprint TEXT,
                    performed_by TEXT NOT NULL
                )
                """.trimIndent()
            )
            db.execSQL("CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_trail(event_timestamp DESC)")
            db.execSQL("CREATE INDEX IF NOT EXISTS idx_audit_specimen ON audit_trail(specimen_id)")
        }

        override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
            if (oldVersion < 2) {
                db.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS reference_labels (
                        specimen_id TEXT PRIMARY KEY,
                        ground_truth_label INTEGER NOT NULL,
                        ground_truth_class TEXT NOT NULL,
                        cohort_id TEXT NOT NULL,
                        source TEXT,
                        imported_at INTEGER NOT NULL
                    )
                    """.trimIndent()
                )
                db.execSQL("CREATE INDEX IF NOT EXISTS idx_ref_cohort ON reference_labels(cohort_id)")
            }
            if (oldVersion < 3) {
                db.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS audit_trail (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_timestamp INTEGER NOT NULL,
                        action_type TEXT NOT NULL,
                        specimen_id TEXT,
                        details TEXT,
                        sha256_fingerprint TEXT,
                        performed_by TEXT NOT NULL
                    )
                    """.trimIndent()
                )
                db.execSQL("CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_trail(event_timestamp DESC)")
                db.execSQL("CREATE INDEX IF NOT EXISTS idx_audit_specimen ON audit_trail(specimen_id)")
            }
            if (oldVersion < 4) {
                db.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS report_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        report_id TEXT UNIQUE NOT NULL,
                        case_id INTEGER NOT NULL,
                        inference_run_id INTEGER,
                        template_version TEXT NOT NULL,
                        created_at INTEGER NOT NULL,
                        export_status TEXT NOT NULL,
                        file_uri TEXT,
                        file_path TEXT,
                        model_version TEXT,
                        sha256_hash TEXT,
                        error_message TEXT,
                        FOREIGN KEY(case_id) REFERENCES cases(id) ON DELETE CASCADE
                    )
                    """.trimIndent()
                )
                db.execSQL("CREATE INDEX IF NOT EXISTS idx_reports_case ON report_history(case_id, created_at DESC)")
                db.execSQL("CREATE UNIQUE INDEX IF NOT EXISTS idx_reports_uid ON report_history(report_id)")
            }
        }
    }
}
