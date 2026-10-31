package org.crc.multimodal.data

import org.crc.multimodal.inference.InferenceProtocol
import java.util.concurrent.atomic.AtomicLong

/**
 * Thread-safe in-memory repository implementation for unit testing and fast verification.
 */
class InMemoryWorkstationRepository : WorkstationRepository {

    private val caseIdGen = AtomicLong(1L)
    private val inputIdGen = AtomicLong(1L)
    private val runIdGen = AtomicLong(1L)
    private val noteIdGen = AtomicLong(1L)
    private val reportIdGen = AtomicLong(1L)

    private val casesMap = LinkedHashMap<Long, ResearchCase>()
    private val inputsMap = LinkedHashMap<Long, SpecimenInput>()
    private val runsMap = LinkedHashMap<Long, InferenceRun>()
    private val notesMap = LinkedHashMap<Long, ReviewNote>()
    private val reportsMap = LinkedHashMap<Long, ResearchReport>()
    private val reportHistoryMap = LinkedHashMap<String, PersistentReportRecord>()
    private val reportRecordIdGen = AtomicLong(1L)

    private val lock = Any()

    override fun insertCase(researchCase: ResearchCase): Long = synchronized(lock) {
        val existing = casesMap.values.firstOrNull {
            it.caseIdentifier.equals(researchCase.caseIdentifier.trim(), ignoreCase = true)
        }
        if (existing != null) {
            throw IllegalArgumentException("Case with identifier '${researchCase.caseIdentifier}' already exists.")
        }
        val id = caseIdGen.getAndIncrement()
        val created = researchCase.copy(id = id)
        casesMap[id] = created
        id
    }

    override fun updateCase(researchCase: ResearchCase): Boolean = synchronized(lock) {
        if (!casesMap.containsKey(researchCase.id)) return false
        val duplicate = casesMap.values.firstOrNull {
            it.id != researchCase.id && it.caseIdentifier.equals(researchCase.caseIdentifier.trim(), ignoreCase = true)
        }
        if (duplicate != null) {
            throw IllegalArgumentException("Another case with identifier '${researchCase.caseIdentifier}' already exists.")
        }
        casesMap[researchCase.id] = researchCase.copy(updatedAt = System.currentTimeMillis())
        true
    }

    override fun deleteCase(caseId: Long): Boolean = synchronized(lock) {
        if (!casesMap.containsKey(caseId)) return false
        casesMap.remove(caseId)
        inputsMap.values.filter { it.caseId == caseId }.forEach { inputsMap.remove(it.id) }
        runsMap.values.filter { it.caseId == caseId }.forEach { runsMap.remove(it.id) }
        notesMap.values.filter { it.caseId == caseId }.forEach { notesMap.remove(it.id) }
        reportsMap.values.filter { it.caseId == caseId }.forEach { reportsMap.remove(it.id) }
        reportHistoryMap.values.filter { it.caseId == caseId }.forEach { reportHistoryMap.remove(it.reportId) }
        true
    }

    override fun getCaseById(id: Long): ResearchCase? = synchronized(lock) {
        casesMap[id]
    }

    override fun getCaseByIdentifier(identifier: String): ResearchCase? = synchronized(lock) {
        casesMap.values.firstOrNull { it.caseIdentifier.equals(identifier.trim(), ignoreCase = true) }
    }

    override fun getAllCases(): List<ResearchCase> = synchronized(lock) {
        casesMap.values.toList()
    }

    override fun getCasesFiltered(
        status: CaseStatus?,
        searchQuery: String?,
        sortBy: CaseSortOption
    ): List<ResearchCase> = synchronized(lock) {
        var list = casesMap.values.asSequence()

        if (status != null) {
            list = list.filter { it.status == status }
        }

        if (!searchQuery.isNullOrBlank()) {
            val q = searchQuery.trim().lowercase()
            list = list.filter {
                it.caseIdentifier.lowercase().contains(q) ||
                        it.specimenDescription.lowercase().contains(q) ||
                        it.researchContext.lowercase().contains(q)
            }
        }

        val sorted = when (sortBy) {
            CaseSortOption.DATE_NEWEST -> list.sortedByDescending { it.updatedAt }
            CaseSortOption.DATE_OLDEST -> list.sortedBy { it.updatedAt }
            CaseSortOption.CASE_IDENTIFIER -> list.sortedBy { it.caseIdentifier }
        }

        sorted.toList()
    }

    override fun getRecentCases(limit: Int): List<ResearchCase> = synchronized(lock) {
        casesMap.values.sortedByDescending { it.updatedAt }.take(limit)
    }

    override fun getDashboardCounts(): DashboardCounts = synchronized(lock) {
        val all = casesMap.values
        DashboardCounts(
            totalCases = all.size,
            draftCount = all.count { it.status == CaseStatus.DRAFT },
            readyCount = all.count { it.status == CaseStatus.READY_FOR_ANALYSIS },
            analyzedCount = all.count { it.status == CaseStatus.ANALYZED },
            underReviewCount = all.count { it.status == CaseStatus.UNDER_REVIEW },
            reviewedCount = all.count { it.status == CaseStatus.REVIEWED }
        )
    }

    override fun insertSpecimenInput(input: SpecimenInput): Long = synchronized(lock) {
        val id = inputIdGen.getAndIncrement()
        inputsMap[id] = input.copy(id = id)
        id
    }

    override fun getInputsForCase(caseId: Long): List<SpecimenInput> = synchronized(lock) {
        inputsMap.values.filter { it.caseId == caseId }
    }

    override fun deleteInput(inputId: Long): Boolean = synchronized(lock) {
        inputsMap.remove(inputId) != null
    }

    override fun insertInferenceRun(run: InferenceRun): Long = synchronized(lock) {
        val id = runIdGen.getAndIncrement()
        runsMap[id] = run.copy(id = id)
        id
    }

    override fun getRunsForCase(caseId: Long): List<InferenceRun> = synchronized(lock) {
        runsMap.values.filter { it.caseId == caseId }.sortedByDescending { it.executionTimestamp }
    }

    override fun getLatestRunForCase(caseId: Long): InferenceRun? = synchronized(lock) {
        runsMap.values.filter { it.caseId == caseId }.maxByOrNull { it.executionTimestamp }
    }

    override fun getAllInferenceRuns(): List<InferenceRun> = synchronized(lock) {
        runsMap.values.sortedByDescending { it.executionTimestamp }
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
        val existingCase = casesMap[caseId]
            ?: throw IllegalStateException("Cannot persist inference result: Case $caseId not found.")
        val id = runIdGen.getAndIncrement()
        runsMap[id] = run.copy(id = id)
        casesMap[caseId] = existingCase.copy(
            status = newStatus,
            patchCount = run.numPatches,
            protocol = run.protocol,
            updatedAt = System.currentTimeMillis()
        )
        id
    }

    override fun recordInferenceFailure(caseId: Long, errorMessage: String): Unit = synchronized(lock) {
        val existingCase = casesMap[caseId]
        if (existingCase != null) {
            casesMap[caseId] = existingCase.copy(
                status = CaseStatus.READY_FOR_ANALYSIS,
                updatedAt = System.currentTimeMillis()
            )
        }
    }

    override fun recoverInterruptedCases(): Unit = synchronized(lock) {
        for ((id, cCase) in casesMap) {
            if (cCase.status == CaseStatus.ANALYZED) {
                val hasSuccessRun = runsMap.values.any { it.caseId == id && it.status == "SUCCESS" }
                if (!hasSuccessRun) {
                    casesMap[id] = cCase.copy(
                        status = CaseStatus.READY_FOR_ANALYSIS,
                        updatedAt = System.currentTimeMillis()
                    )
                }
            }
        }
    }

    override fun insertReviewNote(note: ReviewNote): Long = synchronized(lock) {
        val id = noteIdGen.getAndIncrement()
        notesMap[id] = note.copy(id = id)
        id
    }

    override fun updateReviewNote(note: ReviewNote): Boolean = synchronized(lock) {
        if (!notesMap.containsKey(note.id)) return false
        notesMap[note.id] = note.copy(updatedAt = System.currentTimeMillis())
        true
    }

    override fun deleteReviewNote(noteId: Long): Boolean = synchronized(lock) {
        notesMap.remove(noteId) != null
    }

    override fun getNotesForCase(caseId: Long): List<ReviewNote> = synchronized(lock) {
        notesMap.values.filter { it.caseId == caseId }.sortedByDescending { it.createdAt }
    }

    override fun insertOrUpdateReport(report: ResearchReport): Long = synchronized(lock) {
        val existing = reportsMap.values.firstOrNull { it.caseId == report.caseId }
        val id = existing?.id ?: reportIdGen.getAndIncrement()
        reportsMap[id] = report.copy(id = id, generatedAt = System.currentTimeMillis())
        id
    }

    override fun getReportForCase(caseId: Long): ResearchReport? = synchronized(lock) {
        reportsMap.values.firstOrNull { it.caseId == caseId }
    }

    override fun insertReportRecord(record: PersistentReportRecord): Long = synchronized(lock) {
        val id = reportRecordIdGen.getAndIncrement()
        val toStore = record.copy(id = id)
        reportHistoryMap[record.reportId] = toStore
        id
    }

    override fun updateReportRecordStatus(
        reportId: String,
        status: ReportExportStatus,
        fileUri: String?,
        filePath: String?,
        errorMessage: String?
    ): Boolean = synchronized(lock) {
        val existing = reportHistoryMap[reportId] ?: return false
        reportHistoryMap[reportId] = existing.copy(
            exportStatus = status,
            fileUri = fileUri ?: existing.fileUri,
            filePath = filePath ?: existing.filePath,
            errorMessage = errorMessage ?: existing.errorMessage
        )
        true
    }

    override fun getReportHistoryForCase(caseId: Long): List<PersistentReportRecord> = synchronized(lock) {
        reportHistoryMap.values
            .filter { it.caseId == caseId }
            .sortedByDescending { it.createdAt }
    }

    override fun getReportRecordById(reportId: String): PersistentReportRecord? = synchronized(lock) {
        reportHistoryMap[reportId]
    }

    override fun deleteReportRecord(reportId: String): Boolean = synchronized(lock) {
        reportHistoryMap.remove(reportId) != null
    }

    private val referenceLabelsMap = LinkedHashMap<String, org.crc.multimodal.evaluation.ReferenceLabel>()

    override fun insertReferenceLabel(label: org.crc.multimodal.evaluation.ReferenceLabel): Long = synchronized(lock) {
        val key = label.specimenId.trim().lowercase()
        referenceLabelsMap[key] = label
        1L
    }

    override fun insertReferenceLabels(labels: List<org.crc.multimodal.evaluation.ReferenceLabel>): Int = synchronized(lock) {
        var count = 0
        for (l in labels) {
            val key = l.specimenId.trim().lowercase()
            referenceLabelsMap[key] = l
            count++
        }
        count
    }

    override fun getReferenceLabel(specimenId: String): org.crc.multimodal.evaluation.ReferenceLabel? = synchronized(lock) {
        referenceLabelsMap[specimenId.trim().lowercase()]
    }

    override fun getAllReferenceLabels(): List<org.crc.multimodal.evaluation.ReferenceLabel> = synchronized(lock) {
        referenceLabelsMap.values.toList()
    }

    override fun deleteReferenceLabel(specimenId: String): Boolean = synchronized(lock) {
        referenceLabelsMap.remove(specimenId.trim().lowercase()) != null
    }

    override fun clearReferenceLabels(): Int = synchronized(lock) {
        val count = referenceLabelsMap.size
        referenceLabelsMap.clear()
        count
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
            cohortId = cohortId ?: "IN_MEMORY_COHORT"
        )
    }

    override fun clearAllData(): Unit = synchronized(lock) {
        casesMap.clear()
        inputsMap.clear()
        runsMap.clear()
        notesMap.clear()
        reportsMap.clear()
        reportHistoryMap.clear()
        referenceLabelsMap.clear()
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

    private val auditIdGen = AtomicLong(1L)
    private val auditList = mutableListOf<org.crc.multimodal.security.AuditTrailRecord>()

    override fun recordAuditEvent(event: org.crc.multimodal.security.AuditTrailRecord): Long = synchronized(lock) {
        val id = auditIdGen.getAndIncrement()
        val record = event.copy(id = id)
        auditList.add(record)
        id
    }

    override fun getAuditTrail(limit: Int): List<org.crc.multimodal.security.AuditTrailRecord> = synchronized(lock) {
        auditList.sortedWith(compareByDescending<org.crc.multimodal.security.AuditTrailRecord> { it.eventTimestamp }.thenByDescending { it.id }).take(limit)
    }

    override fun getAuditTrailForSpecimen(specimenId: String): List<org.crc.multimodal.security.AuditTrailRecord> = synchronized(lock) {
        auditList.filter { it.specimenId.equals(specimenId.trim(), ignoreCase = true) }
            .sortedWith(compareByDescending<org.crc.multimodal.security.AuditTrailRecord> { it.eventTimestamp }.thenByDescending { it.id })
    }

    override fun clearAuditTrail(): Int = synchronized(lock) {
        val count = auditList.size
        auditList.clear()
        count
    }
}
