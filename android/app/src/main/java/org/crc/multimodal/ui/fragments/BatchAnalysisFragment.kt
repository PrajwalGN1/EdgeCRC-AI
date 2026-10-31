package org.crc.multimodal.ui.fragments

import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.View
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.core.content.FileProvider
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import com.google.android.material.chip.ChipGroup
import com.google.android.material.textfield.TextInputEditText
import org.crc.multimodal.R
import org.crc.multimodal.batch.BatchCoordinator
import org.crc.multimodal.batch.ExportedReportBundle
import org.crc.multimodal.batch.BatchReportGenerator
import org.crc.multimodal.batch.BatchRun
import org.crc.multimodal.batch.BatchSpecimenItem
import org.crc.multimodal.batch.BatchSpecimenStatus
import org.crc.multimodal.batch.BatchStatus
import org.crc.multimodal.inference.BundleStorageManager
import org.crc.multimodal.inference.CrcLogger
import org.crc.multimodal.inference.InferenceProtocol
import org.crc.multimodal.inference.ThresholdLogic
import org.crc.multimodal.security.AuditActionType
import org.crc.multimodal.security.AuditTrailRecord
import org.crc.multimodal.ui.WorkstationNavigationHost
import org.crc.multimodal.ui.adapters.BatchSpecimenAdapter
import java.io.File
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class BatchAnalysisFragment : Fragment(R.layout.fragment_batch_analysis) {

    private var navHost: WorkstationNavigationHost? = null
    private var currentBatch: BatchRun? = null
    private lateinit var adapter: BatchSpecimenAdapter
    private val inferenceExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val backgroundExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    @Volatile
    private var cancellationRequested = false

    private var pendingExportFile: File? = null

    private val selectFilesLauncher = registerForActivityResult(
        ActivityResultContracts.OpenMultipleDocuments()
    ) { uris ->
        if (uris != null && uris.isNotEmpty()) {
            handleSelectedUris(uris)
        }
    }

    private lateinit var savePdfLauncher: ActivityResultLauncher<String>

    companion object {
        private const val ARG_BATCH_ID = "arg_batch_id"

        fun newInstance(batchId: String? = null): BatchAnalysisFragment {
            return BatchAnalysisFragment().apply {
                arguments = Bundle().apply {
                    if (batchId != null) putString(ARG_BATCH_ID, batchId)
                }
            }
        }
    }

    override fun onAttach(context: Context) {
        super.onAttach(context)
        if (context is WorkstationNavigationHost) {
            navHost = context
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        savePdfLauncher = registerForActivityResult(ActivityResultContracts.CreateDocument("application/pdf")) { destinationUri: Uri? ->
            if (destinationUri != null) {
                handleSavePdfResult(destinationUri)
            } else {
                Toast.makeText(requireContext(), "Save operation cancelled", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        navHost?.setTopToolbar(
            title = getString(R.string.title_batch_analysis),
            subtitle = getString(R.string.subtitle_batch_analysis),
            showBackButton = true
        )

        val rvSpecimens: RecyclerView = view.findViewById(R.id.rvBatchSpecimens)
        rvSpecimens.layoutManager = LinearLayoutManager(requireContext())
        adapter = BatchSpecimenAdapter(
            onItemClickListener = { item ->
                if (adapter.isSelectionMode) {
                    adapter.toggleSelection(item.specimenId)
                } else {
                    showSpecimenDetailDialog(item)
                }
            }
        )
        rvSpecimens.adapter = adapter

        // Basic Batch Action Buttons
        val btnSelectFiles: MaterialButton = view.findViewById(R.id.btnSelectBatchFiles)
        val btnStartBatch: MaterialButton = view.findViewById(R.id.btnStartBatch)
        val btnCancelBatch: MaterialButton = view.findViewById(R.id.btnCancelBatch)
        val btnExportReport: MaterialButton = view.findViewById(R.id.btnExportReport)

        // Filter and Search
        val chipGroupFilter: ChipGroup = view.findViewById(R.id.chipGroupFilter)
        val etBatchSearch: TextInputEditText = view.findViewById(R.id.etBatchSearch)

        // Multi-Selection Controls
        val btnToggleSelection: MaterialButton = view.findViewById(R.id.btnToggleSelection)
        val cardSelectionToolbar: MaterialCardView = view.findViewById(R.id.cardSelectionToolbar)
        val tvBatchSelectedCount: TextView = view.findViewById(R.id.tvBatchSelectedCount)
        val btnBatchSelectAll: MaterialButton = view.findViewById(R.id.btnBatchSelectAll)
        val btnBatchClearSelection: MaterialButton = view.findViewById(R.id.btnBatchClearSelection)
        val btnBulkExportPdf: MaterialButton = view.findViewById(R.id.btnBulkExportPdf)
        val btnBulkArchive: MaterialButton = view.findViewById(R.id.btnBulkArchive)
        val btnBulkDelete: MaterialButton = view.findViewById(R.id.btnBulkDelete)

        btnSelectFiles.setOnClickListener {
            selectFilesLauncher.launch(arrayOf("*/*"))
        }

        btnStartBatch.setOnClickListener {
            val batch = currentBatch ?: return@setOnClickListener
            val classifier = navHost?.getClassifier()
            if (classifier == null || !navHost!!.isClassifierReady()) {
                Toast.makeText(requireContext(), "Inference engine is still initializing. Please wait...", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            startBatchExecution(batch)
        }

        btnCancelBatch.setOnClickListener {
            cancellationRequested = true
            Toast.makeText(requireContext(), "Cancellation requested. Halting after current specimen...", Toast.LENGTH_SHORT).show()
        }

        btnExportReport.setOnClickListener {
            val batch = currentBatch ?: return@setOnClickListener
            exportBatchReport(batch)
        }

        // Selection Listener
        adapter.onSelectionChangedListener = { count, _ ->
            tvBatchSelectedCount.text = "$count selected"
            val hasSelection = count > 0
            btnBulkExportPdf.isEnabled = hasSelection
            btnBulkArchive.isEnabled = hasSelection
            btnBulkDelete.isEnabled = hasSelection
        }

        // Toggle Selection Mode
        btnToggleSelection.setOnClickListener {
            val enterMode = !adapter.isSelectionMode
            adapter.setSelectionMode(enterMode)
            btnToggleSelection.text = if (enterMode) "Exit" else "Select"
            cardSelectionToolbar.visibility = if (enterMode) View.VISIBLE else View.GONE
            if (!enterMode) {
                tvBatchSelectedCount.text = "0 selected"
            }
        }

        // Select All / Clear Selection
        btnBatchSelectAll.setOnClickListener {
            adapter.selectAll()
        }

        btnBatchClearSelection.setOnClickListener {
            adapter.clearSelection()
        }

        // Search Query Listener
        etBatchSearch.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                adapter.setSearchQuery(s?.toString() ?: "")
            }
            override fun afterTextChanged(s: Editable?) {}
        })

        // Status Filter Chips
        chipGroupFilter.setOnCheckedChangeListener { _, checkedId ->
            when (checkedId) {
                R.id.chipFilterMsi -> {
                    adapter.setFilter(BatchSpecimenAdapter.FilterOption.MSI_ONLY)
                    btnBulkArchive.text = "Archive"
                }
                R.id.chipFilterMss -> {
                    adapter.setFilter(BatchSpecimenAdapter.FilterOption.MSS_ONLY)
                    btnBulkArchive.text = "Archive"
                }
                R.id.chipFilterExcluded -> {
                    adapter.setFilter(BatchSpecimenAdapter.FilterOption.EXCLUDED_ONLY)
                    btnBulkArchive.text = "Archive"
                }
                R.id.chipFilterArchived -> {
                    adapter.setFilter(BatchSpecimenAdapter.FilterOption.ARCHIVED_ONLY)
                    btnBulkArchive.text = "Restore"
                }
                else -> {
                    adapter.setFilter(BatchSpecimenAdapter.FilterOption.ALL)
                    btnBulkArchive.text = "Archive"
                }
            }
        }

        // Bulk Archive / Restore Action
        btnBulkArchive.setOnClickListener {
            val batch = currentBatch ?: return@setOnClickListener
            val selectedItems = adapter.getSelectedItems()
            if (selectedItems.isEmpty()) {
                Toast.makeText(requireContext(), "No specimens selected", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val isArchivedFilter = chipGroupFilter.checkedChipId == R.id.chipFilterArchived
            if (isArchivedFilter) {
                AlertDialog.Builder(requireContext())
                    .setTitle("Restore Selected Specimens")
                    .setMessage("Restore ${selectedItems.size} specimen(s) to the active batch list and cohort metrics?")
                    .setPositiveButton("Restore") { _, _ ->
                        performRestoreSpecimens(batch, selectedItems)
                    }
                    .setNegativeButton("Cancel", null)
                    .show()
            } else {
                AlertDialog.Builder(requireContext())
                    .setTitle("Archive Selected Specimens")
                    .setMessage(
                        "Archive ${selectedItems.size} specimen(s)?\n\n" +
                        "Archived specimens will be removed from the active batch list. " +
                        "Their inference records remain preserved and recoverable under the 'Archived' filter."
                    )
                    .setPositiveButton("Archive") { _, _ ->
                        performArchiveSpecimens(batch, selectedItems)
                    }
                    .setNegativeButton("Cancel", null)
                    .show()
            }
        }

        // Bulk Permanent Deletion Action
        btnBulkDelete.setOnClickListener {
            val batch = currentBatch ?: return@setOnClickListener
            val selectedItems = adapter.getSelectedItems()
            if (selectedItems.isEmpty()) {
                Toast.makeText(requireContext(), "No specimens selected", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val idList = selectedItems.take(5).joinToString(", ") { it.specimenId }
            val extraCount = if (selectedItems.size > 5) " and ${selectedItems.size - 5} more" else ""

            AlertDialog.Builder(requireContext())
                .setTitle("Permanently Delete Specimens?")
                .setMessage(
                    "WARNING: You are about to permanently delete ${selectedItems.size} specimen(s):\n" +
                    "$idList$extraCount\n\n" +
                    "The following data will be PERMANENTLY REMOVED:\n" +
                    "• Batch specimen records and inference results\n" +
                    "• Associated shared Cases registry entry and inference runs\n" +
                    "• Specimen tensor bundles\n\n" +
                    "Shared ONNX models, neural network weights, and historical audit logs will NOT be affected.\n\n" +
                    "This operation CANNOT be undone. Are you sure you want to proceed?"
                )
                .setPositiveButton("Permanently Delete") { _, _ ->
                    performDeleteSpecimens(batch, selectedItems)
                }
                .setNegativeButton("Cancel", null)
                .show()
        }

        // Bulk PDF Export Action
        btnBulkExportPdf.setOnClickListener {
            val batch = currentBatch ?: return@setOnClickListener
            val selectedItems = adapter.getSelectedItems()
            if (selectedItems.isEmpty()) {
                Toast.makeText(requireContext(), "No specimens selected", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val eligibleItems = selectedItems.filter {
                it.status == BatchSpecimenStatus.COMPLETED && it.probability != null
            }
            val ineligibleItems = selectedItems.filter {
                it.status != BatchSpecimenStatus.COMPLETED || it.probability == null
            }

            if (eligibleItems.isEmpty()) {
                AlertDialog.Builder(requireContext())
                    .setTitle("Cannot Export PDF")
                    .setMessage("None of the selected specimens have completed inference results. Rejected, pending, or failed specimens cannot be exported.")
                    .setPositiveButton("OK", null)
                    .show()
                return@setOnClickListener
            }

            if (ineligibleItems.isNotEmpty()) {
                val ineligiblesText = ineligibleItems.take(3).joinToString(", ") { it.specimenId } +
                        (if (ineligibleItems.size > 3) " and ${ineligibleItems.size - 3} more" else "")
                AlertDialog.Builder(requireContext())
                    .setTitle("Ineligible Specimens Selected")
                    .setMessage(
                        "Selected specimens contain ${ineligibleItems.size} item(s) without valid inference results ($ineligiblesText).\n\n" +
                        "Bulk export will proceed with ${eligibleItems.size} eligible specimen(s).\n\n" +
                        "Do you want to continue?"
                    )
                    .setPositiveButton("Proceed") { _, _ ->
                        performBulkExportPdf(batch, eligibleItems)
                    }
                    .setNegativeButton("Cancel", null)
                    .show()
            } else {
                performBulkExportPdf(batch, eligibleItems)
            }
        }

        // Restore batch: explicit ID if passed, or most recent batch deterministically
        val repo = navHost?.getBatchRepository()
        if (repo != null) {
            val initialBatchId = arguments?.getString(ARG_BATCH_ID)
            val batchToDisplay = BatchCoordinator.resolveBatchForDisplay(repo, initialBatchId)
            if (batchToDisplay != null) {
                displayBatch(batchToDisplay)
            }
        }
    }

    private fun handleSelectedUris(uris: List<android.net.Uri>) {
        val host = navHost ?: return
        val coordinator = BatchCoordinator(
            repository = host.getBatchRepository(),
            workstationRepository = host.getRepository()
        )

        try {
            val batchRun = coordinator.ingestUris(
                context = requireContext(),
                uris = uris,
                title = "Batch Evaluation (${uris.size} files)"
            )
            displayBatch(batchRun)
        } catch (e: Exception) {
            Toast.makeText(requireContext(), "Failed to ingest files: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun displayBatch(batchRun: BatchRun) {
        currentBatch = batchRun
        val view = view ?: return

        // Synchronize completed specimens with shared Cases registry
        val host = navHost
        if (host != null) {
            BatchCoordinator.syncBatchToCases(requireContext(), batchRun, host.getRepository())
        }

        val layoutSummary: LinearLayout = view.findViewById(R.id.layoutIngestSummary)
        val tvCounts: TextView = view.findViewById(R.id.tvIngestSummaryCounts)
        val tvBatchId: TextView = view.findViewById(R.id.tvBatchIdLabel)
        val btnStart: MaterialButton = view.findViewById(R.id.btnStartBatch)
        val tvEmpty: TextView = view.findViewById(R.id.tvEmptyBatchList)
        val cardSummary: MaterialCardView = view.findViewById(R.id.cardBatchSummary)
        val cardProgress: MaterialCardView = view.findViewById(R.id.cardBatchProgress)
        val tvSummaryArchivedCounts: TextView? = view.findViewById(R.id.tvSummaryArchivedCounts)

        layoutSummary.visibility = View.VISIBLE
        val stats = batchRun.statistics
        tvCounts.text = "Selected: ${stats.totalSelected} | Accepted: ${stats.totalAccepted} | Rejected: ${stats.totalRejected}"
        tvBatchId.text = "Batch ID: ${batchRun.batchId}"

        adapter.setItems(batchRun.items)
        tvEmpty.visibility = if (batchRun.items.isEmpty()) View.VISIBLE else View.GONE

        // Enable execution button if there are validated items ready to process
        val hasPending = batchRun.items.any { it.status == BatchSpecimenStatus.VALIDATED }
        btnStart.isEnabled = hasPending

        // If completed or partially completed, show summary metrics card
        if (batchRun.status == BatchStatus.COMPLETED || batchRun.status == BatchStatus.PARTIALLY_COMPLETED) {
            cardSummary.visibility = View.VISIBLE
            cardProgress.visibility = View.GONE
            val tvMsi: TextView = view.findViewById(R.id.tvSummaryMsiCounts)
            val tvLat: TextView = view.findViewById(R.id.tvSummaryLatencies)

            val msiPct = String.format(Locale.US, "%.1f", stats.msiProportion * 100)
            val mssPct = String.format(Locale.US, "%.1f", stats.mssProportion * 100)
            tvMsi.text = "MSI: ${stats.msiCount} ($msiPct%) | MSS: ${stats.mssCount} ($mssPct%)"
            tvLat.text = "Mean Latency: ${String.format(Locale.US, "%.1f", stats.meanLatencyMs)}ms | Median: ${String.format(Locale.US, "%.1f", stats.medianLatencyMs)}ms | Duration: ${stats.totalDurationMs}ms"

            if (stats.totalArchived > 0) {
                tvSummaryArchivedCounts?.visibility = View.VISIBLE
                tvSummaryArchivedCounts?.text = "Archived Specimens: ${stats.totalArchived} (Excluded from active cohort)"
            } else {
                tvSummaryArchivedCounts?.visibility = View.GONE
            }
        }
    }

    private fun startBatchExecution(batchRun: BatchRun) {
        val host = navHost ?: return
        val classifier = host.getClassifier() ?: return
        val coordinator = BatchCoordinator(
            repository = host.getBatchRepository(),
            workstationRepository = host.getRepository()
        )
        val view = view ?: return

        cancellationRequested = false
        val cardProgress: MaterialCardView = view.findViewById(R.id.cardBatchProgress)
        val cardSummary: MaterialCardView = view.findViewById(R.id.cardBatchSummary)
        val btnStart: MaterialButton = view.findViewById(R.id.btnStartBatch)
        val progressBar: ProgressBar = view.findViewById(R.id.progressBarBatch)
        val tvStatus: TextView = view.findViewById(R.id.tvBatchProgressStatus)
        val tvCounts: TextView = view.findViewById(R.id.tvBatchProgressCounts)

        cardProgress.visibility = View.VISIBLE
        cardSummary.visibility = View.GONE
        btnStart.isEnabled = false
        progressBar.progress = 0

        inferenceExecutor.execute {
            val finalizedBatch = coordinator.executeBatchInference(
                context = requireContext(),
                batchRun = batchRun,
                classifier = classifier,
                onProgress = { completed, total, currentSpecimen ->
                    activity?.runOnUiThread {
                        val pct = if (total > 0) (completed * 100) / total else 0
                        progressBar.progress = pct
                        tvStatus.text = "Processing specimen ($completed/$total): $currentSpecimen"
                        tvCounts.text = "Completed: $completed | Remaining: ${total - completed}"
                    }
                },
                isCancelled = { cancellationRequested }
            )

            activity?.runOnUiThread {
                displayBatch(finalizedBatch)
                Toast.makeText(
                    requireContext(),
                    "Batch execution finished: ${finalizedBatch.status.name}",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }
    }

    private fun performArchiveSpecimens(batch: BatchRun, items: List<BatchSpecimenItem>) {
        val host = navHost ?: return
        val batchRepo = host.getBatchRepository()
        val caseRepo = host.getRepository()

        backgroundExecutor.execute {
            var successCount = 0
            var failCount = 0
            for (item in items) {
                try {
                    val ok = batchRepo.archiveBatchItem(batch.batchId, item.specimenId)
                    if (ok) {
                        caseRepo.recordAuditEvent(
                            AuditTrailRecord(
                                actionType = AuditActionType.SPECIMEN_ARCHIVED,
                                specimenId = item.specimenId,
                                details = "Archived specimen ${item.specimenId} from active batch ${batch.batchId}"
                            )
                        )
                        successCount++
                    } else {
                        failCount++
                    }
                } catch (t: Throwable) {
                    failCount++
                    CrcLogger.e("Failed to archive specimen ${item.specimenId}", t)
                }
            }

            val updatedBatch = batchRepo.getBatchRun(batch.batchId)
            activity?.runOnUiThread {
                if (!isAdded || isDetached) return@runOnUiThread
                if (updatedBatch != null) {
                    displayBatch(updatedBatch)
                }
                adapter.clearSelection()
                Toast.makeText(
                    requireContext(),
                    if (failCount == 0) "$successCount specimen(s) archived" else "$successCount archived · $failCount failed",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }
    }

    private fun performRestoreSpecimens(batch: BatchRun, items: List<BatchSpecimenItem>) {
        val host = navHost ?: return
        val batchRepo = host.getBatchRepository()
        val caseRepo = host.getRepository()

        backgroundExecutor.execute {
            var successCount = 0
            var failCount = 0
            for (item in items) {
                try {
                    val ok = batchRepo.unarchiveBatchItem(batch.batchId, item.specimenId)
                    if (ok) {
                        caseRepo.recordAuditEvent(
                            AuditTrailRecord(
                                actionType = AuditActionType.SPECIMEN_RESTORED,
                                specimenId = item.specimenId,
                                details = "Restored specimen ${item.specimenId} to active batch ${batch.batchId}"
                            )
                        )
                        successCount++
                    } else {
                        failCount++
                    }
                } catch (t: Throwable) {
                    failCount++
                    CrcLogger.e("Failed to restore specimen ${item.specimenId}", t)
                }
            }

            val updatedBatch = batchRepo.getBatchRun(batch.batchId)
            activity?.runOnUiThread {
                if (!isAdded || isDetached) return@runOnUiThread
                if (updatedBatch != null) {
                    displayBatch(updatedBatch)
                }
                adapter.clearSelection()
                Toast.makeText(
                    requireContext(),
                    if (failCount == 0) "$successCount specimen(s) restored" else "$successCount restored · $failCount failed",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }
    }

    private fun performDeleteSpecimens(batch: BatchRun, items: List<BatchSpecimenItem>) {
        val host = navHost ?: return
        val batchRepo = host.getBatchRepository()
        val caseRepo = host.getRepository()
        val appContext = requireContext().applicationContext

        backgroundExecutor.execute {
            var successCount = 0
            var failCount = 0
            for (item in items) {
                try {
                    // 1. Delete from batch runs DB
                    batchRepo.deleteBatchItem(batch.batchId, item.specimenId)

                    // 2. Delete from shared Cases repository if present
                    val caseEntity = caseRepo.getCaseByIdentifier(item.specimenId)
                    if (caseEntity != null) {
                        caseRepo.deleteCase(caseEntity.id)
                    }

                    // 3. Delete tensor bundle if owned exclusively
                    BundleStorageManager.deleteBundle(appContext, item.specimenId)

                    // 4. Record audit event
                    caseRepo.recordAuditEvent(
                        AuditTrailRecord(
                            actionType = AuditActionType.SPECIMEN_DELETED,
                            specimenId = item.specimenId,
                            details = "Permanently deleted specimen ${item.specimenId} from batch ${batch.batchId} and shared cases registry"
                        )
                    )
                    successCount++
                } catch (t: Throwable) {
                    failCount++
                    CrcLogger.e("Failed to delete specimen ${item.specimenId}", t)
                }
            }

            val updatedBatch = batchRepo.getBatchRun(batch.batchId)
            activity?.runOnUiThread {
                if (!isAdded || isDetached) return@runOnUiThread
                if (updatedBatch != null) {
                    displayBatch(updatedBatch)
                }
                adapter.clearSelection()
                Toast.makeText(
                    requireContext(),
                    if (failCount == 0) "$successCount specimen(s) permanently deleted" else "$successCount deleted · $failCount failed",
                    Toast.LENGTH_LONG
                ).show()
            }
        }
    }

    private fun performBulkExportPdf(batch: BatchRun, eligibleItems: List<BatchSpecimenItem>) {
        val appContext = requireContext().applicationContext

        Toast.makeText(requireContext(), "Generating combined PDF for ${eligibleItems.size} specimen(s)…", Toast.LENGTH_SHORT).show()

        backgroundExecutor.execute {
            try {
                val reportBundle = BatchReportGenerator.exportCombinedReport(
                    context = appContext,
                    batchId = batch.batchId,
                    selectedItems = eligibleItems
                )

                activity?.runOnUiThread {
                    if (!isAdded || isDetached) return@runOnUiThread
                    showBulkExportSuccessDialog(batch, eligibleItems, reportBundle)
                }
            } catch (t: Throwable) {
                CrcLogger.e("Failed to generate bulk PDF", t)
                activity?.runOnUiThread {
                    if (!isAdded || isDetached) return@runOnUiThread
                    Toast.makeText(requireContext(), "Export failed: ${t.localizedMessage}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun showBulkExportSuccessDialog(
        batch: BatchRun,
        items: List<BatchSpecimenItem>,
        bundle: ExportedReportBundle
    ) {
        val pdfFile = bundle.pdfFile
        if (pdfFile == null || !pdfFile.exists()) {
            Toast.makeText(requireContext(), "PDF report generation failed", Toast.LENGTH_LONG).show()
            return
        }

        val dialogBuilder = AlertDialog.Builder(requireContext())
        dialogBuilder.setTitle("Bulk PDF Report Exported")

        val message = StringBuilder()
        message.append("Batch ID: ${bundle.batchId}\n")
        message.append("Specimens Exported: ${items.size}\n\n")
        message.append("PDF Report:\n${pdfFile.name}\n")
        message.append("Size: ${pdfFile.length() / 1024} KB\n")
        message.append("SHA-256 Digest:\n${bundle.pdfSha256?.take(24)}…\n\n")
        message.append("Integrity Manifest:\n${bundle.manifestFile.name}")

        dialogBuilder.setMessage(message.toString())

        dialogBuilder.setPositiveButton("Preview PDF") { _, _ ->
            previewPdfFile(pdfFile)
        }
        dialogBuilder.setNeutralButton("Save PDF (SAF)") { _, _ ->
            pendingExportFile = pdfFile
            savePdfLauncher.launch(pdfFile.name)
        }
        dialogBuilder.setNegativeButton("Share") { _, _ ->
            sharePdfFile(pdfFile, "CRC Batch Report - ${batch.batchId}")
        }

        dialogBuilder.show()
    }

    private fun previewPdfFile(file: File) {
        try {
            val uri = FileProvider.getUriForFile(
                requireContext(),
                "org.crc.multimodal.fileprovider",
                file
            )
            val intent = Intent(Intent.ACTION_VIEW).apply {
                setDataAndType(uri, "application/pdf")
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            startActivity(intent)
        } catch (e: ActivityNotFoundException) {
            AlertDialog.Builder(requireContext())
                .setTitle("No PDF Viewer Found")
                .setMessage("No application is installed to preview PDF files. You can save or share the file to view it.")
                .setPositiveButton("OK", null)
                .show()
        } catch (t: Throwable) {
            Toast.makeText(requireContext(), "Failed to open preview: ${t.localizedMessage}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun sharePdfFile(file: File, subject: String) {
        try {
            val uri = FileProvider.getUriForFile(
                requireContext(),
                "org.crc.multimodal.fileprovider",
                file
            )
            val shareIntent = Intent(Intent.ACTION_SEND).apply {
                type = "application/pdf"
                putExtra(Intent.EXTRA_STREAM, uri)
                putExtra(Intent.EXTRA_SUBJECT, subject)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivity(Intent.createChooser(shareIntent, "Share Batch PDF Report"))
        } catch (t: Throwable) {
            Toast.makeText(requireContext(), "Failed to share PDF: ${t.localizedMessage}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun handleSavePdfResult(destinationUri: Uri) {
        val sourceFile = pendingExportFile ?: return
        val host = navHost ?: return
        val contentResolver = requireContext().contentResolver

        Toast.makeText(requireContext(), "Saving PDF to selected destination…", Toast.LENGTH_SHORT).show()

        backgroundExecutor.execute {
            try {
                contentResolver.openOutputStream(destinationUri)?.use { outStream ->
                    sourceFile.inputStream().use { inStream ->
                        inStream.copyTo(outStream)
                    }
                } ?: throw IllegalStateException("Could not open output stream for selected destination.")

                host.getRepository().recordAuditEvent(
                    AuditTrailRecord(
                        actionType = AuditActionType.REPORT_COMPILED,
                        specimenId = currentBatch?.batchId,
                        details = "Bulk PDF report successfully saved via SAF to: $destinationUri"
                    )
                )

                activity?.runOnUiThread {
                    if (!isAdded || isDetached) return@runOnUiThread
                    Toast.makeText(requireContext(), "Report successfully saved!", Toast.LENGTH_SHORT).show()
                }
            } catch (t: Throwable) {
                CrcLogger.e("Failed to save PDF via SAF: $destinationUri", t)
                activity?.runOnUiThread {
                    if (!isAdded || isDetached) return@runOnUiThread
                    Toast.makeText(requireContext(), "Failed to save PDF: ${t.localizedMessage}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun showSpecimenDetailDialog(item: BatchSpecimenItem) {
        val host = navHost
        val caseRepo = host?.getRepository()
        val dialogBuilder = AlertDialog.Builder(requireContext())
        dialogBuilder.setTitle("Specimen: ${item.specimenId}")

        val info = StringBuilder()
        info.append("Status: ${item.status.name}\n")
        if (item.isArchived) {
            info.append("Archived: Yes (Excluded from active cohort)\n")
        }
        if (item.sourceUri.isNotBlank()) {
            info.append("Source: ${item.sourceUri}\n")
        }
        if (item.patchCount > 0) {
            info.append("Patch Count: ${item.patchCount}\n")
            info.append("Protocol: ${item.protocol}\n")
        }
        if (item.probability != null) {
            val prob = item.probability
            val label = if (prob >= ThresholdLogic.OPERATING_THRESHOLD) "MSI" else "MSS"
            info.append("Prediction: $label (P = ${String.format(Locale.US, "%.4f", prob)})\n")
            info.append("Inference Latency: ${item.inferenceLatencyMs} ms\n")
        }
        val err = item.validationError ?: item.errorDetails
        if (!err.isNullOrBlank()) {
            info.append("\nNote / Rejection: $err\n")
        }

        dialogBuilder.setMessage(info.toString())

        if (caseRepo != null && (item.status == BatchSpecimenStatus.COMPLETED || item.status == BatchSpecimenStatus.REJECTED)) {
            val caseEntity = caseRepo.getCaseByIdentifier(item.specimenId)
            if (caseEntity != null) {
                dialogBuilder.setPositiveButton("View in Cases") { _, _ ->
                    host.navigateToCaseDetails(caseEntity.id)
                }
                if (item.status == BatchSpecimenStatus.COMPLETED) {
                    dialogBuilder.setNeutralButton("View Case Report") { _, _ ->
                        host.navigateToReport(caseEntity.id)
                    }
                }
            }
        }

        dialogBuilder.setNegativeButton("Close", null)
        dialogBuilder.show()
    }

    private fun exportBatchReport(batchRun: BatchRun) {
        try {
            val reportBundle = BatchReportGenerator.exportReport(requireContext(), batchRun)
            val message = StringBuilder()
            message.append("Batch ID: ${reportBundle.batchId}\n\n")
            message.append("Report JSON:\n${reportBundle.jsonFile.name}\n")
            message.append("SHA-256: ${reportBundle.jsonSha256.take(20)}…\n\n")

            if (reportBundle.pdfFile != null) {
                message.append("Report PDF:\n${reportBundle.pdfFile.name}\n")
                message.append("SHA-256: ${reportBundle.pdfSha256?.take(20) ?: "N/A"}…\n\n")
            }

            message.append("Integrity Manifest:\n${reportBundle.manifestFile.name}\n\n")
            message.append("All artifacts saved to app storage:\n${reportBundle.manifestFile.parentFile?.absolutePath}")

            val dialog = AlertDialog.Builder(requireContext())
                .setTitle("Research Report Exported")
                .setMessage(message.toString())
                .setPositiveButton("OK", null)

            if (reportBundle.pdfFile != null) {
                dialog.setNeutralButton("Preview PDF") { _, _ ->
                    previewPdfFile(reportBundle.pdfFile)
                }
            }

            dialog.show()

        } catch (e: Exception) {
            Toast.makeText(requireContext(), "Failed to export report: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        cancellationRequested = true
    }

    override fun onDestroy() {
        super.onDestroy()
        inferenceExecutor.shutdownNow()
        backgroundExecutor.shutdownNow()
    }
}

