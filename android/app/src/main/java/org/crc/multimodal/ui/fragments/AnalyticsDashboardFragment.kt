package org.crc.multimodal.ui.fragments

import android.content.Context
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.button.MaterialButton
import com.google.android.material.chip.ChipGroup
import com.google.android.material.textfield.TextInputEditText
import org.crc.multimodal.R
import org.crc.multimodal.analytics.AnalyticsFilterStatus
import org.crc.multimodal.analytics.AnalyticsSortOption
import org.crc.multimodal.analytics.ResearchAnalyticsEngine
import org.crc.multimodal.analytics.ResearchAnalyticsSummary
import org.crc.multimodal.analytics.SpecimenPredictionRecord
import org.crc.multimodal.ui.WorkstationNavigationHost
import org.crc.multimodal.ui.adapters.AnalyticsSpecimenAdapter
import java.util.Locale

class AnalyticsDashboardFragment : Fragment(R.layout.fragment_analytics_dashboard) {

    private var navHost: WorkstationNavigationHost? = null
    private lateinit var specimenAdapter: AnalyticsSpecimenAdapter
    private var allRecords: List<SpecimenPredictionRecord> = emptyList()

    private var currentSearchQuery: String = ""
    private var currentFilter: AnalyticsFilterStatus = AnalyticsFilterStatus.ALL
    private var currentSort: AnalyticsSortOption = AnalyticsSortOption.MARGIN_SMALLEST

    override fun onAttach(context: Context) {
        super.onAttach(context)
        if (context is WorkstationNavigationHost) {
            navHost = context
        }
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        navHost?.setTopToolbar(
            title = "Research Analytics",
            subtitle = "Population Metrics & Decision Margin Dashboard",
            showBackButton = true
        )

        val host = navHost ?: return
        val repository = host.getRepository()
        val batchRepository = host.getBatchRepository()

        // Bind Summary Metric Views
        val tvTotalSpecimens: TextView = view.findViewById(R.id.tvAnalyticsTotalSpecimens)
        val tvSuccessfulCount: TextView = view.findViewById(R.id.tvAnalyticsSuccessfulCount)
        val tvMsiSummary: TextView = view.findViewById(R.id.tvAnalyticsMsiSummary)
        val tvMssSummary: TextView = view.findViewById(R.id.tvAnalyticsMssSummary)

        val tvMinMargin: TextView = view.findViewById(R.id.tvAnalyticsMinMargin)
        val tvMeanMargin: TextView = view.findViewById(R.id.tvAnalyticsMeanMargin)
        val tvNearBoundaryCount: TextView = view.findViewById(R.id.tvAnalyticsNearBoundaryCount)
        val tvMarginBrackets: TextView = view.findViewById(R.id.tvAnalyticsMarginBrackets)

        val tvProbabilityBuckets: TextView = view.findViewById(R.id.tvAnalyticsProbabilityBuckets)

        val tvMedianLatency: TextView = view.findViewById(R.id.tvAnalyticsMedianLatency)
        val tvMeanLatency: TextView = view.findViewById(R.id.tvAnalyticsMeanLatency)
        val tvMeanPatches: TextView = view.findViewById(R.id.tvAnalyticsMeanPatches)
        val tvPatchDistribution: TextView = view.findViewById(R.id.tvAnalyticsPatchDistribution)

        // Bind Explorer Views
        val etSearch: TextInputEditText = view.findViewById(R.id.etAnalyticsSearch)
        val chipGroupFilter: ChipGroup = view.findViewById(R.id.chipGroupAnalyticsFilter)
        val tvCountBanner: TextView = view.findViewById(R.id.tvAnalyticsCountBanner)
        val btnSortToggle: MaterialButton = view.findViewById(R.id.btnAnalyticsSortToggle)
        val layoutEmpty: LinearLayout = view.findViewById(R.id.layoutAnalyticsEmpty)
        val rvSpecimens: RecyclerView = view.findViewById(R.id.rvAnalyticsSpecimens)

        // Setup RecyclerView
        rvSpecimens.layoutManager = LinearLayoutManager(requireContext())
        specimenAdapter = AnalyticsSpecimenAdapter(emptyList()) { record ->
            if (record.caseId != null) {
                host.navigateToCaseDetails(record.caseId)
            } else if (record.batchId != null) {
                host.navigateToBatchAnalysis(record.batchId)
            }
        }
        rvSpecimens.adapter = specimenAdapter

        // Fetch data
        val summary: ResearchAnalyticsSummary = repository.getResearchAnalyticsSummary(batchRepository)
        allRecords = repository.getUnifiedPredictionRecords(batchRepository)

        // Render Summary
        tvTotalSpecimens.text = summary.totalSpecimens.toString()
        tvSuccessfulCount.text = summary.totalSuccessful.toString()

        val msiPct = String.format(Locale.US, "%.1f", summary.msiProportion * 100)
        val mssPct = String.format(Locale.US, "%.1f", summary.mssProportion * 100)
        tvMsiSummary.text = "${summary.msiCount} ($msiPct%)"
        tvMssSummary.text = "${summary.mssCount} ($mssPct%)"

        if (summary.totalSuccessful > 0) {
            tvMinMargin.text = String.format(Locale.US, "%.4f", summary.marginMetrics.minMargin)
            tvMeanMargin.text = String.format(Locale.US, "%.4f", summary.marginMetrics.meanMargin)
            tvNearBoundaryCount.text = "${summary.marginMetrics.nearThresholdCount} cases"
        } else {
            tvMinMargin.text = "N/A"
            tvMeanMargin.text = "N/A"
            tvNearBoundaryCount.text = "0 cases"
        }

        // Margin bracket text
        val mDist = summary.marginMetrics.distribution
        tvMarginBrackets.text = buildString {
            append("[0.00-0.05]: ${mDist["[0.00 - 0.05]"] ?: 0}  ·  ")
            append("(0.05-0.15]: ${mDist["(0.05 - 0.15]"] ?: 0}  ·  ")
            append("(0.15-0.30]: ${mDist["(0.15 - 0.30]"] ?: 0}  ·  ")
            append(">0.30: ${mDist["> 0.30"] ?: 0}")
        }

        // Probability buckets text
        val pDist = summary.probabilityBuckets
        tvProbabilityBuckets.text = buildString {
            appendLine("[0.00 - 0.20): ${pDist["[0.00 - 0.20)"] ?: 0} specimens")
            appendLine("[0.20 - 0.40): ${pDist["[0.20 - 0.40)"] ?: 0} specimens")
            appendLine("[0.40 - 0.60): ${pDist["[0.40 - 0.60)"] ?: 0} specimens")
            appendLine("[0.60 - 0.80): ${pDist["[0.60 - 0.80)"] ?: 0} specimens")
            appendLine("[0.80 - 0.83) (Near-threshold MSS): ${pDist["[0.80 - 0.83)"] ?: 0} specimens")
            append("[0.83 - 1.00] (MSI Call): ${pDist["[0.83 - 1.00]"] ?: 0} specimens")
        }

        // Latency and patch metrics
        tvMedianLatency.text = String.format(Locale.US, "%.1f ms", summary.latencyMetrics.medianMs)
        tvMeanLatency.text = String.format(Locale.US, "%.1f ms", summary.latencyMetrics.meanMs)
        tvMeanPatches.text = String.format(Locale.US, "%.1f", summary.patchMetrics.meanPatches)

        val patchDist = summary.patchMetrics.distribution
        tvPatchDistribution.text = buildString {
            append("1-100: ${patchDist["1 - 100"] ?: 0}  ·  ")
            append("101-500: ${patchDist["101 - 500"] ?: 0}  ·  ")
            append("501-1000: ${patchDist["501 - 1000"] ?: 0}  ·  ")
            append(">1000: ${patchDist["> 1000"] ?: 0}")
        }

        // Setup Search Listener
        etSearch.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                currentSearchQuery = s?.toString().orEmpty()
                refreshExplorer(tvCountBanner, layoutEmpty)
            }
            override fun afterTextChanged(s: Editable?) {}
        })

        // Setup Filter Chips Listener
        chipGroupFilter.setOnCheckedChangeListener { _, checkedId ->
            currentFilter = when (checkedId) {
                R.id.chipFilterMsi -> AnalyticsFilterStatus.MSI_ONLY
                R.id.chipFilterMss -> AnalyticsFilterStatus.MSS_ONLY
                R.id.chipFilterNearBoundary -> AnalyticsFilterStatus.NEAR_THRESHOLD_ONLY
                R.id.chipFilterFailed -> AnalyticsFilterStatus.FAILED_OR_CANCELLED
                else -> AnalyticsFilterStatus.ALL
            }
            refreshExplorer(tvCountBanner, layoutEmpty)
        }

        // Setup Sort Toggle Listener
        btnSortToggle.setOnClickListener {
            currentSort = when (currentSort) {
                AnalyticsSortOption.MARGIN_SMALLEST -> {
                    btnSortToggle.text = "Sort: Largest Margin"
                    AnalyticsSortOption.MARGIN_LARGEST
                }
                AnalyticsSortOption.MARGIN_LARGEST -> {
                    btnSortToggle.text = "Sort: Date (Newest)"
                    AnalyticsSortOption.DATE_NEWEST
                }
                AnalyticsSortOption.DATE_NEWEST -> {
                    btnSortToggle.text = "Sort: Probability (Highest)"
                    AnalyticsSortOption.PROBABILITY_HIGHEST
                }
                AnalyticsSortOption.PROBABILITY_HIGHEST -> {
                    btnSortToggle.text = "Sort: Smallest Margin"
                    AnalyticsSortOption.MARGIN_SMALLEST
                }
                else -> {
                    btnSortToggle.text = "Sort: Smallest Margin"
                    AnalyticsSortOption.MARGIN_SMALLEST
                }
            }
            refreshExplorer(tvCountBanner, layoutEmpty)
        }

        btnSortToggle.text = "Sort: Smallest Margin"
        refreshExplorer(tvCountBanner, layoutEmpty)
    }

    private fun refreshExplorer(tvCountBanner: TextView, emptyLayout: LinearLayout) {
        val filtered = ResearchAnalyticsEngine.filterAndSort(
            records = allRecords,
            searchQuery = currentSearchQuery,
            filterStatus = currentFilter,
            sortOption = currentSort
        )

        tvCountBanner.text = "Showing ${filtered.size} of ${allRecords.size} specimens"

        if (filtered.isEmpty()) {
            emptyLayout.visibility = View.VISIBLE
            specimenAdapter.submitList(emptyList())
        } else {
            emptyLayout.visibility = View.GONE
            specimenAdapter.submitList(filtered)
        }
    }

    companion object {
        fun newInstance(): AnalyticsDashboardFragment {
            return AnalyticsDashboardFragment()
        }
    }
}
