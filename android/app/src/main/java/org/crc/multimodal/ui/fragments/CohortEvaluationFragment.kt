package org.crc.multimodal.ui.fragments

import android.content.Context
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageButton
import android.widget.TextView
import android.widget.Toast
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.button.MaterialButton
import com.google.android.material.chip.ChipGroup
import com.google.android.material.textfield.TextInputEditText
import org.crc.multimodal.R
import org.crc.multimodal.evaluation.CohortEvaluationEngine
import org.crc.multimodal.evaluation.CohortEvaluationSummary
import org.crc.multimodal.evaluation.EvaluationFilterStatus
import org.crc.multimodal.evaluation.EvaluationSortOption
import org.crc.multimodal.evaluation.MetricValue
import org.crc.multimodal.evaluation.ReferenceLabelImporter
import org.crc.multimodal.ui.WorkstationNavigationHost
import org.crc.multimodal.ui.adapters.CohortSpecimenAdapter
import java.util.Locale

class CohortEvaluationFragment : Fragment() {

    private var navHost: WorkstationNavigationHost? = null
    private var cohortIdParam: String? = null

    private lateinit var adapter: CohortSpecimenAdapter
    private var currentSummary: CohortEvaluationSummary? = null
    private var currentFilter: EvaluationFilterStatus = EvaluationFilterStatus.ALL
    private var currentSort: EvaluationSortOption = EvaluationSortOption.SPECIMEN_ID
    private var currentSearch: String = ""

    override fun onAttach(context: Context) {
        super.onAttach(context)
        if (context is WorkstationNavigationHost) {
            navHost = context
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        cohortIdParam = arguments?.getString(ARG_COHORT_ID)
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.fragment_cohort_evaluation, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        navHost?.setTopToolbar(
            title = "Cohort Evaluation",
            subtitle = "Reference Benchmarks & Discrimination Curves",
            showBackButton = true
        )

        val rvSpecimens: RecyclerView = view.findViewById(R.id.rvCohortSpecimens)
        rvSpecimens.layoutManager = LinearLayoutManager(requireContext())
        adapter = CohortSpecimenAdapter()
        rvSpecimens.adapter = adapter

        // Setup filter chips
        val chipGroup: ChipGroup = view.findViewById(R.id.chipGroupCohortFilters)
        chipGroup.setOnCheckedChangeListener { _, checkedId ->
            currentFilter = when (checkedId) {
                R.id.chipFilterTP -> EvaluationFilterStatus.TP_ONLY
                R.id.chipFilterTN -> EvaluationFilterStatus.TN_ONLY
                R.id.chipFilterFP -> EvaluationFilterStatus.FP_ONLY
                R.id.chipFilterFN -> EvaluationFilterStatus.FN_ONLY
                R.id.chipFilterErrors -> EvaluationFilterStatus.MISCLASSIFIED_ONLY
                else -> EvaluationFilterStatus.ALL
            }
            updateSpecimenList(view)
        }

        // Setup search input
        val etSearch: TextInputEditText = view.findViewById(R.id.etCohortSearch)
        etSearch.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {
                currentSearch = s?.toString().orEmpty()
                updateSpecimenList(view)
            }
            override fun afterTextChanged(s: Editable?) {}
        })

        // Setup sort button toggle
        val btnSort: MaterialButton = view.findViewById(R.id.btnCohortSortToggle)
        btnSort.setOnClickListener {
            currentSort = when (currentSort) {
                EvaluationSortOption.SPECIMEN_ID -> {
                    btnSort.text = "Margin (Asc)"
                    EvaluationSortOption.MARGIN_SMALLEST_FIRST
                }
                EvaluationSortOption.MARGIN_SMALLEST_FIRST -> {
                    btnSort.text = "Margin (Desc)"
                    EvaluationSortOption.MARGIN_LARGEST_FIRST
                }
                EvaluationSortOption.MARGIN_LARGEST_FIRST -> {
                    btnSort.text = "Prob (Desc)"
                    EvaluationSortOption.PROBABILITY_DESCENDING
                }
                EvaluationSortOption.PROBABILITY_DESCENDING -> {
                    btnSort.text = "Specimen ID"
                    EvaluationSortOption.SPECIMEN_ID
                }
                EvaluationSortOption.PROBABILITY_ASCENDING -> {
                    btnSort.text = "Specimen ID"
                    EvaluationSortOption.SPECIMEN_ID
                }
            }
            updateSpecimenList(view)
        }

        // Load TCGA Holdout labels button
        val btnLoadHoldout: MaterialButton = view.findViewById(R.id.btnLoadHoldoutLabels)
        btnLoadHoldout.setOnClickListener {
            loadAssetReferenceLabels(view)
        }

        // Clear reference labels button
        val btnClearLabels: MaterialButton = view.findViewById(R.id.btnClearReferenceLabels)
        btnClearLabels.setOnClickListener {
            val repo = navHost?.getRepository() ?: return@setOnClickListener
            val cleared = repo.clearReferenceLabels()
            Toast.makeText(requireContext(), "Cleared $cleared reference labels.", Toast.LENGTH_SHORT).show()
            refreshEvaluation(view)
        }

        // Initial evaluation load
        refreshEvaluation(view)
    }

    private fun loadAssetReferenceLabels(rootView: View) {
        val host = navHost ?: return
        val repo = host.getRepository()

        try {
            val assetStream = requireContext().assets.open("reference_labels/tcga_holdout_reference_labels.json")
            val importResult = ReferenceLabelImporter.parseJsonStream(assetStream)

            if (importResult.isSuccess) {
                val inserted = repo.insertReferenceLabels(importResult.importedLabels)
                Toast.makeText(
                    requireContext(),
                    "Loaded $inserted reference labels for ${importResult.cohortId}",
                    Toast.LENGTH_SHORT
                ).show()
                refreshEvaluation(rootView)
            } else {
                Toast.makeText(
                    requireContext(),
                    "Import failed: ${importResult.errorMessages.firstOrNull()}",
                    Toast.LENGTH_LONG
                ).show()
            }
        } catch (e: Exception) {
            Toast.makeText(
                requireContext(),
                "Error reading reference labels asset: ${e.message}",
                Toast.LENGTH_LONG
            ).show()
        }
    }

    private fun refreshEvaluation(rootView: View) {
        val host = navHost ?: return
        val repo = host.getRepository()
        val batchRepo = host.getBatchRepository()

        val summary = repo.evaluateCohort(cohortId = cohortIdParam, batchRepo = batchRepo)
        currentSummary = summary

        bindSummaryViews(rootView, summary)
        updateSpecimenList(rootView)
    }

    private fun bindSummaryViews(rootView: View, summary: CohortEvaluationSummary) {
        val tvCohortTitle: TextView = rootView.findViewById(R.id.tvCohortTitle)
        val tvMatchedStatus: TextView = rootView.findViewById(R.id.tvCohortMatchedStatus)
        val tvRefLabelsBadge: TextView = rootView.findViewById(R.id.tvRefLabelsBadge)
        val tvPredictionsAvailableBadge: TextView = rootView.findViewById(R.id.tvPredictionsAvailableBadge)
        val tvUnmatchedRefBadge: TextView = rootView.findViewById(R.id.tvUnmatchedRefBadge)

        tvCohortTitle.text = "Reference Cohort: ${summary.cohortId}"
        tvMatchedStatus.text = "${summary.matchedCount} Matched"
        tvRefLabelsBadge.text = "Labels: ${summary.totalReferenceLabels}"
        tvPredictionsAvailableBadge.text = "Predictions: ${summary.totalPredictionsAvailable}"
        tvUnmatchedRefBadge.text = "Unmatched: ${summary.unmatchedReferenceCount}"

        // Sample Size Warning Banner
        val tvSampleWarning: TextView? = rootView.findViewById(R.id.tvCohortSampleWarning)
        if (tvSampleWarning != null) {
            when {
                summary.matchedCount == 0 -> {
                    tvSampleWarning.visibility = View.VISIBLE
                    tvSampleWarning.text = "⚠ No Matched Specimens: 0 predictions match reference cohort labels. Load TCGA labels or run batch analysis to match specimens."
                }
                summary.matchedCount < 10 -> {
                    tvSampleWarning.visibility = View.VISIBLE
                    tvSampleWarning.text = "⚠ Limited Sample Size: ${summary.matchedCount} matched specimen (${summary.unmatchedReferenceCount} unmatched reference labels). 100% accuracy on a single matched negative case is not statistically meaningful evidence of general model performance."
                }
                summary.unmatchedReferenceCount > summary.matchedCount -> {
                    tvSampleWarning.visibility = View.VISIBLE
                    tvSampleWarning.text = "Notice: ${summary.unmatchedReferenceCount} reference labels remain unmatched. Results reflect ${summary.matchedCount} matched specimens."
                }
                else -> {
                    tvSampleWarning.visibility = View.GONE
                }
            }
        }

        // Confusion Matrix
        rootView.findViewById<TextView>(R.id.tvMatrixTN).text = "TN: ${summary.confusionMatrix.tn}"
        rootView.findViewById<TextView>(R.id.tvMatrixFP).text = "FP: ${summary.confusionMatrix.fp}"
        rootView.findViewById<TextView>(R.id.tvMatrixFN).text = "FN: ${summary.confusionMatrix.fn}"
        rootView.findViewById<TextView>(R.id.tvMatrixTP).text = "TP: ${summary.confusionMatrix.tp}"
        rootView.findViewById<TextView>(R.id.tvMatrixTotalMSS).text = "${summary.confusionMatrix.actualNegative}"
        rootView.findViewById<TextView>(R.id.tvMatrixTotalMSI).text = "${summary.confusionMatrix.actualPositive}"
        rootView.findViewById<TextView>(R.id.tvMatrixPredMSS).text = "${summary.confusionMatrix.predictedNegative}"
        rootView.findViewById<TextView>(R.id.tvMatrixPredMSI).text = "${summary.confusionMatrix.predictedPositive}"
        rootView.findViewById<TextView>(R.id.tvMatrixGrandTotal).text = "${summary.confusionMatrix.total}"

        // Diagnostic Metrics
        bindMetric(
            valueView = rootView.findViewById(R.id.tvMetricAccuracy),
            noteView = rootView.findViewById(R.id.tvMetricAccuracyNote),
            metric = summary.accuracy,
            isPercentage = true,
            defaultFormula = "(TP + TN) / Total"
        )
        bindMetric(
            valueView = rootView.findViewById(R.id.tvMetricBalancedAcc),
            noteView = rootView.findViewById(R.id.tvMetricBalancedAccNote),
            metric = summary.balancedAccuracy,
            isPercentage = true,
            defaultFormula = "(Sens + Spec) / 2"
        )
        bindMetric(
            valueView = rootView.findViewById(R.id.tvMetricPrecision),
            noteView = rootView.findViewById(R.id.tvMetricPrecisionNote),
            metric = summary.precision,
            isPercentage = true,
            defaultFormula = "TP / (TP + FP)"
        )
        bindMetric(
            valueView = rootView.findViewById(R.id.tvMetricSensitivity),
            noteView = rootView.findViewById(R.id.tvMetricSensitivityNote),
            metric = summary.recallSensitivity,
            isPercentage = true,
            defaultFormula = "TP / (TP + FN)"
        )
        bindMetric(
            valueView = rootView.findViewById(R.id.tvMetricSpecificity),
            noteView = rootView.findViewById(R.id.tvMetricSpecificityNote),
            metric = summary.specificity,
            isPercentage = true,
            defaultFormula = "TN / (TN + FP)"
        )
        bindMetric(
            valueView = rootView.findViewById(R.id.tvMetricF1),
            noteView = rootView.findViewById(R.id.tvMetricF1Note),
            metric = summary.f1Score,
            isPercentage = false,
            defaultFormula = "Harmonic Mean (P, R)"
        )
        bindMetric(
            valueView = rootView.findViewById(R.id.tvMetricNPV),
            noteView = rootView.findViewById(R.id.tvMetricNPVNote),
            metric = summary.npv,
            isPercentage = true,
            defaultFormula = "TN / (TN + FN)"
        )

        // Discrimination Curves (ROC-AUC & PR-AUC)
        bindCurveMetric(
            valueView = rootView.findViewById(R.id.tvMetricRocAuc),
            noteView = rootView.findViewById(R.id.tvRocPointsSummary),
            metric = summary.rocAuc,
            defaultPointsText = "${summary.rocCurvePoints.size} curve points evaluated"
        )
        bindCurveMetric(
            valueView = rootView.findViewById(R.id.tvMetricPrAuc),
            noteView = rootView.findViewById(R.id.tvPrPointsSummary),
            metric = summary.prAuc,
            defaultPointsText = "${summary.prCurvePoints.size} precision-recall steps"
        )
    }

    private fun bindMetric(
        valueView: TextView,
        noteView: TextView,
        metric: MetricValue,
        isPercentage: Boolean,
        defaultFormula: String
    ) {
        val context = valueView.context
        when (metric) {
            is MetricValue.Defined -> {
                valueView.text = if (isPercentage) {
                    String.format(Locale.US, "%.1f%%", metric.value * 100.0)
                } else {
                    String.format(Locale.US, "%.4f", metric.value)
                }
                valueView.setTextColor(ContextCompat.getColor(context, R.color.text_primary))
                noteView.text = String.format(Locale.US, "Score: %.4f | %s", metric.value, defaultFormula)
            }
            is MetricValue.Undefined -> {
                valueView.text = "Undefined"
                valueView.setTextColor(ContextCompat.getColor(context, R.color.status_warning))
                noteView.text = metric.reason
            }
        }
    }

    private fun bindCurveMetric(
        valueView: TextView,
        noteView: TextView,
        metric: MetricValue,
        defaultPointsText: String
    ) {
        val context = valueView.context
        when (metric) {
            is MetricValue.Defined -> {
                valueView.text = String.format(Locale.US, "%.4f", metric.value)
                valueView.setTextColor(ContextCompat.getColor(context, R.color.text_primary))
                noteView.text = defaultPointsText
            }
            is MetricValue.Undefined -> {
                valueView.text = "Unavailable"
                valueView.setTextColor(ContextCompat.getColor(context, R.color.status_warning))
                noteView.text = if (metric.reason.contains("no positive cases", ignoreCase = true)) {
                    "Unavailable — no positive reference cases"
                } else {
                    metric.reason
                }
            }
        }
    }

    private fun updateSpecimenList(rootView: View) {
        val summary = currentSummary ?: return
        val filtered = CohortEvaluationEngine.filterAndSortMatches(
            matches = summary.specimenMatches,
            filter = currentFilter,
            sortOption = currentSort,
            searchQuery = currentSearch
        )

        adapter.submitList(filtered)
        val tvEmptyState: TextView = rootView.findViewById(R.id.tvCohortEmptyState)
        tvEmptyState.visibility = if (filtered.isEmpty()) View.VISIBLE else View.GONE
    }

    companion object {
        private const val ARG_COHORT_ID = "arg_cohort_id"

        fun newInstance(cohortId: String? = null): CohortEvaluationFragment {
            val fragment = CohortEvaluationFragment()
            if (cohortId != null) {
                val args = Bundle().apply {
                    putString(ARG_COHORT_ID, cohortId)
                }
                fragment.arguments = args
            }
            return fragment
        }
    }
}
