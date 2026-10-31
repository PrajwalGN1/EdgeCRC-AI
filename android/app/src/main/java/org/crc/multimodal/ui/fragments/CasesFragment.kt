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
import org.crc.multimodal.data.CaseSortOption
import org.crc.multimodal.data.CaseStatus
import org.crc.multimodal.ui.WorkstationNavigationHost
import org.crc.multimodal.ui.adapters.CaseAdapter

class CasesFragment : Fragment(R.layout.fragment_cases) {

    private var navHost: WorkstationNavigationHost? = null
    private lateinit var caseAdapter: CaseAdapter

    private var currentFilterStatus: CaseStatus? = null
    private var currentSearchQuery: String = ""
    private var currentSortOption: CaseSortOption = CaseSortOption.DATE_NEWEST

    override fun onAttach(context: Context) {
        super.onAttach(context)
        if (context is WorkstationNavigationHost) {
            navHost = context
        }
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        navHost?.setTopToolbar(
            title = "Case Management",
            subtitle = "Active Cohort & Specimen Records",
            showBackButton = false
        )

        val host = navHost ?: return
        val repository = host.getRepository()

        val etCaseSearch: TextInputEditText = view.findViewById(R.id.etCaseSearch)
        val chipGroupCaseFilter: ChipGroup = view.findViewById(R.id.chipGroupCaseFilter)
        val tvCasesCountBanner: TextView = view.findViewById(R.id.tvCasesCountBanner)
        val btnCaseSortToggle: MaterialButton = view.findViewById(R.id.btnCaseSortToggle)
        val layoutCasesEmptyState: LinearLayout = view.findViewById(R.id.layoutCasesEmptyState)
        val btnCasesCreateNew: MaterialButton = view.findViewById(R.id.btnCasesCreateNew)

        // RecyclerView
        val rvCasesList: RecyclerView = view.findViewById(R.id.rvCasesList)
        rvCasesList.layoutManager = LinearLayoutManager(requireContext())
        caseAdapter = CaseAdapter(emptyList(), repository) { selectedCase ->
            host.navigateToCaseDetails(selectedCase.id)
        }
        rvCasesList.adapter = caseAdapter

        // Search text watcher
        etCaseSearch.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                currentSearchQuery = s?.toString().orEmpty()
                applyFilters(tvCasesCountBanner, layoutCasesEmptyState)
            }
            override fun afterTextChanged(s: Editable?) {}
        })

        // Filter chips listener
        chipGroupCaseFilter.setOnCheckedChangeListener { _, checkedId ->
            currentFilterStatus = when (checkedId) {
                R.id.chipFilterDraft -> CaseStatus.DRAFT
                R.id.chipFilterReady -> CaseStatus.READY_FOR_ANALYSIS
                R.id.chipFilterAnalyzed -> CaseStatus.ANALYZED
                R.id.chipFilterUnderReview -> CaseStatus.UNDER_REVIEW
                R.id.chipFilterReviewed -> CaseStatus.REVIEWED
                else -> null
            }
            applyFilters(tvCasesCountBanner, layoutCasesEmptyState)
        }

        // Sort toggle listener
        btnCaseSortToggle.setOnClickListener {
            currentSortOption = when (currentSortOption) {
                CaseSortOption.DATE_NEWEST -> {
                    btnCaseSortToggle.text = "Date (Oldest)"
                    CaseSortOption.DATE_OLDEST
                }
                CaseSortOption.DATE_OLDEST -> {
                    btnCaseSortToggle.text = "Case Identifier"
                    CaseSortOption.CASE_IDENTIFIER
                }
                CaseSortOption.CASE_IDENTIFIER -> {
                    btnCaseSortToggle.text = "Date (Newest)"
                    CaseSortOption.DATE_NEWEST
                }
            }
            applyFilters(tvCasesCountBanner, layoutCasesEmptyState)
        }

        btnCasesCreateNew.setOnClickListener {
            host.navigateToNewSpecimen()
        }

        applyFilters(tvCasesCountBanner, layoutCasesEmptyState)
    }

    private fun applyFilters(tvCountBanner: TextView, emptyLayout: LinearLayout) {
        val host = navHost ?: return
        val repository = host.getRepository()

        val filtered = repository.getCasesFiltered(
            status = currentFilterStatus,
            searchQuery = currentSearchQuery,
            sortBy = currentSortOption
        )

        tvCountBanner.text = "Showing ${filtered.size} research cases"

        if (filtered.isEmpty()) {
            emptyLayout.visibility = View.VISIBLE
            caseAdapter.submitList(emptyList())
        } else {
            emptyLayout.visibility = View.GONE
            caseAdapter.submitList(filtered)
        }
    }

    override fun onResume() {
        super.onResume()
        view?.let { v ->
            val tvCountBanner: TextView = v.findViewById(R.id.tvCasesCountBanner)
            val emptyLayout: LinearLayout = v.findViewById(R.id.layoutCasesEmptyState)
            applyFilters(tvCountBanner, emptyLayout)
        }
    }
}
