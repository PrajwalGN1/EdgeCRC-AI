package org.crc.multimodal.ui.fragments

import android.content.Context
import android.os.Bundle
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.button.MaterialButton
import com.google.android.material.chip.ChipGroup
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.textfield.TextInputEditText
import com.google.android.material.textfield.TextInputLayout
import org.crc.multimodal.R
import org.crc.multimodal.data.CaseStatus
import org.crc.multimodal.data.ResearchCase
import org.crc.multimodal.data.ReviewNote
import org.crc.multimodal.data.ReviewNoteCategory
import org.crc.multimodal.ui.WorkstationNavigationHost
import org.crc.multimodal.ui.adapters.ReviewNoteAdapter

class ReviewNotesFragment : Fragment(R.layout.fragment_review_notes) {

    private var navHost: WorkstationNavigationHost? = null
    private var caseId: Long = 0L
    private lateinit var noteAdapter: ReviewNoteAdapter

    override fun onAttach(context: Context) {
        super.onAttach(context)
        if (context is WorkstationNavigationHost) {
            navHost = context
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        caseId = arguments?.getLong(ARG_CASE_ID) ?: 0L
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        navHost?.setTopToolbar(
            title = "Review Questions & Notes",
            subtitle = "Pathology Case Review Log",
            showBackButton = true
        )

        val host = navHost ?: return
        val repository = host.getRepository()
        var currentCase = repository.getCaseById(caseId) ?: run {
            host.navigateToWorkspace()
            return
        }

        val tvNotesCaseId: TextView = view.findViewById(R.id.tvNotesCaseId)
        val tvNotesCaseStatus: TextView = view.findViewById(R.id.tvNotesCaseStatus)
        val btnMarkUnderReview: MaterialButton = view.findViewById(R.id.btnMarkUnderReview)
        val btnMarkReviewed: MaterialButton = view.findViewById(R.id.btnMarkReviewed)

        val chipGroupNoteCategory: ChipGroup = view.findViewById(R.id.chipGroupNoteCategory)
        val etNoteAuthor: TextInputEditText = view.findViewById(R.id.etNoteAuthor)
        val tilNoteContent: TextInputLayout = view.findViewById(R.id.tilNoteContent)
        val etNoteContent: TextInputEditText = view.findViewById(R.id.etNoteContent)
        val btnAddNoteSubmit: MaterialButton = view.findViewById(R.id.btnAddNoteSubmit)

        val layoutNotesEmpty: LinearLayout = view.findViewById(R.id.layoutNotesEmpty)
        val rvReviewNotesList: RecyclerView = view.findViewById(R.id.rvReviewNotesList)

        tvNotesCaseId.text = currentCase.caseIdentifier
        tvNotesCaseStatus.text = currentCase.status.displayName

        // Status update buttons
        btnMarkUnderReview.setOnClickListener {
            val updated = currentCase.copy(status = CaseStatus.UNDER_REVIEW, updatedAt = System.currentTimeMillis())
            repository.updateCase(updated)
            currentCase = updated
            tvNotesCaseStatus.text = currentCase.status.displayName
            Toast.makeText(requireContext(), "Case marked as Under Review", Toast.LENGTH_SHORT).show()
        }

        btnMarkReviewed.setOnClickListener {
            val updated = currentCase.copy(status = CaseStatus.REVIEWED, updatedAt = System.currentTimeMillis())
            repository.updateCase(updated)
            currentCase = updated
            tvNotesCaseStatus.text = currentCase.status.displayName
            Toast.makeText(requireContext(), "Case marked as Reviewed", Toast.LENGTH_SHORT).show()
        }

        // RecyclerView setup
        rvReviewNotesList.layoutManager = LinearLayoutManager(requireContext())
        noteAdapter = ReviewNoteAdapter(emptyList()) { noteToDelete ->
            MaterialAlertDialogBuilder(requireContext())
                .setTitle("Delete Review Note?")
                .setMessage("Are you sure you want to delete this observation?")
                .setPositiveButton("Delete") { _, _ ->
                    repository.deleteReviewNote(noteToDelete.id)
                    loadNotes(rvReviewNotesList, layoutNotesEmpty)
                    Toast.makeText(requireContext(), "Note deleted", Toast.LENGTH_SHORT).show()
                }
                .setNegativeButton("Cancel", null)
                .show()
        }
        rvReviewNotesList.adapter = noteAdapter

        // Add Note Action
        btnAddNoteSubmit.setOnClickListener {
            val contentText = etNoteContent.text?.toString()?.trim().orEmpty()
            if (contentText.isBlank()) {
                tilNoteContent.error = "Please enter an observation or clinical note"
                return@setOnClickListener
            }
            tilNoteContent.error = null

            val author = etNoteAuthor.text?.toString()?.trim().orEmpty().ifBlank { "Research Pathologist" }
            val selectedCategory = when (chipGroupNoteCategory.checkedChipId) {
                R.id.chipCatUncertainty -> ReviewNoteCategory.MODEL_UNCERTAINTY
                R.id.chipCatInputQuality -> ReviewNoteCategory.INPUT_QUALITY_ISSUE
                R.id.chipCatTechnical -> ReviewNoteCategory.TECHNICAL_ISSUE
                R.id.chipCatGeneral -> ReviewNoteCategory.GENERAL_RESEARCH_NOTE
                else -> ReviewNoteCategory.HISTOLOGICAL_OBSERVATION
            }

            val newNote = ReviewNote(
                caseId = currentCase.id,
                category = selectedCategory,
                authorLabel = author,
                content = contentText
            )
            repository.insertReviewNote(newNote)

            etNoteContent.setText("")
            loadNotes(rvReviewNotesList, layoutNotesEmpty)
            Toast.makeText(requireContext(), "Review note logged successfully", Toast.LENGTH_SHORT).show()
        }

        loadNotes(rvReviewNotesList, layoutNotesEmpty)
    }

    private fun loadNotes(rv: RecyclerView, emptyLayout: LinearLayout) {
        val host = navHost ?: return
        val notes = host.getRepository().getNotesForCase(caseId)
        if (notes.isEmpty()) {
            emptyLayout.visibility = View.VISIBLE
            noteAdapter.submitList(emptyList())
        } else {
            emptyLayout.visibility = View.GONE
            noteAdapter.submitList(notes)
        }
    }

    companion object {
        private const val ARG_CASE_ID = "arg_case_id"

        fun newInstance(caseId: Long): ReviewNotesFragment {
            return ReviewNotesFragment().apply {
                arguments = Bundle().apply {
                    putLong(ARG_CASE_ID, caseId)
                }
            }
        }
    }
}
