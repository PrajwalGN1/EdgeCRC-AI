package org.crc.multimodal.ui.adapters

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.RecyclerView
import org.crc.multimodal.R
import org.crc.multimodal.data.CaseStatus
import org.crc.multimodal.data.ResearchCase
import org.crc.multimodal.data.WorkstationRepository
import org.crc.multimodal.inference.InferenceProtocol
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class CaseAdapter(
    private var cases: List<ResearchCase>,
    private val repository: WorkstationRepository,
    private val onCaseClicked: (ResearchCase) -> Unit
) : RecyclerView.Adapter<CaseAdapter.CaseViewHolder>() {

    private val dateFormat = SimpleDateFormat("MM/dd HH:mm", Locale.US)

    fun submitList(newList: List<ResearchCase>) {
        cases = newList
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): CaseViewHolder {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.item_case_card, parent, false)
        return CaseViewHolder(view)
    }

    override fun onBindViewHolder(holder: CaseViewHolder, position: Int) {
        holder.bind(cases[position])
    }

    override fun getItemCount(): Int = cases.size

    inner class CaseViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val tvCaseIdentifier: TextView = itemView.findViewById(R.id.tvCaseIdentifier)
        private val tvCaseStatusBadge: TextView = itemView.findViewById(R.id.tvCaseStatusBadge)
        private val tvCaseDate: TextView = itemView.findViewById(R.id.tvCaseDate)
        private val tvCaseSpecimen: TextView = itemView.findViewById(R.id.tvCaseSpecimen)
        private val tvCaseContextSnippet: TextView = itemView.findViewById(R.id.tvCaseContextSnippet)
        private val tvCasePatchesAndProtocol: TextView = itemView.findViewById(R.id.tvCasePatchesAndProtocol)
        private val tvCaseInferenceFinding: TextView = itemView.findViewById(R.id.tvCaseInferenceFinding)

        fun bind(item: ResearchCase) {
            val context = itemView.context
            tvCaseIdentifier.text = item.caseIdentifier
            tvCaseDate.text = dateFormat.format(Date(item.updatedAt))
            tvCaseSpecimen.text = item.specimenDescription.ifBlank { "No specimen description recorded" }
            tvCaseContextSnippet.text = item.researchContext.ifBlank { "No clinical context recorded" }

            val protocolName = InferenceProtocol.formatProtocol(item.protocol)
            tvCasePatchesAndProtocol.text = "${item.patchCount} patches · $protocolName"

            // Status Badge Coloring
            tvCaseStatusBadge.text = item.status.displayName
            when (item.status) {
                CaseStatus.DRAFT -> {
                    tvCaseStatusBadge.backgroundTintList = ContextCompat.getColorStateList(context, R.color.status_warning_bg)
                    tvCaseStatusBadge.setTextColor(ContextCompat.getColor(context, R.color.status_warning))
                }
                CaseStatus.READY_FOR_ANALYSIS -> {
                    tvCaseStatusBadge.backgroundTintList = ContextCompat.getColorStateList(context, R.color.teal_light)
                    tvCaseStatusBadge.setTextColor(ContextCompat.getColor(context, R.color.secondary))
                }
                CaseStatus.ANALYZED -> {
                    tvCaseStatusBadge.backgroundTintList = ContextCompat.getColorStateList(context, R.color.status_success_bg)
                    tvCaseStatusBadge.setTextColor(ContextCompat.getColor(context, R.color.status_success))
                }
                CaseStatus.UNDER_REVIEW -> {
                    tvCaseStatusBadge.backgroundTintList = ContextCompat.getColorStateList(context, R.color.status_pending_bg)
                    tvCaseStatusBadge.setTextColor(ContextCompat.getColor(context, R.color.status_pending))
                }
                CaseStatus.REVIEWED -> {
                    tvCaseStatusBadge.backgroundTintList = ContextCompat.getColorStateList(context, R.color.teal_light)
                    tvCaseStatusBadge.setTextColor(ContextCompat.getColor(context, R.color.teal_dark))
                }
            }

            // Inference Findings Check
            val latestRun = repository.getLatestRunForCase(item.id)
            if (latestRun != null) {
                val probFormatted = String.format(Locale.US, "%.4f", latestRun.probability)
                tvCaseInferenceFinding.text = "${latestRun.predictedLabel} (P = $probFormatted)"
                val findingColor = if (latestRun.predictedClass == 1) R.color.secondary else R.color.mss_color
                tvCaseInferenceFinding.setTextColor(ContextCompat.getColor(context, findingColor))
            } else {
                tvCaseInferenceFinding.text = "Inference Pending"
                tvCaseInferenceFinding.setTextColor(ContextCompat.getColor(context, R.color.text_muted_light))
            }

            itemView.setOnClickListener {
                onCaseClicked(item)
            }
        }
    }
}
