package org.crc.multimodal.ui.adapters

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import org.crc.multimodal.R
import org.crc.multimodal.evaluation.EvaluationOutcome
import org.crc.multimodal.evaluation.SpecimenEvaluationMatch
import java.util.Locale

class CohortSpecimenAdapter :
    ListAdapter<SpecimenEvaluationMatch, CohortSpecimenAdapter.CohortSpecimenViewHolder>(DiffCallback) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): CohortSpecimenViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_cohort_specimen, parent, false)
        return CohortSpecimenViewHolder(view)
    }

    override fun onBindViewHolder(holder: CohortSpecimenViewHolder, position: Int) {
        holder.bind(getItem(position))
    }

    class CohortSpecimenViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val tvSpecimenId: TextView = itemView.findViewById(R.id.tvCohortSpecimenId)
        private val tvOutcomeBadge: TextView = itemView.findViewById(R.id.tvCohortOutcomeBadge)
        private val tvGroundTruth: TextView = itemView.findViewById(R.id.tvCohortGroundTruth)
        private val tvPrediction: TextView = itemView.findViewById(R.id.tvCohortPrediction)
        private val tvMargin: TextView = itemView.findViewById(R.id.tvCohortMargin)
        private val tvSource: TextView = itemView.findViewById(R.id.tvCohortSource)

        fun bind(item: SpecimenEvaluationMatch) {
            val context = itemView.context
            tvSpecimenId.text = item.specimenId

            when (item.outcome) {
                EvaluationOutcome.TRUE_POSITIVE -> {
                    tvOutcomeBadge.text = "TP"
                    tvOutcomeBadge.setTextColor(ContextCompat.getColor(context, R.color.status_success))
                }
                EvaluationOutcome.TRUE_NEGATIVE -> {
                    tvOutcomeBadge.text = "TN"
                    tvOutcomeBadge.setTextColor(ContextCompat.getColor(context, R.color.status_success))
                }
                EvaluationOutcome.FALSE_POSITIVE -> {
                    tvOutcomeBadge.text = "FP (Type I Error)"
                    tvOutcomeBadge.setTextColor(ContextCompat.getColor(context, R.color.status_error))
                }
                EvaluationOutcome.FALSE_NEGATIVE -> {
                    tvOutcomeBadge.text = "FN (Type II Error)"
                    tvOutcomeBadge.setTextColor(ContextCompat.getColor(context, R.color.status_error))
                }
            }

            tvGroundTruth.text = "Actual: ${item.groundTruthClass} (${item.groundTruthLabel})"
            tvPrediction.text = String.format(
                Locale.US,
                "Pred: %s (P = %.4f)",
                item.predictedLabel,
                item.predictedProbability
            )

            val signedDistance = item.predictedProbability - 0.8300
            val signStr = if (signedDistance >= 0) "+" else ""
            tvMargin.text = String.format(
                Locale.US,
                "Margin: %s%.4f (dist to tau=0.8300)",
                signStr,
                signedDistance
            )

            tvSource.text = item.source ?: "Verified Label"
        }
    }

    companion object {
        private val DiffCallback = object : DiffUtil.ItemCallback<SpecimenEvaluationMatch>() {
            override fun areItemsTheSame(
                oldItem: SpecimenEvaluationMatch,
                newItem: SpecimenEvaluationMatch
            ): Boolean = oldItem.specimenId == newItem.specimenId

            override fun areContentsTheSame(
                oldItem: SpecimenEvaluationMatch,
                newItem: SpecimenEvaluationMatch
            ): Boolean = oldItem == newItem
        }
    }
}
