package org.crc.multimodal.ui.adapters

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.RecyclerView
import org.crc.multimodal.R
import org.crc.multimodal.analytics.ResearchAnalyticsEngine
import org.crc.multimodal.analytics.SpecimenPredictionRecord
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class AnalyticsSpecimenAdapter(
    private var items: List<SpecimenPredictionRecord>,
    private val onItemClick: (SpecimenPredictionRecord) -> Unit
) : RecyclerView.Adapter<AnalyticsSpecimenAdapter.ViewHolder>() {

    private val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US)

    fun submitList(newItems: List<SpecimenPredictionRecord>) {
        items = newItems
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_analytics_specimen, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        holder.bind(items[position])
    }

    override fun getItemCount(): Int = items.size

    inner class ViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val tvSpecimenId: TextView = itemView.findViewById(R.id.tvAnalyticsItemSpecimenId)
        private val tvClassBadge: TextView = itemView.findViewById(R.id.tvAnalyticsItemClassBadge)
        private val tvOriginAndDate: TextView = itemView.findViewById(R.id.tvAnalyticsItemOriginAndDate)
        private val tvProbability: TextView = itemView.findViewById(R.id.tvAnalyticsItemProbability)
        private val tvMargin: TextView = itemView.findViewById(R.id.tvAnalyticsItemMargin)
        private val tvMetrics: TextView = itemView.findViewById(R.id.tvAnalyticsItemMetrics)
        private val tvBoundaryTag: TextView = itemView.findViewById(R.id.tvAnalyticsItemBoundaryTag)

        fun bind(record: SpecimenPredictionRecord) {
            val context = itemView.context

            tvSpecimenId.text = record.specimenId

            // Classification badge
            if (record.status == "SUCCESS" && record.predictedClass != null) {
                if (record.predictedClass == 1) {
                    tvClassBadge.text = "MSI (Class 1)"
                    tvClassBadge.setTextColor(ContextCompat.getColor(context, R.color.msi_color))
                    tvClassBadge.backgroundTintList = ContextCompat.getColorStateList(context, R.color.msi_bg_light)
                } else {
                    tvClassBadge.text = "MSS (Class 0)"
                    tvClassBadge.setTextColor(ContextCompat.getColor(context, R.color.mss_color))
                    tvClassBadge.backgroundTintList = ContextCompat.getColorStateList(context, R.color.mss_bg_light)
                }
            } else {
                tvClassBadge.text = record.status
                tvClassBadge.setTextColor(ContextCompat.getColor(context, R.color.text_secondary_light))
                tvClassBadge.backgroundTintList = ContextCompat.getColorStateList(context, R.color.surface_variant_light)
            }

            // Origin & Date
            val originStr = if (record.origin == "BATCH") {
                "Batch: ${record.batchId ?: "Session"}"
            } else {
                "Single Case"
            }
            val dateStr = if (record.executionTimestamp > 0) {
                dateFormat.format(Date(record.executionTimestamp))
            } else {
                "Recent"
            }
            tvOriginAndDate.text = "$originStr · $dateStr"

            // Probability & Margin
            if (record.probability != null) {
                tvProbability.text = String.format(Locale.US, "P(MSI): %.4f", record.probability)
            } else {
                tvProbability.text = "P(MSI): Not Evaluated"
            }

            if (record.absMargin != null) {
                val signStr = if ((record.signedMargin ?: 0f) >= 0f) "+" else ""
                tvMargin.text = String.format(
                    Locale.US,
                    "Dist: %.4f (%s%.4f vs 0.83)",
                    record.absMargin,
                    signStr,
                    record.signedMargin ?: 0f
                )
            } else {
                tvMargin.text = "Margin: N/A"
            }

            // Metrics
            tvMetrics.text = "${record.patchCount} patches · ${String.format(Locale.US, "%.1f", record.latencyMs)} ms"

            // Near boundary tag
            if (record.absMargin != null && record.absMargin <= ResearchAnalyticsEngine.NEAR_THRESHOLD_TOLERANCE) {
                tvBoundaryTag.visibility = View.VISIBLE
            } else {
                tvBoundaryTag.visibility = View.GONE
            }

            itemView.setOnClickListener {
                onItemClick(record)
            }
        }
    }
}
