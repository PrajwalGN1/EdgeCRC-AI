package org.crc.multimodal.ui.adapters

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.RecyclerView
import org.crc.multimodal.R
import org.crc.multimodal.data.InferenceRun
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class InferenceRunAdapter(
    private var runs: List<InferenceRun>
) : RecyclerView.Adapter<InferenceRunAdapter.RunViewHolder>() {

    private val timeFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)

    fun submitList(newList: List<InferenceRun>) {
        runs = newList
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RunViewHolder {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.item_inference_run, parent, false)
        return RunViewHolder(view)
    }

    override fun onBindViewHolder(holder: RunViewHolder, position: Int) {
        holder.bind(runs[position])
    }

    override fun getItemCount(): Int = runs.size

    inner class RunViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val tvRunClassificationBadge: TextView = itemView.findViewById(R.id.tvRunClassificationBadge)
        private val tvRunScore: TextView = itemView.findViewById(R.id.tvRunScore)
        private val tvRunTimestamp: TextView = itemView.findViewById(R.id.tvRunTimestamp)
        private val tvRunMargin: TextView = itemView.findViewById(R.id.tvRunMargin)
        private val tvRunLatencyAndPatches: TextView = itemView.findViewById(R.id.tvRunLatencyAndPatches)

        fun bind(item: InferenceRun) {
            val context = itemView.context
            val isMsi = item.predictedClass == 1
            tvRunClassificationBadge.text = "${item.predictedLabel} (Class ${item.predictedClass})"

            if (isMsi) {
                tvRunClassificationBadge.backgroundTintList = ContextCompat.getColorStateList(context, R.color.teal_light)
                tvRunClassificationBadge.setTextColor(ContextCompat.getColor(context, R.color.secondary))
            } else {
                tvRunClassificationBadge.backgroundTintList = ContextCompat.getColorStateList(context, R.color.mss_bg_light)
                tvRunClassificationBadge.setTextColor(ContextCompat.getColor(context, R.color.mss_color))
            }

            tvRunScore.text = String.format(Locale.US, "P = %.4f", item.probability)
            tvRunTimestamp.text = timeFormat.format(Date(item.executionTimestamp))

            val marginStr = if (item.margin >= 0) {
                String.format(Locale.US, "+%.4f above 0.83 threshold", item.margin)
            } else {
                String.format(Locale.US, "%.4f below 0.83 threshold", item.margin)
            }
            tvRunMargin.text = marginStr
            tvRunLatencyAndPatches.text = String.format(Locale.US, "%.2f ms · %d patches", item.latencyMs, item.numPatches)
        }
    }
}
