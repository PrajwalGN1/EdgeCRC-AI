package org.crc.multimodal.ui.adapters

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import org.crc.multimodal.R
import org.crc.multimodal.data.ReviewNote
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class ReviewNoteAdapter(
    private var notes: List<ReviewNote>,
    private val onDeleteClicked: (ReviewNote) -> Unit
) : RecyclerView.Adapter<ReviewNoteAdapter.NoteViewHolder>() {

    private val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US)

    fun submitList(newList: List<ReviewNote>) {
        notes = newList
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): NoteViewHolder {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.item_review_note, parent, false)
        return NoteViewHolder(view)
    }

    override fun onBindViewHolder(holder: NoteViewHolder, position: Int) {
        holder.bind(notes[position])
    }

    override fun getItemCount(): Int = notes.size

    inner class NoteViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val tvNoteCategory: TextView = itemView.findViewById(R.id.tvNoteCategory)
        private val tvNoteDate: TextView = itemView.findViewById(R.id.tvNoteDate)
        private val tvNoteAuthor: TextView = itemView.findViewById(R.id.tvNoteAuthor)
        private val tvNoteContent: TextView = itemView.findViewById(R.id.tvNoteContent)
        private val btnDeleteNote: ImageView = itemView.findViewById(R.id.btnDeleteNote)

        fun bind(item: ReviewNote) {
            tvNoteCategory.text = item.category.displayName
            tvNoteDate.text = dateFormat.format(Date(item.createdAt))
            tvNoteAuthor.text = "By: ${item.authorLabel}"
            tvNoteContent.text = item.content

            btnDeleteNote.setOnClickListener {
                onDeleteClicked(item)
            }
        }
    }
}
