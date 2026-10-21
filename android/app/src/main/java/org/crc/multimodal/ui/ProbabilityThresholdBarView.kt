package org.crc.multimodal.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import org.crc.multimodal.R

/**
 * Enterprise healthcare custom visualization component.
 *
 * Renders an accessible, bounded probability bar indicating the model's predicted
 * probability alongside the frozen decision threshold marker at 0.83.
 *
 * Visual Rules:
 * - Operating Threshold is fixed at 0.83f (83% of the horizontal bar).
 * - Values >= 0.83f evaluate to Class 1 (MSI) and are rendered with the clinical teal accent.
 * - Values < 0.83f evaluate to Class 0 (MSS) and are rendered with the clinical slate/blue accent.
 * - Full system font scaling, night mode adaptation, and screen-reader accessibility are supported.
 */
class ProbabilityThresholdBarView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var probability: Float = 0.0f
    private var threshold: Float = 0.83f
    private var isDataSet: Boolean = false

    private val trackPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val thresholdLinePaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val zonePaint = Paint(Paint.ANTI_ALIAS_FLAG)

    private val barRect = RectF()
    private val progressRect = RectF()

    init {
        trackPaint.style = Paint.Style.FILL
        fillPaint.style = Paint.Style.FILL

        thresholdLinePaint.style = Paint.Style.STROKE
        thresholdLinePaint.strokeWidth = dpToPx(2.5f)
        thresholdLinePaint.color = ContextCompat.getColor(context, R.color.threshold_marker)

        textPaint.isAntiAlias = true
        textPaint.textSize = spToPx(10.5f)
        textPaint.style = Paint.Style.FILL

        zonePaint.style = Paint.Style.FILL

        importantForAccessibility = IMPORTANT_FOR_ACCESSIBILITY_YES
        updateAccessibilityDescription()
    }

    /**
     * Updates the visualization with verified inference outputs.
     *
     * @param prob The predicted output probability in [0.0, 1.0].
     * @param thresh The decision threshold (default: 0.83f).
     */
    fun setProbability(prob: Float, thresh: Float = 0.83f) {
        require(!prob.isNaN() && !prob.isInfinite()) { "Probability must be a finite float" }
        require(!thresh.isNaN() && !thresh.isInfinite()) { "Threshold must be a finite float" }

        this.probability = prob.coerceIn(0.0f, 1.0f)
        this.threshold = thresh.coerceIn(0.0f, 1.0f)
        this.isDataSet = true

        updateAccessibilityDescription()
        invalidate()
    }

    /**
     * Resets the visualization to initial idle/unexecuted state.
     */
    fun reset() {
        this.probability = 0.0f
        this.isDataSet = false
        updateAccessibilityDescription()
        invalidate()
    }

    private fun updateAccessibilityDescription() {
        contentDescription = if (isDataSet) {
            val scorePct = (probability * 100).toInt()
            val threshPct = (threshold * 100).toInt()
            val classification = if (probability >= threshold) "MSI" else "MSS"
            "Probability gauge: score is $scorePct percent, threshold is $threshPct percent. Evaluated classification: $classification."
        } else {
            "Probability gauge awaiting model inference."
        }
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val minHeight = dpToPx(48f).toInt()
        val height = resolveSizeAndState(minHeight, heightMeasureSpec, 0)
        val width = MeasureSpec.getSize(widthMeasureSpec)
        setMeasuredDimension(width, height)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val w = width.toFloat()
        val h = height.toFloat()
        if (w <= 0 || h <= 0) return

        val paddingL = paddingLeft.toFloat() + dpToPx(4f)
        val paddingR = paddingRight.toFloat() + dpToPx(4f)
        val availableWidth = (w - paddingL - paddingR).coerceAtLeast(0f)

        // The bar occupies the central vertical region
        val barHeight = dpToPx(14f)
        val barTop = (h - barHeight) / 2f
        val barBottom = barTop + barHeight
        val cornerRadius = barHeight / 2f

        barRect.set(paddingL, barTop, paddingL + availableWidth, barBottom)

        // 1. Draw track background
        trackPaint.color = ContextCompat.getColor(context, R.color.card_stroke_light)
        canvas.drawRoundRect(barRect, cornerRadius, cornerRadius, trackPaint)

        // 2. Draw subtle MSI zone indicator on the track (from threshold to 1.0)
        val thresholdX = paddingL + (availableWidth * threshold)
        if (thresholdX < barRect.right) {
            zonePaint.color = ContextCompat.getColor(context, R.color.teal_light)
            val zoneRect = RectF(thresholdX, barTop, barRect.right, barBottom)
            canvas.drawRoundRect(zoneRect, cornerRadius, cornerRadius, zonePaint)
        }

        // 3. Draw active probability progress fill if data is set
        if (isDataSet && probability > 0.001f) {
            val fillRight = paddingL + (availableWidth * probability)
            progressRect.set(paddingL, barTop, fillRight.coerceAtMost(barRect.right), barBottom)

            fillPaint.color = if (probability >= threshold) {
                ContextCompat.getColor(context, R.color.msi_color)
            } else {
                ContextCompat.getColor(context, R.color.mss_color)
            }
            canvas.drawRoundRect(progressRect, cornerRadius, cornerRadius, fillPaint)
        }

        // 4. Draw Threshold Marker Line at 0.83
        val markerTop = barTop - dpToPx(5f)
        val markerBottom = barBottom + dpToPx(5f)
        canvas.drawLine(thresholdX, markerTop, thresholdX, markerBottom, thresholdLinePaint)

        // 5. Draw Labels
        textPaint.color = ContextCompat.getColor(context, R.color.text_secondary_light)
        textPaint.textAlign = Paint.Align.LEFT
        canvas.drawText("0.0 (MSS)", paddingL, barBottom + dpToPx(13f), textPaint)

        textPaint.textAlign = Paint.Align.RIGHT
        canvas.drawText("1.0 (MSI)", paddingL + availableWidth, barBottom + dpToPx(13f), textPaint)

        // Threshold annotation at top
        textPaint.color = ContextCompat.getColor(context, R.color.threshold_marker)
        textPaint.textAlign = Paint.Align.CENTER
        val thresholdLabel = String.format("Threshold %.2f", threshold)
        val clampedLabelX = thresholdX.coerceIn(paddingL + dpToPx(36f), paddingL + availableWidth - dpToPx(36f))
        canvas.drawText(thresholdLabel, clampedLabelX, barTop - dpToPx(7f), textPaint)
    }

    private fun dpToPx(dp: Float): Float = dp * resources.displayMetrics.density
    private fun spToPx(sp: Float): Float = android.util.TypedValue.applyDimension(
        android.util.TypedValue.COMPLEX_UNIT_SP,
        sp,
        resources.displayMetrics
    )
}
