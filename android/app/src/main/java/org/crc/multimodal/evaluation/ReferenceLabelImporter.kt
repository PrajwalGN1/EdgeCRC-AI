package org.crc.multimodal.evaluation

import org.json.JSONObject
import java.io.InputStream
import java.nio.charset.StandardCharsets

/**
 * Validates and imports independently established reference ground-truth labels from JSON or CSV.
 */
object ReferenceLabelImporter {

    data class ImportResult(
        val isSuccess: Boolean,
        val importedLabels: List<ReferenceLabel>,
        val errorMessages: List<String>,
        val cohortId: String
    )

    /**
     * Parses and validates a JSON string conforming to the reference labels schema.
     */
    fun parseJson(jsonString: String): ImportResult {
        val errors = mutableListOf<String>()
        val labels = mutableListOf<ReferenceLabel>()
        var cohortId = "UNKNOWN_COHORT"

        try {
            val root = JSONObject(jsonString)
            cohortId = root.optString("cohort_id", "REFERENCE_COHORT").trim()
            val labelsArray = root.optJSONArray("labels")
            if (labelsArray == null) {
                errors.add("Invalid JSON format: missing 'labels' array.")
                return ImportResult(false, emptyList(), errors, cohortId)
            }

            for (i in 0 until labelsArray.length()) {
                val item = labelsArray.getJSONObject(i)
                val specimenId = item.optString("specimen_id").trim()
                if (specimenId.isBlank()) {
                    errors.add("Item at index $i has a blank or missing 'specimen_id'.")
                    continue
                }

                if (!item.has("ground_truth_label")) {
                    errors.add("Specimen '$specimenId' lacks 'ground_truth_label'.")
                    continue
                }

                val gtLabel = item.getInt("ground_truth_label")
                if (gtLabel != 0 && gtLabel != 1) {
                    errors.add("Specimen '$specimenId' has invalid ground_truth_label: $gtLabel (must be 0 or 1).")
                    continue
                }

                val gtClass = item.optString("ground_truth_class", if (gtLabel == 1) "MSI" else "MSS").trim().uppercase()
                val expectedClass = if (gtLabel == 1) "MSI" else "MSS"
                if (gtClass != expectedClass) {
                    errors.add("Specimen '$specimenId' has mismatched label/class: label=$gtLabel but class='$gtClass'.")
                    continue
                }

                val source = item.optString("source", "Reference file").trim()
                labels.add(
                    ReferenceLabel(
                        specimenId = specimenId,
                        groundTruthLabel = gtLabel,
                        groundTruthClass = gtClass,
                        cohortId = cohortId,
                        source = source
                    )
                )
            }
        } catch (e: Exception) {
            errors.add("Failed to parse JSON: ${e.message}")
        }

        return ImportResult(
            isSuccess = errors.isEmpty() && labels.isNotEmpty(),
            importedLabels = labels,
            errorMessages = errors,
            cohortId = cohortId
        )
    }

    /**
     * Parses and validates a CSV string with header `specimen_id,ground_truth_label,ground_truth_class,source`.
     */
    fun parseCsv(csvString: String, defaultCohortId: String = "CSV_COHORT"): ImportResult {
        val errors = mutableListOf<String>()
        val labels = mutableListOf<ReferenceLabel>()
        val lines = csvString.lines().map { it.trim() }.filter { it.isNotBlank() }

        if (lines.isEmpty()) {
            return ImportResult(false, emptyList(), listOf("CSV content is empty."), defaultCohortId)
        }

        // Header check
        val header = lines.first().lowercase()
        val hasHeader = header.contains("specimen_id") || header.contains("label")
        val dataLines = if (hasHeader) lines.drop(1) else lines

        for ((idx, line) in dataLines.withIndex()) {
            val tokens = line.split(",").map { it.trim().trim('"', '\'') }
            if (tokens.size < 2) {
                errors.add("Line ${idx + 1} has insufficient columns (expected at least 2: specimen_id, label).")
                continue
            }

            val specimenId = tokens[0]
            if (specimenId.isBlank()) {
                errors.add("Line ${idx + 1} has blank specimen_id.")
                continue
            }

            val labelParsed = tokens[1].toIntOrNull()
            if (labelParsed == null || (labelParsed != 0 && labelParsed != 1)) {
                // Check if string "MSI" or "MSS" was given in the second column
                val stringLabel = tokens[1].uppercase()
                val inferredLabel = when (stringLabel) {
                    "MSI", "1" -> 1
                    "MSS", "0" -> 0
                    else -> null
                }
                if (inferredLabel == null) {
                    errors.add("Line ${idx + 1} ('$specimenId') has invalid label '${tokens[1]}'. Must be 0/1 or MSS/MSI.")
                    continue
                }
            }

            val gtLabel = if (tokens[1].toIntOrNull() != null) tokens[1].toInt() else (if (tokens[1].equals("MSI", true)) 1 else 0)
            val expectedClass = if (gtLabel == 1) "MSI" else "MSS"
            val gtClass = if (tokens.size > 2 && tokens[2].isNotBlank()) tokens[2].uppercase() else expectedClass

            if (gtClass != expectedClass) {
                errors.add("Line ${idx + 1} ('$specimenId') has mismatched class '$gtClass' for label $gtLabel.")
                continue
            }

            val source = if (tokens.size > 3) tokens[3] else "CSV Import"
            labels.add(
                ReferenceLabel(
                    specimenId = specimenId,
                    groundTruthLabel = gtLabel,
                    groundTruthClass = gtClass,
                    cohortId = defaultCohortId,
                    source = source
                )
            )
        }

        return ImportResult(
            isSuccess = errors.isEmpty() && labels.isNotEmpty(),
            importedLabels = labels,
            errorMessages = errors,
            cohortId = defaultCohortId
        )
    }

    /**
     * Reads and parses input stream as UTF-8 JSON.
     */
    fun parseJsonStream(inputStream: InputStream): ImportResult {
        val text = inputStream.bufferedReader(StandardCharsets.UTF_8).use { it.readText() }
        return parseJson(text)
    }

    /**
     * Reads and parses input stream as UTF-8 CSV.
     */
    fun parseCsvStream(inputStream: InputStream, defaultCohortId: String = "CSV_COHORT"): ImportResult {
        val text = inputStream.bufferedReader(StandardCharsets.UTF_8).use { it.readText() }
        return parseCsv(text, defaultCohortId)
    }
}
