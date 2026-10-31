package org.crc.multimodal.security

import java.util.regex.Pattern

/**
 * Diagnostic and log sanitization filter preventing Protected Health Information (PHI),
 * Personally Identifiable Information (PII), and sensitive clinical narrative text from
 * inadvertently leaking into Android Logcat or release telemetry.
 *
 * Implements strict offline security requirements mandated by WP7.5.
 */
object SensitiveLogRedactor {

    private const val MAX_LOG_LENGTH = 500

    private val EMAIL_PATTERN = Pattern.compile(
        """\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"""
    )

    private val PHONE_PATTERN = Pattern.compile(
        """\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"""
    )

    private val MRN_PATIENT_ID_PATTERN = Pattern.compile(
        """(?i)\b(?:mrn|patient[_\s]*id|pt[_\s]*id|ssn|dob|date[_\s]*of[_\s]*birth)\s*[:=]\s*([A-Za-z0-9\-]+)"""
    )

    private val CLINICAL_NARRATIVE_PATTERN = Pattern.compile(
        """(?i)\b(?:adenocarcinoma|carcinoma|resection|biopsy\s+specimen|pathology\s+report|patient\s+presents|clinical\s+history|histological\s+diagnosis)[:\s]+([^,\n;]+)"""
    )

    /**
     * Sanitizes a log string by scrubbing potential identifiers, clinical notes, and truncating
     * excessively long text payloads.
     */
    fun redact(message: String?): String {
        if (message == null) return "null"
        if (message.isBlank()) return message

        var sanitized = message

        // 1. Scrub Emails
        sanitized = EMAIL_PATTERN.matcher(sanitized).replaceAll("[REDACTED_EMAIL]")

        // 2. Scrub Phone Numbers
        sanitized = PHONE_PATTERN.matcher(sanitized).replaceAll("[REDACTED_PHONE]")

        // 3. Scrub MRN / Patient IDs
        val mrnMatcher = MRN_PATIENT_ID_PATTERN.matcher(sanitized)
        val mrnSb = StringBuffer()
        while (mrnMatcher.find()) {
            val fullMatch = mrnMatcher.group(0).orEmpty()
            val idVal = mrnMatcher.group(1).orEmpty()
            val replacement = if (idVal.isNotEmpty()) fullMatch.replace(idVal, "[REDACTED_ID]") else fullMatch
            mrnMatcher.appendReplacement(mrnSb, java.util.regex.Matcher.quoteReplacement(replacement))
        }
        mrnMatcher.appendTail(mrnSb)
        sanitized = mrnSb.toString()

        // 4. Scrub Clinical Free-Text Narrative Snippets
        val clinicalMatcher = CLINICAL_NARRATIVE_PATTERN.matcher(sanitized)
        val clinSb = StringBuffer()
        while (clinicalMatcher.find()) {
            val fullMatch = clinicalMatcher.group(0).orEmpty()
            val narrativeVal = clinicalMatcher.group(1).orEmpty()
            val replacement = if (narrativeVal.isNotEmpty()) fullMatch.replace(narrativeVal, "[REDACTED_CLINICAL_NARRATIVE]") else fullMatch
            clinicalMatcher.appendReplacement(clinSb, java.util.regex.Matcher.quoteReplacement(replacement))
        }
        clinicalMatcher.appendTail(clinSb)
        sanitized = clinSb.toString()

        // 5. Length Cap Guard (prevent inadvertent raw tensor or document dumps into logcat)
        if (sanitized.length > MAX_LOG_LENGTH) {
            sanitized = sanitized.substring(0, MAX_LOG_LENGTH) + " ... [TRUNCATED_TO_PREVENT_PII_LEAK]"
        }

        return sanitized
    }
}
