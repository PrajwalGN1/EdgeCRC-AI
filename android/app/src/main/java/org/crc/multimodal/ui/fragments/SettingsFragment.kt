package org.crc.multimodal.ui.fragments

import android.content.Context
import android.os.Bundle
import android.view.View
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatDelegate
import androidx.fragment.app.Fragment
import com.google.android.material.button.MaterialButton
import com.google.android.material.button.MaterialButtonToggleGroup
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import org.crc.multimodal.R
import org.crc.multimodal.ui.WorkstationNavigationHost

class SettingsFragment : Fragment(R.layout.fragment_settings) {

    private var navHost: WorkstationNavigationHost? = null

    override fun onAttach(context: Context) {
        super.onAttach(context)
        if (context is WorkstationNavigationHost) {
            navHost = context
        }
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        navHost?.setTopToolbar(
            title = "Settings & About",
            subtitle = "Engine Configuration & Preferences",
            showBackButton = false
        )

        val host = navHost ?: return
        val repository = host.getRepository()

        val toggleGroupTheme: MaterialButtonToggleGroup = view.findViewById(R.id.toggleGroupTheme)
        val tvSettingsDatabaseSummary: TextView = view.findViewById(R.id.tvSettingsDatabaseSummary)
        val btnSettingsSeedReferences: MaterialButton = view.findViewById(R.id.btnSettingsSeedReferences)
        val btnSettingsClearData: MaterialButton = view.findViewById(R.id.btnSettingsClearData)

        // Setup Theme Toggle
        when (AppCompatDelegate.getDefaultNightMode()) {
            AppCompatDelegate.MODE_NIGHT_NO -> toggleGroupTheme.check(R.id.btnThemeLight)
            AppCompatDelegate.MODE_NIGHT_YES -> toggleGroupTheme.check(R.id.btnThemeDark)
            else -> toggleGroupTheme.check(R.id.btnThemeSystem)
        }

        toggleGroupTheme.addOnButtonCheckedListener { _, checkedId, isChecked ->
            if (isChecked) {
                when (checkedId) {
                    R.id.btnThemeLight -> AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_NO)
                    R.id.btnThemeDark -> AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_YES)
                    else -> AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM)
                }
            }
        }

        // Database Summary
        updateDbSummary(tvSettingsDatabaseSummary)

        btnSettingsSeedReferences.setOnClickListener {
            repository.seedReferenceCases()
            updateDbSummary(tvSettingsDatabaseSummary)
            Toast.makeText(requireContext(), "Holdout reference research cases loaded", Toast.LENGTH_SHORT).show()
        }

        btnSettingsClearData.setOnClickListener {
            MaterialAlertDialogBuilder(requireContext())
                .setTitle("Clear All Workstation Records?")
                .setMessage("Are you sure you want to delete all saved cases, runs, and review notes? This cannot be undone.")
                .setPositiveButton("Clear All") { _, _ ->
                    repository.clearAllData()
                    updateDbSummary(tvSettingsDatabaseSummary)
                    Toast.makeText(requireContext(), "All workstation records cleared", Toast.LENGTH_SHORT).show()
                }
                .setNegativeButton("Cancel", null)
                .show()
        }
    }

    private fun updateDbSummary(tv: TextView) {
        val host = navHost ?: return
        val casesCount = host.getRepository().getAllCases().size
        val auditCount = host.getRepository().getAuditTrail(1000).size
        tv.text = "Local Database: crc_workstation.db (v3) · $casesCount cases · $auditCount audit events"
    }
}
