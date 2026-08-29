; Runs before Tauri's own file-copy macros during install/upgrade.
;
; Real defect reproduced live: on a Windows currentUser NSIS install, the
; bundler copies every file the current build produces into $INSTDIR, but
; never removes files from a PREVIOUS install that a new build no longer
; produces under the SAME name. Vite emits content-hashed chunk filenames
; (e.g. dist/assets/index-<hash>.js) that change on every source edit, so a
; reinstall/upgrade leaves the previous build's now-orphaned chunk files
; sitting next to the new ones under $INSTDIR\_up_\dist\assets. The signed
; release manifest's protected_assets set is generated from a SINGLE build's
; file list, so it can never match an install directory holding the union of
; two builds' outputs -- the app's own FINAL-phase integrity self-check
; (security/integrity.py::verify_release_assets, mirrored in
; src-tauri/src/release_integrity.rs) correctly refuses to start, panicking
; with "extra protected asset: dist/assets/index-<stale-hash>.js" on first
; launch after ANY reinstall over an existing install. Not a bypassable
; check -- the fix is to never let the accumulation happen: purge the
; entire `_up_` resource tree (everything Tauri's `../`-escaped resources
; land under -- see src-tauri/src/main.rs's resolve_resource_root) before
; copying the new build's files in. `_up_` holds only build-derived
; resources (sidecar exe, frontend dist, release manifests, the bundled DB
; seed) -- never user data, which lives entirely under
; %APPDATA%\BharatTerminal\data\, outside $INSTDIR. Safe to delete
; unconditionally on every install/upgrade.
!macro NSIS_HOOK_PREINSTALL
  ; Real defect reproduced live: bharatterminal-compute.exe at $INSTDIR root
  ; (the Tauri externalBin sidecar copy) is also the executable the
  ; Task Scheduler-launched collector (BharatTerminal_SmartCollector) and its
  ; watchdog (BharatTerminal_SmartCollector_Watchdog) run from directly --
  ; independently of whether the Tauri host app is open. If either task's
  ; process tree is still running during an upgrade, Windows cannot replace
  ; the locked exe: the installer copy step silently no-ops on that one file
  ; (no error, exit code 0) and the collector keeps running the OLD sidecar
  ; binary indefinitely after an "upgrade" that the release manifest and
  ; every other artifact correctly reflect as new.
  ;
  ; A second version of this hook ended the watchdog before the main task
  ; (fixing the two-/end-calls race) but was STILL caught by a longer-range
  ; real race, reproduced live across three consecutive full installs: the
  ; `_up_` resource copy that follows this hook -- in particular the
  ; terminal.db.gz seed blob, hundreds of MB -- takes minutes, easily long
  ; enough for the watchdog's own normal 5-minute schedule to elapse and
  ; refire on its own, re-launching and re-locking $INSTDIR\bharatterminal-
  ; compute.exe between this hook's one-time /end calls and the `File
  ; /oname=bharatterminal-compute.exe` instruction that runs near the end of
  ; Section Install. NSIS's File instruction silently no-ops on a share
  ; violation (no error, exit code 0) -- every other artifact in the same
  ; install updates correctly while this one exe alone keeps serving the OLD
  ; build indefinitely, exactly the original bug, just on a longer clock.
  ; /end only stops an already-running instance; it does nothing to stop a
  ; LATER trigger from firing again before the section finishes. /disable
  ; closes that for the section's entire duration -- NSIS_HOOK_POSTINSTALL
  ; (below) re-enables both tasks once the new binary is safely in place, so
  ; normal collection resumes with no manual step required.
  nsExec::ExecToLog 'schtasks /change /tn "BharatTerminal_SmartCollector_Watchdog" /disable'
  nsExec::ExecToLog 'schtasks /change /tn "BharatTerminal_SmartCollector" /disable'
  nsExec::ExecToLog 'schtasks /end /tn "BharatTerminal_SmartCollector_Watchdog"'
  Sleep 500
  nsExec::ExecToLog 'schtasks /end /tn "BharatTerminal_SmartCollector"'
  Sleep 500
  nsExec::ExecToLog 'schtasks /end /tn "BharatTerminal_SmartCollector"'
  Sleep 500
  RMDir /r "$INSTDIR\_up_"
!macroend

; Re-enables both scheduled tasks after the new binaries are fully in place.
; Runs after the externalBin `File` copy (see NSIS_HOOK_PREINSTALL above) --
; leaving the tasks disabled after a successful install would silently stop
; all future data collection with no visible error.
!macro NSIS_HOOK_POSTINSTALL
  nsExec::ExecToLog 'schtasks /change /tn "BharatTerminal_SmartCollector" /enable'
  nsExec::ExecToLog 'schtasks /change /tn "BharatTerminal_SmartCollector_Watchdog" /enable'
!macroend
