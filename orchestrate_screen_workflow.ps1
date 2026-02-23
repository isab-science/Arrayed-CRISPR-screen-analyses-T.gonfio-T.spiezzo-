param(
    [string]$RawDir,

    [string]$LayoutCsv,

    [string]$GenomicsExcel,

    [string]$OutputDir = "results",
    [string]$SkylineSheet = "skylineplot2",
    [int]$SkipFret = 38,
    [int]$SkipGlo = 9,
    [string]$HeatmapPlate = "1",
    [switch]$Gui,
    [switch]$NoAutoScan,
    [switch]$DebugMode
)

$ErrorActionPreference = "Stop"
$debugEnv = if ($null -eq $env:PRPCSCREEN_DEBUG) { "" } else { $env:PRPCSCREEN_DEBUG.ToString().Trim().ToLower() }
$script:IsDebugEnabled = $DebugMode -or @("1", "true", "yes", "on").Contains($debugEnv)

function Resolve-DefaultDataRoot {
    $suffix = "Neuropathology - Manuscripts\TrevisanWang2024\Data\ScreenResults"
    $base = $env:USERPROFILE

    # Prefer an existing filesystem directory matching "... UZH" to avoid any encoding issues.
    $candidates = Get-ChildItem -Path $base -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "Universit* UZH" } |
        ForEach-Object { Join-Path $_.FullName $suffix }

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    $universityFolder = "Universit$([char]0x00E4)t Z$([char]0x00FC)rich UZH"
    return Join-Path $base "$universityFolder\$suffix"
}

$DefaultDataRoot = Resolve-DefaultDataRoot

function Emit-LogLine {
    param(
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Message,
        [scriptblock]$Logger
    )
    Write-Host $Message
    if ($Logger) {
        & $Logger $Message
    }
}

function Emit-DebugLine {
    param(
        [Parameter(Mandatory = $true)][string]$Message,
        [scriptblock]$Logger
    )
    if (-not $script:IsDebugEnabled) {
        return
    }
    $debugText = "[DEBUG] $Message"
    Emit-LogLine $debugText $Logger
}

function Invoke-PythonTask {
    param(
        [Parameter(Mandatory = $true)][string]$PythonExe,
        [string[]]$PythonPrefixArgs = @(),
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$ScriptPath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [switch]$EnableDebug,
        [scriptblock]$Logger
    )

    # Compose the final argument list, optionally enabling script-level debug mode.
    $taskArgs = @($Arguments)
    if ($EnableDebug) {
        $taskArgs += "--debug"
    }

    Emit-LogLine "" $Logger
    Emit-LogLine "==> $Name" $Logger
    $displayCmd = @($PythonExe) + $PythonPrefixArgs + @($ScriptPath) + ($taskArgs | ForEach-Object { """$_""" })
    Emit-LogLine ("    {0}" -f ($displayCmd -join " ")) $Logger
    Emit-DebugLine "Launching $Name with script path $ScriptPath" $Logger

    $previousErrorAction = $ErrorActionPreference
    $hasNativeErrorPreference = Test-Path Variable:PSNativeCommandUseErrorActionPreference
    if ($hasNativeErrorPreference) {
        $previousNativeErrorPreference = $PSNativeCommandUseErrorActionPreference
        $PSNativeCommandUseErrorActionPreference = $false
    }
    try {
        $ErrorActionPreference = "Continue"
        & $PythonExe @PythonPrefixArgs $ScriptPath @taskArgs 2>&1 | ForEach-Object {
            Emit-LogLine "    $_" $Logger
        }
    } finally {
        $ErrorActionPreference = $previousErrorAction
        if ($hasNativeErrorPreference) {
            $PSNativeCommandUseErrorActionPreference = $previousNativeErrorPreference
        }
    }
    $exitCode = $LASTEXITCODE
    Emit-DebugLine "Raw exit code for ${Name}: $exitCode" $Logger
    if ($exitCode -ne 0) {
        throw "Step failed: $Name (exit code $exitCode)"
    }
    Emit-DebugLine "Completed $Name successfully" $Logger
}

function Start-ScreenWorkflow {
    param(
        [Parameter(Mandatory = $true)][string]$RawDirPath,
        [Parameter(Mandatory = $true)][string]$LayoutCsvPath,
        [Parameter(Mandatory = $true)][string]$GenomicsExcelPath,
        [Parameter(Mandatory = $true)][string]$OutputDirPath,
        [Parameter(Mandatory = $true)][string]$SkylineSheetName,
        [Parameter(Mandatory = $true)][int]$SkipFretValue,
        [Parameter(Mandatory = $true)][int]$SkipGloValue,
        [Parameter(Mandatory = $true)][string]$HeatmapPlateValue,
        [scriptblock]$Logger
    )

    # Resolve interpreter precedence so the same script works across environments.
    $pythonExe = $null
    $pythonPrefixArgs = @()
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $pythonExe = "python"
    } elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
        $pythonExe = "python3"
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        $pythonExe = "py"
        $pythonPrefixArgs = @("-3")
    } else {
        throw "Python is not available on PATH. Install Python 3.10+ and retry."
    }
    Emit-DebugLine "Selected Python executable: $pythonExe $($pythonPrefixArgs -join ' ')" $Logger

    # Validate required input paths before launching any analysis stage.
    if (-not (Test-Path $RawDirPath)) { throw "Raw input not found: $RawDirPath" }
    if (-not (Test-Path $LayoutCsvPath)) { throw "LayoutCsv not found: $LayoutCsvPath" }
    if (-not (Test-Path $GenomicsExcelPath)) { throw "GenomicsExcel not found: $GenomicsExcelPath" }

    $rawPathItem = Get-Item -LiteralPath $RawDirPath
    $rawDirForIntegration = if ($rawPathItem.PSIsContainer) { $rawPathItem.FullName } else { Split-Path -Parent $rawPathItem.FullName }
    if (-not $rawPathItem.PSIsContainer) {
        Emit-LogLine "Raw input is a file; using parent directory for integration: $rawDirForIntegration" $Logger
    }
    Emit-DebugLine "Raw integration directory: $rawDirForIntegration" $Logger

    # Prepare deterministic output locations used by all downstream stages.
    $figDir = Join-Path $OutputDirPath "figures"
    New-Item -ItemType Directory -Force -Path $OutputDirPath, $figDir | Out-Null

    $integratedCsv = Join-Path $OutputDirPath "01_integrated.csv"
    $analyzedCsv = Join-Path $OutputDirPath "02_analyzed.csv"
    $hitsCsv = Join-Path $OutputDirPath "03_hits.csv"
    Emit-DebugLine "Output files: integrated=$integratedCsv analyzed=$analyzedCsv hits=$hitsCsv" $Logger

    # Fail fast when the provided workbook cannot drive skyline plotting.
    try {
        Invoke-PythonTask $pythonExe $pythonPrefixArgs "Validate genomics workbook for skyline" "prpcscreen/scripts/plot_genomic_signal_skyline.py" @($GenomicsExcelPath, "--sheet", $SkylineSheetName, "--validate-only") -EnableDebug:$script:IsDebugEnabled $Logger
    } catch {
        throw "Genomics XLSX is not skyline-compatible: $GenomicsExcelPath. Expected columns are Gene_symbol, Mean_log2FC, Chromosome, Start_Position in the requested sheet (or any fallback sheet)."
    }

    # Execute the pipeline as explicit stages to preserve traceability and restartability.
    $steps = @(
        @{
            Name = "Integrate raw data"
            Script = "prpcscreen/scripts/merge_assay_exports.py"
            Arguments = @($rawDirForIntegration, $LayoutCsvPath, $integratedCsv, "--skip-fret", "$SkipFretValue", "--skip-glo", "$SkipGloValue")
        },
        @{
            Name = "Analyze integrated data"
            Script = "prpcscreen/scripts/compute_screen_metrics.py"
            Arguments = @($integratedCsv, $analyzedCsv, "--hits_csv", $hitsCsv)
        },
        @{
            Name = "Plate quality controls"
            Script = "prpcscreen/scripts/plot_plate_health.py"
            Arguments = @($analyzedCsv, (Join-Path $figDir "plate_qc_ssmd_controls.png"), "--interactive-only")
        },
        @{
            Name = "Plate well trajectory plot"
            Script = "prpcscreen/scripts/plot_well_trajectories.py"
            Arguments = @($analyzedCsv, (Join-Path $figDir "plate_well_series_raw_rep1.png"), "--column", "Raw_rep1", "--interactive-only")
        },
        @{
            Name = "Replicate agreement diagnostics"
            Script = "prpcscreen/scripts/plot_replicate_agreement.py"
            Arguments = @($analyzedCsv, (Join-Path $figDir "replicate_agreement_log2fc.png"), "--stem", "Log2FC")
        },
        @{
            Name = "Signal distribution histogram"
            Script = "prpcscreen/scripts/plot_signal_distributions.py"
            Arguments = @(
                $analyzedCsv,
                "--output_html",
                (Join-Path $figDir "distribution_log2fc_rep1_interactive.html"),
                "--column",
                "Log2FC_rep1"
            )
        },
        @{
            Name = "Candidate landscape plots"
            Script = "prpcscreen/scripts/plot_candidate_landscape.py"
            Arguments = @(
                $analyzedCsv,
                (Join-Path $figDir "candidate_flashlight_ranked_meanlog2.png"),
                "--volcano_html",
                (Join-Path $figDir "candidate_volcano_interactive.html"),
                "--genomics_excel",
                $GenomicsExcelPath
            )
        },
        @{
            Name = "Heatmap + violin/box plot"
            Script = "prpcscreen/scripts/plot_spatial_and_group_views.py"
            Arguments = @($analyzedCsv, (Join-Path $figDir "plate_heatmap_raw_rep1.png"), (Join-Path $figDir "grouped_boxplot_raw_rep1.png"), "--plate", $HeatmapPlateValue)
        },
        @{
            Name = "Genomic signal skyline plot"
            Script = "prpcscreen/scripts/plot_genomic_signal_skyline.py"
            Arguments = @($GenomicsExcelPath, (Join-Path $figDir "genomic_skyline_meanlog2fc.png"), "--sheet", $SkylineSheetName)
        }
    )

    $progressActivity = "PrPC Screen Workflow"
    $totalSteps = $steps.Count
    $pipelineTimer = [System.Diagnostics.Stopwatch]::StartNew()
    Emit-LogLine ("Pipeline has {0} steps." -f $totalSteps) $Logger

    try {
        for ($i = 0; $i -lt $totalSteps; $i++) {
            $step = $steps[$i]
            $stepNumber = $i + 1
            $startPercent = [int][Math]::Floor((($stepNumber - 1) / $totalSteps) * 100)
            $donePercent = [int][Math]::Floor(($stepNumber / $totalSteps) * 100)

            Write-Progress -Activity $progressActivity -Status ("Step {0}/{1}: {2}" -f $stepNumber, $totalSteps, $step.Name) -PercentComplete $startPercent
            Emit-LogLine ("Progress [{0}/{1}] Starting: {2}" -f $stepNumber, $totalSteps, $step.Name) $Logger

            $stepTimer = [System.Diagnostics.Stopwatch]::StartNew()
            Invoke-PythonTask $pythonExe $pythonPrefixArgs $step.Name $step.Script $step.Arguments -EnableDebug:$script:IsDebugEnabled $Logger
            $stepTimer.Stop()

            Write-Progress -Activity $progressActivity -Status ("Completed step {0}/{1}: {2}" -f $stepNumber, $totalSteps, $step.Name) -PercentComplete $donePercent
            Emit-LogLine ("Progress [{0}/{1}] Done: {2} ({3:N1}s)" -f $stepNumber, $totalSteps, $step.Name, $stepTimer.Elapsed.TotalSeconds) $Logger
        }
    } finally {
        Write-Progress -Activity $progressActivity -Completed
    }

    $pipelineTimer.Stop()

    Emit-LogLine "" $Logger
    Emit-LogLine "Pipeline completed." $Logger
    Emit-LogLine ("Total runtime: {0:N1} seconds" -f $pipelineTimer.Elapsed.TotalSeconds) $Logger
    Emit-LogLine "Outputs:" $Logger
    Emit-LogLine "  $integratedCsv" $Logger
    Emit-LogLine "  $analyzedCsv" $Logger
    Emit-LogLine "  $hitsCsv" $Logger
    Emit-LogLine "  $figDir" $Logger
}

function Open-ScreenWorkflowGui {
    # Build a simple operator UI for selecting inputs, launching the workflow, and viewing figures.
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing

    $form = New-Object System.Windows.Forms.Form
    $form.Text = "PrPC Screen Runner"
    $form.Size = New-Object System.Drawing.Size(1280, 820)
    $form.StartPosition = "CenterScreen"

    $font = New-Object System.Drawing.Font("Segoe UI", 9)
    $form.Font = $font

    $leftPanel = New-Object System.Windows.Forms.Panel
    $leftPanel.Location = New-Object System.Drawing.Point(10, 10)
    $leftPanel.Size = New-Object System.Drawing.Size(560, 760)
    $leftPanel.Anchor = "Top,Bottom,Left"
    $form.Controls.Add($leftPanel)

    $rightPanel = New-Object System.Windows.Forms.Panel
    $rightPanel.Location = New-Object System.Drawing.Point(580, 10)
    $rightPanel.Size = New-Object System.Drawing.Size(680, 760)
    $rightPanel.Anchor = "Top,Bottom,Left,Right"
    $form.Controls.Add($rightPanel)

    function New-Label([string]$text, [int]$x, [int]$y, [int]$w = 120) {
        $lbl = New-Object System.Windows.Forms.Label
        $lbl.Text = $text
        $lbl.Location = New-Object System.Drawing.Point($x, $y)
        $lbl.Size = New-Object System.Drawing.Size($w, 22)
        return $lbl
    }

    function New-TextBox([int]$x, [int]$y, [int]$w = 330, [string]$value = "") {
        $tb = New-Object System.Windows.Forms.TextBox
        $tb.Location = New-Object System.Drawing.Point($x, $y)
        $tb.Size = New-Object System.Drawing.Size($w, 24)
        $tb.Text = $value
        return $tb
    }

    function New-Combo([int]$x, [int]$y, [int]$w = 330) {
        $cb = New-Object System.Windows.Forms.ComboBox
        $cb.Location = New-Object System.Drawing.Point($x, $y)
        $cb.Size = New-Object System.Drawing.Size($w, 24)
        $cb.DropDownStyle = "DropDown"
        return $cb
    }

    function New-Button([string]$text, [int]$x, [int]$y, [int]$w = 85) {
        $btn = New-Object System.Windows.Forms.Button
        $btn.Text = $text
        $btn.Location = New-Object System.Drawing.Point($x, $y)
        $btn.Size = New-Object System.Drawing.Size($w, 24)
        return $btn
    }

    $leftPanel.Controls.Add((New-Label "Data root" 0 5))
    $tbRoot = New-TextBox 125 0 330 $DefaultDataRoot
    $btnRoot = New-Button "Browse..." 465 0
    $leftPanel.Controls.Add($tbRoot)
    $leftPanel.Controls.Add($btnRoot)

    $leftPanel.Controls.Add((New-Label "Raw dir/file" 0 38))
    $cbRaw = New-Combo 125 33
    $btnRaw = New-Button "Browse..." 465 33
    $leftPanel.Controls.Add($cbRaw)
    $leftPanel.Controls.Add($btnRaw)

    $leftPanel.Controls.Add((New-Label "Layout CSV" 0 71))
    $cbLayout = New-Combo 125 66
    $btnLayout = New-Button "Browse..." 465 66
    $leftPanel.Controls.Add($cbLayout)
    $leftPanel.Controls.Add($btnLayout)

    $leftPanel.Controls.Add((New-Label "Genomics XLSX" 0 104))
    $cbExcel = New-Combo 125 99
    $btnExcel = New-Button "Browse..." 465 99
    $leftPanel.Controls.Add($cbExcel)
    $leftPanel.Controls.Add($btnExcel)

    $leftPanel.Controls.Add((New-Label "Output dir" 0 137))
    $tbOutput = New-TextBox 125 132 330 "results"
    $btnOutput = New-Button "Browse..." 465 132
    $leftPanel.Controls.Add($tbOutput)
    $leftPanel.Controls.Add($btnOutput)

    $leftPanel.Controls.Add((New-Label "Sheet" 0 170))
    $tbSheet = New-TextBox 125 165 120 "skylineplot2"
    $leftPanel.Controls.Add($tbSheet)

    $leftPanel.Controls.Add((New-Label "Skip FRET" 260 170 70))
    $tbSkipFret = New-TextBox 330 165 50 "38"
    $leftPanel.Controls.Add($tbSkipFret)

    $leftPanel.Controls.Add((New-Label "Skip GLO" 390 170 70))
    $tbSkipGlo = New-TextBox 450 165 45 "9"
    $leftPanel.Controls.Add($tbSkipGlo)

    $leftPanel.Controls.Add((New-Label "Heatmap plate" 0 203))
    $tbPlate = New-TextBox 125 198 60 "1"
    $leftPanel.Controls.Add($tbPlate)

    $btnScan = New-Button "Auto-fill Paths" 210 198 120
    $btnRun = New-Button "Run Pipeline" 310 198 120
    $leftPanel.Controls.Add($btnScan)
    $leftPanel.Controls.Add($btnRun)

    $logBox = New-Object System.Windows.Forms.TextBox
    $logBox.Location = New-Object System.Drawing.Point(0, 235)
    $logBox.Size = New-Object System.Drawing.Size(550, 520)
    $logBox.Multiline = $true
    $logBox.ScrollBars = "Vertical"
    $logBox.ReadOnly = $true
    $logBox.Anchor = "Top,Bottom,Left,Right"
    $leftPanel.Controls.Add($logBox)

    $rightPanel.Controls.Add((New-Label "Generated figures" 0 5 130))

    $figList = New-Object System.Windows.Forms.ListBox
    $figList.Location = New-Object System.Drawing.Point(0, 30)
    $figList.Size = New-Object System.Drawing.Size(260, 700)
    $figList.Anchor = "Top,Bottom,Left"
    $rightPanel.Controls.Add($figList)

    $picture = New-Object System.Windows.Forms.PictureBox
    $picture.Location = New-Object System.Drawing.Point(270, 30)
    $picture.Size = New-Object System.Drawing.Size(400, 700)
    $picture.SizeMode = "Zoom"
    $picture.BorderStyle = "FixedSingle"
    $picture.Anchor = "Top,Bottom,Left,Right"
    $rightPanel.Controls.Add($picture)

    $btnOpenFigure = New-Button "Open Figure" 150 4 110
    $rightPanel.Controls.Add($btnOpenFigure)

    $toolTip = New-Object System.Windows.Forms.ToolTip
    $toolTip.AutoPopDelay = 60000
    $toolTip.InitialDelay = 300
    $toolTip.ReshowDelay = 200
    $toolTip.ShowAlways = $true
    $toolTip.SetToolTip($tbRoot, "Top-level folder used by 'Auto-fill Paths' to find candidates. Set this to your campaign folder (for example ...\\Data\\ScreenResults).")
    $toolTip.SetToolTip($btnRoot, "Browse for Data root.")
    $toolTip.SetToolTip($cbRaw, "Required. Raw assay source for integration. Use a folder containing exports (recommended) or a single raw file. The pipeline searches this location for FRET/GLO files.")
    $toolTip.SetToolTip($btnRaw, "Browse for Raw dir/file. You can pick a file first, then a folder if needed.")
    $toolTip.SetToolTip($cbLayout, "Required. Plate layout/annotation CSV. Must contain layout/control columns (for example Well_number_384, Plate_number_384, Is_NT_ctrl, Is_pos_ctrl).")
    $toolTip.SetToolTip($btnLayout, "Browse for the layout CSV.")
    $toolTip.SetToolTip($cbExcel, "Required. Genomics workbook for Skyline plot. Must contain Gene_symbol, Mean_log2FC, Chromosome, Start_Position (not Layout.xlsx).")
    $toolTip.SetToolTip($btnExcel, "Browse for genomics workbook (.xlsx/.xls).")
    $toolTip.SetToolTip($tbOutput, "Required. Output directory for all generated files. Main outputs: 01_integrated.csv, 02_analyzed.csv, 03_hits.csv, and figures\\*.png.")
    $toolTip.SetToolTip($btnOutput, "Choose output folder.")
    $toolTip.SetToolTip($tbSheet, "Optional. Worksheet name for Skyline plot. Default: skylineplot2. If missing or incompatible, pipeline auto-selects a compatible sheet and logs a warning.")
    $toolTip.SetToolTip($tbSkipFret, "Optional integer. Header lines to skip when reading FRET exports. Default: 38. Change only if your instrument export format differs.")
    $toolTip.SetToolTip($tbSkipGlo, "Optional integer. Header lines to skip when reading GLO exports. Default: 9. Change only if your export format differs.")
    $toolTip.SetToolTip($tbPlate, "Optional selector for heatmap plates (step 8). Supported: single number (1), range (1-4), series (1,2,6), or 'all'.")
    $toolTip.SetToolTip($btnScan, "Scan Data root and auto-fill candidate values for Raw/Layout/Genomics fields.")
    $toolTip.SetToolTip($btnRun, "Run all 9 pipeline steps with current values.")
    $toolTip.SetToolTip($figList, "Generated figures in output\\figures. Select one to preview.")
    $toolTip.SetToolTip($btnOpenFigure, "Open the selected figure with your default image viewer.")

    $setLog = {
        param([string]$line)
        $logBox.AppendText("$line`r`n")
        $logBox.SelectionStart = $logBox.TextLength
        $logBox.ScrollToCaret()
        [System.Windows.Forms.Application]::DoEvents()
    }

    $showInstructions = {
        & $setLog "How to use this window:"
        & $setLog "1) Data root (required for scan): folder where your campaign files live."
        & $setLog "   Example: ...\\Data\\ScreenResults"
        & $setLog "2) Click 'Auto-fill Paths' to auto-populate file candidates."
        & $setLog "3) Raw dir/file (required): folder containing raw exports (recommended) or one raw file."
        & $setLog "4) Layout CSV (required): annotation/layout table with plate + control metadata."
        & $setLog "5) Genomics XLSX (required): workbook for the Skyline figure."
        & $setLog "   Required columns: Gene_symbol, Mean_log2FC, Chromosome, Start_Position."
        & $setLog "   Do not use the plate layout workbook (Layout.xlsx)."
        & $setLog "6) Output dir (required): destination for CSV outputs and figures."
        & $setLog ""
        & $setLog "Optional parameters:"
        & $setLog "- Sheet: worksheet for Skyline plot. Default: skylineplot2."
        & $setLog "  If the sheet is missing or incompatible, a compatible sheet is auto-selected with a warning."
        & $setLog "- Skip FRET: header rows skipped when reading FRET files. Default: 38."
        & $setLog "- Skip GLO: header rows skipped when reading GLO files. Default: 9."
        & $setLog "- Heatmap plate: selector for heatmap output. Default: 1."
        & $setLog "  Supported: single number (1), range (1-4), series (1,2,6), or 'all'."
        & $setLog ""
        & $setLog "What happens when you click 'Run Pipeline':"
        & $setLog "- Step 1: Integrate raw data + layout -> 01_integrated.csv"
        & $setLog "- Step 2: Compute metrics/hits -> 02_analyzed.csv + 03_hits.csv"
        & $setLog "- Steps 3-9: Generate figures in output\\figures\\"
        & $setLog "Use mouse hover on each field for detailed guidance."
        & $setLog ""
    }

    $validateInputs = {
        $issues = @()

        $rawValue = $cbRaw.Text.Trim()
        $layoutValue = $cbLayout.Text.Trim()
        $excelValue = $cbExcel.Text.Trim()
        $outputValue = $tbOutput.Text.Trim()
        $sheetValue = $tbSheet.Text.Trim()
        $plateValue = $tbPlate.Text.Trim()

        if ([string]::IsNullOrWhiteSpace($rawValue)) {
            $issues += "Raw dir/file is required."
        } elseif (-not (Test-Path -LiteralPath $rawValue)) {
            $issues += "Raw dir/file not found: $rawValue"
        }

        if ([string]::IsNullOrWhiteSpace($layoutValue)) {
            $issues += "Layout CSV path is required."
        } elseif (-not (Test-Path -LiteralPath $layoutValue)) {
            $issues += "Layout CSV not found: $layoutValue"
        } else {
            try {
                $layoutHeader = (Get-Content -LiteralPath $layoutValue -TotalCount 1 -ErrorAction Stop)
                $h = if ($null -eq $layoutHeader) { "" } else { $layoutHeader.ToString().ToLowerInvariant() }
                $looksLikeLayout = $h.Contains("well_number_384") -or $h.Contains("plate_number_384") -or $h.Contains("is_nt_ctrl") -or $h.Contains("is_pos_ctrl")
                if (-not $looksLikeLayout) {
                    $issues += "Layout CSV does not look like a layout table (missing expected columns in header). Selected file may be a raw export."
                }
            } catch {
                $issues += "Layout CSV header could not be read: $layoutValue"
            }
        }

        if ([string]::IsNullOrWhiteSpace($excelValue)) {
            $issues += "Genomics XLSX path is required."
        } elseif (-not (Test-Path -LiteralPath $excelValue)) {
            $issues += "Genomics XLSX not found: $excelValue"
        } else {
            $excelLeaf = [System.IO.Path]::GetFileName($excelValue).ToLowerInvariant()
            $excelPathText = $excelValue.ToLowerInvariant()
            $looksLikeLayoutWorkbook = ($excelLeaf -match "^layout(\\.xlsx|\\.xls)?$") -or ($excelPathText -match "\\\\layout\\\\")
            if ($looksLikeLayoutWorkbook) {
                $issues += "Genomics XLSX appears to be a layout workbook. Select a genomics workbook with Gene_symbol, Mean_log2FC, Chromosome, Start_Position (for example GeneticLocation.xlsx)."
            }
        }

        if ([string]::IsNullOrWhiteSpace($outputValue)) {
            $issues += "Output dir is required."
        }

        if ([string]::IsNullOrWhiteSpace($sheetValue)) {
            $issues += "Sheet name is required."
        }

        if ([string]::IsNullOrWhiteSpace($plateValue)) {
            $issues += "Heatmap plate is required."
        } else {
            $plateNorm = $plateValue.ToLowerInvariant()
            $isHeatmapSelectorValid = $plateNorm -eq "all" -or `
                ($plateValue -match "^\d+$") -or `
                ($plateValue -match "^\d+\s*-\s*\d+$") -or `
                ($plateValue -match "^\d+(\s*,\s*\d+)+$")
            if (-not $isHeatmapSelectorValid) {
                $issues += "Heatmap plate must be one number (1), a range (1-4), a series (1,2,6), or 'all'."
            }
        }

        $tmp = 0
        if (-not [int]::TryParse($tbSkipFret.Text, [ref]$tmp)) {
            $issues += "Skip FRET must be an integer."
        }
        if (-not [int]::TryParse($tbSkipGlo.Text, [ref]$tmp)) {
            $issues += "Skip GLO must be an integer."
        }

        return ,$issues
    }

    $refreshFigures = {
        $figList.Items.Clear()
        $figDir = Join-Path $tbOutput.Text "figures"
        if (Test-Path $figDir) {
            Get-ChildItem -Path $figDir -File -Filter *.png | Sort-Object Name | ForEach-Object {
                [void]$figList.Items.Add($_.FullName)
            }
        }
    }

    $scanRoot = {
        try {
            $scanTimer = [System.Diagnostics.Stopwatch]::StartNew()
            $root = $tbRoot.Text.Trim()
            $cbRaw.Items.Clear()
            $cbLayout.Items.Clear()
            $cbExcel.Items.Clear()

            if ([string]::IsNullOrWhiteSpace($root)) {
                throw "Data root is empty."
            }
            if (-not (Test-Path -LiteralPath $root)) {
                [System.Windows.Forms.MessageBox]::Show("Data root not found: $root", "Auto-fill Paths", "OK", "Warning") | Out-Null
                return
            }

            & $setLog ("Scanning root: {0}" -f $root)

            [void]$cbRaw.Items.Add($root)

            $dirs = Get-ChildItem -LiteralPath $root -Directory -ErrorAction SilentlyContinue | Sort-Object Name
            foreach ($dir in $dirs) {
                [void]$cbRaw.Items.Add($dir.FullName)
            }

            $parentRoot = Split-Path -Parent $root
            if (-not [string]::IsNullOrWhiteSpace($parentRoot) -and (Test-Path -LiteralPath $parentRoot) -and ($parentRoot -ne $root)) {
                $parentDirs = Get-ChildItem -LiteralPath $parentRoot -Directory -ErrorAction SilentlyContinue |
                    Where-Object { $_.FullName -ne $root } |
                    Sort-Object Name
                foreach ($dir in $parentDirs) {
                    [void]$cbRaw.Items.Add($dir.FullName)
                }
            }

            $allFiles = @()
            # Root is scanned recursively; parent is scanned as direct files only.
            # This keeps auto-fill responsive even when parent has many large sibling trees.
            $allFiles += Get-ChildItem -LiteralPath $root -File -Recurse -ErrorAction SilentlyContinue
            if (-not [string]::IsNullOrWhiteSpace($parentRoot) -and (Test-Path -LiteralPath $parentRoot) -and ($parentRoot -ne $root)) {
                $allFiles += Get-ChildItem -LiteralPath $parentRoot -File -ErrorAction SilentlyContinue
                $parentChildFiles = Get-ChildItem -LiteralPath $parentRoot -Directory -ErrorAction SilentlyContinue |
                    Where-Object { $_.FullName -ne $root } |
                    ForEach-Object {
                        Get-ChildItem -LiteralPath $_.FullName -File -ErrorAction SilentlyContinue
                    }
                $allFiles += $parentChildFiles
            }
            $allFiles = $allFiles |
                Where-Object {
                    @(".csv", ".txt", ".tsv", ".xlsx", ".xls") -contains $_.Extension.ToLowerInvariant()
                } |
                Sort-Object FullName -Unique
            $rawFiles = $allFiles | Where-Object { $_.Extension -in ".csv", ".txt", ".tsv" } | Sort-Object FullName
            $layoutFiles = $allFiles | Where-Object { $_.Extension -ieq ".csv" } | Sort-Object FullName
            $excelFiles = $allFiles | Where-Object { $_.Extension -in ".xlsx", ".xls" } | Sort-Object FullName

            $rankedLayout = foreach ($item in $layoutFiles) {
                $name = $item.Name.ToLowerInvariant()
                $pathText = $item.FullName.ToLowerInvariant()
                $score = 0
                if ($name -match "layout|annotation|annot|plate|map") { $score += 3 }
                if ($pathText -match "\\\\layout\\\\") { $score += 2 }
                if ($name -match "fret|tr-fret|glo|raw|edge") { $score -= 3 }
                [PSCustomObject]@{
                    FullName = $item.FullName
                    Score = $score
                }
            }
            $rankedLayout = @(
                $rankedLayout | Sort-Object @{ Expression = "Score"; Descending = $true }, @{ Expression = "FullName"; Descending = $false }
            )
            for ($i = 0; $i -lt $rankedLayout.Count; $i++) {
                $entry = $rankedLayout[$i]
                try {
                    $header = (Get-Content -LiteralPath $entry.FullName -TotalCount 1 -ErrorAction Stop)
                    $headerText = if ($null -eq $header) { "" } else { $header.ToString().ToLowerInvariant() }
                    if ($headerText.Contains("well_number_384")) { $entry.Score += 6 }
                    if ($headerText.Contains("plate_number_384")) { $entry.Score += 4 }
                    if ($headerText.Contains("is_nt_ctrl")) { $entry.Score += 4 }
                    if ($headerText.Contains("is_pos_ctrl")) { $entry.Score += 4 }
                } catch {
                    $entry.Score -= 1
                }
            }
            $rankedLayout = $rankedLayout | Sort-Object @{ Expression = "Score"; Descending = $true }, @{ Expression = "FullName"; Descending = $false }
            $rankedExcel = foreach ($item in $excelFiles) {
                $name = $item.Name.ToLowerInvariant()
                $pathText = $item.FullName.ToLowerInvariant()
                $score = 0
                if ($name -match "genetic|genomic|chrom|location|skyline") { $score += 6 }
                if ($name -match "layout|plate|map") { $score -= 4 }
                if ($pathText -match "\\\\layout\\\\") { $score -= 4 }
                [PSCustomObject]@{
                    FullName = $item.FullName
                    Score = $score
                }
            }
            $rankedExcel = $rankedExcel | Sort-Object @{ Expression = "Score"; Descending = $true }, @{ Expression = "FullName"; Descending = $false }

            foreach ($item in $rawFiles) {
                [void]$cbRaw.Items.Add($item.FullName)
            }
            foreach ($item in $rankedLayout) {
                [void]$cbLayout.Items.Add($item.FullName)
            }
            foreach ($item in $rankedExcel) {
                [void]$cbExcel.Items.Add($item.FullName)
            }

            if ($cbRaw.Items.Count -gt 0 -and [string]::IsNullOrWhiteSpace($cbRaw.Text)) { $cbRaw.Text = [string]$cbRaw.Items[0] }
            if ($cbLayout.Items.Count -gt 0) {
                $cbLayout.Text = [string]$cbLayout.Items[0]
            }
            if ($cbExcel.Items.Count -gt 0) {
                $currentExcel = $cbExcel.Text
                if ([string]::IsNullOrWhiteSpace($currentExcel) -or -not ($cbExcel.Items -contains $currentExcel)) {
                    $cbExcel.Text = [string]$cbExcel.Items[0]
                }
            }

            $scanTimer.Stop()
            & $setLog ("Scan complete in {3:N2}s. Raw candidates: {0}; Layout CSVs: {1}; Genomics Excel files: {2}" -f $cbRaw.Items.Count, $cbLayout.Items.Count, $cbExcel.Items.Count, $scanTimer.Elapsed.TotalSeconds)
            if ($cbLayout.Items.Count -gt 0) {
                & $setLog ("Auto-selected Layout CSV: {0}" -f $cbLayout.Text)
            }
            if ($cbExcel.Items.Count -gt 0) {
                & $setLog ("Auto-selected Genomics XLSX: {0}" -f $cbExcel.Text)
            }
        } catch {
            $msg = $_.Exception.Message
            & $setLog ("Scan failed: {0}" -f $msg)
            [System.Windows.Forms.MessageBox]::Show(("Auto-fill Paths failed:`r`n{0}" -f $msg), "Auto-fill Paths Error", "OK", "Error") | Out-Null
        }
    }

    $btnRoot.Add_Click({
        $dlg = New-Object System.Windows.Forms.FolderBrowserDialog
        $dlg.SelectedPath = $tbRoot.Text
        if ($dlg.ShowDialog() -eq "OK") {
            $tbRoot.Text = $dlg.SelectedPath
            & $scanRoot
        }
    })
    $btnRaw.Add_Click({
        $fileDlg = New-Object System.Windows.Forms.OpenFileDialog
        $fileDlg.InitialDirectory = if (Test-Path $tbRoot.Text) { $tbRoot.Text } else { (Get-Location).Path }
        $fileDlg.Title = "Select raw data file (Cancel to choose a folder)"
        $fileDlg.Filter = "Data files (*.csv;*.tsv;*.txt)|*.csv;*.tsv;*.txt|All files (*.*)|*.*"
        if ($fileDlg.ShowDialog() -eq "OK") {
            $cbRaw.Text = $fileDlg.FileName
            return
        }

        $folderDlg = New-Object System.Windows.Forms.FolderBrowserDialog
        $folderDlg.SelectedPath = if ([string]::IsNullOrWhiteSpace($cbRaw.Text)) { $tbRoot.Text } else { $cbRaw.Text }
        if ($folderDlg.ShowDialog() -eq "OK") { $cbRaw.Text = $folderDlg.SelectedPath }
    })
    $btnLayout.Add_Click({
        $dlg = New-Object System.Windows.Forms.OpenFileDialog
        $dlg.InitialDirectory = if (Test-Path $tbRoot.Text) { $tbRoot.Text } else { (Get-Location).Path }
        $dlg.Filter = "CSV files (*.csv)|*.csv|All files (*.*)|*.*"
        if ($dlg.ShowDialog() -eq "OK") { $cbLayout.Text = $dlg.FileName }
    })
    $btnExcel.Add_Click({
        $dlg = New-Object System.Windows.Forms.OpenFileDialog
        $dlg.InitialDirectory = if (Test-Path $tbRoot.Text) { $tbRoot.Text } else { (Get-Location).Path }
        $dlg.Filter = "Excel files (*.xlsx;*.xls)|*.xlsx;*.xls|All files (*.*)|*.*"
        if ($dlg.ShowDialog() -eq "OK") { $cbExcel.Text = $dlg.FileName }
    })
    $btnOutput.Add_Click({
        $dlg = New-Object System.Windows.Forms.FolderBrowserDialog
        $dlg.SelectedPath = if ([string]::IsNullOrWhiteSpace($tbOutput.Text)) { (Get-Location).Path } else { $tbOutput.Text }
        if ($dlg.ShowDialog() -eq "OK") { $tbOutput.Text = $dlg.SelectedPath }
    })
    $btnScan.Add_Click({ & $scanRoot })

    $figList.Add_SelectedIndexChanged({
        if ($figList.SelectedItem) {
            $path = [string]$figList.SelectedItem
            if (Test-Path $path) {
                if ($picture.Image) {
                    $old = $picture.Image
                    $picture.Image = $null
                    $old.Dispose()
                }
                $stream = [System.IO.File]::Open($path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
                $image = [System.Drawing.Image]::FromStream($stream)
                $picture.Image = New-Object System.Drawing.Bitmap($image)
                $image.Dispose()
                $stream.Close()
            }
        }
    })

    $btnOpenFigure.Add_Click({
        if ($figList.SelectedItem -and (Test-Path ([string]$figList.SelectedItem))) {
            Start-Process ([string]$figList.SelectedItem)
        }
    })

    $btnRun.Add_Click({
        $issues = & $validateInputs
        if ($issues.Count -gt 0) {
            & $setLog "Validation failed. Please fix the following:"
            foreach ($issue in $issues) {
                & $setLog " - $issue"
            }
            [System.Windows.Forms.MessageBox]::Show(("Validation failed:`r`n- " + ($issues -join "`r`n- ")), "Validation", "OK", "Warning") | Out-Null
            return
        }

        $btnRun.Enabled = $false
        $btnScan.Enabled = $false
        try {
            Start-ScreenWorkflow -RawDirPath $cbRaw.Text.Trim() `
                -LayoutCsvPath $cbLayout.Text.Trim() `
                -GenomicsExcelPath $cbExcel.Text.Trim() `
                -OutputDirPath $tbOutput.Text.Trim() `
                -SkylineSheetName $tbSheet.Text.Trim() `
                -SkipFretValue ([int]$tbSkipFret.Text) `
                -SkipGloValue ([int]$tbSkipGlo.Text) `
                -HeatmapPlateValue $tbPlate.Text.Trim() `
                -Logger $setLog
            & $refreshFigures
            [System.Windows.Forms.MessageBox]::Show("Pipeline completed.", "Done", "OK", "Information") | Out-Null
        } catch {
            & $setLog "ERROR: $($_.Exception.Message)"
            [System.Windows.Forms.MessageBox]::Show("Pipeline failed: $($_.Exception.Message)", "Error", "OK", "Error") | Out-Null
        } finally {
            $btnRun.Enabled = $true
            $btnScan.Enabled = $true
        }
    })

    & $showInstructions
    if (-not $NoAutoScan) {
        & $scanRoot
    }
    & $refreshFigures
    [void]$form.ShowDialog()
}

$hasRawDirArg = -not [string]::IsNullOrWhiteSpace($RawDir)
$hasLayoutArg = -not [string]::IsNullOrWhiteSpace($LayoutCsv)
$hasGenomicsArg = -not [string]::IsNullOrWhiteSpace($GenomicsExcel)
$hasAnyCliArgs = $hasRawDirArg -or $hasLayoutArg -or $hasGenomicsArg
$hasAllCliArgs = $hasRawDirArg -and $hasLayoutArg -and $hasGenomicsArg

if ($Gui -or -not $hasAnyCliArgs) {
    if (-not $Gui -and -not $hasAnyCliArgs) {
        Emit-LogLine "No CLI arguments provided. Launching GUI mode without auto-scan. Use 'Auto-fill Paths' to load candidates." $null
        $NoAutoScan = $true
    }
    Emit-DebugLine "Launching GUI mode" $null
    Open-ScreenWorkflowGui
    exit 0
}

if (-not $hasAllCliArgs) {
    throw "Missing required CLI arguments: -RawDir, -LayoutCsv, -GenomicsExcel. Use -Gui to launch interactive mode."
}

Emit-DebugLine "Launching CLI mode" $null
Start-ScreenWorkflow -RawDirPath $RawDir `
    -LayoutCsvPath $LayoutCsv `
    -GenomicsExcelPath $GenomicsExcel `
    -OutputDirPath $OutputDir `
    -SkylineSheetName $SkylineSheet `
    -SkipFretValue $SkipFret `
    -SkipGloValue $SkipGlo `
    -HeatmapPlateValue $HeatmapPlate


