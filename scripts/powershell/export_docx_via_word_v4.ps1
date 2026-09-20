param(
    [string]$Source = "D:\DING PROJECT\FINAL PAPER\Manuscript_single_file\v4_2026-07-28\CPEC_manuscript_v4_section_and_language_refined_2026-07-28.docx",
    [string]$OutputPdf = "D:\DING PROJECT\FINAL PAPER\Manuscript_single_file\v4_2026-07-28\qa\CPEC_manuscript_v4_word_render.pdf"
)

$ErrorActionPreference = "Stop"

$temporaryDocument = "C:\Windows\Temp\CPEC_v4_QA.docx"

Copy-Item -LiteralPath $Source -Destination $temporaryDocument -Force

$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.Options.UpdateLinksAtOpen = $false

    $document = $word.Documents.Open($temporaryDocument, $false, $true, $false)
    Write-Output "Opened the V4 document in Microsoft Word."

    $pages = $document.ComputeStatistics(2)
    Write-Output "Pages=$pages"

    $document.ExportAsFixedFormat($OutputPdf, 17)
    Write-Output "PDF=$OutputPdf"
}
finally {
    if ($document) {
        $document.Close([ref]0)
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($document) | Out-Null
    }
    if ($word) {
        $word.Quit()
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
