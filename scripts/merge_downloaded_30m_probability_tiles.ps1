param(
    [string]$TilesDir = "D:\DING PROJECT\04_maps\drive_tiles_30m",
    [string]$OutDir = "D:\DING PROJECT\04_maps\probability_rasters_30m_merged"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $TilesDir)) {
    New-Item -ItemType Directory -Force -Path $TilesDir | Out-Null
}
if (-not (Test-Path -LiteralPath $OutDir)) {
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
}

$tiles = Get-ChildItem -LiteralPath $TilesDir -Filter "*.tif" -File
if ($tiles.Count -eq 0) {
    Write-Host "No GeoTIFF tiles found in $TilesDir"
    Write-Host "Download the Google Drive folder GEE_CPEC_LSM_RASTERS_30M into that folder, then run this script again."
    exit 1
}

Write-Host "Found $($tiles.Count) GeoTIFF tiles. Merging..."
python "D:\DING PROJECT\06_scripts\python\merge_probability_tiles_to_single_raster.py" --tiles-dir $TilesDir --out-dir $OutDir
Write-Host "Done. Merged rasters are in $OutDir"
