param(
    [string]$EnvName = "ioi-env"
)

Write-Host "Creating/activating virtual environment '$EnvName'..."

# Create venv folder if it doesn't exist
if (-not (Test-Path $EnvName)) {
    python -m venv $EnvName
}

# Activate venv in current session
. .\$EnvName\Scripts\Activate.ps1

Write-Host "Installing Python dependencies from requirements.txt..."
Write-Host "(use `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124` if you have a compatible NVIDIA GPU for better performance)" -ForegroundColor Yellow

pip install --upgrade pip
pip install -r requirements.txt

Write-Host "Environment setup complete. To activate later, run:" -ForegroundColor Green
Write-Host ". .\\$EnvName\\Scripts\\Activate.ps1" -ForegroundColor Yellow
Write-Host "Then run:  py .\\path_patching\\IOI_pathpatching_v2.py" -ForegroundColor Yellow
