param(
    [string]$ProjectId,
    [string]$Region = "us-central1",
    [string]$ServiceName = "negation-rephraser",
    [string]$GeminiApiKey = "",
    [string]$PipelineConfigPath = "config/pipeline_config.json"
)

if ([string]::IsNullOrWhiteSpace($ProjectId)) {
    throw "Pass -ProjectId YOUR_PROJECT_ID"
}

gcloud config set project $ProjectId
gcloud config set run/region $Region

$envArgs = @()
if (-not [string]::IsNullOrWhiteSpace($GeminiApiKey)) {
    $envArgs += "GEMINI_API_KEY=$GeminiApiKey"
}

if (-not [string]::IsNullOrWhiteSpace($PipelineConfigPath)) {
    $envArgs += "PIPELINE_CONFIG_PATH=$PipelineConfigPath"
}

$joinedEnvArgs = $envArgs -join ","

if ([string]::IsNullOrWhiteSpace($joinedEnvArgs)) {
    gcloud run deploy $ServiceName --source . --allow-unauthenticated --port 8000
} else {
    gcloud run deploy $ServiceName --source . --allow-unauthenticated --port 8000 --set-env-vars $joinedEnvArgs
}
