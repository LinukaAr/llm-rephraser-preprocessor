param(
    [string]$ProjectId,
    [string]$Region = "us-central1",
    [string]$ServiceName = "negation-rephraser",
    [string]$GeminiApiKey = "",
    [string]$SimilarityModel = "token-overlap",
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

if (-not [string]::IsNullOrWhiteSpace($SimilarityModel)) {
    $envArgs += "SIMILARITY_MODEL=$SimilarityModel"
}

if (-not [string]::IsNullOrWhiteSpace($PipelineConfigPath)) {
    $envArgs += "PIPELINE_CONFIG_PATH=$PipelineConfigPath"
}

$joinedEnvArgs = $envArgs -join ","

if ([string]::IsNullOrWhiteSpace($joinedEnvArgs)) {
    gcloud run deploy $ServiceName --source . --allow-unauthenticated --port 8000 --memory 1Gi
} else {
    gcloud run deploy $ServiceName --source . --allow-unauthenticated --port 8000 --memory 1Gi --set-env-vars $joinedEnvArgs
}
