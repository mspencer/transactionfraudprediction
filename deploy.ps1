$PROJECT_ID = "fraud-detection-service-504610"
$SERVICE_NAME = "predictfraud"
$REGION = "europe-west2"
$IMAGE = "europe-west2-docker.pkg.dev/$PROJECT_ID/frauddetection-service/frauddetection-api:latest"

$MEMORY = "2Gi"
$CPU = "2"
$MIN_INSTANCES = "1"
$MAX_INSTANCES = "10"
$CONCURRENCY = "20"
$TIMEOUT = "300s"
$PORT = "8080"

$gcloudArgs = @(
    "run", "deploy", $SERVICE_NAME,
    "--project=$PROJECT_ID",
    "--image=$IMAGE",
    "--region=$REGION",
    "--platform=managed",
    "--port=$PORT",
    "--memory=$MEMORY",
    "--cpu=$CPU",
    "--min-instances=$MIN_INSTANCES",
    "--max-instances=$MAX_INSTANCES",
    "--concurrency=$CONCURRENCY",
    "--timeout=$TIMEOUT",
    "--execution-environment=gen2",
    "--no-cpu-throttling",
    "--allow-unauthenticated"
)

Write-Host "Deploying $SERVICE_NAME to Cloud Run in region $REGION..."

gcloud @gcloudArgs

Write-Host "Deployment complete!"