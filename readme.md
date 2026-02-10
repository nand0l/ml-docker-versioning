# ml-docker-versioning

## Model Versioning

Version format:
`iris-logreg-YYYY-MM-DD-###`

Convention:

- `iris`: dataset name
- `logreg`: model type
- `YYYY-MM-DD`: training date
- `###`: 3-digit sequence for same-day retrains (e.g. `001`, `002`, `003`)

## GIT

```bash
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/nand0l/ml-docker-versioning.git
git push -u origin main
```

## Docker

Build v1 image:

```bash
docker build -t ml-demo:v1 .
```

Build v1 image with training overrides:

```bash
docker build --build-arg MODEL_C=0.8 --build-arg MODEL_MAX_ITER=500 -t ml-demo:v1 .
```

Run v1 container:

```bash
docker run -d --name ml-demo -p 8080:8080 ml-demo:v1
```

Validate endpoints:

```bash
curl.exe http://localhost:8080/health
curl.exe -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d "{\"features\":[5.1,3.5,1.4,0.2]}"
```
