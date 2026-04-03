# Sum API

A minimal ASP.NET Core API that calculates the sum of two numbers.

## Endpoint

- `GET /sum?x={number}&y={number}` — returns JSON `{ "sum": x + y }`

## Local run (SDK)

```bash
dotnet restore
dotnet run --urls http://0.0.0.0:8080
```

Test:

```bash
curl "http://localhost:8080/sum?x=3&y=4"
```

## Docker

Build:

```bash
docker build -t dotnet-sum-api:local .
```

Run:

```bash
docker run --rm -p 8080:8080 dotnet-sum-api:local
```

Test:

```bash
curl "http://localhost:8080/sum?x=3&y=4"
```
