# Build stage
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

# Copy project files
COPY SumApi.csproj .
RUN dotnet restore

# Copy remaining source code
COPY . .

# Build and publish
RUN dotnet publish -c Release -o /app/publish

# Runtime stage
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS runtime
WORKDIR /app
EXPOSE 8080

# Copy published files
COPY --from=build /app/publish .

# Set entry point
ENTRYPOINT ["dotnet", "SumApi.dll"]
