var builder = WebApplication.CreateBuilder(args);
builder.WebHost.ConfigureKestrel(serverOptions =>
{
    serverOptions.ListenAnyIP(8080);
});

var app = builder.Build();

app.MapGet("/sum", (int x, int y) => new { sum = x + y });

app.Run();
