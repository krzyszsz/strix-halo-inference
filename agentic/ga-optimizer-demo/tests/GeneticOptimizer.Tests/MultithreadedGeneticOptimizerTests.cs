using System.Collections.Concurrent;
using GeneticOptimizer;

namespace GeneticOptimizer.Tests;

public sealed class MultithreadedGeneticOptimizerTests
{
    private static OptimizationSettings DefaultSettings(int seed = 42, int generations = 40, int population = 64)
        => new(
            PopulationSize: population,
            Generations: generations,
            MutationRate: 0.20,
            CrossoverRate: 0.85,
            TournamentSize: 3,
            EliteCount: 2,
            Seed: seed,
            MaxDegreeOfParallelism: 4
        );

    [Fact]
    public void Optimize_OnSphereProblem_FindsNearZero()
    {
        var optimizer = new MultithreadedGeneticOptimizer();
        var problem = new OptimizationProblem(
            Dimensions: 3,
            Objective: v => v.Sum(x => x * x),
            LowerBounds: [-5.0, -5.0, -5.0],
            UpperBounds: [5.0, 5.0, 5.0]
        );

        var result = optimizer.Optimize(problem, DefaultSettings(generations: 50, population: 80));

        Assert.True(result.BestScore < 0.3, $"Expected score < 0.3, got {result.BestScore}");
        Assert.Equal(3, result.BestVector.Length);
    }

    [Fact]
    public void Optimize_RespectsBounds()
    {
        var optimizer = new MultithreadedGeneticOptimizer();
        var lower = new[] { -1.0, 2.0, -3.0, 0.5 };
        var upper = new[] { 1.0, 4.0, -1.0, 1.5 };

        var problem = new OptimizationProblem(
            Dimensions: 4,
            Objective: v => v.Select(Math.Abs).Sum(),
            LowerBounds: lower,
            UpperBounds: upper
        );

        var result = optimizer.Optimize(problem, DefaultSettings(generations: 25, population: 48));

        for (var i = 0; i < 4; i++)
        {
            Assert.InRange(result.BestVector[i], lower[i], upper[i]);
        }
    }

    [Fact]
    public void Optimize_SameSeed_IsDeterministic()
    {
        var optimizer = new MultithreadedGeneticOptimizer();
        var problem = new OptimizationProblem(
            Dimensions: 2,
            Objective: v =>
            {
                var x = v[0];
                var y = v[1];
                return (x - 1.2) * (x - 1.2) + (y + 0.7) * (y + 0.7);
            },
            LowerBounds: [-10.0, -10.0],
            UpperBounds: [10.0, 10.0]
        );

        var settings = DefaultSettings(seed: 123, generations: 35, population: 70);
        var r1 = optimizer.Optimize(problem, settings);
        var r2 = optimizer.Optimize(problem, settings);

        Assert.Equal(r1.BestScore, r2.BestScore, precision: 8);
        Assert.Equal(r1.BestVector[0], r2.BestVector[0], precision: 8);
        Assert.Equal(r1.BestVector[1], r2.BestVector[1], precision: 8);
    }

    [Fact]
    public void Optimize_UsesMultipleThreads_WhenConfigured()
    {
        var optimizer = new MultithreadedGeneticOptimizer();
        var threadIds = new ConcurrentDictionary<int, byte>();

        var problem = new OptimizationProblem(
            Dimensions: 2,
            Objective: v =>
            {
                threadIds.TryAdd(Environment.CurrentManagedThreadId, 0);
                var x = v[0];
                var y = v[1];
                return x * x + y * y;
            },
            LowerBounds: [-8.0, -8.0],
            UpperBounds: [8.0, 8.0]
        );

        optimizer.Optimize(problem, DefaultSettings(generations: 20, population: 120));

        Assert.True(threadIds.Count >= 2, $"Expected >=2 threads, got {threadIds.Count}");
    }
}
