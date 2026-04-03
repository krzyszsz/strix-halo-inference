namespace GeneticOptimizer;

public sealed record OptimizationProblem(
    int Dimensions,
    Func<double[], double> Objective,
    double[] LowerBounds,
    double[] UpperBounds
);

public sealed record OptimizationSettings(
    int PopulationSize,
    int Generations,
    double MutationRate,
    double CrossoverRate,
    int TournamentSize,
    int EliteCount,
    int Seed,
    int MaxDegreeOfParallelism
);

public sealed record OptimizationResult(
    double[] BestVector,
    double BestScore,
    int GenerationsEvaluated
);

public interface IGeneticOptimizer
{
    OptimizationResult Optimize(OptimizationProblem problem, OptimizationSettings settings, CancellationToken cancellationToken = default);
}
