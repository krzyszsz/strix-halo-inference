namespace GeneticOptimizer;

public sealed class MultithreadedGeneticOptimizer : IGeneticOptimizer
{
    public OptimizationResult Optimize(OptimizationProblem problem, OptimizationSettings settings, CancellationToken cancellationToken = default)
    {
        Validate(problem, settings);

        var rng = new Random(settings.Seed);
        var population = InitializePopulation(problem, settings, rng);
        var scores = EvaluatePopulation(problem, population, settings, cancellationToken);

        var bestIndex = IndexOfBest(scores);
        var bestScore = scores[bestIndex];
        var bestVector = (double[])population[bestIndex].Clone();

        for (var generation = 0; generation < settings.Generations; generation++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var nextPopulation = Evolve(problem, settings, population, scores, rng);
            var nextScores = EvaluatePopulation(problem, nextPopulation, settings, cancellationToken);

            population = nextPopulation;
            scores = nextScores;

            var genBestIndex = IndexOfBest(scores);
            if (scores[genBestIndex] < bestScore)
            {
                bestScore = scores[genBestIndex];
                bestVector = (double[])population[genBestIndex].Clone();
            }
        }

        return new OptimizationResult(bestVector, bestScore, settings.Generations);
    }

    private static void Validate(OptimizationProblem problem, OptimizationSettings settings)
    {
        if (problem.Dimensions <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(problem.Dimensions), "Dimensions must be > 0.");
        }
        if (problem.LowerBounds.Length != problem.Dimensions || problem.UpperBounds.Length != problem.Dimensions)
        {
            throw new ArgumentException("Bounds length must match dimensions.");
        }
        if (problem.Objective is null)
        {
            throw new ArgumentNullException(nameof(problem.Objective));
        }
        if (settings.PopulationSize < 4)
        {
            throw new ArgumentOutOfRangeException(nameof(settings.PopulationSize), "PopulationSize must be >= 4.");
        }
        if (settings.Generations < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(settings.Generations), "Generations must be >= 1.");
        }
        if (settings.MutationRate < 0 || settings.MutationRate > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(settings.MutationRate), "MutationRate must be in [0,1].");
        }
        if (settings.CrossoverRate < 0 || settings.CrossoverRate > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(settings.CrossoverRate), "CrossoverRate must be in [0,1].");
        }
        if (settings.TournamentSize < 2 || settings.TournamentSize > settings.PopulationSize)
        {
            throw new ArgumentOutOfRangeException(nameof(settings.TournamentSize), "TournamentSize must be in [2,PopulationSize].");
        }
        if (settings.EliteCount < 0 || settings.EliteCount >= settings.PopulationSize)
        {
            throw new ArgumentOutOfRangeException(nameof(settings.EliteCount), "EliteCount must be in [0, PopulationSize).");
        }
        for (var i = 0; i < problem.Dimensions; i++)
        {
            if (problem.LowerBounds[i] > problem.UpperBounds[i])
            {
                throw new ArgumentException($"LowerBounds[{i}] > UpperBounds[{i}].");
            }
        }
    }

    private static double[][] InitializePopulation(OptimizationProblem problem, OptimizationSettings settings, Random rng)
    {
        var population = new double[settings.PopulationSize][];
        for (var i = 0; i < population.Length; i++)
        {
            population[i] = new double[problem.Dimensions];
            for (var d = 0; d < problem.Dimensions; d++)
            {
                population[i][d] = NextInRange(rng, problem.LowerBounds[d], problem.UpperBounds[d]);
            }
        }
        return population;
    }

    private static double[] EvaluatePopulation(
        OptimizationProblem problem,
        double[][] population,
        OptimizationSettings settings,
        CancellationToken cancellationToken)
    {
        var scores = new double[population.Length];
        var options = new ParallelOptions
        {
            MaxDegreeOfParallelism = settings.MaxDegreeOfParallelism <= 0
                ? Environment.ProcessorCount
                : settings.MaxDegreeOfParallelism,
            CancellationToken = cancellationToken
        };

        Parallel.For(0, population.Length, options, i =>
        {
            options.CancellationToken.ThrowIfCancellationRequested();
            var candidate = population[i];
            var copy = new double[candidate.Length];
            Array.Copy(candidate, copy, candidate.Length);
            scores[i] = problem.Objective(copy);
        });

        return scores;
    }

    private static double[][] Evolve(
        OptimizationProblem problem,
        OptimizationSettings settings,
        double[][] population,
        double[] scores,
        Random rng)
    {
        var order = Enumerable.Range(0, scores.Length)
            .OrderBy(i => scores[i])
            .ToArray();

        var next = new double[population.Length][];
        var cursor = 0;

        for (; cursor < settings.EliteCount; cursor++)
        {
            next[cursor] = (double[])population[order[cursor]].Clone();
        }

        while (cursor < next.Length)
        {
            var p1 = population[TournamentSelect(scores, settings.TournamentSize, rng)];
            var p2 = population[TournamentSelect(scores, settings.TournamentSize, rng)];

            var childA = new double[problem.Dimensions];
            var childB = new double[problem.Dimensions];
            Crossover(problem, settings, p1, p2, childA, childB, rng);
            Mutate(problem, settings, childA, rng);
            Mutate(problem, settings, childB, rng);

            next[cursor++] = childA;
            if (cursor < next.Length)
            {
                next[cursor++] = childB;
            }
        }

        return next;
    }

    private static int TournamentSelect(double[] scores, int tournamentSize, Random rng)
    {
        var best = rng.Next(scores.Length);
        var bestScore = scores[best];

        for (var i = 1; i < tournamentSize; i++)
        {
            var idx = rng.Next(scores.Length);
            if (scores[idx] < bestScore)
            {
                best = idx;
                bestScore = scores[idx];
            }
        }

        return best;
    }

    private static void Crossover(
        OptimizationProblem problem,
        OptimizationSettings settings,
        double[] p1,
        double[] p2,
        double[] childA,
        double[] childB,
        Random rng)
    {
        for (var d = 0; d < problem.Dimensions; d++)
        {
            if (rng.NextDouble() < settings.CrossoverRate)
            {
                var alpha = rng.NextDouble();
                childA[d] = alpha * p1[d] + (1 - alpha) * p2[d];
                childB[d] = alpha * p2[d] + (1 - alpha) * p1[d];
            }
            else
            {
                childA[d] = p1[d];
                childB[d] = p2[d];
            }
        }
    }

    private static void Mutate(
        OptimizationProblem problem,
        OptimizationSettings settings,
        double[] vector,
        Random rng)
    {
        for (var d = 0; d < problem.Dimensions; d++)
        {
            if (rng.NextDouble() < settings.MutationRate)
            {
                var sigma = Math.Max((problem.UpperBounds[d] - problem.LowerBounds[d]) * 0.08, 1e-9);
                var delta = NextGaussian(rng) * sigma;
                vector[d] = Clamp(vector[d] + delta, problem.LowerBounds[d], problem.UpperBounds[d]);
            }
            else
            {
                vector[d] = Clamp(vector[d], problem.LowerBounds[d], problem.UpperBounds[d]);
            }
        }
    }

    private static int IndexOfBest(double[] scores)
    {
        var idx = 0;
        var best = scores[0];
        for (var i = 1; i < scores.Length; i++)
        {
            if (scores[i] < best)
            {
                best = scores[i];
                idx = i;
            }
        }
        return idx;
    }

    private static double NextInRange(Random rng, double min, double max)
    {
        return min + rng.NextDouble() * (max - min);
    }

    private static double NextGaussian(Random rng)
    {
        // Box-Muller transform.
        var u1 = 1.0 - rng.NextDouble();
        var u2 = 1.0 - rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    private static double Clamp(double value, double min, double max)
    {
        if (value < min)
        {
            return min;
        }
        if (value > max)
        {
            return max;
        }
        return value;
    }
}
