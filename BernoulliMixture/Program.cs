using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Compiler;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace BernoulliMixture;

class Program
{
    static void Main(string[] args)
    {
        // Configure Infer.NET to use Roslyn compiler instead of CodeDom
        // This is required for .NET 8.0 as CodeDom is not supported
        ConfigureRoslynCompiler();
        
        // Parse command-line arguments
        if (args.Length < 2)
        {
            PrintUsage();
            return;
        }

        string inputFile = args[0];
        if (!int.TryParse(args[1], out int numClusters) || numClusters < 2)
        {
            Console.WriteLine("Error: Number of clusters must be an integer >= 2");
            PrintUsage();
            return;
        }

        string outputWeightsFile = args.Length > 2 ? args[2] : "cluster_weights.csv";
        string outputAssignmentsFile = args.Length > 3 ? args[3] : "cluster_assignments.csv";

        // Validate input file
        if (!File.Exists(inputFile))
        {
            Console.WriteLine($"Error: Input file not found: {inputFile}");
            return;
        }

        Console.WriteLine("=== Bernoulli Mixture Model - Inference ===\n");
        
        // Load and validate data
        Console.WriteLine($"Loading data from: {inputFile}");
        bool[][] data;
        try
        {
            data = LoadCsvData(inputFile);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error loading data: {ex.Message}");
            return;
        }

        int numData = data.Length;
        int numDims = data[0].Length;
        
        // Validate that we have enough data points for clustering
        if (numData < numClusters)
        {
            Console.WriteLine($"Error: Not enough data points ({numData}) for {numClusters} clusters.");
            Console.WriteLine("Number of data points must be >= number of clusters.");
            return;
        }
        
        // Validate dimensionality
        if (numDims == 0)
        {
            Console.WriteLine("Error: Data has zero dimensions.");
            return;
        }
        
        Console.WriteLine($"  Data points: {numData}");
        Console.WriteLine($"  Dimensions: {numDims}");
        Console.WriteLine($"  Clusters: {numClusters}");
        Console.WriteLine();

        //
        // Define model variables
        //

        Variable<int> nItems = Variable.New<int>();

        Range n = new Range(nItems);
        Range k = new Range(numClusters);
        Range d = new Range(numDims);

        double[] piPrior = new double[numClusters];
        for (int i = 0; i < numClusters; i++)
            piPrior[i] = 1.0;

        // Define latent variables
        var pi = Variable.Dirichlet(k, piPrior).Named("pi");
        var c = Variable.Array<int>(n).Named("c");
        var t = Variable.Array(Variable.Array<double>(d), k).Named("t");
        var x = Variable.Array(Variable.Array<bool>(d), n).Named("x");

        // Cluster-specific parameters (probability of each feature being true in each cluster)
        t[k][d] = Variable.Beta(1, 1).ForEach(k).ForEach(d);

        // Data generation model
        using (Variable.ForEach(n))
        {
            c[n] = Variable.Discrete(pi);
            using (Variable.Switch(c[n]))
            {
                using (Variable.ForEach(d))
                    x[n][d] = Variable.Bernoulli(t[c[n]][d]);
            }
        }

        // Symmetry breaking -- assign clusters to examples randomly
        Discrete[] cinit = new Discrete[numData];
        for (int i = 0; i < cinit.Length; i++)
            cinit[i] = Discrete.PointMass(Rand.Int(k.SizeAsInt), k.SizeAsInt);
        c.InitialiseTo(Distribution<int>.Array(cinit));

        //
        // Inference
        //
        Console.WriteLine("Performing inference...");

        InferenceEngine engine = new InferenceEngine()
        {
            NumberOfIterations = 50,
            ShowProgress = true
        };

        // Try to configure the engine's compiler to use Roslyn
        ConfigureEngineCompiler(engine);

        // Set observed data
        nItems.ObservedValue = numData;
        x.ObservedValue = data;

        // Perform inference with error handling
        Dirichlet piMarginal;
        Beta[][] tMarginal;
        Discrete[] cMarginal;
        
        try
        {
            piMarginal = engine.Infer<Dirichlet>(pi);
            tMarginal = engine.Infer<Beta[][]>(t);
            cMarginal = engine.Infer<Discrete[]>(c);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\nError during inference: {ex.Message}");
            Console.WriteLine("This may indicate numerical instability or model convergence issues.");
            return;
        }
        
        Console.WriteLine("Inference complete!\n");

        //
        // Write results to CSV files
        //
        try
        {
            Console.WriteLine($"Writing cluster weights to: {outputWeightsFile}");
            WriteClusterWeights(outputWeightsFile, piMarginal, tMarginal);

            Console.WriteLine($"Writing cluster assignments to: {outputAssignmentsFile}");
            WriteClusterAssignments(outputAssignmentsFile, cMarginal);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\nError writing output files: {ex.Message}");
            return;
        }

        Console.WriteLine("\n=== Done ===");
    }

    /// <summary>
    /// Prints usage information for the program.
    /// </summary>
    static void PrintUsage()
    {
        Console.WriteLine("Usage: BernoulliMixture <input_csv> <num_clusters> [output_weights_csv] [output_assignments_csv]");
        Console.WriteLine();
        Console.WriteLine("Arguments:");
        Console.WriteLine("  input_csv              : Path to input CSV file with binary data (0 or 1 values)");
        Console.WriteLine("  num_clusters           : Number of clusters to infer (must be >= 2)");
        Console.WriteLine("  output_weights_csv     : (Optional) Output file for cluster weights (default: cluster_weights.csv)");
        Console.WriteLine("  output_assignments_csv : (Optional) Output file for cluster assignments (default: cluster_assignments.csv)");
        Console.WriteLine();
        Console.WriteLine("Input CSV format:");
        Console.WriteLine("  - Each row is a data point");
        Console.WriteLine("  - Each column is a feature dimension");
        Console.WriteLine("  - Values must be 0 or 1 (binary)");
        Console.WriteLine("  - No header row");
        Console.WriteLine();
        Console.WriteLine("Example:");
        Console.WriteLine("  dotnet run data.csv 5");
        Console.WriteLine("  dotnet run data.csv 10 weights.csv assignments.csv");
    }

    /// <summary>
    /// Loads binary data from a CSV file.
    /// Each row is a data point, each column is a feature.
    /// Values must be 0 or 1.
    /// </summary>
    static bool[][] LoadCsvData(string filePath)
    {
        var lines = File.ReadAllLines(filePath);
        if (lines.Length == 0)
        {
            throw new InvalidDataException("Input file is empty");
        }

        var data = new List<bool[]>();
        int? numDims = null;

        for (int lineNum = 0; lineNum < lines.Length; lineNum++)
        {
            var line = lines[lineNum].Trim();
            if (string.IsNullOrWhiteSpace(line))
                continue; // Skip empty lines

            var values = line.Split(',');
            
            // Validate dimensionality consistency
            if (numDims == null)
            {
                numDims = values.Length;
            }
            else if (values.Length != numDims.Value)
            {
                throw new InvalidDataException(
                    $"Line {lineNum + 1}: Expected {numDims.Value} dimensions but found {values.Length}");
            }

            var row = new bool[values.Length];
            for (int i = 0; i < values.Length; i++)
            {
                var trimmedValue = values[i].Trim();
                if (trimmedValue == "0")
                {
                    row[i] = false;
                }
                else if (trimmedValue == "1")
                {
                    row[i] = true;
                }
                else
                {
                    throw new InvalidDataException(
                        $"Line {lineNum + 1}, column {i + 1}: Invalid value '{trimmedValue}'. " +
                        $"Expected 0 or 1.");
                }
            }
            data.Add(row);
        }

        if (data.Count == 0)
        {
            throw new InvalidDataException("No valid data found in file");
        }

        return data.ToArray();
    }

    /// <summary>
    /// Writes cluster weights and feature probabilities to a CSV file.
    /// </summary>
    static void WriteClusterWeights(string filePath, Dirichlet piMarginal, Beta[][] tMarginal)
    {
        if (piMarginal == null)
        {
            throw new ArgumentNullException(nameof(piMarginal), "Cluster weights cannot be null.");
        }
        if (tMarginal == null || tMarginal.Length == 0)
        {
            throw new ArgumentException("Feature probabilities cannot be null or empty.", nameof(tMarginal));
        }
        
        using var writer = new StreamWriter(filePath);
        
        // Write header
        writer.WriteLine("Cluster,Weight,FeatureDimension,FeatureProbability");
        
        // Get cluster weights
        var weights = piMarginal.GetMean();
        
        // Write each cluster's weight and feature probabilities
        for (int cluster = 0; cluster < tMarginal.Length; cluster++)
        {
            for (int dim = 0; dim < tMarginal[cluster].Length; dim++)
            {
                double featureProb = tMarginal[cluster][dim].GetMean();
                writer.WriteLine($"{cluster},{weights[cluster]:F6},{dim},{featureProb:F6}");
            }
        }
    }

    /// <summary>
    /// Writes cluster assignment probabilities for each data point to a CSV file.
    /// </summary>
    static void WriteClusterAssignments(string filePath, Discrete[] cMarginal)
    {
        if (cMarginal == null || cMarginal.Length == 0)
        {
            throw new ArgumentException("Cluster assignments cannot be null or empty.", nameof(cMarginal));
        }
        
        using var writer = new StreamWriter(filePath);
        
        // Write header
        int numClusters = cMarginal[0].Dimension;
        writer.Write("DataPoint");
        for (int k = 0; k < numClusters; k++)
        {
            writer.Write($",Cluster{k}_Prob");
        }
        writer.Write(",AssignedCluster");
        writer.WriteLine();
        
        // Write probabilities for each data point
        for (int i = 0; i < cMarginal.Length; i++)
        {
            var probs = cMarginal[i].GetProbs();
            writer.Write(i);
            
            // Write probability for each cluster
            for (int k = 0; k < numClusters; k++)
            {
                writer.Write($",{probs[k]:F6}");
            }
            
            // Write most likely cluster assignment
            int maxCluster = 0;
            double maxProb = probs[0];
            for (int k = 1; k < numClusters; k++)
            {
                if (probs[k] > maxProb)
                {
                    maxProb = probs[k];
                    maxCluster = k;
                }
            }
            writer.WriteLine($",{maxCluster}");
        }
    }

    /// <summary>
    /// Configures Infer.NET to use Roslyn compiler instead of CodeDom.
    /// This is required for .NET 8.0+ where CodeDom is not supported.
    /// Called at startup - actual configuration happens in ConfigureEngineCompiler.
    /// </summary>
    static void ConfigureRoslynCompiler()
    {
        // Intentionally empty - configuration happens per-engine
    }

    /// <summary>
    /// Configures a specific InferenceEngine instance to use Roslyn compiler.
    /// </summary>
    static void ConfigureEngineCompiler(InferenceEngine engine)
    {
        try
        {
            // The InferenceEngine has a Compiler property that returns an IAlgorithm
            // We need to access the underlying compiler settings
            var compilerProperty = engine.GetType().GetProperty("Compiler", 
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            
            if (compilerProperty != null)
            {
                var compiler = compilerProperty.GetValue(engine);
                
                // Try to find and modify compiler settings
                if (compiler != null)
                {
                    var compilerType = compiler.GetType();
                    var compilerAssembly = compilerType.Assembly;
                    
                    // Find CompilerChoice enum
                    var compilerChoiceType = compilerAssembly.GetTypes()
                        .FirstOrDefault(t => t.Name == "CompilerChoice" && t.IsEnum);
                    
                    if (compilerChoiceType != null)
                    {
                        var roslynValue = Enum.Parse(compilerChoiceType, "Roslyn");
                        
                        // Look for a field or property that holds the compiler choice
                        var allFields = compilerType.GetFields(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
                        
                        foreach (var field in allFields)
                        {
                            if (field.FieldType == compilerChoiceType)
                            {
                                field.SetValue(compiler, roslynValue);
                                return;
                            }
                        }
                    }
                }
            }
        }
        catch
        {
            // Silently fail - compiler configuration is best-effort
        }
    }
}
