# Bernoulli Mixture Model (BMM)

A probabilistic clustering model implementation using Microsoft Infer.NET for learning mixtures of Bernoulli distributions with variational inference (Expectation Propagation). **Modernized for .NET 8.0 with Roslyn compiler support.**

## Quick Start

### Prerequisites
- [.NET 8.0 SDK](https://dotnet.microsoft.com/download/dotnet/8.0) or later

### Basic Usage

```bash
# Build the project
dotnet build

# Run with your CSV data
dotnet run --project BernoulliMixture/BernoulliMixture.csproj -- <input.csv> <num_clusters>

# Example: Cluster data.csv into 5 clusters
dotnet run --project BernoulliMixture/BernoulliMixture.csproj -- data.csv 5

# Specify custom output files
dotnet run --project BernoulliMixture/BernoulliMixture.csproj -- data.csv 5 weights.csv assignments.csv
```

### Command-Line Arguments

```
Usage: BernoulliMixture <input_csv> <num_clusters> [output_weights_csv] [output_assignments_csv]

Arguments:
  input_csv              : Path to input CSV file with binary data (0 or 1 values)
  num_clusters           : Number of clusters to infer (must be >= 2)
  output_weights_csv     : (Optional) Output file for cluster weights (default: cluster_weights.csv)
  output_assignments_csv : (Optional) Output file for cluster assignments (default: cluster_assignments.csv)
```

### Input Data Format

The model accepts CSV files with binary (0 or 1) values:

- **Format**: CSV file with no header row
- **Rows**: Each row represents one data point
- **Columns**: Each column represents one feature dimension
- **Values**: Must be 0 or 1 (binary)
- **Encoding**: UTF-8 or ASCII

#### Example Input File

```csv
1,0,1,0,1
0,1,0,1,0
1,1,0,0,1
0,0,1,1,0
```

This represents 4 data points with 5 binary features each.

#### Data Validation

The program automatically validates:
- File exists and is readable
- All rows have the same number of columns
- All values are 0 or 1
- At least one data point exists

Invalid data will produce clear error messages.

### Output Files

The program generates two CSV files:

#### 1. Cluster Weights (`cluster_weights.csv`)

Contains cluster mixture weights and feature probabilities:

```csv
Cluster,Weight,FeatureDimension,FeatureProbability
0,0.333333,0,0.850000
0,0.333333,1,0.120000
0,0.333333,2,0.900000
1,0.333333,0,0.150000
1,0.333333,1,0.880000
...
```

- **Cluster**: Cluster ID (0 to K-1)
- **Weight**: Overall mixture weight for this cluster (sums to 1.0)
- **FeatureDimension**: Feature index (0 to D-1)
- **FeatureProbability**: Probability that this feature is 1 in this cluster

#### 2. Cluster Assignments (`cluster_assignments.csv`)

Contains per-data-point cluster probabilities:

```csv
DataPoint,Cluster0_Prob,Cluster1_Prob,Cluster2_Prob,AssignedCluster
0,0.950000,0.030000,0.020000,0
1,0.010000,0.980000,0.010000,1
2,0.020000,0.025000,0.955000,2
...
```

- **DataPoint**: Data point index (row number in input file, 0-indexed)
- **ClusterX_Prob**: Probability that this point belongs to cluster X
- **AssignedCluster**: Most likely cluster (argmax of probabilities)

### Console Output Example

```
=== Bernoulli Mixture Model - Inference ===

Loading data from: sample_data.csv
  Data points: 12
  Dimensions: 8
  Clusters: 3

Performing inference...
Compiling model...done.
Iterating: 
.........|.........|.........|.........|.........| 50
Inference complete!

Writing cluster weights to: cluster_weights.csv
Writing cluster assignments to: cluster_assignments.csv

=== Done ===
```

## Model Description

### Mathematical Background

A Bernoulli Mixture Model (BMM) is a probabilistic model for clustering binary data. It assumes that:

1. **Data Generation Process**:
   - Each data point belongs to one of K clusters
   - Cluster assignment is drawn from a categorical distribution with weights π
   - Given a cluster assignment, each feature is independently generated from a Bernoulli distribution

2. **Generative Process**:
   ```
   π ~ Dirichlet(α)                    # Cluster mixture weights
   θ[k,d] ~ Beta(1, 1)                # Feature probabilities for cluster k, dimension d
   
   For each data point i:
     z[i] ~ Categorical(π)            # Cluster assignment
     x[i,d] ~ Bernoulli(θ[z[i], d])  # Binary features
   ```

3. **Variables**:
   - **π** (pi): K-dimensional probability vector representing cluster mixture weights
   - **θ** (theta): K × D matrix where θ[k,d] is the probability that feature d is "true" in cluster k
   - **z** (or c in code): Cluster assignments (latent variables)
   - **x**: Observed binary data (N × D matrix)

4. **Inference Goal**:
   Learn the posterior distributions of π and θ given observed data x

### Use Cases

BMM is particularly useful for:
- **Document clustering** (with bag-of-words representations)
- **Image clustering** (with binarized pixel values)
- **Genetic data analysis** (with SNP markers)
- **Recommendation systems** (with binary user preferences)
- **Any clustering task with binary features**

## How the Model is Built

### Model Construction (Infer.NET)

The model is built using Infer.NET's factor graph representation:

```csharp
// Define dimensions
Range n = new Range(nItems);      // Number of data points
Range k = new Range(numClusters); // Number of clusters
Range d = new Range(numDims);     // Number of features

// Cluster mixture weights
var pi = Variable.Dirichlet(k, piPrior).Named("pi");

// Cluster-specific feature parameters
var t = Variable.Array(Variable.Array<double>(d), k).Named("t");
t[k][d] = Variable.Beta(1, 1).ForEach(k).ForEach(d);

// Data generation
var c = Variable.Array<int>(n).Named("c");      // Cluster assignments
var x = Variable.Array(Variable.Array<bool>(d), n).Named("x");  // Data

using (Variable.ForEach(n))
{
    c[n] = Variable.Discrete(pi);  // Sample cluster
    using (Variable.Switch(c[n]))
    {
        using (Variable.ForEach(d))
            x[n][d] = Variable.Bernoulli(t[c[n]][d]);  // Sample features
    }
}
```

### Inference with Expectation Propagation

The implementation uses **Expectation Propagation (EP)** for variational inference:

1. **Batch Processing**: All data is processed in a single batch
   - Suitable for datasets that fit in memory
   - For larger datasets, consider data subsampling or mini-batch approaches

2. **Expectation Propagation**: Infer.NET uses EP algorithm for inference
   - Iteratively refines approximate posterior distributions
   - Converges to a local optimum
   - Default: 50 iterations (configurable)

### Symmetry Breaking

To avoid local minima where all clusters are identical:
```csharp
// Initialize cluster assignments randomly
Discrete[] cinit = new Discrete[batchSize];
for (int i = 0; i < cinit.Length; i++)
    cinit[i] = Discrete.PointMass(Rand.Int(k.SizeAsInt), k.SizeAsInt);
c.InitialiseTo(Distribution<int>.Array(cinit));
```

## How to Run and Do Inference

### Basic Inference

```csharp
// 1. Create inference engine
InferenceEngine engine = new InferenceEngine()
{
    NumberOfIterations = 50,  // EP iterations
    ShowProgress = false
};

// 2. Set observed data
nItems.ObservedValue = numData;
x.ObservedValue = data;

// 3. Perform inference
Dirichlet piMarginal = engine.Infer<Dirichlet>(pi);
Beta[][] tMarginal = engine.Infer<Beta[][]>(t);

// piMarginal contains the posterior over cluster weights
// tMarginal contains the posterior over feature probabilities for each cluster
```

### Customizing the Model

#### Adjust Number of Clusters
```csharp
const int numClusters = 20;  // Increase for more fine-grained clustering
```

#### Change Prior Distributions
```csharp
// Informative prior for cluster weights (e.g., prefer uniform clusters)
double[] piPrior = Enumerable.Repeat(10.0, numClusters).ToArray();

// Informative prior for features (e.g., sparsity)
t[k][d] = Variable.Beta(0.5, 2.0).ForEach(k).ForEach(d);  // Bias toward 0
```

#### Adjust Inference Iterations
```csharp
engine.NumberOfIterations = 100;  // More iterations = better convergence
engine.ShowProgress = true;       // Display iteration progress
```

### Using Your Own Data

To use custom binary data:

```csharp
// Your data: N samples × D dimensions
bool[][] myData = LoadMyData();

// Set dimensions
const int numData = myData.Length;
const int numDims = myData[0].Length;
const int numClusters = 10;  // Choose appropriate K

// Use myData instead of MakeData()
bool[][] data = myData;

// Continue with rest of the inference code...
```

For non-binary data, binarize first:
```csharp
// Example: threshold continuous features
bool[][] BinarizeData(double[][] continuousData, double threshold = 0.5)
{
    return continuousData.Select(row => 
        row.Select(val => val > threshold).ToArray()
    ).ToArray();
}
```

## Project Structure

```
BernoulliMixture/
├── BernoulliMixture/
│   ├── BernoulliMixture.csproj   # Modern .NET 8.0 SDK project
│   └── Program.cs                # Main implementation with Roslyn configuration
├── BernoulliMixture.sln          # Solution file
├── mnist_test.csv                # Optional: MNIST test data
├── CHANGELOG.md                  # Migration history
└── README.md                     # This file
```

## Dependencies

- **Microsoft.ML.Probabilistic** (v0.4.2301.301): Core probabilistic programming framework
- **Microsoft.ML.Probabilistic.Compiler** (v0.4.2301.301): Model compilation and inference
- **Microsoft.CodeAnalysis.CSharp** (v4.11.0): Roslyn compiler for runtime code generation

All packages are automatically restored via NuGet.

## .NET 8.0 and Roslyn Compiler

This project is configured to run on .NET 8.0 using the Roslyn compiler instead of the legacy CodeDom compiler. The code uses reflection to automatically configure Infer.NET's `ModelCompiler` to use Roslyn at runtime:

```csharp
// Automatically executed when creating an InferenceEngine
ConfigureEngineCompiler(engine);
```

This configuration happens transparently - you don't need to manually set anything. The compiler choice is changed from "Auto" (which would try to use unsupported CodeDom) to "Roslyn" before inference begins.

## Technical Details

### Inference Algorithm

The implementation uses **Expectation Propagation (EP)** for approximate Bayesian inference:

1. **EP Algorithm**: Approximates intractable posterior distributions
   - Iteratively refines factor approximations
   - Minimizes KL divergence between true and approximate posteriors
   - Iterates until convergence or max iterations reached

2. **Inference Process**:
   - Initialize cluster assignments randomly (symmetry breaking)
   - Iterate EP updates for specified number of iterations
   - Returns approximate posterior distributions over parameters

### Computational Complexity

- **Per iteration**: O(N × K × D) where N=dataset size, K=clusters, D=dimensions
- **Total**: O(I × N × K × D) where I=EP iterations
- **Memory**: O(N × K + K × D) for data and parameters

### Convergence

Monitor convergence by:
1. Increasing number of iterations until posterior distributions stabilize
2. Examining the evidence (model likelihood) if available
3. Validating cluster quality on held-out data
4. Visual inspection of learned cluster parameters

## References

- [Infer.NET Documentation](https://dotnet.github.io/infer/)
- [Expectation Propagation](https://tminka.github.io/papers/ep/)
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Chapter 9: Mixture Models.

## License

This project uses [MIT License](LICENSE) (update as appropriate).

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
