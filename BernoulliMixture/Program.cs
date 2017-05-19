using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer;

namespace BernoulliMixture
{
    class Program
    {
        static void Main(string[] args)
        {
            //
            // constants
            //

            const int numData = 10000;          // total number of examples in a dataset
            const int numDims = 784;            // dimensionality of an example

            const int numClusters = 10;         // requested number of clusters

            const int batchSize = 100;          // number of examples to use in a batch

            //
            // load the data
            //

            // Get data from: https://pjreddie.com/projects/mnist-in-csv/
            string filePath = @"C:\Users\vladi\Documents\Visual Studio 2017\Projects\BernoulliMixture\mnist_test.csv";
            StreamReader sr = new StreamReader(filePath);

            bool[][] data = new bool[numData][];

            int row = 0;
            while (!sr.EndOfStream)
            {
                string[] line = sr.ReadLine().Split(',');
                data[row] = new bool[numDims];
                for (int i=1; i < numDims; i++)
                {
                    if (line[i] == "0")
                        data[row][i - 1] = false;
                    else
                        data[row][i - 1] = true;
                }
                row++;
            }

            //
            // Define model variables
            //

            Variable<int> nItems = Variable.New<int>();

            Range n = new Range(nItems);
            Range k = new Range(numClusters);
            Range d = new Range(numDims);

            double[] piPrior = new double[numClusters];
            for (int i = 0; i < numClusters; i++)
                piPrior[i] = 1.0 / numClusters;

            // define latent variables
            var pi = Variable.Dirichlet(k, piPrior).Named("pi");
            var c = Variable.Array<int>(n).Named("c");
            var t = Variable.Array(Variable.Array<double>(d), k).Named("t");
            var x = Variable.Array(Variable.Array<bool>(d), n).Named("x");

            // cluster-specific parameters
            t[k][d] = Variable.Beta(1, 1).ForEach(k).ForEach(d);

            // attach accumulator for pi variable
            Variable<Dirichlet> piMessage = Variable.Observed<Dirichlet>(Dirichlet.Uniform(numClusters));
            Variable.ConstrainEqualRandom(pi, piMessage);
            pi.AddAttribute(QueryTypes.Marginal);
            pi.AddAttribute(QueryTypes.MarginalDividedByPrior);

            // attach accumulator for each variable in t array
            var tMessage = Variable.Array(Variable.Array<Beta>(d), k);
            using (Variable.ForEach(k))
            {
                using (Variable.ForEach(d))
                {
                    tMessage[k][d] = Variable.Observed<Beta>(Beta.Uniform());
                    Variable.ConstrainEqualRandom(t[k][d], tMessage[k][d]);
                    t[k][d].AddAttribute(QueryTypes.Marginal);
                    t[k][d].AddAttribute(QueryTypes.MarginalDividedByPrior);
                }
            }

            // data generation model
            using (Variable.ForEach(n))
            {
                c[n] = Variable.Discrete(pi);
                using (Variable.Switch(c[n]))
                {
                    using (Variable.ForEach(d))
                        x[n][d] = Variable.Bernoulli(t[c[n]][d]);
                }
            }

            // symmetry breaking -- assign clusters to examples randomly
            Discrete[] cinit = new Discrete[batchSize];
            for (int i = 0; i < cinit.Length; i++)
                cinit[i] = Discrete.PointMass(Rand.Int(k.SizeAsInt), k.SizeAsInt);
            c.InitialiseTo(Distribution<int>.Array(cinit));

            //
            // inference
            //

            InferenceEngine engine = new InferenceEngine();

            // marginals to infer (and update in each batch)
            Dirichlet piMarginal = Dirichlet.Uniform(numClusters);

            Beta[][] tMarginal = new Beta[numClusters][];
            for (int i = 0; i < numClusters; i++)
            {
                tMarginal[i] = new Beta[numDims];
                for (int j = 0; j < numDims; j++)
                    tMarginal[i][j] = Beta.Uniform();
            }

            // online learning in batches
            bool[][] batch = new bool[batchSize][];
            for (int b = 0; b < numData / batchSize; b++)
            {
                nItems.ObservedValue = batchSize;

                // fill the batch with data
                batch = data.Skip(b * batchSize).Take(batchSize).ToArray();
                x.ObservedValue = batch;

                piMarginal = engine.Infer<Dirichlet>(pi);
                tMarginal = engine.Infer<Beta[][]>(t);

                piMessage.ObservedValue = engine.Infer<Dirichlet>(pi, QueryTypes.MarginalDividedByPrior);
                tMessage.ObservedValue = engine.Infer<Beta[][]>(t, QueryTypes.MarginalDividedByPrior);

                Console.WriteLine("Batch {0}, pi Marginal: {1}", b, piMarginal);
            }

            Console.Read();
        }
    }
}
