using System;
using System.IO;
using System.Linq;
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
            /*
            //
            // constants
            //

            const int numData = 10000;          // total number of examples in a dataset
            const int numDims = 784;            // dimensionality of an example

            const int numClusters = 20;         // requested number of clusters

            const int batchSize = 500;          // number of examples to use in a batch

            //
            // load the data
            //

            // Get data from: https://pjreddie.com/projects/mnist-in-csv/
            string filePath = @"/Users/vlad/Projects/BernoulliMixture/mnist_test.csv";
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

            // shuffle the data
            Random rnd = new Random();
            data = data.OrderBy(z => rnd.Next()).ToArray();

            */

            //
            // Generate synthetic data
            //

            const int numData = 10000;
            const int numClusters = 7;
            const int numDims = 10;

            const int batchSize = 200;

            bool[][] data = MakeData();

            Random rnd = new Random();

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
                    tMessage[k][d] = Variable.Observed<Beta>(new Beta(1, 1));
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

            InferenceEngine engine = new InferenceEngine()
            {
                NumberOfIterations = 50,
                ShowProgress = true
            };

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

				//piMarginal = engine.Infer<Dirichlet>(pi);
				//tMarginal = engine.Infer<Beta[][]>(t);

                var temp = engine.Infer<Dirichlet>(pi, QueryTypes.MarginalDividedByPrior);
				//piMessage.ObservedValue = engine.Infer<Dirichlet>(pi, QueryTypes.MarginalDividedByPrior);
				tMessage.ObservedValue = engine.Infer<Beta[][]>(t, QueryTypes.MarginalDividedByPrior);
                piMessage.ObservedValue = temp;

                //Console.WriteLine("\tBatch {0}, pi Marginal: {1}", b, engine.Infer<Dirichlet>(pi));
                //Console.WriteLine("\tBatch {0}, pi Marginal: {1}", b, engine.Infer<Beta[][]>(t));
			}

            piMarginal = engine.Infer<Dirichlet>(pi);
            tMarginal = engine.Infer<Beta[][]>(t);

            Console.WriteLine(piMarginal.ToString());

            for (int i = 0; i < numClusters; i++) {
                string output = "";
                for (int j = 0; j < numDims; j++) {
                    output = output + " " + string.Format("{0:N2}",tMarginal[i][j].GetMean()) + " ";
                }
                Console.WriteLine(output);
            }

            Console.Read();
        }

        static bool[][] MakeData() {
            int numPoints = 10000;
            int numDims = 10;
            int numClusters = 5;

            // prior and pi RV
            double[] piParams = { 0.2, 0.15, 0.15, 0.4, 0.1 };
            Discrete pi = new Discrete(piParams);

            // prior and t RV array of arrays
            double[][] tParams = {
                new double[] {0.1, 0.1, 0.9, 0.9, 0.5, 0.5, 0.2, 0.1, 0.9, 0.2}, //
                new double[] {0.5, 0.9, 0.5, 0.1, 0.1, 0.2, 0.5, 0.1, 0.4, 0.9},
                new double[] {0.9, 0.9, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.1, 0.7}, //
                new double[] {0.1, 0.2, 0.3, 0.3, 0.7, 0.1, 0.9, 0.9, 0.3, 0.1}, //
                new double[] {0.7, 0.3, 0.9, 0.2, 0.2, 0.1, 0.2, 0.3, 0.4, 0.5} //
            };
            Bernoulli[][] t = new Bernoulli[numClusters][];
            for (int i = 0; i < numClusters; i++) {
                t[i] = new Bernoulli[numDims];
                for (int j = 0; j < numDims; j++) {
                    t[i][j] = new Bernoulli(tParams[i][j]);
                }
            }

            // declare and init data
            bool[][] data = new bool[numPoints][];
            for (int i = 0; i < numPoints; i++)
                data[i] = new bool[numDims];

            // generate data
            for (int i = 0; i < numPoints; i++) {
                int c = pi.Sample();
                for (int j = 0; j < numDims; j++) {
                    data[i][j] = t[c][j].Sample();
                }
            }

            return data;
        }
    }
}
