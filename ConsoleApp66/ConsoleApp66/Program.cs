// Gerekli tüm kütüphaneler
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using Newtonsoft.Json;

// Projenizin ana namespace'i
namespace NeuralNetworkVisualizer
{
    public class MlpModelData
    {
        public int InputSize { get; set; }
        public int HiddenSize1 { get; set; }
        public int HiddenSize2 { get; set; }
        public int OutputSize { get; set; }
        public double LearningRate { get; set; }
        public double Momentum { get; set; }
        public double[][] WeightsInputToHidden1 { get; set; }
        public double[][] WeightsHidden1ToHidden2 { get; set; }
        public double[][] WeightsHidden2ToOutput { get; set; }
        public double[] BiasHidden1 { get; set; }
        public double[] BiasHidden2 { get; set; }
        public double[] BiasOutput { get; set; }
    }

    public class NetworkActivations
    {
        public double[] InputActivations { get; set; }
        public double[] Hidden1Activations { get; set; }
        public double[] Hidden2Activations { get; set; }
        public double[] OutputActivations { get; set; }
    }

    public class MultiLayerPerceptron
    {
        private readonly int _inputSize;
        private readonly int _hiddenSize1;
        private readonly int _hiddenSize2;
        private readonly int _outputSize;
        private readonly double _learningRate;
        private readonly double _momentum;
        private double[,] _weightsInputToHidden1;
        private double[,] _weightsHidden1ToHidden2;
        private double[,] _weightsHidden2ToOutput;
        private double[] _biasHidden1;
        private double[] _biasHidden2;
        private double[] _biasOutput;
        private double[,] _prevUpdateInputToHidden1;
        private double[,] _prevUpdateHidden1ToHidden2;
        private double[,] _prevUpdateHidden2ToOutput;
        private double[] _prevUpdateBiasHidden1;
        private double[] _prevUpdateBiasHidden2;
        private double[] _prevUpdateBiasOutput;
        private readonly Random _random = new Random();

        private NetworkActivations _lastActivations;

        public MultiLayerPerceptron(int inputSize, int hiddenSize1, int hiddenSize2, int outputSize, double learningRate, double momentum)
        {
            _inputSize = inputSize;
            _hiddenSize1 = hiddenSize1;
            _hiddenSize2 = hiddenSize2;
            _outputSize = outputSize;
            _learningRate = learningRate;
            _momentum = momentum;
            _weightsInputToHidden1 = new double[inputSize, hiddenSize1];
            _weightsHidden1ToHidden2 = new double[hiddenSize1, hiddenSize2];
            _weightsHidden2ToOutput = new double[hiddenSize2, outputSize];
            _biasHidden1 = new double[hiddenSize1];
            _biasHidden2 = new double[hiddenSize2];
            _biasOutput = new double[outputSize];
            _prevUpdateInputToHidden1 = new double[inputSize, hiddenSize1];
            _prevUpdateHidden1ToHidden2 = new double[hiddenSize1, hiddenSize2];
            _prevUpdateHidden2ToOutput = new double[hiddenSize2, outputSize];
            _prevUpdateBiasHidden1 = new double[hiddenSize1];
            _prevUpdateBiasHidden2 = new double[hiddenSize2];
            _prevUpdateBiasOutput = new double[outputSize];
            InitializeWeights(_weightsInputToHidden1);
            InitializeWeights(_weightsHidden1ToHidden2);
            InitializeWeights(_weightsHidden2ToOutput);
            InitializeBiases(_biasHidden1);
            InitializeBiases(_biasHidden2);
            InitializeBiases(_biasOutput);
        }

        public NetworkActivations GetLastActivations() => _lastActivations;

        public double[,] GetWeightsInputToHidden1() => _weightsInputToHidden1;
        public double[,] GetWeightsHidden1ToHidden2() => _weightsHidden1ToHidden2;
        public double[,] GetWeightsHidden2ToOutput() => _weightsHidden2ToOutput;

        private void InitializeWeights(double[,] weights)
        {
            for (int i = 0; i < weights.GetLength(0); i++)
                for (int j = 0; j < weights.GetLength(1); j++)
                    weights[i, j] = (_random.NextDouble() - 0.5) * 2;
        }

        private void InitializeBiases(double[] biases)
        {
            for (int i = 0; i < biases.Length; i++)
                biases[i] = (_random.NextDouble() - 0.5) * 2;
        }

        private double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
        private double SigmoidDerivative(double sigmoidOutput) => sigmoidOutput * (1 - sigmoidOutput);

        public double[] FeedForward(double[] inputs)
        {
            _lastActivations = new NetworkActivations { InputActivations = (double[])inputs.Clone() };

            double[] hidden1Outputs = new double[_hiddenSize1];
            for (int j = 0; j < _hiddenSize1; j++)
            {
                double sum = _biasHidden1[j];
                for (int i = 0; i < _inputSize; i++)
                {
                    sum += inputs[i] * _weightsInputToHidden1[i, j];
                }
                hidden1Outputs[j] = Sigmoid(sum);
            }
            _lastActivations.Hidden1Activations = (double[])hidden1Outputs.Clone();

            double[] hidden2Outputs = new double[_hiddenSize2];
            for (int j = 0; j < _hiddenSize2; j++)
            {
                double sum = _biasHidden2[j];
                for (int i = 0; i < _hiddenSize1; i++)
                {
                    sum += hidden1Outputs[i] * _weightsHidden1ToHidden2[i, j];
                }
                hidden2Outputs[j] = Sigmoid(sum);
            }
            _lastActivations.Hidden2Activations = (double[])hidden2Outputs.Clone();

            double[] finalOutputs = new double[_outputSize];
            for (int k = 0; k < _outputSize; k++)
            {
                double sum = _biasOutput[k];
                for (int j = 0; j < _hiddenSize2; j++)
                {
                    sum += hidden2Outputs[j] * _weightsHidden2ToOutput[j, k];
                }
                finalOutputs[k] = Sigmoid(sum);
            }
            _lastActivations.OutputActivations = (double[])finalOutputs.Clone();

            return finalOutputs;
        }

        public void Train(double[][] trainingSet, double[][] targetSet, int epochs)
        {
            Console.WriteLine($"Eğitim Başlatılıyor.. {_inputSize}-{_hiddenSize1}-{_hiddenSize2}-{_outputSize}, Devir: {epochs}, Öğrenme Oranı: {_learningRate}, Momentum: {_momentum})");
            Console.WriteLine(new string('-', 70));
            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                double epochError = 0;
                for (int i = 0; i < trainingSet.Length; i++)
                {
                    double[] inputs = trainingSet[i];
                    double[] targets = targetSet[i];
                    double[] hidden1_outputs = new double[_hiddenSize1];
                    for (int j = 0; j < _hiddenSize1; j++)
                    {
                        double sum = _biasHidden1[j];
                        for (int k = 0; k < _inputSize; k++)
                        {
                            sum += inputs[k] * _weightsInputToHidden1[k, j];
                        }
                        hidden1_outputs[j] = Sigmoid(sum);
                    }
                    double[] hidden2_outputs = new double[_hiddenSize2];
                    for (int j = 0; j < _hiddenSize2; j++)
                    {
                        double sum = _biasHidden2[j];
                        for (int k = 0; k < _hiddenSize1; k++)
                        {
                            sum += hidden1_outputs[k] * _weightsHidden1ToHidden2[k, j];
                        }
                        hidden2_outputs[j] = Sigmoid(sum);
                    }
                    double[] finalOutputs = new double[_outputSize];
                    for (int j = 0; j < _outputSize; j++)
                    {
                        double sum = _biasOutput[j];
                        for (int k = 0; k < _hiddenSize2; k++)
                        {
                            sum += hidden2_outputs[k] * _weightsHidden2ToOutput[k, j];
                        }
                        finalOutputs[j] = Sigmoid(sum);
                    }
                    for (int j = 0; j < _outputSize; j++)
                    {
                        epochError += 0.5 * Math.Pow(targets[j] - finalOutputs[j], 2);
                    }
                    double[] outputGradients = new double[_outputSize];
                    for (int j = 0; j < _outputSize; j++)
                    {
                        double errorDerivative = targets[j] - finalOutputs[j];
                        double activationDerivative = SigmoidDerivative(finalOutputs[j]);
                        outputGradients[j] = errorDerivative * activationDerivative;
                    }
                    double[] hidden2_Gradients = new double[_hiddenSize2];
                    for (int j = 0; j < _hiddenSize2; j++)
                    {
                        double propagatedError = 0;
                        for (int k = 0; k < _outputSize; k++)
                        {
                            propagatedError += outputGradients[k] * _weightsHidden2ToOutput[j, k];
                        }
                        double activationDerivative = SigmoidDerivative(hidden2_outputs[j]);
                        hidden2_Gradients[j] = propagatedError * activationDerivative;
                    }
                    double[] hidden1_Gradients = new double[_hiddenSize1];
                    for (int j = 0; j < _hiddenSize1; j++)
                    {
                        double propagatedError = 0;
                        for (int k = 0; k < _hiddenSize2; k++)
                        {
                            propagatedError += hidden2_Gradients[k] * _weightsHidden1ToHidden2[j, k];
                        }
                        double activationDerivative = SigmoidDerivative(hidden1_outputs[j]);
                        hidden1_Gradients[j] = propagatedError * activationDerivative;
                    }
                    for (int j = 0; j < _outputSize; j++)
                    {
                        double biasUpdate = _learningRate * outputGradients[j];
                        double biasVelocity = biasUpdate + _momentum * _prevUpdateBiasOutput[j];
                        _biasOutput[j] += biasVelocity;
                        _prevUpdateBiasOutput[j] = biasVelocity;
                        for (int k = 0; k < _hiddenSize2; k++)
                        {
                            double update = _learningRate * outputGradients[j] * hidden2_outputs[k];
                            double velocity = update + _momentum * _prevUpdateHidden2ToOutput[k, j];
                            _weightsHidden2ToOutput[k, j] += velocity;
                            _prevUpdateHidden2ToOutput[k, j] = velocity;
                        }
                    }
                    for (int j = 0; j < _hiddenSize2; j++)
                    {
                        double biasUpdate = _learningRate * hidden2_Gradients[j];
                        double biasVelocity = biasUpdate + _momentum * _prevUpdateBiasHidden2[j];
                        _biasHidden2[j] += biasVelocity;
                        _prevUpdateBiasHidden2[j] = biasVelocity;
                        for (int k = 0; k < _hiddenSize1; k++)
                        {
                            double update = _learningRate * hidden2_Gradients[j] * hidden1_outputs[k];
                            double velocity = update + _momentum * _prevUpdateHidden1ToHidden2[k, j];
                            _weightsHidden1ToHidden2[k, j] += velocity;
                            _prevUpdateHidden1ToHidden2[k, j] = velocity;
                        }
                    }
                    for (int j = 0; j < _hiddenSize1; j++)
                    {
                        double biasUpdate = _learningRate * hidden1_Gradients[j];
                        double biasVelocity = biasUpdate + _momentum * _prevUpdateBiasHidden1[j];
                        _biasHidden1[j] += biasVelocity;
                        _prevUpdateBiasHidden1[j] = biasVelocity;
                        for (int k = 0; k < _inputSize; k++)
                        {
                            double update = _learningRate * hidden1_Gradients[j] * inputs[k];
                            double velocity = update + _momentum * _prevUpdateInputToHidden1[k, j];
                            _weightsInputToHidden1[k, j] += velocity;
                            _prevUpdateInputToHidden1[k, j] = velocity;
                        }
                    }
                }
                if (epoch % 500 == 0)
                {
                    Console.WriteLine($"Devir {epoch,5}/{epochs} | Ortalama Hata: {epochError / trainingSet.Length:F8}");
                }
            }
            Console.WriteLine(new string('-', 70));
            Console.WriteLine("Eğitim Tamamlandı.");
        }

        public void SaveModel(string filePath)
        {
            var modelData = new MlpModelData
            {
                InputSize = _inputSize,
                HiddenSize1 = _hiddenSize1,
                HiddenSize2 = _hiddenSize2,
                OutputSize = _outputSize,
                LearningRate = _learningRate,
                Momentum = _momentum,
                BiasHidden1 = _biasHidden1,
                BiasHidden2 = _biasHidden2,
                BiasOutput = _biasOutput,
                WeightsInputToHidden1 = ConvertToJaggedArray(_weightsInputToHidden1),
                WeightsHidden1ToHidden2 = ConvertToJaggedArray(_weightsHidden1ToHidden2),
                WeightsHidden2ToOutput = ConvertToJaggedArray(_weightsHidden2ToOutput)
            };
            string json = JsonConvert.SerializeObject(modelData, Formatting.Indented);
            File.WriteAllText(filePath, json);
        }

        public static MultiLayerPerceptron LoadModel(string filePath)
        {
            string json = File.ReadAllText(filePath);
            var modelData = JsonConvert.DeserializeObject<MlpModelData>(json);
            var mlp = new MultiLayerPerceptron(modelData.InputSize, modelData.HiddenSize1, modelData.HiddenSize2, modelData.OutputSize, modelData.LearningRate, modelData.Momentum);
            mlp._biasHidden1 = modelData.BiasHidden1;
            mlp._biasHidden2 = modelData.BiasHidden2;
            mlp._biasOutput = modelData.BiasOutput;
            mlp._weightsInputToHidden1 = ConvertToMultiDimensionalArray(modelData.WeightsInputToHidden1);
            mlp._weightsHidden1ToHidden2 = ConvertToMultiDimensionalArray(modelData.WeightsHidden1ToHidden2);
            mlp._weightsHidden2ToOutput = ConvertToMultiDimensionalArray(modelData.WeightsHidden2ToOutput);
            return mlp;
        }

        private static double[][] ConvertToJaggedArray(double[,] multiArray)
        {
            int rows = multiArray.GetLength(0);
            int cols = multiArray.GetLength(1);
            double[][] jaggedArray = new double[rows][];
            for (int i = 0; i < rows; i++)
            {
                jaggedArray[i] = new double[cols];
                for (int j = 0; j < cols; j++)
                {
                    jaggedArray[i][j] = multiArray[i, j];
                }
            }
            return jaggedArray;
        }

        private static double[,] ConvertToMultiDimensionalArray(double[][] jaggedArray)
        {
            int rows = jaggedArray.Length;
            int cols = jaggedArray.Length > 0 ? jaggedArray[0].Length : 0;
            double[,] multiArray = new double[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    multiArray[i, j] = jaggedArray[i][j];
                }
            }
            return multiArray;
        }
    }

    public partial class Form1 : Form
    {
        private MultiLayerPerceptron mlp;
        private MlpModelData modelData;
        private List<PointF>[] neuronPositions;
        private NetworkActivations currentActivations;
        private Panel topPanel;
        private Button btnTrain;
        private Button btnLoad;
        private Button btnSave;
        private Label lblInfo;
        private RichTextBox logTextBox;
        private Panel networkPanel;
        private GroupBox predictionGroupBox;
        private Label lblInput1;
        private NumericUpDown numInput1;
        private Label lblInput2;
        private NumericUpDown numInput2;
        private Label lblInput3;
        private NumericUpDown numInput3;
        private Button btnPredict;
        private Label lblPredictionResult;

        public Form1()
        {
            InitializeComponentProgrammatically();
            this.networkPanel.Paint += new PaintEventHandler(networkPanel_Paint);
            this.networkPanel.Resize += (s, e) => this.networkPanel.Invalidate();
            this.btnTrain.Click += new EventHandler(btnTrain_Click);
            this.btnLoad.Click += new EventHandler(btnLoad_Click);
            this.btnSave.Click += new EventHandler(btnSave_Click);
            this.btnPredict.Click += new EventHandler(btnPredict_Click);
        }

        private async void btnTrain_Click(object sender, EventArgs e)
        {
            logTextBox.Clear();
            SetUIState(false, "Yeni model eğitiliyor, lütfen bekleyin...");

            currentActivations = null;

            await Task.Run(() =>
            {
                Console.SetOut(new TextBoxStreamWriter(logTextBox));

                mlp = new MultiLayerPerceptron(inputSize: 3, hiddenSize1: 8, hiddenSize2: 6, outputSize: 1, learningRate: 0.7, momentum: 0.9);

                double[][] inputs = new double[][]
                {
                    new double[] {0, 0, 0}, // -> 0
                    new double[] {0, 0, 1}, // -> 1
                    new double[] {0, 1, 0}, // -> 1
                    new double[] {0, 1, 1}, // -> 0
                    new double[] {1, 0, 0}, // -> 1
                    new double[] {1, 0, 1}, // -> 0
                    new double[] {1, 1, 0}, // -> 0
                    new double[] {1, 1, 1}  // -> 1
                };

                double[][] targets = new double[][]
                {
                    new double[] {0},
                    new double[] {1},
                    new double[] {1},
                    new double[] {0},
                    new double[] {1},
                    new double[] {0},
                    new double[] {0},
                    new double[] {1}
                };

                mlp.Train(inputs, targets, 8000);

                var tempDataPath = Path.GetTempFileName();
                mlp.SaveModel(tempDataPath);
                string json = File.ReadAllText(tempDataPath);
                modelData = JsonConvert.DeserializeObject<MlpModelData>(json);
                File.Delete(tempDataPath);
            });
            SetUIState(true, "Model eğitildi. Canlı tahmin yapabilirsiniz.");
            networkPanel.Invalidate();
        }

        private void btnLoad_Click(object sender, EventArgs e)
        {
            using (OpenFileDialog ofd = new OpenFileDialog { Filter = "JSON Model Dosyası (*.json)|*.json", Title = "Eğitilmiş Bir Model Dosyası Seçin" })
            {
                if (ofd.ShowDialog() == DialogResult.OK)
                {
                    try
                    {
                        SetUIState(false, "Model yükleniyor...");
                        mlp = MultiLayerPerceptron.LoadModel(ofd.FileName);
                        string json = File.ReadAllText(ofd.FileName);
                        modelData = JsonConvert.DeserializeObject<MlpModelData>(json);

                        currentActivations = null;

                        logTextBox.Clear();
                        logTextBox.AppendText("Model başarıyla yüklendi.\n");
                        logTextBox.AppendText($"Mimari: {modelData.InputSize}-{modelData.HiddenSize1}-{modelData.HiddenSize2}-{modelData.OutputSize}\n");

                        bool is3InputModel = modelData.InputSize == 3;
                        predictionGroupBox.Visible = is3InputModel;
                        if (!is3InputModel)
                        {
                            MessageBox.Show("Yüklenen model 3 girişli değildir. Tahmin paneli devre dışı bırakıldı.", "Uyarı", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                        }

                        SetUIState(true, "Model yüklendi. Canlı tahmin yapabilirsiniz.");
                        networkPanel.Invalidate();
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show("Model yüklenirken bir hata oluştu: " + ex.Message, "Hata", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        SetUIState(true, "Model yüklenemedi.");
                    }
                }
            }
        }

        private void btnSave_Click(object sender, EventArgs e)
        {
            if (mlp == null) { MessageBox.Show("Kaydedilecek bir model yok.", "Uyarı", MessageBoxButtons.OK, MessageBoxIcon.Warning); return; }
            using (SaveFileDialog sfd = new SaveFileDialog { Filter = "JSON Model Dosyası (*.json)|*.json", FileName = "mlp_3input_model.json", Title = "Modeli Kaydet" })
            {
                if (sfd.ShowDialog() == DialogResult.OK)
                {
                    try { mlp.SaveModel(sfd.FileName); MessageBox.Show("Model başarıyla kaydedildi.", "Başarılı", MessageBoxButtons.OK, MessageBoxIcon.Information); }
                    catch (Exception ex) { MessageBox.Show("Model kaydedilirken bir hata oluştu: " + ex.Message, "Hata", MessageBoxButtons.OK, MessageBoxIcon.Error); }
                }
            }
        }

        private void btnPredict_Click(object sender, EventArgs e)
        {
            if (mlp == null) { MessageBox.Show("Lütfen önce bir model eğitin veya yükleyin.", "Model Yok", MessageBoxButtons.OK, MessageBoxIcon.Warning); return; }

            double[] inputs = { (double)numInput1.Value, (double)numInput2.Value, (double)numInput3.Value };

            double[] predictionRaw = mlp.FeedForward(inputs);
            double rawOutput = predictionRaw[0];
            int roundedOutput = (rawOutput > 0.5) ? 1 : 0;

            currentActivations = mlp.GetLastActivations();

            lblPredictionResult.Text = $"Ham Çıktı: {rawOutput:F4}\n\nTahmin: {roundedOutput}";
            if (roundedOutput == 1) { lblPredictionResult.BackColor = Color.LightGreen; lblPredictionResult.ForeColor = Color.DarkGreen; }
            else { lblPredictionResult.BackColor = Color.LightCoral; lblPredictionResult.ForeColor = Color.DarkRed; }

            networkPanel.Invalidate();
        }

        private void SetUIState(bool isEnabled, string message) => this.Invoke((MethodInvoker)delegate {
            btnTrain.Enabled = isEnabled;
            btnLoad.Enabled = isEnabled;
            btnSave.Enabled = isEnabled && mlp != null;
            predictionGroupBox.Enabled = isEnabled && mlp != null;
            lblInfo.Text = message;
        });

        private void networkPanel_Paint(object sender, PaintEventArgs e)
        {
            base.OnPaint(e);
            Graphics g = e.Graphics;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            if (modelData == null) return;
            CalculateNeuronPositions();
            DrawConnections(g);
            DrawNeurons(g);
        }

        private void CalculateNeuronPositions()
        {
            int[] layerSizes = { modelData.InputSize, modelData.HiddenSize1, modelData.HiddenSize2, modelData.OutputSize };
            neuronPositions = new List<PointF>[layerSizes.Length];
            float layerSpacing = (float)networkPanel.Width / (layerSizes.Length);
            float maxNeuronsInLayer = layerSizes.Max();
            float neuronRadius = Math.Min(30f, (networkPanel.Height * 0.8f) / (maxNeuronsInLayer * 2.5f));
            for (int i = 0; i < layerSizes.Length; i++)
            {
                neuronPositions[i] = new List<PointF>();
                float layerX = layerSpacing / 2 + i * layerSpacing;
                float totalHeightForLayer = layerSizes[i] * neuronRadius * 2.5f;
                float startY = (networkPanel.Height - totalHeightForLayer) / 2;
                for (int j = 0; j < layerSizes[i]; j++)
                {
                    float neuronY = startY + neuronRadius + j * neuronRadius * 2.5f;
                    neuronPositions[i].Add(new PointF(layerX, neuronY));
                }
            }
        }

        private void DrawConnections(Graphics g)
        {
            DrawLayerConnections(g, 0, mlp?.GetWeightsInputToHidden1() ?? ConvertToMultiDimensionalArray(modelData.WeightsInputToHidden1));
            DrawLayerConnections(g, 1, mlp?.GetWeightsHidden1ToHidden2() ?? ConvertToMultiDimensionalArray(modelData.WeightsHidden1ToHidden2));
            DrawLayerConnections(g, 2, mlp?.GetWeightsHidden2ToOutput() ?? ConvertToMultiDimensionalArray(modelData.WeightsHidden2ToOutput));
        }

        private void DrawLayerConnections(Graphics g, int fromLayerIndex, double[,] weights)
        {
            if (neuronPositions == null || neuronPositions.Length <= fromLayerIndex + 1) return;

            double[] fromActivations = null;
            double[] toActivations = null;

            if (currentActivations != null)
            {
                switch (fromLayerIndex)
                {
                    case 0:
                        fromActivations = currentActivations.InputActivations;
                        toActivations = currentActivations.Hidden1Activations;
                        break;
                    case 1:
                        fromActivations = currentActivations.Hidden1Activations;
                        toActivations = currentActivations.Hidden2Activations;
                        break;
                    case 2:
                        fromActivations = currentActivations.Hidden2Activations;
                        toActivations = currentActivations.OutputActivations;
                        break;
                }
            }

            for (int i = 0; i < neuronPositions[fromLayerIndex].Count; i++)
            {
                for (int j = 0; j < neuronPositions[fromLayerIndex + 1].Count; j++)
                {
                    PointF startPoint = neuronPositions[fromLayerIndex][i];
                    PointF endPoint = neuronPositions[fromLayerIndex + 1][j];
                    double weight = weights[i, j];

                    double signalStrength = 0;
                    if (fromActivations != null && toActivations != null)
                    {
                        signalStrength = fromActivations[i] * weight;
                    }

                    float baseThickness = 0.2f + (float)(Math.Abs(weight) / 10.0);
                    float thickness = baseThickness;
                    Color connectionColor = GetColorForWeight(weight);

                    if (currentActivations != null)
                    {
                        thickness = Math.Max(0.1f, baseThickness * (float)Math.Abs(signalStrength));
                        int alpha = 60 + (int)(Math.Min(Math.Abs(signalStrength), 1.0) * 195);
                        alpha = Math.Max(0, Math.Min(255, alpha));
                        connectionColor = signalStrength > 0 ?
                            Color.FromArgb(alpha, 34, 139, 34) :
                            Color.FromArgb(alpha, 220, 20, 60);
                    }

                    using (Pen pen = new Pen(connectionColor, thickness))
                    {
                        g.DrawLine(pen, startPoint, endPoint);
                    }

                    PointF textPos = new PointF((startPoint.X + endPoint.X) / 2, (startPoint.Y + endPoint.Y) / 2 - 12);
                    using (Font font = new Font("Arial", 7))
                    using (var bgBrush = new SolidBrush(Color.FromArgb(180, Color.White)))
                    using (var textBrush = new SolidBrush(Color.Black))
                    {
                        string weightText = weight.ToString("F2");
                        if (currentActivations != null)
                        {
                            weightText += $"\n({signalStrength:F2})";
                        }

                        var textSize = g.MeasureString(weightText, font);
                        g.FillRectangle(bgBrush, textPos.X - textSize.Width / 2, textPos.Y, textSize.Width, textSize.Height);
                        g.DrawString(weightText, font, textBrush, textPos.X - textSize.Width / 2, textPos.Y);
                    }
                }
            }
        }

        private void DrawNeurons(Graphics g)
        {
            if (neuronPositions == null) return;
            float neuronRadius = (neuronPositions[0].Count > 1 ? neuronPositions[0][1].Y - neuronPositions[0][0].Y : 75f) / 2.5f;
            neuronRadius = Math.Min(30f, neuronRadius);
            string[] layerPrefix = { "Giriş", "Gizli 1", "Gizli 2", "Çıkış" };

            using (Font font = new Font("Segoe UI", 8, FontStyle.Bold))
            using (StringFormat stringFormat = new StringFormat { Alignment = StringAlignment.Center, LineAlignment = StringAlignment.Center })
            {
                for (int i = 0; i < neuronPositions.Length; i++)
                {
                    double[] layerActivations = null;
                    if (currentActivations != null)
                    {
                        switch (i)
                        {
                            case 0: layerActivations = currentActivations.InputActivations; break;
                            case 1: layerActivations = currentActivations.Hidden1Activations; break;
                            case 2: layerActivations = currentActivations.Hidden2Activations; break;
                            case 3: layerActivations = currentActivations.OutputActivations; break;
                        }
                    }

                    for (int j = 0; j < neuronPositions[i].Count; j++)
                    {
                        PointF pos = neuronPositions[i][j];
                        RectangleF rect = new RectangleF(pos.X - neuronRadius, pos.Y - neuronRadius, 2 * neuronRadius, 2 * neuronRadius);

                        Color centerColor = Color.LightSkyBlue;
                        Color surroundColor = Color.CornflowerBlue;

                        if (layerActivations != null && j < layerActivations.Length)
                        {
                            double activation = layerActivations[j];
                            int intensity = (int)(activation * 255);
                            centerColor = Color.FromArgb(255, 255 - intensity, intensity, intensity);
                            surroundColor = Color.FromArgb(255, 128 - intensity / 2, intensity / 2, intensity / 2);
                        }

                        using (var path = new GraphicsPath())
                        {
                            path.AddEllipse(rect);
                            using (var brush = new PathGradientBrush(path)
                            {
                                CenterColor = centerColor,
                                SurroundColors = new[] { surroundColor }
                            })
                            {
                                g.FillEllipse(brush, rect);
                            }
                        }
                        g.DrawEllipse(Pens.Black, rect);

                        string label = $"{layerPrefix[i]}\n#{j + 1}";

                        if (layerActivations != null && j < layerActivations.Length)
                        {
                            label += $"\n{layerActivations[j]:F2}";
                        }

                        g.DrawString(label, font, Brushes.White, rect, stringFormat);
                    }
                }
            }
        }

        private Color GetColorForWeight(double weight)
        {
            int alpha = 60 + (int)(Math.Min(Math.Abs(weight), 2.5) / 2.5 * 195);
            alpha = Math.Max(0, Math.Min(255, alpha));
            return weight > 0 ? Color.FromArgb(alpha, 34, 139, 34) : Color.FromArgb(alpha, 220, 20, 60);
        }

        private static double[,] ConvertToMultiDimensionalArray(double[][] jaggedArray)
        {
            int rows = jaggedArray.Length;
            int cols = jaggedArray.Length > 0 ? jaggedArray[0].Length : 0;
            double[,] multiArray = new double[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    multiArray[i, j] = jaggedArray[i][j];
                }
            }
            return multiArray;
        }

        private void InitializeComponentProgrammatically()
        {
            this.topPanel = new Panel();
            this.btnTrain = new Button();
            this.btnLoad = new Button();
            this.btnSave = new Button();
            this.lblInfo = new Label();
            this.logTextBox = new RichTextBox();
            this.networkPanel = new Panel();
            this.predictionGroupBox = new GroupBox();
            this.lblInput1 = new Label();
            this.numInput1 = new NumericUpDown();
            this.lblInput2 = new Label();
            this.numInput2 = new NumericUpDown();
            this.lblInput3 = new Label();
            this.numInput3 = new NumericUpDown();
            this.btnPredict = new Button();
            this.lblPredictionResult = new Label();
            this.topPanel.SuspendLayout();
            this.predictionGroupBox.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numInput1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numInput2)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numInput3)).BeginInit();
            this.SuspendLayout();
            this.topPanel.Controls.Add(this.btnSave); this.topPanel.Controls.Add(this.btnLoad); this.topPanel.Controls.Add(this.btnTrain); this.topPanel.Controls.Add(this.lblInfo); this.topPanel.Dock = DockStyle.Top; this.topPanel.Location = new Point(0, 0); this.topPanel.Name = "topPanel"; this.topPanel.Size = new Size(1200, 55);
            this.btnTrain.Font = new Font("Segoe UI", 9F, FontStyle.Bold); this.btnTrain.Location = new Point(12, 12); this.btnTrain.Size = new Size(210, 32); this.btnTrain.Text = "Yeni 3-Girişli Model Eğit";
            this.btnLoad.Font = new Font("Segoe UI", 9F); this.btnLoad.Location = new Point(228, 12); this.btnLoad.Size = new Size(210, 32); this.btnLoad.Text = "Model Yükle ve Görselleştir";
            this.btnSave.Enabled = false; this.btnSave.Font = new Font("Segoe UI", 9F); this.btnSave.Location = new Point(444, 12); this.btnSave.Size = new Size(165, 32); this.btnSave.Text = "Mevcut Modeli Kaydet";
            this.lblInfo.Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right; this.lblInfo.Font = new Font("Segoe UI Semibold", 9.75F, FontStyle.Bold); this.lblInfo.Location = new Point(615, 0); this.lblInfo.Size = new Size(580, 55); this.lblInfo.Text = "Başlamak için 3-Girişli bir model eğitin veya yükleyin."; this.lblInfo.TextAlign = ContentAlignment.MiddleCenter;
            this.logTextBox.BackColor = Color.FromArgb(30, 30, 30); this.logTextBox.BorderStyle = BorderStyle.None; this.logTextBox.Dock = DockStyle.Bottom; this.logTextBox.Font = new Font("Consolas", 9F); this.logTextBox.ForeColor = Color.Gainsboro; this.logTextBox.Location = new Point(0, 600); this.logTextBox.ReadOnly = true; this.logTextBox.ScrollBars = RichTextBoxScrollBars.Vertical; this.logTextBox.Size = new Size(1200, 161); this.logTextBox.Text = "Uygulama başlatıldı. Lütfen bir işlem seçin.";
            this.predictionGroupBox.Controls.Add(this.lblPredictionResult); this.predictionGroupBox.Controls.Add(this.btnPredict); this.predictionGroupBox.Controls.Add(this.numInput3); this.predictionGroupBox.Controls.Add(this.lblInput3); this.predictionGroupBox.Controls.Add(this.numInput2); this.predictionGroupBox.Controls.Add(this.lblInput2); this.predictionGroupBox.Controls.Add(this.numInput1); this.predictionGroupBox.Controls.Add(this.lblInput1); this.predictionGroupBox.Dock = DockStyle.Right; this.predictionGroupBox.Enabled = false; this.predictionGroupBox.Font = new Font("Segoe UI", 9F, FontStyle.Bold); this.predictionGroupBox.Location = new Point(950, 55); this.predictionGroupBox.Name = "predictionGroupBox"; this.predictionGroupBox.Size = new Size(250, 545); this.predictionGroupBox.Text = "Canlı Tahmin";
            this.lblInput1.AutoSize = true; this.lblInput1.Font = new Font("Segoe UI", 9F); this.lblInput1.Location = new Point(20, 40); this.lblInput1.Text = "Giriş 1:";
            this.numInput1.Font = new Font("Segoe UI", 12F); this.numInput1.Location = new Point(23, 60); this.numInput1.Maximum = 1;
            this.lblInput2.AutoSize = true; this.lblInput2.Font = new Font("Segoe UI", 9F); this.lblInput2.Location = new Point(20, 100); this.lblInput2.Text = "Giriş 2:";
            this.numInput2.Font = new Font("Segoe UI", 12F); this.numInput2.Location = new Point(23, 120); this.numInput2.Maximum = 1;
            this.lblInput3.AutoSize = true; this.lblInput3.Font = new Font("Segoe UI", 9F); this.lblInput3.Location = new Point(20, 160); this.lblInput3.Text = "Giriş 3:";
            this.numInput3.Font = new Font("Segoe UI", 12F); this.numInput3.Location = new Point(23, 180); this.numInput3.Maximum = 1;
            this.btnPredict.Font = new Font("Segoe UI", 10F, FontStyle.Bold); this.btnPredict.Location = new Point(23, 230); this.btnPredict.Size = new Size(204, 40); this.btnPredict.Text = "Tahmin Et";
            this.lblPredictionResult.BackColor = Color.Gainsboro; this.lblPredictionResult.BorderStyle = BorderStyle.FixedSingle; this.lblPredictionResult.Font = new Font("Segoe UI", 12F, FontStyle.Bold); this.lblPredictionResult.Location = new Point(23, 290); this.lblPredictionResult.Size = new Size(204, 100); this.lblPredictionResult.Text = "Sonuç Bekleniyor..."; this.lblPredictionResult.TextAlign = ContentAlignment.MiddleCenter;
            this.networkPanel.BackColor = Color.White; this.networkPanel.BorderStyle = BorderStyle.FixedSingle; this.networkPanel.Dock = DockStyle.Fill; this.networkPanel.Location = new Point(0, 55);
            this.ClientSize = new Size(1200, 761); this.Controls.Add(this.networkPanel); this.Controls.Add(this.predictionGroupBox); this.Controls.Add(this.logTextBox); this.Controls.Add(this.topPanel); this.MinimumSize = new Size(900, 600); this.Name = "Form1"; this.StartPosition = FormStartPosition.CenterScreen; this.Text = "Gelişmiş Yapay Sinir Ağı Görselleştiricisi (3-Girişli) - Canlı Aktivasyon";
            this.topPanel.ResumeLayout(false); this.predictionGroupBox.ResumeLayout(false); this.predictionGroupBox.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numInput1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numInput2)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numInput3)).EndInit();
            this.ResumeLayout(false);
        }
    }

    public class TextBoxStreamWriter : TextWriter
    {
        private RichTextBox _output;
        public TextBoxStreamWriter(RichTextBox output) => _output = output;
        public override void Write(char value) => _output.Invoke((MethodInvoker)delegate { _output.AppendText(value.ToString()); _output.ScrollToCaret(); });
        public override System.Text.Encoding Encoding => System.Text.Encoding.UTF8;
    }

    public static class Program
    {
        [STAThread]
        public static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Form1());
        }
    }
}