![Neural Network Visualized](https://i.imgur.com/UDAJdZu.png)

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
