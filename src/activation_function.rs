use ndarray::{Array2, Axis};

pub enum LayerActivation {
    ReLU,
    Softmax
}

pub struct ActivationFunction {
    activation: LayerActivation
}

impl ActivationFunction {
    pub fn new(activation: LayerActivation) -> ActivationFunction {
        ActivationFunction {
            activation
        }
    }

    fn relu(&self, inputs: Array2<f64>) -> Array2<f64> {
        inputs.mapv(|x| x.max(0.0))
    }

    /**
     * Used as the activation function within the output layer for classification models. 
     * Outputs a probability distribution for each row of inputs.
     */
    fn softmax(&self, inputs: Array2<f64>) -> Array2<f64> {
        // find the maximum value in the inputs
        let max = inputs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // exponentiate each item
        let exps = inputs.mapv(|x| (x - max).exp());

        // sum each of the exponentiated items in each row
        let sums = exps.sum_axis(Axis(1));

        // divide each item by the sum of the row
        let res = exps / &sums.insert_axis(Axis(1));

        res
    }

    pub fn forward(&self, inputs: Array2<f64>) -> Array2<f64> {
        match self.activation {
            LayerActivation::ReLU => self.relu(inputs),
            LayerActivation::Softmax => self.softmax(inputs)
        }
    }
}