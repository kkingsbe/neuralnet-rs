use std::f64::MIN;

use ndarray::prelude::*;
use ndarray_rand::{rand_distr::Normal, RandomExt};

#[derive(Debug)]
pub enum LayerArrayError {
    IncorrectDimension(String)
}

pub enum LayerActivation {
    ReLU,
    Softmax
}

/**
 * NEURONS: The total number of neurons in this layer
 * FAN_IN: The number of inputs to each neuron within this layer. There must be one weight per input, so len(weights[0]) == FAN_IN
 */
pub struct Layer<const NEURONS: usize, const FAN_IN: usize> {
    weights: Array2<f64>, // Weights for each neuron are stored as a column
    biases: Array1<f64>,
    activation: LayerActivation
}

impl <const NEURONS: usize, const FAN_IN: usize> Layer<NEURONS, FAN_IN> {
    pub fn new(activation: LayerActivation) -> Layer<NEURONS, FAN_IN> {
        Layer { 
            weights: 0.01 * Layer::<NEURONS, FAN_IN>::random_weights(),
            biases: Array::zeros(NEURONS),
            activation
        }
    }

    fn random_weights() -> Array2<f64> {
        let normal = Normal::new(0.0, 1.0).unwrap();
        Array::random((FAN_IN, NEURONS), normal)
    }

    pub fn relu(&self, inputs: Array2<f64>) -> Array2<f64> {
        inputs.mapv(|x| x.max(0.0))
    }

    pub fn softmax(&self, inputs: Array2<f64>) -> Array2<f64> {
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

    pub fn set_weights(&mut self, new_weights: Array2<f64>) -> Result<(), LayerArrayError> {
        if new_weights.dim().0 != NEURONS {
            return Err(LayerArrayError::IncorrectDimension(
                format!(
                    "{} neurons exist in this layer, but {} sets of weights were provided.", 
                    NEURONS, 
                    new_weights.dim().0
                ).to_string()
            ))
        }

        if new_weights.dim().1 != FAN_IN {
            return Err(LayerArrayError::IncorrectDimension(
                format!(
                    "Neurons in this layer have {} weights each, but {} weights were provided in each set.", 
                    FAN_IN, 
                    new_weights.dim().1
                ).to_string()
            ))
        }
        
        self.weights = new_weights;

        Ok(())
    }

    pub fn set_biases(&mut self, new_biases: Array1<f64>) -> Result<(), LayerArrayError> {
        if new_biases.dim() != NEURONS {
            return Err(LayerArrayError::IncorrectDimension(
                format!(
                    "{} neurons exist in this layer, but {} biases were provided.", 
                    NEURONS, 
                    new_biases.dim()
                ).to_string()
            ))
        }

        self.biases = new_biases;

        Ok(())
    }

    pub fn forward(&self, inputs: Array2<f64>) -> Array2<f64> {
        let intermediate_output = inputs.dot(&self.weights) + &self.biases;

        match self.activation {
            LayerActivation::ReLU => self.relu(intermediate_output),
            LayerActivation::Softmax => self.softmax(intermediate_output)
        }
    }
}

#[cfg(test)]
mod layer_tests {
    use ndarray::prelude::*;
    use crate::layer::LayerActivation;

    use super::Layer;

    #[test]
    fn create_layer() {
        const NEURONS: usize = 3;
        const FAN_IN: usize = 4;
        let layer = Layer::<NEURONS, FAN_IN>::new(LayerActivation::ReLU);
    }

    #[test]
    fn weights_validation() {
        const NEURONS: usize = 3;
        const FAN_IN: usize = 4;
        let mut layer = Layer::<NEURONS, FAN_IN>::new(LayerActivation::ReLU);

        // Each weight set should have a length equal to the FAN_IN
        let test_weights_1 = array![
            [0.1, -0.14, 0.5],    // [neuron 1 weight 1, neuron 2 weight 1, neuron 3 weight 1]
            [-0.5, 0.12, -0.33],  // [neuron 1 weight 2, neuron 2 weight 2, neuron 3 weight 2]
            [-0.44, 0.73, -0.13], // [neuron 1 weight 3, neuron 2 weight 3, neuron 3 weight 3]
        ];

        let set_weights_res_1 = layer.set_weights(test_weights_1);
        assert!(set_weights_res_1.is_err());

        // The number of weight sets should match the number of neurons
        let test_weights_2 = array![
            [0.1, -0.14, 0.5, 1.0],
            [-0.5, 0.12, -0.33, 2.5],
        ];

        let set_weights_res_2 = layer.set_weights(test_weights_2);
        assert!(set_weights_res_2.is_err());

        // Valid weights set
        let test_weights_3 = array![
            [0.1, -0.14, 0.5, 1.0],
            [-0.5, 0.12, -0.33, -0.5],
            [-0.44, 0.73, -0.13, 0.2],
        ];

        let set_weights_res_3 = layer.set_weights(test_weights_3);
        assert!(set_weights_res_3.is_ok());
    }

    #[test]
    fn biases_validation() {
        const NEURONS: usize = 3;
        const FAN_IN: usize = 4;
        let mut layer = Layer::<NEURONS, FAN_IN>::new(LayerActivation::ReLU);

        // The bias set should have a length equal to the number of neurons
        let test_biases_1 = array![0.1, 0.2];

        let set_biases_res_1 = layer.set_biases(test_biases_1);
        assert!(set_biases_res_1.is_err());

        // Valid biases set
        let test_biases_2 = array![2., 3., 0.5];

        let set_biases_res_2 = layer.set_biases(test_biases_2);
        assert!(set_biases_res_2.is_ok());
    }

    #[test]
    fn softmax() {
        let test_inputs = array![
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.]
        ];

        let layer = Layer::<3, 2>::new(LayerActivation::Softmax);
        let output = layer.softmax(test_inputs);
    }
}