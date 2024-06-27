pub mod cross_entropy;

use cross_entropy::CrossEntropy;
use ndarray::{Array1, Array2};

pub trait LossCalculation {
    fn forward_onehot(&self, outputs: Array2<f64>, target_outputs: Array2<f64>) -> Array1<f64>;
    fn forward_sparse(&self, outputs: Array2<f64>, target_outputs: Array1<usize>) -> Array1<f64>;
}

pub trait LossTargetData {
    fn new_onehot(data: Array2<f64>) -> Self;
    fn new_sparse(data: Array1<usize>) -> Self;
    fn encoding(&self) -> LossFunctionTargetEncoding;
    fn get_onehot(&self) -> Array2<f64>;
    fn get_sparse(&self) -> Array1<usize>;
}

pub struct OneHotLossTargetData {
    data: Array2<f64>
}

impl LossTargetData for OneHotLossTargetData {
    fn new_onehot(data: Array2<f64>) -> OneHotLossTargetData {
        OneHotLossTargetData {
            data
        }
    }

    fn new_sparse(data: Array1<usize>) -> OneHotLossTargetData {
        panic!("Attempted to load sparse target data into struct configured for one-hot");
    }

    fn encoding(&self) -> LossFunctionTargetEncoding {
        LossFunctionTargetEncoding::OneHot
    }

    fn get_onehot(&self) -> Array2<f64> {
        self.data.clone()
    }

    fn get_sparse(&self) -> Array1<usize> {
        panic!("Attempted to get sparse target data from struct configured for one-hot");
    }
}

pub struct SparseLossTargetData {
    data: Array1<usize>
}

impl LossTargetData for SparseLossTargetData {
    fn new_onehot(data: Array2<f64>) -> SparseLossTargetData {
        panic!("Attempted to load one-hot target data into struct configured for sparse");
    }

    fn new_sparse(data: Array1<usize>) -> SparseLossTargetData {
        SparseLossTargetData {
            data
        }
    }

    fn encoding(&self) -> LossFunctionTargetEncoding {
        LossFunctionTargetEncoding::Sparse
    }

    fn get_onehot(&self) -> Array2<f64> {
        panic!("Attempted to get one-hot target data from struct configured for sparse");
    }

    fn get_sparse(&self) -> Array1<usize> {
        self.data.clone()
    }
}

pub enum LossFunctionType {
    CrossEntropy
}

pub enum LossFunctionTargetEncoding {
    OneHot,
    Sparse
}

pub struct LossFunction {
    function: Box<dyn LossCalculation>
}

impl LossFunction {
    pub fn new(function_type: LossFunctionType) -> LossFunction {
        LossFunction {
            function: match function_type {
                LossFunctionType::CrossEntropy => Box::new(CrossEntropy::new()) as Box<dyn LossCalculation>
            }
        }
    }

    pub fn calculate<T>(&self, outputs: Array2<f64>, target_outputs: T) -> f64 
    where
        T: LossTargetData,
    {
        match target_outputs.encoding() {
            LossFunctionTargetEncoding::OneHot => {
                self.function.forward_onehot(outputs, target_outputs.get_onehot()).mean().unwrap_or(0.0)
            },
            LossFunctionTargetEncoding::Sparse => {
                self.function.forward_sparse(outputs, target_outputs.get_sparse()).mean().unwrap_or(0.0)
            }
        }
    }
}