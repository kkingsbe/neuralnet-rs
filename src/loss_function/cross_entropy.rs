use ndarray::{Array1, Array2, Axis};

use super::LossCalculation;

pub enum CrossEntropyEncoding {
    Sparse,
    OneHot
}

pub struct CrossEntropy {}

impl CrossEntropy {
    pub fn new() -> CrossEntropy {
        CrossEntropy {}
    }

    fn determine_correct_confidences_sparse(&self, outputs: Array2<f64>, target_output_set: Array1<usize>) -> Vec<f64> {
        outputs
            .outer_iter()
            .zip(target_output_set.iter())
            .map(|(output, target_class)| output[target_class.to_owned()]) //Get the corresponding output for the correct class
            .collect::<Vec<f64>>()
    }

    fn determine_correct_confidences_one_hot(&self, outputs: Array2<f64>, target_output_set: Array2<f64>) -> Vec<f64> {
        (outputs * target_output_set).sum_axis(Axis(1)).to_owned().into_iter().collect::<Vec<f64>>()
    }

    fn cross_entropy(&self, confidences: Vec<f64>) -> Array1<f64> {
        Array1::from(confidences.iter().map(|confidence| -1.0 * confidence.ln()).collect::<Vec<f64>>())
    }

    /**
     * Clips the provided values to prevent them from equaling 0. 
     * Symmetrically clips to prevent skewing the confidence values, 
     * so it also clips the values to prevent them from equalling 1
     */
    fn clip_values(&self, values: Array2<f64>) -> Array2<f64> {
        values.mapv(|val| val.clamp(1E-7, 1. - 1E-7))
    }

    /**
     * `target_output_set` of format [[0, 1, 0], [1, 0, 0], etc]. The correct class is given a "1" while the others are given a 0
     */
    pub fn cross_entropy_one_hot(&self, outputs: Array2<f64>, target_output_set: Array2<f64>) -> Array1<f64> {
        let correct_confidences = self.determine_correct_confidences_one_hot(outputs, target_output_set);
        self.cross_entropy(correct_confidences)
    }

    /**
     * `target_output_set` of format [1, 0, 3, etc]. Each item represents the index of the correct class
     */
    pub fn cross_entropy_sparse(&self, outputs: Array2<f64>, target_output_set: Array1<usize>) -> Array1<f64> {
        let correct_confidences = self.determine_correct_confidences_sparse(outputs, target_output_set);
        self.cross_entropy(correct_confidences)
    }
}

impl LossCalculation for CrossEntropy {
    fn forward_sparse(&self, outputs: Array2<f64>, target_outputs: Array1<usize>) -> Array1<f64> {
        self.cross_entropy_sparse(self.clip_values(outputs), target_outputs)
    }

    fn forward_onehot(&self, outputs: Array2<f64>, target_outputs: Array2<f64>) -> Array1<f64> {
        self.cross_entropy_one_hot(self.clip_values(outputs), target_outputs)
    }
}

#[cfg(test)]
mod cross_entropy_confidences {
    use ndarray::array;

    use super::CrossEntropy;


    #[test]
    fn determine_confidences_sparse() {
        let outputs = array![
            [0.7, 0.1, 0.2],
            [0.1, 0.5, 0.4],
            [0.02, 0.9, 0.08]
        ];

        let class_targets = array![0, 1, 1];

        let func = CrossEntropy::new();
        let correct_confidences = func.determine_correct_confidences_sparse(outputs, class_targets);

        assert!(correct_confidences.len() == 3);
        assert!(correct_confidences[0] == 0.7);
        assert!(correct_confidences[1] == 0.5);
        assert!(correct_confidences[2] == 0.9);
    }

    #[test]
    fn determine_confidences_onehot() {
        let outputs = array![
            [0.7, 0.1, 0.2],
            [0.1, 0.5, 0.4],
            [0.02, 0.9, 0.08]
        ];

        let class_targets = array![
            [1.0, 0., 0.],
            [0., 1.0, 0.],
            [0., 1.0, 0.]
        ];

        let func = CrossEntropy::new();
        let correct_confidences = func.determine_correct_confidences_one_hot(outputs, class_targets);

        assert!(correct_confidences.len() == 3);
        assert!(correct_confidences[0] == 0.7);
        assert!(correct_confidences[1] == 0.5);
        assert!(correct_confidences[2] == 0.9);
    }
}