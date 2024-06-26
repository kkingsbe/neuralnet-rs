use std::fs;

use ndarray::{prelude::*, stack, OwnedRepr};
mod layer;

use layer::Layer;
use ndarray_rand::{rand_distr::{num_traits::Float, Normal}, RandomExt};
use charts::{Chart, ScaleBand, ScaleLinear, ScatterView, VerticalBarView};

#[derive(Debug)]
pub struct SpiralItem {
    x: f64,
    y: f64,
    class: i32
}

fn spiral_data() -> Vec<SpiralItem> {
    let file = fs::File::open("./src/datasets/spiral.json").expect("File should open read only");
    let json: serde_json::Value = serde_json::from_reader(file).expect("File should be proper JSON");
    let data = json["data"].as_array().expect("Data should be an array");

    let items: Vec<SpiralItem> = data.iter().map(|item| {
        let x = item["x"].as_f64().expect("X should be a float");
        let y = item["y"].as_f64().expect("Y should be a float");
        let class = item["class"].as_i64().expect("Class should be an integer") as i32;

        SpiralItem { x, y, class }
    }).collect();

    items
}

fn plot() {
    let width = 800;
    let height = 600;
    let (top, right, bottom, left) = (90, 40, 50, 60);

    let x = ScaleLinear::new()
        .set_domain(vec![-1.0, 1.0])
        .set_range(vec![height - top - bottom, 0]);

    let y = ScaleLinear::new()
        .set_domain(vec![-1.0, 1.0])
        .set_range(vec![height - top - bottom, 0]);

    let data = spiral_data();

    let data_vec: Vec<(f32, f32)> = data.iter().map(|item| {
        (item.x as f32, item.y as f32)
    }).collect();

    let view = ScatterView::new()
        .set_x_scale(&x)
        .set_y_scale(&y)
        .load_data(&data_vec).unwrap();

    Chart::new()
        .set_width(width)
        .set_height(height)
        .set_margins(top, right, bottom, left)
        .add_title(String::from("Spiral Data"))
        .add_view(&view)
        .add_axis_bottom(&x)
        .add_axis_left(&y)
        .save("scatter.svg").unwrap()
}

fn main() {
    let dataset = spiral_data();
    let dataset_arr = arr2(&dataset.iter().map(|item| {
        [item.x, item.y]
    }).collect::<Vec<[f64; 2]>>());

    let dense_1 = Layer::<3,2>::new(layer::LayerActivation::ReLU);
    let dense_2 = Layer::<3,3>::new(layer::LayerActivation::Softmax);

    let dense_1_outputs = dense_1.forward(dataset_arr);

    println!("{:#?}", dense_1_outputs.slice(s![0..5, ..]));

    let dense_2_outputs = dense_2.forward(dense_1_outputs);

    println!("{:#?}", dense_2_outputs.slice(s![0..5, ..]));
}
