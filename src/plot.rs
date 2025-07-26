use std::path::{Path, PathBuf};

use capitalize::Capitalize;
use itertools::Itertools;
use log::info;
use ndarray::{Array1, ArrayBase, Dimension, OwnedRepr};
use num_ordinal::{Ordinal, Osize};
use plotly::{
    Plot, Scatter,
    common::{Mode, Title},
    plotly_static,
};

pub fn plot_t_traces<D: Dimension, P: AsRef<Path>>(
    t_values: ArrayBase<OwnedRepr<f64>, D>,
    t_threshold: Option<f64>,
    abs_values: bool,
    output_dir: P,
    show_plots: bool,
    plots_config: &plotly::Configuration,
    image_exporter: &mut plotly_static::StaticExporter,
) -> miette::Result<()> {
    let threshold_lines = if let Some(t) = t_threshold {
        let pos = plotly::layout::Shape::new()
            .shape_type(plotly::layout::ShapeType::Line)
            .x0(0)
            .x1(t_values.len_of(ndarray::Axis(1)) - 1)
            .y0(t)
            .y1(t)
            .line(
                plotly::layout::ShapeLine::new()
                    .color(plotly::color::NamedColor::Red)
                    .width(1.0)
                    .dash(plotly::common::DashType::Dot),
            );
        if !abs_values {
            let neg = plotly::layout::Shape::new()
                .shape_type(plotly::layout::ShapeType::Line)
                .x0(0)
                .x1(t_values.len_of(ndarray::Axis(1)) - 1)
                .y0(-t)
                .y1(-t)
                .line(
                    plotly::layout::ShapeLine::new()
                        .color(plotly::color::NamedColor::Red)
                        .width(1.0)
                        .dash(plotly::common::DashType::Dot),
                );
            vec![pos, neg]
        } else {
            vec![pos]
        }
    } else {
        vec![]
    };

    let lines = threshold_lines.clone();
    let y_label = if abs_values { "|t|" } else { "t-value" };
    let t_traces = t_values
        .rows()
        .into_iter()
        .enumerate()
        .map(|(i, order_t1_values)| {
            let d = i + 1;
            // map inf to 0 and also to absolute value
            let ord_t_values = order_t1_values.map(|x| {
                if x.is_finite() {
                    if abs_values { x.abs() } else { *x }
                } else {
                    0.0
                }
            });

            info!("Plotting t-values for d={d}");
            //plot the t-values
            let mut t_plot = Plot::new();

            let t_trace = Scatter::from_array(
                Array1::range(0., ord_t_values.len() as f32, 1.0),
                ord_t_values.clone(),
            )
            .mode(Mode::Lines)
            .line(
                plotly::common::Line::new()
                    .width(2.0)
                    .auto_color_scale(true),
            )
            .x_axis(y_label)
            .y_axis("time (cycles)");
            t_plot.add_trace(t_trace.clone());
            let y_axis = plotly::layout::Axis::new().title(Title::with_text(y_label));
            // y_max is: if t_threshold is Some(t) => Some(v) where v is the maximum if t and the absolute value of the t-values, otherwise its None
            let max_y = if let Some(t) = t_threshold {
                Some(
                    ord_t_values
                        .iter()
                        .fold(t, |acc, &x| acc.max(x.abs()))
                        .max(1.5 * t)
                        + 0.5,
                )
            } else {
                None
            };
            log::info!(
                "Max y for d={d}: {}",
                max_y.map_or("None".to_string(), |v| v.to_string())
            );
            let y_axis = if let Some(max_y) = max_y {
                y_axis
                    .range(vec![if abs_values { 0.0 } else { -max_y }, max_y])
                    .auto_range(false)
            } else {
                y_axis.auto_range(true)
            };
            t_plot.set_layout(
                plotly::Layout::new()
                    .shapes(lines.clone())
                    .x_axis(plotly::layout::Axis::new().title("Time (cycles)"))
                    .y_axis(y_axis),
            );

            let file_stem = PathBuf::from(format!("t_test_d{d}"));
            t_plot.set_configuration(plots_config.clone());
            let html_output_path = output_dir.as_ref().join(file_stem.with_extension("html"));
            info!("Writing t_plot to {}", html_output_path.display());
            t_plot.write_html(html_output_path);
            let image_output_path = output_dir.as_ref().join(file_stem.with_extension("svg"));
            info!("Writing t_plot to {}", image_output_path.display());
            if let Err(e) = t_plot.write_image_with_exporter(
                image_exporter,
                image_output_path,
                plotly_static::ImageFormat::SVG,
                800,
                600,
                1.0,
            ) {
                log::error!("Failed to write t_plot to PDF: {}", e);
            }
            let t_plot_json_path = output_dir.as_ref().join(file_stem.with_extension("json"));
            std::fs::write(t_plot_json_path, t_plot.to_json())
                .expect("Failed to write t_plot to JSON file");
            if show_plots {
                t_plot.show();
            }
            t_trace
        });
    let t_plots_file_stem = PathBuf::from("all_t_values");
    let mut all_t_plot = Plot::new();
    all_t_plot.set_configuration(plots_config.clone());
    all_t_plot.set_layout(
        plotly::Layout::new()
            .x_axis(plotly::layout::Axis::new().title("time (cycles)"))
            .y_axis(plotly::layout::Axis::new().title(Title::with_text(y_label)))
            .shapes(threshold_lines),
    );
    for (i, t_trace) in t_traces.enumerate() {
        let d = i + 1;
        all_t_plot.add_trace(t_trace.name(format!("d={d}")));
    }
    let html_output_path = output_dir
        .as_ref()
        .join(t_plots_file_stem.with_extension("html"));
    info!("Writing all_t_plot to {}", html_output_path.display());
    all_t_plot.write_html(html_output_path);
    if show_plots {
        all_t_plot.show();
    }
    Ok(())
}

pub fn plot_max_t_values(
    max_t_values: Vec<Vec<f64>>,
    num_traces_so_far: Vec<usize>,
    t_threshold: Option<f64>,
    output_dir: &Path,
    show_plots: bool,
    plots_config: &plotly::Configuration,
    image_exporter: &mut plotly_static::StaticExporter,
) -> miette::Result<()> {
    assert!(num_traces_so_far.len() == max_t_values[0].len());

    // plot max t-values
    let mut max_t_plot = Plot::new();
    max_t_plot.set_configuration(plots_config.clone());

    let threshold_line = t_threshold.map(|t| {
        plotly::layout::Shape::new()
            .shape_type(plotly::layout::ShapeType::Line)
            .x0(0)
            .x1(num_traces_so_far.last().expect("No traces found") - 1)
            .y0(t)
            .y1(t)
            .line(
                plotly::layout::ShapeLine::new()
                    .color(plotly::color::NamedColor::Red)
                    .width(1.0)
                    .dash(plotly::common::DashType::Dot),
            )
    });

    max_t_plot.set_layout(
        plotly::Layout::new()
            // .title("Max t-values (max(|t|)) vs Number of Traces")
            .x_axis(plotly::layout::Axis::new().title("Number of Traces"))
            .y_axis(plotly::layout::Axis::new().title("max(|t|)"))
            .shapes(threshold_line.into_iter().collect()),
    );
    for (i, max_tvals) in max_t_values.into_iter().enumerate() {
        let d = i + 1;
        println!(
            "Max t-value for d={d}: {:.03}",
            max_tvals
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(&0.0)
        );
        let max_t_trace = Scatter::new(num_traces_so_far.clone(), max_tvals)
            .mode(Mode::Lines)
            .name(format!("d={d}"))
            .line(
                plotly::common::Line::new()
                    .width(1.0)
                    .auto_color_scale(true),
            );
        max_t_plot.add_trace(max_t_trace);
    }
    let max_t_plot_file_stem = PathBuf::from("max_t_values");
    let max_t_plot_file_stem = output_dir.join(max_t_plot_file_stem);
    let html_output_path = max_t_plot_file_stem.with_extension("html");
    info!("Writing max_t_plot to {}", html_output_path.display());
    max_t_plot.write_html(html_output_path);
    // max_t_plot.write_image(
    //     output_dir.join(max_t_plot_file_stem.with_extension("png")),
    //     plotly::ImageFormat::PNG,
    //     1920,
    //     400,
    //     1.0,
    // );
    let max_t_plot_json_path = max_t_plot_file_stem.with_extension("json");
    info!("Writing max_t_plot to {}", max_t_plot_json_path.display());
    std::fs::write(max_t_plot_json_path, max_t_plot.to_json())
        .expect("Failed to write max_t_plot to JSON file");
    info!(
        "Writing max_t_plot to {}",
        max_t_plot_file_stem.with_extension("svg").display()
    );
    if let Err(e) = max_t_plot.write_image_with_exporter(
        image_exporter,
        max_t_plot_file_stem.with_extension("svg"),
        plotly_static::ImageFormat::SVG,
        800,
        600,
        1.0,
    ) {
        log::error!("Failed to write max_t_plot to SVG: {}", e);
    }
    if show_plots {
        max_t_plot.show();
    }
    Ok(())
}
