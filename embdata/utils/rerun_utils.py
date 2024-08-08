import rerun as rr
import rerun.blueprint as rrb

def get_blueprint() -> rrb.Blueprint:

    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView( 
                    name=f"Scene",
                    background=[0.0, 0.0, 0.0, 0.0],
                    origin=f"scene",
                    visible=True,
                ),
                rrb.Spatial2DView(
                    name=f"Augmented",
                    background=[0.0, 0.0, 0.0, 0.0],
                    origin=f"augmented",
                    visible=True,
                ),
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(
                    name=f"Actions",
                    origin=f"action",
                    visible=True,
                    axis_y=rrb.ScalarAxis(range=(-0.5, 0.5), zoom_lock=True),
                    plot_legend=rrb.PlotLegend(visible=True),
                    time_ranges=[rrb.VisibleTimeRange("timeline0", start=rrb.TimeRangeBoundary.cursor_relative(seq=-100), end=rrb.TimeRangeBoundary.cursor_relative())],
                ),
            ),
            row_shares=[2, 1]
        ),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.TimePanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
    )
    return blueprint