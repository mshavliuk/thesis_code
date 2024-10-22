from matplotlib.transforms import Bbox


def get_fig_box(fig):
    fig.canvas.draw()
    fig_box = fig.get_tightbbox()
    
    margin_px = 1
    margin_in_inches = margin_px / fig.dpi
    
    fig_box_expanded = Bbox.from_bounds(
        fig_box.x0 - margin_in_inches,
        fig_box.y0 - margin_in_inches,
        fig_box.width + 2 * margin_in_inches,
        fig_box.height + 2 * margin_in_inches
    )
    
    return fig_box_expanded
