import numpy as np
import plotly.graph_objects as go

# Module to be tested
import ipywgt_tool


#
# Confirm the plotly behavior assumptions
#
def test_obj_persistence():
    """Check persistence of objects inside 'data', 'annotations' and 'shapes' tuples"""
    fig = go.Figure()

    # Check persistence of fig.data entries (scatters)
    assert len(fig.data) == 0, 'Test expects empty data'
    fig.add_scatter()
    obj0 = fig.data[0]
    fig.add_scatter()
    assert obj0 is fig.data[0], 'fig.data changed after add_scatter()'

    # Check persistence of fig.layout.annotations entries (annotations)
    assert len(fig.layout.annotations) == 0, 'Test expects empty annotations'
    fig.add_annotation()
    annots = fig.layout.annotations
    obj0 = fig.layout.annotations[0]
    fig.add_annotation()
    #assert annots is fig.layout.annotations, 'fig.layout.annotations changed after add_annotation()'
    if annots is fig.layout.annotations:
        print('Warning: fig.layout.annotations NOT changed after add_annotation()')
    #assert obj0 is fig.layout.annotations[0], 'fig.layout.annotations[0] changed after add_annotation()'
    if obj0 is fig.layout.annotations[0]:
        print('Warning: fig.layout.annotations[0] NOT changed after add_annotation()')

    # Check persistence of fig.layout.shapes entries (shapes)
    assert len(fig.layout.shapes) == 0, 'Test expects empty shapes'
    fig.add_shape()
    shapes = fig.layout.shapes
    obj0 = fig.layout.shapes[0]
    fig.add_shape()
    #assert shapes is fig.layout.shapes, 'fig.layout.shapes changed after add_shape()'
    if shapes is fig.layout.shapes:
        print('Warning: fig.layout.shapes NOT changed after add_shape()')
    #assert obj0 is fig.layout.shapes[0], 'fig.layout.shapes[0] changed after add_shape()'
    if obj0 is fig.layout.shapes[0]:
        print('Warning: fig.layout.shapes[0] NOT changed after add_shape()')

#
# utility function  tests
#
def test_utils():
    """Check utility functions"""
    # Test split_complex()
    cvals = np.exp(np.arange(15) * .5j).reshape(3, -1)
    REAL_IMAG = np.array([1,1j])[:, np.newaxis]

    res = ipywgt_tool.split_complex(cvals)
    assert res.shape == (3, 2, 5), 'Split complex array shape'
    assert ((res * REAL_IMAG).sum(-2) == cvals).all()

    res = ipywgt_tool.split_complex(cvals, np.arange(15).reshape(3, -1))
    assert res.shape == (3, 3, 5), 'Split complex with aux array shape'

    res = ipywgt_tool.split_complex(cvals, np.arange(5), broadcast=True)
    assert res.shape == (3, 3, 5), 'Split complex with broadcasted aux array'
    res = ipywgt_tool.split_complex(cvals[0], np.arange(15).reshape(3, -1), broadcast=True)
    assert res.shape == (3, 3, 5), 'Split broadcasted complex with aux array'
    assert ((res[..., :2, :] * REAL_IMAG).sum(-2) == cvals[0]).all()

    # Test split_xyz()
    res = ipywgt_tool.split_xyz(xy=np.arange(6).reshape((2,-1)))
    assert (res['x'] == np.arange(0, 3)).all(), 'Split xy'
    assert (res['y'] == np.arange(0, 3) + 3).all(), 'Split xy'

    # Test flatten_join()
    res = ipywgt_tool.flatten_join(np.arange(12.).reshape(3, -1), np.nan)
    assert res.shape == (12 + 3-1,), 'Flatten by join array'
    res = ipywgt_tool.flatten_join(np.arange(2*3*4.).reshape(2, 3, -1), np.nan)
    assert res.shape == (2*3 + 2-1, 4) and np.isnan(res[3]).all(), 'Flatten by join array'
    res = ipywgt_tool.flatten_join(np.arange(2*3*4.).reshape(2, 3, -1), np.nan, num_axes=2)
    assert res.shape == (2*3*4 + 2*3-1,) and np.isnan(res[4::5]).all(), 'Flatten by join array'

#
# plot_data class tests
#
def test_plot_data_scatters():
    """Check scatters handling by plot_data class"""
    fig = go.Figure()

    # Create data objects
    scatt_1 = ipywgt_tool.add_scatter(fig, mode='lines', line_dash='dash', name='first')
    scatt_2 = ipywgt_tool.add_scatter(fig, mode='lines+markers', line_dash='dot', name='second')
    assert fig.data[0].mode == 'lines' and fig.data[1].name == 'second'

    # Update scatter objects
    scatt_1.x = (1,2,3)
    scatt_1.y = (10,20,30)
    assert fig.data[0].x == (1,2,3) and fig.data[0].y == (10,20,30)
    # Use iterables and numpy arrays
    scatt_2.x, scatt_2.y = np.arange(10), np.arange(10) + 100
    assert (fig.data[1].y == np.arange(10) + 100).all(), 'Non-iterated update'
    # Update list of objects
    ipywgt_tool.update_fig_objs([scatt_1, scatt_2],
            x=np.arange(10).reshape((2,-1)),    # Iterable exact
            y=np.arange(10).reshape((1,-1)),    # Iterable exhaustible
            mode='markers',                     # Single (string)
            marker_size=7)                      # Single (int)
    assert (fig.data[0].x == np.arange(5)).all(), 'Iterables must spread around'
    assert (fig.data[0].y == np.arange(10)).all()
    assert fig.data[0].marker.size == 7 and fig.data[1].marker.size == 7, 'Non-iterables must be reused'
    assert fig.data[0].mode == 'markers' and fig.data[1].mode == 'markers', 'Strings must treated as non-iterables'
    assert fig.data[1].y is None, 'Exhausted iterable must assigns None'

    # Try to call some functions
    scatt_1.on_click(lambda x: None)
    scatt_1.on_selection(lambda x: None)
    scatt_1.on_deselect(lambda x: None)

def test_plot_data_layout():
    """Check annotations and shapes handling by plot_data class"""
    fig = go.Figure()

    # Create some objects
    annot_1 = ipywgt_tool.add_annotation(fig, name='first')
    shape_1 = ipywgt_tool.add_shape(fig, name='shape_1')
    shape_2 = ipywgt_tool.add_shape(fig, name='shape_2')
    annot_2 = ipywgt_tool.add_annotation(fig, name='second')
    assert fig.layout.annotations[0].name == 'first' and fig.layout.annotations[1].name == 'second'
    assert fig.layout.shapes[0].name == 'shape_1' and fig.layout.shapes[1].name == 'shape_2'

    # Update annotation/shape objects
    annot_1.name = 'First'
    annot_1.x = 5
    assert fig.layout.annotations[0].name == 'First' and fig.layout.annotations[0].x == 5
    shape_2.name, shape_2.y0 = 'Shape_2', 10
    assert fig.layout.shapes[1].name == 'Shape_2' and fig.layout.shapes[1].y0 == 10
    # Use iterables and range()
    ipywgt_tool.update_fig_objs([shape_2, shape_1],
            x0=(0, 1),          # Iterable exact
            x1=range(1,5),      # Iterable exact
            y0=range(10,11),    # Iterable exhaustible
            xanchor='paper')    # Single
    assert fig.layout.shapes[0].x0 == 1 and fig.layout.shapes[1].x0 == 0, 'Iterables must spread around'
    assert fig.layout.shapes[0].x1 == 2 and fig.layout.shapes[1].x1 == 1
    assert fig.layout.shapes[0].y0 == None and fig.layout.shapes[1].y0 == 10, 'Exhausted iterable must assigns None'
    assert fig.layout.shapes[0].xanchor == 'paper' and fig.layout.shapes[1].xanchor == 'paper', 'Non-iterables must be reused'

def test_multiple():
    """Check add_multiple() functionality"""
    fig = go.FigureWidget()

    # Create multiple objects
    scatts = ipywgt_tool.add_multiple(fig, ipywgt_tool.add_scatter3d,
            x=[[0], [1], [3]],                  # Iterable exact (each x must be iterable)
            line_dash='dash',                   # Single
            mode=['lines', 'markers'],          # Iterable exhaustible
            name=['first', 'second', 'third'])  # Iterable exact
    assert fig.data[0].x == (0,) and fig.data[1].x == (1,) and fig.data[2].x == (3,)
    assert fig.data[0].mode == 'lines' and fig.data[1].mode == 'markers' and fig.data[2].mode == None
    assert fig.data[0].name == 'first' and fig.data[1].name == 'second' and fig.data[2].name == 'third'

    # Update using combined numpy xyz coordinates
    ipywgt_tool.update_fig_objs_xyz(scatts,
            xyz=np.arange(24).reshape((2,3,-1)),  # xyz iterable
            mode='lines')
    assert (fig.data[0].x == np.arange(0, 4)).all() \
            and (fig.data[1].x == np.arange(12, 16)).all() \
            and fig.data[2].x is None
    assert (fig.data[0].z == np.arange(8, 12)).all() \
            and (fig.data[1].z == np.arange(20, 24)).all() \
            and fig.data[2].z is None

    # Create mapbox scatter objects
    scatts = ipywgt_tool.add_multiple(fig, ipywgt_tool.add_scattermapbox,
            name=['first_mbox', 'second_mbox'])   # Iterable exact
    assert (fig.data[3].name == 'first_mbox' and fig.data[4].name == 'second_mbox')

    # Update using combined lon_at coordinates
    ipywgt_tool.update_fig_objs_xyz(scatts,
            **{'lon_lat': np.arange(12).reshape((2,2,-1))},  # lonxyz iterable
            mode='lines')
    assert (fig.data[3].lon == np.arange(3)).all() \
            and (fig.data[4].lon == np.arange(3) + 6).all()
    assert (fig.data[3].lat == np.arange(3) + 3).all() \
            and (fig.data[4].lat == np.arange(3) + 9).all()

    # Create mapbox density objects
    scatts = ipywgt_tool.add_multiple(fig, ipywgt_tool.add_densitymapbox,
            name=['first_mbox', 'second_mbox'])   # Iterable exact
    assert (fig.data[3].name == 'first_mbox' and fig.data[4].name == 'second_mbox')

    # Update using combined lon_lat_z coordinates
    ipywgt_tool.update_fig_objs_xyz(scatts,
            **{'lon_lat_z': np.arange(18).reshape((2,3,-1))},  # lonxyz iterable
            radius=10)
    assert (fig.data[5].lon == np.arange(3)).all() \
            and (fig.data[6].lon == np.arange(3) + 9).all()
    assert (fig.data[5].lat == np.arange(3) + 3).all() \
            and (fig.data[6].lat == np.arange(3) + 12).all()
    assert (fig.data[5].z == np.arange(3) + 6).all() \
            and (fig.data[6].z == np.arange(3) + 15).all()
    assert fig.data[5].radius == 10 and fig.data[6].radius == 10
