"""Utility class/functions to create interactive plotly plots with ipywidgets
"""
import numpy as np

def _is_iterable(var):
    """Test for iterable that is not string or bytes"""
    return hasattr(var, '__iter__') and not isinstance(var, (str, bytes))

def split_complex(cvals: np.array, aux_vals: np.array or None=None, /,
        axis: int=-2, roll: int=0, broadcast: bool=False) -> np.array:
    """Split complex values in two reals, stack a 3-rd auxiliary value"""
    # cvals can be some generic python list
    cvals = np.asarray(cvals)

    if aux_vals is None:
        # Only splitted cvals
        arr_list = cvals.real, cvals.imag
    elif broadcast:
        # Broadcast both arrays
        arr_list = np.broadcast_arrays(cvals.real, cvals.imag, aux_vals)
    else:
        # Both arrays non-broadcased
        arr_list = cvals.real, cvals.imag, aux_vals

    # Combine final array
    cvals = np.stack(arr_list, axis=axis)

    # Roll the new axis, 'roll=1' puts 'aux_vals' at index 0
    if roll:
        cvals = np.roll(cvals, roll, axis=axis)
    return cvals

def flatten_join(arr: np.array, values: np.array, num_axes=1):
    """Join first 'num_axes' with a separator value in between"""
    # Flatten the axes to be joined into a single
    arr = arr.reshape((-1,) + arr.shape[num_axes:])
    # Select where to insert the separators, after merging first two axes
    sep_idx = np.arange(1, arr.shape[0]) * arr.shape[1]
    return np.insert(arr.reshape((-1,) + arr.shape[2:]), sep_idx, values, axis=0)

class plot_data:
    """Interactive plot context"""
    def __init__(self, fig: object):
        self.fig = fig

    def add_data(self, add_func: callable, **kwargs) -> int:
        """Create a fig.data entry and return its index"""
        idx = len(self.fig.data)
        add_func(**kwargs)
        return idx

    def add_scatter(self, **kwargs) -> int:
        """Wraps Figure.add_scatter()"""
        return self.add_data(self.fig.add_scatter, **kwargs)

    def add_scatter3d(self, **kwargs) -> int:
        """Wraps Figure.add_scatter3d()"""
        return self.add_data(self.fig.add_scatter3d, **kwargs)

    def add_annotation(self, **kwargs) -> int:
        """Wraps Figure.add_annotation()"""
        idx = len(self.fig.layout.annotations)
        self.fig.add_annotation(**kwargs)
        return idx

    def add_shape(self, **kwargs) -> int:
        """Wraps Figure.add_shape()"""
        idx = len(self.fig.layout.shapes)
        self.fig.add_shape(**kwargs)
        return idx

    @staticmethod
    def add_multiple(add_func: callable, **kwargs):
        """Add multiple figure objects, the number is the max from iterable-value sizes"""
        # Select the number of objects to create
        num = 0
        for val in kwargs.values():
            if _is_iterable(val):
                num = max(num, len(val))

        # Create 'num' object using spread around key-values
        # See _update_fig_obj_list()
        res = []
        for idx in range(num):
            single_kwargs = {}
            for key, val in kwargs.items():
                if _is_iterable(val):
                    single_kwargs[key] = val[idx] if len(val) > idx else None
                else:
                    single_kwargs[key] = val
            res.append(add_func(**single_kwargs))
        return res

    @staticmethod
    def _update_fig_obj(obj, **kwargs):
        """Update a figure object object"""
        for key, val in kwargs.items():
            obj[key] = val

    @staticmethod
    def _update_fig_obj_list(where: tuple[object], obj_idxs: list[int], **kwargs):
        """Update list of figure objects, using separate values"""
        objs = [where[idx] for idx in obj_idxs]
        for key, val in kwargs.items():
            # The key-value can be multiple or single element
            # - multiple: spread around the objects, None after exhaustion
            # - single: the same for each object
            atom_val = None if _is_iterable(val) else val
            for idx, obj in enumerate(objs):
                obj[key] = atom_val or val[idx] if len(val) > idx else None

    def _gen_update(self, where: tuple[object], data_idx: int or list[int], **kwargs):
        """Generic figure object(s) update, like data, annotations or shapes"""
        if _is_iterable(data_idx):
            self._update_fig_obj_list(where, data_idx, **kwargs)
        else:
            self._update_fig_obj(where[data_idx], **kwargs)

    def update_data(self, data_idx: int or list[int], **kwargs):
        """Update existing scatter/scatter3d(s)"""
        return self._gen_update(self.fig.data, data_idx, **kwargs)

    def update_annotation(self, annnot_idx: int or list[int], **kwargs):
        """Update existing annotation(s)"""
        return self._gen_update(self.fig.layout.annotations, annnot_idx, **kwargs)

    def update_shape(self, shape_idx: int or list[int], **kwargs):
        """Update existing shape(s)"""
        return self._gen_update(self.fig.layout.shapes, shape_idx, **kwargs)

    def update_data_xyz(self, data_idx: int or list[int], **kwargs):
        """Update existing scatter/scatter3d(s), by splitting single xy/xyz keys"""
        # Try the suported keys one by one
        for xyz_key in 'xyz', 'zxy', 'yzx', 'xy', 'yx':
            val = kwargs.get(xyz_key)
            if val is not None:
                kwargs = kwargs.copy()
                del kwargs[xyz_key]
                # Create new key from individual 'xyz_key' chars and 'val' axis -2
                for i, key in enumerate(xyz_key):
                    kwargs[key] = val[..., i, :]
                break   # Makes sense to have just one such key
        return self.update_data(data_idx, **kwargs)

#
# Confirm the plotly behaviour assumptions
#
def test_obj_persistance():
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
    obj0 = fig.layout.annotations[0]
    fig.add_annotation()
    #assert obj0 is fig.layout.annotations[0], 'fig.layout.annotations changed after add_annotation()'
    if obj0 is fig.layout.annotations[0]:
        print('Warning: fig.layout.annotations NOT changed after add_annotation()')

    # Check persistence of fig.layout.shapes entries (shapes)
    assert len(fig.layout.shapes) == 0, 'Test expects empty shapes'
    fig.add_shape()
    obj0 = fig.layout.shapes[0]
    fig.add_shape()
    #assert obj0 is fig.layout.shapes[0], 'fig.layout.shapes changed after add_shape()'
    if obj0 is fig.layout.shapes[0]:
        print('Warning: fig.layout.shapes NOT changed after add_shape()')

#
# utility function  tests
#
def test_utils():
    """Check split_complex() utility"""
    cvals = np.exp(np.arange(15) * .5j).reshape(3, -1)
    REAL_IMAG = np.array([1,1j])[:, np.newaxis]

    res = split_complex(cvals)
    assert res.shape == (3, 2, 5), 'Split complex array shape'
    assert ((res * REAL_IMAG).sum(-2) == cvals).all()

    res = split_complex(cvals, np.arange(15).reshape(3, -1))
    assert res.shape == (3, 3, 5), 'Split complex with aux array shape'

    res = split_complex(cvals, np.arange(5), broadcast=True)
    assert res.shape == (3, 3, 5), 'Split complex with broadcasted aux array'
    res = split_complex(cvals[0], np.arange(15).reshape(3, -1), broadcast=True)
    assert res.shape == (3, 3, 5), 'Split broadcasted complex with aux array'
    assert ((res[..., :2, :] * REAL_IMAG).sum(-2) == cvals[0]).all()

    # Test flatten_join()
    res = flatten_join(np.arange(12.).reshape(3, -1), np.nan)
    assert res.shape == (12 + 3-1,), 'Flatten by join array'
    res = flatten_join(np.arange(2*3*4.).reshape(2, 3, -1), np.nan)
    assert res.shape == (2*3 + 2-1, 4) and np.isnan(res[3]).all(), 'Flatten by join array'
    res = flatten_join(np.arange(2*3*4.).reshape(2, 3, -1), np.nan, num_axes=2)
    assert res.shape == (2*3*4 + 2*3-1,) and np.isnan(res[4::5]).all(), 'Flatten by join array'

#
# plot_data class tests
#
def test_plot_data_scatters():
    """Check scatters handling by plot_data class"""
    fig = go.Figure()
    data = plot_data(fig)

    # Create data objects
    scatt_1 = data.add_scatter(mode='lines', line_dash='dash', name='first')
    scatt_2 = data.add_scatter(mode='lines+markers', line_dash='dot', name='second')
    assert fig.data[0].mode == 'lines' and fig.data[1].name == 'second'

    # Update data objects
    data.update_data(scatt_1, x=(1,2,3), y=(10,20,30))
    assert fig.data[0].x == (1,2,3) and fig.data[0].y == (10,20,30)
    # Use iterables and numpy arrays
    data.update_data(scatt_2, x=np.arange(10), y=np.arange(10) + 100)
    assert (fig.data[1].y == np.arange(10) + 100).all(), 'Non-iterated update'
    data.update_data([scatt_1, scatt_2],
            x=np.arange(10).reshape((2,-1)),    # Iterable exact
            y=np.arange(10).reshape((1,-1)),    # Iterable exhaustible
            mode='markers')                     # Single
    assert (fig.data[0].x == np.arange(5)).all(), 'Iterables must spread around'
    assert (fig.data[0].y == np.arange(10)).all()
    assert fig.data[0].mode == 'markers' and fig.data[1].mode == 'markers', 'Non-iterables must be reused'
    assert fig.data[1].y is None, 'Exhausted iterable must assigns None'

def test_plot_data_layout():
    """Check annotations and shapes handling by plot_data class"""
    fig = go.Figure()
    data = plot_data(fig)

    # Create some objects
    annot_1 = data.add_annotation(name='first')
    shape_1 = data.add_shape(name='shape_1')
    shape_2 = data.add_shape(name='shape_2')
    annot_2 = data.add_annotation(name='second')
    assert fig.layout.annotations[0].name == 'first' and fig.layout.annotations[1].name == 'second'

    # Update data objects
    data.update_annotation(annot_1, name='First', x=5)
    assert fig.layout.annotations[0].name == 'First' and fig.layout.annotations[0].x == 5
    data.update_shape(shape_2, name='Shape_2', y0=10)
    assert fig.layout.shapes[1].name == 'Shape_2' and fig.layout.shapes[1].y0 == 10
    # Use iterables and range()
    data.update_shape([shape_2, shape_1],
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
    data = plot_data(fig)

    # Create multiple objects
    scatts = data.add_multiple(data.add_scatter3d,
            x=[[0], [1], [3]],                  # Iterable exact (each x must be iterable)
            line_dash='dash',                   # Single
            mode=['lines', 'markers'],          # Iterable exhaustible
            name=['first', 'second', 'third'])  # Iterable exact
    assert fig.data[0].x == (0,) and fig.data[1].x == (1,) and fig.data[2].x == (3,)
    assert fig.data[0].mode == 'lines' and fig.data[1].mode == 'markers' and fig.data[2].mode == None
    assert fig.data[0].name == 'first' and fig.data[1].name == 'second' and fig.data[2].name == 'third'

    # Update using combined numpy xyz coordinates
    data.update_data_xyz(scatts,
            xyz=np.arange(24).reshape((2,3,-1)),  # xyz iterable
            mode='lines')
    assert (fig.data[0].x == np.arange(0, 4)).all() \
            and (fig.data[1].x == np.arange(12, 16)).all() \
            and fig.data[2].x is None
    assert (fig.data[0].z == np.arange(8, 12)).all() \
            and (fig.data[1].z == np.arange(20, 24)).all() \
            and fig.data[2].z is None

#
# Test scenarios
#
if __name__ == '__main__':
    import plotly.graph_objects as go

    # Check plotly data, annotations, shapes
    test_obj_persistance()

    #
    # utility functions tests
    #
    test_utils()

    #
    # plot_data tests
    #
    test_plot_data_scatters()
    test_plot_data_layout()
    test_multiple()

    print('Done')
