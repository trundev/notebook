"""Utility class/functions to create interactive plotly plots with ipywidgets
"""
import numpy as np
import numpy.typing as npt
from typing import MutableSequence, Callable, Iterable
import plotly.graph_objects


PlotlyObj = MutableSequence     # Generic object to support getitem() and getattr()
PlotlyFigure = plotly.graph_objects.Figure

def _is_iterable(var):
    """Test for iterable that is not string or bytes"""
    return hasattr(var, '__iter__') and not isinstance(var, (str, bytes))

def split_complex(cvals: npt.NDArray[np.complexfloating], aux_vals: npt.NDArray | None=None, /,
        axis: int=-2, roll: int=0, broadcast: bool=False) -> npt.NDArray:
    """Split complex values in two reals, stack a 3-rd auxiliary value"""
    # 'cvals' can be some generic python list
    cvals = np.asarray(cvals)

    if aux_vals is None:
        # Only splitted 'cvals'
        arr_list = cvals.real, cvals.imag
    elif broadcast:
        # Broadcast both arrays
        arr_list = np.broadcast_arrays(cvals.real, cvals.imag, aux_vals)
    else:
        # Both arrays non-broadcasted
        arr_list = cvals.real, cvals.imag, aux_vals

    # Combine final array
    cvals = np.stack(arr_list, axis=axis)

    # Roll the new axis, 'roll=1' puts 'aux_vals' at index 0
    if roll:
        cvals = np.roll(cvals, roll, axis=axis)
    return cvals

def split_xyz(**kwargs):
    """Split single xy/xyz keys into {x, y, z}, lon_lat_z into {lon, lat, z}"""
    # Try the supported keys one by one
    for xyz_key in 'xyz', 'zxy', 'yzx', 'xy', 'yx', 'lon_lat', 'lat_lon', 'lon_lat_z':
        val = kwargs.get(xyz_key)
        if val is not None:
            kwargs = kwargs.copy()
            del kwargs[xyz_key]
            # Create new key from individual 'xyz_key' chars and 'val' axis -2
            if '_' in xyz_key:
                xyz_key = xyz_key.split('_')
            for i, key in enumerate(xyz_key):
                kwargs[key] = val[..., i, :]
            break   # Makes sense to have just one such key
    return kwargs

def flatten_join(arr: npt.NDArray, values: npt.ArrayLike, num_axes=1) -> npt.NDArray:
    """Join first 'num_axes' with a separator value in between"""
    # Flatten the axes to be joined into a single
    arr = arr.reshape((-1,) + arr.shape[num_axes:])
    # Select where to insert the separators, after merging first two axes
    sep_idx = np.arange(1, arr.shape[0]) * arr.shape[1]
    return np.insert(arr.reshape((-1,) + arr.shape[2:]), sep_idx, values, axis=0)

class fig_obj_wrapper:
    """Wrapper for figure objects, like scatter, annotation, shape

    The 'path' specifies the non-persistent hierarchy components.
    This is to workaround the replacement of layout.annotation/scatter lists
    after addition of new elements.
    """
    fobj_base: PlotlyObj
    fobj_path: tuple[str|int, ...]
    fobj_cache: PlotlyObj | None

    def __init__(self, base: object, path: tuple[str|int, ...]):
        # Avoid our __setattr__() override
        self.__dict__['fobj_base'] = base
        self.__dict__['fobj_path'] = path
        self.__dict__['fobj_cache'] = None

    def get_obj(self, *, use_cache: bool = True, make_cache: bool = True) -> PlotlyObj:
        """Obtain actual figure object (could be changed after addition)"""
        if use_cache:
            obj = self.fobj_cache
            if obj is not None:
                return obj

        obj = self.fobj_base
        for n in self.fobj_path:
            obj = obj[n]

        if make_cache:
            # Avoid our __setattr__() override
            self.__dict__['fobj_cache'] = obj
        return obj

    def __getattr__(self, name: str) -> PlotlyObj:
        """Get attribute from the figure object"""
        return getattr(self.get_obj(make_cache=False), name)

    def __setattr__(self, name: str, value: PlotlyObj) -> None:
        """Set attribute to the figure object"""
        setattr(self.get_obj(), name, value)

    def __getitem__(self, item: int) -> PlotlyObj:
        """Get item from the figure object"""
        return self.get_obj(make_cache=False)[item]

    def __setitem__(self, item, value) -> None:
        """Set item to the figure object"""
        self.get_obj()[item] = value

#
# Figure object wrapper creation
#
def add_fig_data(add_func: Callable, **kwargs) -> fig_obj_wrapper:
    """Create a fig.data entry (add_func must create object there)"""
    fig = add_func(**kwargs)
    idx = len(fig.data) - 1
    return fig_obj_wrapper(fig.data[idx], ())

def add_scatter(fig: PlotlyFigure, **kwargs) -> fig_obj_wrapper:
    """Wraps Figure.add_scatter()"""
    return add_fig_data(fig.add_scatter, **kwargs)

def add_scatter3d(fig: PlotlyFigure, **kwargs) -> fig_obj_wrapper:
    """Wraps Figure.add_scatter3d()"""
    return add_fig_data(fig.add_scatter3d, **kwargs)

def add_scattergeo(fig: PlotlyFigure, **kwargs) -> fig_obj_wrapper:
    """Wraps Figure.add_scatter3d()"""
    return add_fig_data(fig.add_scattergeo, **kwargs)

def add_scattermapbox(fig: PlotlyFigure, **kwargs) -> fig_obj_wrapper:
    """Wraps Figure.add_scatter3d()"""
    return add_fig_data(fig.add_scattermapbox, **kwargs)

def add_densitymapbox(fig: PlotlyFigure, **kwargs) -> fig_obj_wrapper:
    """Wraps Figure.add_scatter3d()"""
    return add_fig_data(fig.add_densitymapbox, **kwargs)

def add_annotation(fig: PlotlyFigure, **kwargs) -> fig_obj_wrapper:
    """Wraps Figure.add_annotation()"""
    idx = len(fig.layout.annotations)
    fig.add_annotation(**kwargs)
    return fig_obj_wrapper(fig.layout, ('annotations', idx))

def add_shape(fig: PlotlyFigure, **kwargs) -> fig_obj_wrapper:
    """Wraps Figure.add_shape()"""
    idx = len(fig.layout.shapes)
    fig.add_shape(**kwargs)
    return fig_obj_wrapper(fig.layout, ('shapes', idx))

def add_multiple(fig: PlotlyFigure, add_func: Callable, **kwargs) -> list[fig_obj_wrapper]:
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
        res.append(add_func(fig=fig, **single_kwargs))
    return res

#
# Batch update of figure objects
#
def update_fig_objs(objs: Iterable[fig_obj_wrapper] | fig_obj_wrapper, **kwargs):
    """Update list of figure object(s), using separate values"""
    for key, val in kwargs.items():
        if isinstance(objs, fig_obj_wrapper):
            # Update single object
            objs[key] = val
        else:
            # Update multiple objects
            # The key-value can be multiple or single element
            # - multiple: spread around the objects, None after exhaustion
            # - single: the same for each object
            def val_gen(val):
                if _is_iterable(val):
                    for v in val: yield v
                    val = None  # Exhausted multiple-value
                while True: yield val
            for obj, v in zip(objs, val_gen(val)):
                obj[key] = v

def update_fig_objs_xyz(objs: Iterable[fig_obj_wrapper] | fig_obj_wrapper, **kwargs):
    """Update existing scatter/scatter3d(s), by splitting single xy/xyz keys"""
    return update_fig_objs(objs, **split_xyz(**kwargs))
