# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 22:37:48 2021

@author: PonnuandKuttu
"""


import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from mne.io.pick import  _pick_data_channels,_get_channel_types,_MEG_CH_TYPES_SPLIT
from mne.channels.layout import _find_topomap_coords
from mne.defaults import _EXTRAPOLATE_DEFAULT, _BORDER_DEFAULT
from mne.utils import  _check_option,_check_sphere,_validate_type
from mne.viz.utils import _setup_vmin_vmax   
from mne.defaults import _handle_default
from mne.io.meas_info import Info, _simplify_info
from mne.io.pick import (pick_types, _picks_by_type, pick_info, pick_channels,
                        _picks_to_idx)
import numpy as np
#%%

def Get_topoplot(data, pos):

    res=64
    outlines='head'
    extrapolate=_EXTRAPOLATE_DEFAULT
    sphere=None
    border=_BORDER_DEFAULT
    ch_type='eeg'  
    # data = np.asarray(temp)
    # pos = fake_evoked.info    
    sphere = _check_sphere(None)
    if isinstance(pos, Info):  # infer pos from Info object
        picks = _pick_data_channels(pos, exclude=())  # pick only data channels
        pos = pick_info(pos, picks)
    
        # check if there is only 1 channel type, and n_chans matches the data
        ch_type = _get_channel_types(pos, unique=True)
        info_help = ("Pick Info with e.g. mne.pick_info and "
                     "mne.io.pick.channel_indices_by_type.")
        if len(ch_type) > 1:
            raise ValueError("Multiple channel types in Info structure. " +
                             info_help)
        elif len(pos["chs"]) != data.shape[0]:
            raise ValueError("Number of channels in the Info object (%s) and "
                             "the data array (%s) do not match. "
                             % (len(pos['chs']), data.shape[0]) + info_help)
        else:
            ch_type = ch_type.pop()
    
        if any(type_ in ch_type for type_ in ('planar', 'grad')):
            # deal with grad pairs
            picks = _pair_grad_sensors(pos, topomap_coords=False)
            pos = _find_topomap_coords(pos, picks=picks[::2], sphere=sphere)
            data, _ = _merge_ch_data(data[picks], ch_type, [])
            data = data.reshape(-1)
        else:
            picks = list(range(data.shape[0]))
            pos = _find_topomap_coords(pos, picks=picks, sphere=sphere)
    
    extrapolate = _check_extrapolate(extrapolate, ch_type)
    if data.ndim > 1:
        raise ValueError("Data needs to be array of shape (n_sensors,); got "
                         "shape %s." % str(data.shape))
    
    # Give a helpful error message for common mistakes regarding the position
    # matrix.
    pos_help = ("Electrode positions should be specified as a 2D array with "
                "shape (n_channels, 2). Each row in this matrix contains the "
                "(x, y) position of an electrode.")
    if pos.ndim != 2:
        error = ("{ndim}D array supplied as electrode positions, where a 2D "
                 "array was expected").format(ndim=pos.ndim)
        raise ValueError(error + " " + pos_help)
    elif pos.shape[1] == 3:
        error = ("The supplied electrode positions matrix contains 3 columns. "
                 "Are you trying to specify XYZ coordinates? Perhaps the "
                 "mne.channels.create_eeg_layout function is useful for you.")
        raise ValueError(error + " " + pos_help)
    # No error is raised in case of pos.shape[1] == 4. In this case, it is
    # assumed the position matrix contains both (x, y) and (width, height)
    # values, such as Layout.pos.
    elif pos.shape[1] == 1 or pos.shape[1] > 4:
        raise ValueError(pos_help)
    pos = pos[:, :2]
    
    if len(data) != len(pos):
        raise ValueError("Data and pos need to be of same length. Got data of "
                         "length %s, pos of length %s" % (len(data), len(pos)))
    
    outlines = _make_head_outlines(sphere, pos, outlines, (0., 0.))
    assert isinstance(outlines, dict)
    
    # find mask limits
    extent, Xi, Yi, interp = _setup_interp(
        pos, res, extrapolate, sphere, outlines, border)
    interp.set_values(data)
    Z = interp.set_locations(Xi, Yi)()
    return Z




def _check_extrapolate(extrapolate, ch_type):
    _check_option('extrapolate', extrapolate, ('box', 'local', 'head', 'auto'))
    if extrapolate == 'auto':
        extrapolate = 'local' if ch_type in _MEG_CH_TYPES_SPLIT else 'head'
    return extrapolate


def _make_head_outlines(sphere, pos, outlines, clip_origin):
    """Check or create outlines for topoplot."""
    assert isinstance(sphere, np.ndarray)
    x, y, _, radius = sphere
    del sphere

    if outlines in ('head', 'skirt', None):
        ll = np.linspace(0, 2 * np.pi, 101)
        head_x = np.cos(ll) * radius + x
        head_y = np.sin(ll) * radius + y
        dx = np.exp(np.arccos(np.deg2rad(12)) * 1j)
        dx, dy = dx.real, dx.imag
        nose_x = np.array([-dx, 0, dx]) * radius + x
        nose_y = np.array([dy, 1.15, dy]) * radius + y
        ear_x = np.array([.497, .510, .518, .5299, .5419, .54, .547,
                          .532, .510, .489]) * (radius * 2)
        ear_y = np.array([.0555, .0775, .0783, .0746, .0555, -.0055, -.0932,
                          -.1313, -.1384, -.1199]) * (radius * 2) + y

        if outlines is not None:
            # Define the outline of the head, ears and nose
            outlines_dict = dict(head=(head_x, head_y), nose=(nose_x, nose_y),
                                 ear_left=(ear_x + x, ear_y),
                                 ear_right=(-ear_x + x, ear_y))
        else:
            outlines_dict = dict()

        # Make the figure encompass slightly more than all points
        mask_scale = 1.25 if outlines == 'skirt' else 1.
        # We probably want to ensure it always contains our most
        # extremely positioned channels, so we do:
        mask_scale = max(
            mask_scale, np.linalg.norm(pos, axis=1).max() * 1.01 / radius)
        outlines_dict['mask_pos'] = (mask_scale * head_x, mask_scale * head_y)
        clip_radius = radius * mask_scale
        outlines_dict['clip_radius'] = (clip_radius,) * 2
        outlines_dict['clip_origin'] = clip_origin
        outlines = outlines_dict

    elif isinstance(outlines, dict):
        if 'mask_pos' not in outlines:
            raise ValueError('You must specify the coordinates of the image '
                             'mask.')
    else:
        raise ValueError('Invalid value for `outlines`.')

    return outlines




def _prepare_topomap(pos, ax, check_nonzero=True):
    """Prepare the topomap axis and check positions.

    Hides axis frame and check that position information is present.
    """
    _hide_frame(ax)
    if check_nonzero and not pos.any():
        raise RuntimeError('No position information found, cannot compute '
                           'geometries for topomap.')
        
        
        
        
        
 
def _hide_frame(ax):
    """Hide axis frame for topomaps."""
    ax.get_yticks()
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_frame_on(False)
       
        
        
        
        

def _setup_interp(pos, res, extrapolate, sphere, outlines, border):
 
    xlim = np.inf, -np.inf,
    ylim = np.inf, -np.inf,
    mask_ = np.c_[outlines['mask_pos']]
    clip_radius = outlines['clip_radius']
    clip_origin = outlines.get('clip_origin', (0., 0.))
    xmin, xmax = (np.min(np.r_[xlim[0],
                               mask_[:, 0],
                               clip_origin[0] - clip_radius[0]]),
                  np.max(np.r_[xlim[1],
                               mask_[:, 0],
                               clip_origin[0] + clip_radius[0]]))
    ymin, ymax = (np.min(np.r_[ylim[0],
                               mask_[:, 1],
                               clip_origin[1] - clip_radius[1]]),
                  np.max(np.r_[ylim[1],
                               mask_[:, 1],
                               clip_origin[1] + clip_radius[1]]))
    xi = np.linspace(xmin, xmax, res)
    yi = np.linspace(ymin, ymax, res)
    Xi, Yi = np.meshgrid(xi, yi)
    interp = _GridData(pos, extrapolate, clip_origin, clip_radius, border)
    extent = (xmin, xmax, ymin, ymax)
    return extent, Xi, Yi, interp
        
        
        

class _GridData(object):
    """Unstructured (x,y) data interpolator.

    This class allows optimized interpolation by computing parameters
    for a fixed set of true points, and allowing the values at those points
    to be set independently.
    """

    def __init__(self, pos, extrapolate, origin, radii, border):
        # in principle this works in N dimensions, not just 2
        assert pos.ndim == 2 and pos.shape[1] == 2, pos.shape
        _validate_type(border, ('numeric', str), 'border')

        # check that border, if string, is correct
        if isinstance(border, str):
            _check_option('border', border, ('mean',), extra='when a string')

        # Adding points outside the extremes helps the interpolators
        outer_pts, mask_pts, tri = _get_extra_points(
            pos, extrapolate, origin, radii)
        self.n_extra = outer_pts.shape[0]
        self.mask_pts = mask_pts
        self.border = border
        self.tri = tri

    def set_values(self, v):
        """Set the values at interpolation points."""
        # Rbf with thin-plate is what we used to use, but it's slower and
        # looks about the same:
        #
        #     zi = Rbf(x, y, v, function='multiquadric', smooth=0)(xi, yi)
        #
        # Eventually we could also do set_values with this class if we want,
        # see scipy/interpolate/rbf.py, especially the self.nodes one-liner.
        from scipy.interpolate import CloughTocher2DInterpolator

        if isinstance(self.border, str):
            # we've already checked that border = 'mean'
            n_points = v.shape[0]
            v_extra = np.zeros(self.n_extra)
            indices, indptr = self.tri.vertex_neighbor_vertices
            rng = range(n_points, n_points + self.n_extra)
            used = np.zeros(len(rng), bool)
            for idx, extra_idx in enumerate(rng):
                ngb = indptr[indices[extra_idx]:indices[extra_idx + 1]]
                ngb = ngb[ngb < n_points]
                if len(ngb) > 0:
                    used[idx] = True
                    v_extra[idx] = v[ngb].mean()
            if not used.all() and used.any():
                # Eventually we might want to use the value of the nearest
                # point or something, but this case should hopefully be
                # rare so for now just use the average value of all extras
                v_extra[~used] = np.mean(v_extra[used])
        else:
            v_extra = np.full(self.n_extra, self.border, dtype=float)

        v = np.concatenate((v, v_extra))
        self.interpolator = CloughTocher2DInterpolator(self.tri, v)
        return self

    def set_locations(self, Xi, Yi):
        """Set locations for easier (delayed) calling."""
        self.Xi = Xi
        self.Yi = Yi
        return self

    def __call__(self, *args):
        """Evaluate the interpolator."""
        if len(args) == 0:
            args = [self.Xi, self.Yi]
        return self.interpolator(*args)



def _get_extra_points(pos, extrapolate, origin, radii):
    """Get coordinates of additinal interpolation points."""
    from scipy.spatial.qhull import Delaunay
    radii = np.array(radii, float)
    assert radii.shape == (2,)
    x, y = origin
    # auto should be gone by now
    _check_option('extrapolate', extrapolate, ('head', 'box', 'local'))

    # the old method of placement - large box
    mask_pos = None
    if extrapolate == 'box':
        extremes = np.array([pos.min(axis=0), pos.max(axis=0)])
        diffs = extremes[1] - extremes[0]
        extremes[0] -= diffs
        extremes[1] += diffs
        eidx = np.array(list(itertools.product(
            *([[0] * (pos.shape[1] - 1) + [1]] * pos.shape[1]))))
        pidx = np.tile(np.arange(pos.shape[1])[np.newaxis], (len(eidx), 1))
        outer_pts = extremes[eidx, pidx]
        return outer_pts, mask_pos, Delaunay(np.concatenate((pos, outer_pts)))

    # check if positions are colinear:
    diffs = np.diff(pos, axis=0)
    with np.errstate(divide='ignore'):
        slopes = diffs[:, 1] / diffs[:, 0]
    colinear = ((slopes == slopes[0]).all() or np.isinf(slopes).all())

    # compute median inter-electrode distance
    if colinear or pos.shape[0] < 4:
        dim = 1 if diffs[:, 1].sum() > diffs[:, 0].sum() else 0
        sorting = np.argsort(pos[:, dim])
        pos_sorted = pos[sorting, :]
        diffs = np.diff(pos_sorted, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        distance = np.median(distances)
    else:
        tri = Delaunay(pos, incremental=True)
        idx1, idx2, idx3 = tri.simplices.T
        distances = np.concatenate(
            [np.linalg.norm(pos[i1, :] - pos[i2, :], axis=1)
             for i1, i2 in zip([idx1, idx2], [idx2, idx3])])
        distance = np.median(distances)

    if extrapolate == 'local':
        if colinear or pos.shape[0] < 4:
            # special case for colinear points and when there is too
            # little points for Delaunay (needs at least 3)
            edge_points = sorting[[0, -1]]
            line_len = np.diff(pos[edge_points, :], axis=0)
            unit_vec = line_len / np.linalg.norm(line_len) * distance
            unit_vec_par = unit_vec[:, ::-1] * [[-1, 1]]

            edge_pos = (pos[edge_points, :] +
                        np.concatenate([-unit_vec, unit_vec], axis=0))
            new_pos = np.concatenate([pos + unit_vec_par,
                                      pos - unit_vec_par, edge_pos], axis=0)

            if pos.shape[0] == 3:
                # there may be some new_pos points that are too close
                # to the original points
                new_pos_diff = pos[..., np.newaxis] - new_pos.T[np.newaxis, :]
                new_pos_diff = np.linalg.norm(new_pos_diff, axis=1)
                good_extra = (new_pos_diff > 0.5 * distance).all(axis=0)
                new_pos = new_pos[good_extra]

            tri = Delaunay(np.concatenate([pos, new_pos], axis=0))
            return new_pos, new_pos, tri

        # get the convex hull of data points from triangulation
        hull_pos = pos[tri.convex_hull]

        # extend the convex hull limits outwards a bit
        channels_center = pos.mean(axis=0)
        radial_dir = hull_pos - channels_center
        unit_radial_dir = radial_dir / np.linalg.norm(radial_dir, axis=-1,
                                                      keepdims=True)
        hull_extended = hull_pos + unit_radial_dir * distance
        mask_pos = hull_pos + unit_radial_dir * distance * 0.5
        hull_diff = np.diff(hull_pos, axis=1)[:, 0]
        hull_distances = np.linalg.norm(hull_diff, axis=-1)
        del channels_center

        # Construct a mask
        mask_pos = np.unique(mask_pos.reshape(-1, 2), axis=0)
        mask_center = np.mean(mask_pos, axis=0)
        mask_pos -= mask_center
        mask_pos = mask_pos[
            np.argsort(np.arctan2(mask_pos[:, 1], mask_pos[:, 0]))]
        mask_pos += mask_center

        # add points along hull edges so that the distance between points
        # is around that of average distance between channels
        add_points = list()
        eps = np.finfo('float').eps
        n_times_dist = np.round(0.25 * hull_distances / distance).astype('int')
        for n in range(2, n_times_dist.max() + 1):
            mask = n_times_dist == n
            mult = np.arange(1 / n, 1 - eps, 1 / n)[:, np.newaxis, np.newaxis]
            steps = hull_diff[mask][np.newaxis, ...] * mult
            add_points.append((hull_extended[mask, 0][np.newaxis, ...] +
                               steps).reshape((-1, 2)))

        # remove duplicates from hull_extended
        hull_extended = np.unique(hull_extended.reshape((-1, 2)), axis=0)
        new_pos = np.concatenate([hull_extended] + add_points)
    else:
        assert extrapolate == 'head'
        # return points on the head circle
        angle = np.arcsin(distance / np.mean(radii))
        n_pnts = max(12, int(np.round(2 * np.pi / angle)))
        points_l = np.linspace(0, 2 * np.pi, n_pnts, endpoint=False)
        use_radii = radii * 1.1 + distance
        points_x = np.cos(points_l) * use_radii[0] + x
        points_y = np.sin(points_l) * use_radii[1] + y
        new_pos = np.stack([points_x, points_y], axis=1)
        if colinear or pos.shape[0] == 3:
            tri = Delaunay(np.concatenate([pos, new_pos], axis=0))
            return new_pos, mask_pos, tri
    tri.add_points(new_pos)
    return new_pos, mask_pos, tri


