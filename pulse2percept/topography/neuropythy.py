import numpy as np
import os
from scipy.spatial import cKDTree

from .cortex import CorticalMap

class NeuropythyMap(CorticalMap):

    split_map = False

    def __init__(self, subject, cache_dir=None, **params):
        """
        Uses the visual field maps from the neuropythy package.
        Requires the neuropythy package to be installed (`pip install neuropythy`)

        Parameters
        ----------
        subject : str, path, or :py.class:`neuropythy.mri.core.Subject`
            The subject specific mapping to use. If it is not a loaded
            neuropythy subject object, it will be passed to 
            ny.freesurfer_subject(subject). Neuropythy accepts either a path
            to a freesurfer subject directory, or a string identifying a 
            subject in the benson_winawer_2018 dataset. To get an average
            subject, use 'fsaverage'. 

            The subjects will be downloaded if necessary, which can take a long
            time. You can append to ny.config['freesurfer_subject_paths'] 
            if you have already downloaded the subjects.
            If not set, the subjects will be cached to cache_dir
        cache_dir : str
            The directory to cache the subjects in. If not set, it will be
            ~/.neuropythy_p2p
        regions : list of str
            The regions to include in the map ('v1', 'v2', and/or 'v3'). 
            The default is ['v1'].
        jitter_boundary : bool
            If True, slightly move points at discontinuities. Default is False.
            Note that with neuropythy, there will always be some discontinuities that 
            remain, depending on the subject.
        """
        try:
            import neuropythy as ny
        except ImportError:
            raise ImportError("NeuropythyMap requires the neuropythy package  (`pip install neuropythy`).")
        super().__init__(**params)
        self.cache_dir = os.path.expanduser(os.path.join('~', '.neuropythy_p2p')) if cache_dir is None else os.path.expanduser(cache_dir)
        self.subject = self.parse_subject(subject)
        self.region_meshes = self.load_meshes(self.subject)

    
    def get_default_params(self):
        params = {
            'ndim' : 3,
            'regions' : ['v1'],
            # slightly move points at discontinuities
            'jitter_boundary' : False,
            # jitter points within this close of boundary
            'jitter_thresh' : 0.3,
            # no split map
            'left_offset' : None,
            # max nearest neighbor distance for v*_to_dva
            'cort_nn_thresh' : 1000, # 1mm
        }
        return {**super().get_default_params(), **params}

    def parse_subject(self, subject):
        import neuropythy as ny
        benson_winawer_subjs = ['fsaverage', 'S1201', 'S1202', 'S1203', 'S1204', 
                                'S1205', 'S1206', 'S1207', 'S1208']

        if isinstance(subject, ny.mri.core.Subject):
            return subject
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        if not os.path.exists(os.path.join(self.cache_dir, 'benson_winawer_2018')):
            os.makedirs(os.path.join(self.cache_dir, 'benson_winawer_2018'))
        if ny.config['benson_winawer_2018_path'] is None:
            ny.config['benson_winawer_2018_path'] = os.path.join(self.cache_dir, 'benson_winawer_2018')
        if os.path.join(self.cache_dir, 
                        'benson_winawer_2018', 
                        'freesurfer_subjects') not in ny.config['freesurfer_subject_paths']:
            ny.config['freesurfer_subject_paths'].append(os.path.join(self.cache_dir, 
                                                                      'benson_winawer_2018',
                                                                       'freesurfer_subjects'))
        try:
            return ny.freesurfer_subject(subject)
        except ValueError as ve:
            # For some reason, neuropythy won't download the dataset if the subject 
            # id isn't fsaverage. Download it manually in this case
            if subject in benson_winawer_subjs or subject.upper() in benson_winawer_subjs:
                # force the download (will go to cache_dir)
                _ = ny.data['benson_winawer_2018'].subjects[subject]
                return ny.freesurfer_subject(subject)
            else:
                raise ve
        
    def load_meshes(self, subject):
        """
        Predicts retinotopy and loads submeshes for the given subject.
        Adapted from code courtesy of Noah Benson
        """
        import neuropythy as ny
        left, right = ny.vision.predict_retinotopy(subject, sym_angle=False)

        self.predicted_retinotopy = (left, right)
        cortex_pts = [] # (npoints, 3)
        addr_idxs = [] # (npoints, 2)
        region_idxs = [] # (npoints, 2)
        hemi_idxs = [] # (npoints, 2)
        region_meshes = {}
        for region in self.regions:
            region_lbl = int(region[-1])
            vfmeshes = []
            for (hemi, retinotopy) in zip(['lh','rh'], self.predicted_retinotopy):
                ii = (retinotopy['varea'] == region_lbl)
                (ang, ecc) = (retinotopy['angle'], retinotopy['eccen'])
                ang = np.pi/180 * (90 - ang)
                (x, y) = (ecc*np.cos(ang), ecc*np.sin(ang))
                # doesn't matter what surface is used here for dva_to_v*
                # but grab all the points for a ckdtree for v*_to_dva
                for surface in ['midgray', 'white', 'pial']:
                    submesh = subject.hemis[hemi].surface(surface).submesh(ii)
                    cortex_pts.append(submesh.coordinates.T.astype('float32'))
                    addr_idxs.append(np.arange(submesh.coordinates.shape[1]))
                    region_idxs.append([region for _ in range(submesh.coordinates.shape[1])])
                    hemi_idxs.append([1 if hemi=='rh' else 0 for _ in range(submesh.coordinates.shape[1])])
                ii = submesh.labels
                submesh = submesh.copy(coordinates=[x[ii], y[ii]])
                vfmeshes.append(submesh)
            region_meshes[region] = tuple(vfmeshes)
        
        self.addr_idxs = {'addr' : np.concatenate(addr_idxs), 
                     'region' : np.concatenate(region_idxs),
                     'hemi' : np.concatenate(hemi_idxs)}
        cortex_pts = np.concatenate(cortex_pts)
        self.cortex_tree = cKDTree(cortex_pts)

        return region_meshes
            

    def dva_to_cortex(self, x, y, region='v1', hemi=None, surface='midgray'):
        """Gives the cortex position(s) of the visual field point(s) `(x,y)`.

        Parameters
        ----------
        x, y : float or array_like
            The x and y-coordinate(s) of the visual field point(s) to look up (in dva).
        region : str
            The visual field map to look up the point(s) in. Valid options are 
            'v1', 'v2', and 'v3'. Default is 'v1'.
        hemi : str
            The hemisphere to look up the point(s) in. Valid options are 'lh' and 'rh'.
        surface : str
            The surface to look up the point(s) on. Default is 'midgray'. Other 
            common options include 'pial' and 'white'.
        
        Returns
        -------
        cortex_pts : array_like
            cortical addresses of the visual field points (cortical addresses 
            provide the face containing a point and the barycentric coordinates 
            of the point within that face).

        Adapted from code courtesy of Noah Benson
        """
        if x is None or y is None or x.size==0 or y.size==0:
            return np.array([]), np.array([]), np.array([])
        import neuropythy as ny
        if hemi is None:
            raise ValueError("cannot deduce hemisphere")
        elif hemi == 'lh': h = 0
        elif hemi == 'rh': h = 1
        else: raise ValueError(f"invalid hemisphere: {hemi}")
        if region not in self.region_meshes.keys():
            raise ValueError(f"invalid region: {region} for self.regions={self.regions}")
        meshes = self.region_meshes[region]
        # Look up the addresses of the points in the visual mesh.
        addr = meshes[h].address([x,y])
        if surface is None:
            return addr
        else:
            surf = self.subject.hemis[hemi].surface(surface)
            # Filter out nans so that unaddress doesn't raise an error.
            (faces, bccoords) = ny.address_data(addr, strict=False)
            iinan = ~np.isfinite(bccoords[0])
            bccoords[:,iinan] = 0
            # Convert addresses to surface points.
            surf_pts = surf.unaddress(addr)
            # Fix the nans and return.
            surf_pts[:, iinan] = np.nan
            return np.array(surf_pts * 1000, dtype='float32')
        

    def dva_to_v1(self, x, y, surface='midgray'):
        """
        Gives the 3D cortex position(s) of the visual field point(s) `(x,y)` in v1.

        Parameters
        ----------
        x, y : float or array_like
            The x and y-coordinate(s) of the visual field point(s) to look up (in dva).
        surface : str
            The surface to look up the point(s) on. Default is 'midgray'. Other 
            common options include 'pial' and 'white'.
        """
        x = np.array(x, dtype='float32')
        y = np.array(y, dtype='float32')
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")
        shape = x.shape
        x = x.flatten()
        y = y.flatten()

        if self.jitter_boundary:
            # remove and discontinuities across x axis
            # shift to the same side as existing points
            # this won't get all of them, and really we should be using a threshold based on
            # angle, not x and y coords. But this will be decent for near the fovea
            x[np.isclose(x, 0, rtol=0, atol=self.jitter_thresh)] = np.copysign(self.jitter_thresh, np.mean(x)) 
        ret = np.zeros((3, x.size), dtype='float32')
        idx = x < 0
        # l and r are (3, npoints)
        r = self.dva_to_cortex(x[idx], y[idx], region='v1', hemi='rh', surface=surface)
        l = self.dva_to_cortex(x[~idx], y[~idx], region='v1', hemi='lh', surface=surface)
        ret[:, idx] = r
        ret[:, ~idx] = l
        ret = ret.reshape((3, *shape))
        return ret[0], ret[1], ret[2]
    

    def dva_to_v2(self, x, y, surface='midgray'):
        """
        Gives the 3D cortex position(s) of the visual field point(s) `(x,y)` in v2.

        Parameters
        ----------
        x, y : float or array_like
            The x and y-coordinate(s) of the visual field point(s) to look up (in dva).
        surface : str
            The surface to look up the point(s) on. Default is 'midgray'. Other 
            common options include 'pial' and 'white'.
        """
        x = np.array(x, dtype='float32')
        y = np.array(y, dtype='float32')
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")
        shape = x.shape
        x = x.flatten()
        y = y.flatten()

        if self.jitter_boundary:
            # remove and discontinuities across x axis
            # shift to the same side as existing points
            x[np.isclose(x, 0, rtol=0, atol=self.jitter_thresh)] = np.copysign(self.jitter_thresh, np.mean(x)) 
            y[np.isclose(y, 0, rtol=0, atol=self.jitter_thresh)] = np.copysign(self.jitter_thresh, np.mean(y)) 
        ret = np.zeros((3, x.size), dtype='float32')
        idx = x < 0
        # l and r are (3, npoints)
        r = self.dva_to_cortex(x[idx], y[idx], region='v2', hemi='rh', surface=surface)
        l = self.dva_to_cortex(x[~idx], y[~idx], region='v2', hemi='lh', surface=surface)
        ret[:, idx] = r
        ret[:, ~idx] = l
        ret = ret.reshape((3, *shape))
        return ret[0], ret[1], ret[2]
    
    def dva_to_v3(self, x, y, surface='midgray'):
        """
        Gives the 3D cortex position(s) of the visual field point(s) `(x,y)` in v3.

        Parameters
        ----------
        x, y : float or array_like
            The x and y-coordinate(s) of the visual field point(s) to look up (in dva).
        surface : str
            The surface to look up the point(s) on. Default is 'midgray'. Other 
            common options include 'pial' and 'white'.
        """
        x = np.array(x, dtype='float32')
        y = np.array(y, dtype='float32')
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")
        shape = x.shape
        x = x.flatten()
        y = y.flatten()

        if self.jitter_boundary:
            # remove and discontinuities across x axis
            # shift to the same side as existing points
            x[np.isclose(x, 0, rtol=0, atol=self.jitter_thresh)] = np.copysign(self.jitter_thresh, np.mean(x)) 
            y[np.isclose(y, 0, rtol=0, atol=self.jitter_thresh)] = np.copysign(self.jitter_thresh, np.mean(y)) 
        ret = np.zeros((3, x.size), dtype='float32')
        idx = x < 0
        # l and r are (3, npoints)
        r = self.dva_to_cortex(x[idx], y[idx], region='v3', hemi='rh', surface=surface)
        l = self.dva_to_cortex(x[~idx], y[~idx], region='v3', hemi='lh', surface=surface)
        ret[:, idx] = r
        ret[:, ~idx] = l
        ret = ret.reshape((3, *shape))
        return ret[0], ret[1], ret[2]
    

    def cortex_to_dva(self, xc, yc, zc):
        """
        Gives the visual field position(s) of the cortex point(s) `(xc,yc,zc)`.

        Parameters
        ----------
        xc, yc, zc : float or array_like
            The x, y, and z-coordinate(s) of the cortex point(s) to look up (in mm).
        
        Returns
        -------
        x, y : array_like
            The x and y-coordinate(s) of the visual field point(s) (in dva).
        """
        xc = np.array(xc, dtype='float32')
        yc = np.array(yc, dtype='float32')
        zc = np.array(zc, dtype='float32')
        if np.shape(xc) != np.shape(yc) or np.shape(xc) != np.shape(zc):
            raise ValueError("x, y, and z must have the same shape")
        id_nan = np.isnan(xc) | np.isnan(yc) | np.isnan(zc)
        query = np.stack([np.ravel(xc[~id_nan]), np.ravel(yc[~id_nan]), np.ravel(zc[~id_nan])], axis=-1) / 1000 # convert to mm
        if np.size(query) == 0:
            return np.ones((*np.shape(xc), 2)) * np.nan
        dist, idx = self.cortex_tree.query(query, k=5)#, distance_upper_bound=self.cort_nn_thresh / 1000)
        idx_nan = np.all(dist > self.cort_nn_thresh / 1000, axis=-1)
        dist[dist > self.cort_nn_thresh / 1000] = 999999999 # make high so it doesn't contribute to geometric mean
        neighbors = np.array([[self.region_meshes[self.addr_idxs['region'][i]][self.addr_idxs['hemi'][i]].coordinates[:, self.addr_idxs['addr'][i]] 
                               for i in nb_pts] 
                              for nb_pts in idx])
        # use geometric mean based on distance
        pts = np.sum(neighbors * 1/dist[..., None], axis=1) / np.sum(1/dist, axis=1)[:, None]
        pts[idx_nan] = [np.nan, np.nan]
        out = np.ones((*np.shape(xc), 2)) * np.nan
        out[~id_nan] = pts
        return pts[:, 0], pts[:, 1]
    

    def v1_to_dva(self, xv1, yv1, zv1):
        """
        Convert points in v1 to dva. Uses the mean of the 5 nearest neighbors
        in the cortical mesh, weighted by 1/distance, to interpolate dva. 
        Any points that are more than self.cort_nn_thresh um from the
        nearest neighbor will be set to nan.

        Parameters
        ----------
        xv1, yv1, zv1 : float or array_like
            The x, y, and z-coordinate(s) of the v1 point(s) to look up (in mm).

        Returns
        -------
        x, y : array_like
            The x and y-coordinate(s) of the visual field point(s) (in dva).
        """
        return self.cortex_to_dva(xv1, yv1, zv1)
    
    def v2_to_dva(self, xv2, yv2, zv2):
        """
        Convert points in v2 to dva. Uses the mean of the 5 nearest neighbors
        in the cortical mesh, weighted by 1/distance, to interpolate dva.
        Any points that are more than self.cort_nn_thresh um from the
        nearest neighbor will be set to nan.

        Parameters
        ----------
        xv2, yv2, zv2 : float or array_like
            The x, y, and z-coordinate(s) of the v2 point(s) to look up (in mm).

        Returns
        -------
        x, y : array_like
            The x and y-coordinate(s) of the visual field point(s) (in dva).
        """
        return self.cortex_to_dva(xv2, yv2, zv2)
    
    def v3_to_dva(self, xv3, yv3, zv3):
        """
        Convert points in v3 to dva. Uses the mean of the 5 nearest neighbors
        in the cortical mesh, weighted by 1/distance, to interpolate dva.
        Any points that are more than self.cort_nn_thresh um from the
        nearest neighbor will be set to nan.

        Parameters
        ----------
        xv3, yv3, zv3 : float or array_like
            The x, y, and z-coordinate(s) of the v3 point(s) to look up (in mm).

        Returns
        -------
        x, y : array_like
            The x and y-coordinate(s) of the visual field point(s) (in dva).
        """
        return self.cortex_to_dva(xv3, yv3, zv3)

        

