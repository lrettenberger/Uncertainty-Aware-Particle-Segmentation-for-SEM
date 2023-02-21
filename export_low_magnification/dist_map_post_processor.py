
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import binary_fill_holes
from skimage.segmentation import watershed
from skimage import measure

def get_nucleus_ids(img):
    """ Get nucleus ids in intensity-coded label image.

    :param img: Intensity-coded nuclei image.
        :type:
    :return: List of nucleus ids.
    """

    values = np.unique(img)
    values = values[values > 0]

    return values

class DistMapPostProcessor:
    def __init__(self, 
        sigma_cell,
        th_cell,
        th_seed,
        do_splitting,
        do_area_based_filtering,
        do_fill_holes,
        valid_area_median_factors
        ) -> None:
        self.sigma_cell = sigma_cell
        self.th_cell    = th_cell
        self.th_seed    = th_seed
        self.do_splitting = do_splitting
        self.do_area_based_filtering = do_area_based_filtering
        self.do_fill_holes = do_fill_holes
        self.valid_area_median_factors = valid_area_median_factors

    def process(self, pred_raw, img):
        pred = self._pre_process(pred_raw)
        pred_inst = self._pred2inst(pred)
        pred_inst_final = self._post_process(pred_inst, img)
        return pred_inst_final

    def _pre_process(self, raw_pred):  
        pred = gaussian_filter(raw_pred, sigma=self.sigma_cell)
        return pred

    def _pred2inst(self, pred):
        mask = pred > self.th_cell
        seeds =  pred > self.th_seed
        seeds = measure.label(seeds, background=0)
        if self.do_fill_holes:
            mask = binary_fill_holes(mask)
        pred_inst = watershed(image=-pred, markers=seeds, mask=mask, watershed_line=False)

        if self.do_splitting:
            pred_inst = self._perform_splitting(pred_inst, pred)

        return pred_inst

    def _post_process(self, pred_inst, img):
        if self.do_area_based_filtering:
            pred_inst = self._area_based_filter(pred_inst)

        return pred_inst

    def _area_based_filter(self,pred_inst): 
        area_lst = list()
        region_props = measure.regionprops(pred_inst)
        for prop in region_props:
            area_lst.append(prop.area)
        
        median_area = np.median(area_lst)

        for prop in region_props:
            if not (median_area*self.valid_area_median_factors[0] < prop.area < median_area*self.valid_area_median_factors[1]):
                pred_inst[pred_inst == prop.label] = 0

        pred_inst = measure.label(pred_inst, background=0)
        return pred_inst

    def _perform_splitting(self, pred_inst, pred):
        props = measure.regionprops(pred_inst)
        volumes, nucleus_ids = [], []
        for i in range(len(props)):
            volumes.append(props[i].area)
            nucleus_ids.append(props[i].label)
        volumes = np.array(volumes)
        for i, nucleus_id in enumerate(nucleus_ids):
            if volumes[i] > np.median(volumes) + 2/5 * np.median(volumes):
                nucleus_bin = (pred_inst == nucleus_id)
                cell_prediction_nucleus = pred * nucleus_bin
                for th in [0.50, 0.60, 0.75]:
                    new_seeds = measure.label(cell_prediction_nucleus > th)
                    if np.max(new_seeds) > 1:
                        new_cells = watershed(image=-cell_prediction_nucleus, markers=new_seeds, mask=nucleus_bin,
                                              watershed_line=False)
                        new_ids = get_nucleus_ids(new_cells)
                        for new_id in new_ids:
                            pred_inst[new_cells == new_id] = np.max(pred_inst) + 1
                        break

        return pred_inst