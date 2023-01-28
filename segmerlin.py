#---DEPENDENCIES---------------------------------------------------------------
import subprocess
subprocess.run(["pip", "install", "lightgbm==2.3.1"])
subprocess.run(["pip", "install", "dill"])

import cv2
import numpy as np
import dill
import lightgbm as lgb

#---INFO-----------------------------------------------------------------------
__version__ = 'segmerlin_v1.2'
print(f'>>> SegMerlin Update: Package installed successfully; version-- {__version__}', '\n')

#---SEGMERLIN------------------------------------------------------------------

LGBM_DEFAULTS = {'boosting_type': 'dart',
                'objective': 'binary',
                'metric': 'binary_logloss',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 0,
                'rate_drop': 0.1,
                'num_boost_round': 100}

class SegMerlinWLGBMS:
    
    def __init__(self, feature_extractor = None, init_model = None, input_shape = (720, 1280)):
        if init_model != None:
            with open(init_model, 'rb') as file:
                self.__dict__.update(dill.load(file).__dict__)
            if self.featurex_flag == 0 and feature_extractor != None:
                self.featurex_backbone = None
                print('>>> SegMerlin Alert: Initiated model does not use a feature extractor backbone; self.featurex_backbone set to None', '\n')
            elif self.featurex_flag == 1 and feature_extractor == None:
                print('>>> Segmerlin Alert: Initiated model uses a feature extractor backbone; No backbone attached during initiation', '\n')
                
        else:
            self.Predictors = []
            self.featurex_backbone = None
            self.featurex_flag = 0
            self.input_shape = input_shape
            
    def add_predictor(self, Name, tile_shape, tile_thresh = 0.8, lgbm_params = LGBM_DEFAULTS, pre_process_func = None, pos_process_func = None):
        fxb = self.featurex_backbone
         
        class WLGBMSegmenter:
            
            def __init__(self):
                self.Name = Name
                self.lgbm_params = lgbm_params
                self.tile_shape = tile_shape
                self.pre_process_func = pre_process_func
                self.pos_process_func = pos_process_func
                self.tile_thresh = tile_thresh
                self.pass_thresh = int(self.tile_shape[0] * self.tile_shape[1] * self.tile_thresh)
                self.wlgbmclf = None
                
            def tileer(self, im_lis, ti_sha1):
                M = ti_sha1[0]
                N = ti_sha1[1]
                I = [[i[x:x+M,y:y+N] for x in range(0,i.shape[0],M) for y in range(0,i.shape[1],N)] for i in im_lis]

                return np.array(I)

            def dsr_prep(self, image_list, mask_list = None):
                if self.pre_process_func != None: 
                    image_list = pre_process_func(image_list)
                    
                if fxb != None:
                    image_list = fxb(image_list)
                    
                I = self.tileer(im_lis = image_list, ti_sha1 = self.tile_shape)

                ds = [c.reshape(c.shape[0], -1) for c in I]
                ds = list(np.concatenate(ds))
                
                r_ = None
                if mask_list != None:
                    M = self.tileer(im_lis = mask_list, ti_sha1 = self.tile_shape)
                    r_ = [255 * int((c == 255).sum() >= self.pass_thresh) for m in M for c in m]

                return ds, r_
            
        model = WLGBMSegmenter()
        self.Predictors.append(model)
            
    def fit_predictor(self, predictor_names, images, masks, fit_type = 'refit', nb_rounds_for_trainfit = None, verbose = 0):
        if verbose == 1: print('>>> SegMerlin Update: fitting predictors ...', '\n')
        
        predictors_to_fit = [M for M in self.Predictors if M.Name in predictor_names]
        for predictor in predictors_to_fit:
            ds, r_ = predictor.dsr_prep(images, masks)
            ds, r_ = np.array(ds), np.array(r_)
            
            if predictor.wlgbmclf == None:
                wlgbmclf = lgb.train(predictor.lgbm_params, lgb.Dataset(ds, r_))
                if verbose == 1 and fit_type != 'train':
                    print(">>> SegMerlin Alert: firsthand initiation of LGBM Model; using fit_type == 'train'", '\n')
                predictor.wlgbmclf = wlgbmclf
            else:
                if fit_type == 'refit': predictor.wlgbmclf.refit(ds, r_)
                elif fit_type == 'update': predictor.wlgbmclf.update(ds, r_)
                elif fit_type == 'train':
                    par_ = predictor.lgbm_params
                    if nb_rounds_for_trainfit != None:
                        par_['num_boost_round'] = nb_rounds_for_trainfit
                        new_wlgbmclf = lgb.train(par_, lgb.Dataset(ds, r_), init_model = predictor.wlgbmclf, keep_training_booster = True)
                        predictor.wlgbmclf = new_wlgbmclf
                    else:
                        print('>>> SegMerlin Error: nb_rounds_for_trainfit not specified, this will be the number of trees added to predictor.wlgbmclf LGBM Model; please specify', '\n')
            
            if verbose == 1:
                print(f'>>> SegMerlin Update: predictor-- {predictor.Name} fitted', '\n')
            
        if verbose == 1: print('>>> SegMerlin Update: All specified predictors fitted', '\n')
        
    def predict(self, image_list, predictor_names, filter_strength = 255 / 1.5, predictor_weights = None, voting_type = 'majority'):
        predictors_that_predict = [M for M in self.Predictors if M.Name in predictor_names]
        
        def inner_predict(single_image, predictor_ref):
            ds, r_ = predictor_ref.dsr_prep(single_image)
            pred = predictor_ref.wlgbmclf.predict(ds)
            # LGBM Models predict probablity of True class, multiplying with 255 to maintain format
            pred = pred * 255
            
            pred = pred.reshape((-1, 
                                 int(self.input_shape[0] / predictor_ref.tile_shape[0]), 
                                 int(self.input_shape[1] / predictor_ref.tile_shape[1])
                                 ))
            
            # Convert to binary mask
            pred = [cv2.inRange(i, filter_strength, 255) for i in pred]
            pred = [cv2.resize(k, (self.input_shape[1], self.input_shape[0]), interpolation = cv2.INTER_NEAREST) for k in pred]
            
            if predictor_ref.pos_process_func != None: 
                 pred = predictor_ref.pos_process_func(pred)
                 # Convert to binary mask; revision needed, filter_strength should be a different value
                 pred = [cv2.inRange(i, filter_strength, 255) for i in pred]
            
            return pred
        
        def majority_vote(masks):
            op = np.zeros_like(masks[0])
            for mask in masks:
                op += mask
            op = np.where(op >= len(masks) * 255 / 2.3, 255, 0)
            return op

        def weighted_vote(masks, weights = predictor_weights):
            op = np.zeros_like(masks[0])
            for i in range(len(masks)):
                op += masks[i] * weights[i]
            op = np.where(op >= 255 / 2.5, 255, 0)
            return op
        
        if voting_type == 'weighted': vote = weighted_vote
        elif voting_type == 'majority': vote = majority_vote
        
        predictions = []
        inter_ = []
        for img in image_list:
            p = [inner_predict([img], predic)[0] for predic in predictors_that_predict]
            q = vote(p)
            inter_.append(p)
            predictions.append(q)
            
        return predictions, inter_
    
    def summary(self, summary_type = 'concise'):
        print('>>> SegMerlin Summary:','\n')
        if summary_type == 'exhaustive':
            print("SegMerlin Class Parameters:")
            print("----------------------------")
            print("featurex_backbone: ", self.featurex_backbone)
            print("featurex_flag: ", self.featurex_flag)
            print("\n")
            print("Predictors:")
            print("----------------------------")
            for i, predictor in enumerate(self.Predictors):
                print("Predictor ", i+1, ":")
                print("Name: ", predictor.Name)
                print("lgbm_params: ", predictor.lgbm_params)
                print("tile_shape: ", predictor.tile_shape)
                print("pre_process_func: ", predictor.pre_process_func)
                print("pos_process_func: ", predictor.pos_process_func)
                print("tile_thresh: ", predictor.tile_thresh)
                print("\n")
        elif summary_type == 'concise':
            print("Predictors:")
            print("----------------------------")
            for i, predictor in enumerate(self.Predictors):
                print("Predictor ", i+1, ":")
                print("Name: ", predictor.Name)
                print("tile_shape: ", predictor.tile_shape)
                print("\n")
        else:
            print("SegMerlin Error: Invalid summary_type. Please choose 'exhaustive' or 'concise'.", '\n')

    def save_model(self, path):
        if self.featurex_backbone != None:
            print(">>> SegMerlin Alert: save_model does not save feature extractor backbone; self.featurex_backbone set to None before saving, please save your reference manually", '\n')
            self.featurex_backbone = None
        with open(path, 'wb') as file:
            dill.dump(self, file)
        print(">>> SegMerlin Update: Model saved successfully to ", path, '\n')