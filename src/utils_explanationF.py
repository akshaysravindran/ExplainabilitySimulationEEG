from captum.attr import LayerAttribution
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity
from copy import  deepcopy
from Topoplot_custom import Get_topoplot
import mne
    
    
def attribute_image_features(algorithm, model, input,target_label, g_cam_flag=False):
        model.zero_grad()
        # tensor_attributions = algorithm.attribute(input,
        #                                           target=target_label)
        input_shape = input.shape[-2:]
        if g_cam_flag:
            tensor_attributions = algorithm.attribute(input,
                                                  target=target_label,relu_attributions=False)
            tensor_attributions = LayerAttribution.interpolate(tensor_attributions, input_shape)
            
        else:
            tensor_attributions = algorithm.attribute(input,
                                                  target=target_label)
                
        tensor_out=tensor_attributions.cpu().detach().numpy().squeeze()         
        return tensor_out
    
   

def calc__TOPO_measures( grad_exp : np.ndarray,reference: np.ndarray,fake_evoked,  gt_flag=1, abs_condition = 0, prctile_val=95) -> float:

    """
    Calculate overlap between the model explanation and the true signal heatmap
    Parameters
    ---
    grad_exp (np.ndarray) :  Relevance heatmap 
    reference (np.ndarray): Ground truth 
    data_range            : range of explanations
    abs_condition         : Whether to zero negative values or not
    prctile_val           : top percentile  threshold    
    ---
    Return  Performance measures
    """
    
    grad_exp  = np.mean(grad_exp,axis=2)
    reference = np.mean(reference,axis=2)
    
    explanations, explanations_reference=[], []
    for i in range(np.shape(grad_exp)[0]):   
        explanations.append(Get_topoplot(grad_exp[i],fake_evoked.info))
        explanations_reference.append(Get_topoplot(reference[i],fake_evoked.info))
     
    explanations_reference=np.array(explanations_reference)
    explanations=np.array(explanations)
    
    
    # Ensure the size is the same
    assert explanations_reference.shape == explanations.shape
    
    # Make any values which are nan as zero (no relevance)
    explanations[np.isnan(explanations)]=0
    explanations_reference[np.isnan(explanations_reference)]=0
       
    
    if gt_flag:
        explanations_reference = abs(explanations_reference)
    explanation_abs = deepcopy(explanations)
    if abs_condition:
        explanation_abs = abs(explanation_abs)
    else:
        explanation_abs[explanation_abs<=0] = 0



    
    
    count, k=0,0
    Performance_metric = np.zeros((np.shape(explanations)[0],16))
    for i in range(np.shape(explanations)[0]):     
        # print(i)
        if np.logical_or(np.all(explanation_abs[i]==0),np.all(explanations_reference[i]==0)):
            print('Zeroes detected')
            count=count+1
        else:
            # print(i)
            # Different explanation versions for each example
            explanations_data = deepcopy(explanations[i])        
            explanations_norm_data = (explanations_data-np.min(explanations_data))/(np.max(explanations_data)-np.min(explanations_data))
            
            explanations_abs_data= deepcopy(explanation_abs[i])
            explanations_absnorm_data = (explanations_abs_data-np.min(explanations_abs_data))/(np.max(explanations_abs_data)-np.min(explanations_abs_data))
            
            explanations_ref_data= deepcopy(explanations_reference[i]) 
            explanations_ref_norm_data = (explanations_ref_data-np.min(explanations_ref_data))/(np.max(explanations_ref_data)-np.min(explanations_ref_data))
             
            Exp_top5data=deepcopy(explanations_data)
            Exp_top5data[explanations_data <np.percentile(explanations_data,prctile_val)]=0  
                     
            ExpRef_top5data=deepcopy(explanations_ref_norm_data)
            ExpRef_top5data[explanations_ref_norm_data <np.percentile(explanations_ref_norm_data,prctile_val)]=0    
           
            GT_count = len(np.where(explanations_ref_data>0)[0])
            exp_val = np.sort(explanations_absnorm_data.flatten())[-GT_count]
            
            Exp_topKdata=deepcopy(explanations_absnorm_data)
            Exp_topKdata[explanations_absnorm_data < exp_val]=0  
                               
            # Performance_metric[i,0], pvals         = stats.spearmanr(explanations_norm_data.reshape(-1),explanations_rand_norm_data.reshape(-1),nan_policy='omit')  # Spearman 
            # Performance_metric[i,1], pvalr         = stats.pearsonr(explanations_norm_data.reshape(-1),explanations_rand_norm_data.reshape(-1))  # Pearsons              
            Performance_metric[i,0]             = ssim((ExpRef_top5data), (Exp_top5data), win_size =7 , data_range= (np.nanmax(Exp_top5data) - np.nanmin(Exp_top5data)))    # SSIM_abs
            Performance_metric[i,1]             = ssim(explanations_ref_norm_data, explanations_absnorm_data, win_size = 7, data_range=(np.nanmax(explanations_absnorm_data) - np.nanmin(explanations_absnorm_data))) # SSIM
            Performance_metric[i,2]             = ssim(explanations_ref_norm_data, explanations_norm_data, win_size = 7, data_range= (np.nanmax((explanations_norm_data)) - np.nanmin((explanations_norm_data)))) # SSIM_abs
            Performance_metric[i,3]             = ssim(explanations_ref_norm_data, explanations_data, win_size = 7, data_range= (np.nanmax((explanations_data)) - np.nanmin((explanations_data)))) # SSIM_abs
            
                      
            
            
            Performance_metric[i,4]             = (cosine_similarity(ExpRef_top5data.reshape((1,-1)),Exp_top5data.reshape((1,-1))))[0][0]
            Performance_metric[i,5]             = (cosine_similarity(explanations_ref_norm_data.reshape((1,-1)),explanations_absnorm_data.reshape((1,-1))))[0][0]
            Performance_metric[i,6]             = (cosine_similarity(explanations_ref_norm_data.reshape((1,-1)),explanations_norm_data.reshape((1,-1))))[0][0]
            Performance_metric[i,7]             = (cosine_similarity(explanations_ref_norm_data.reshape((1,-1)),explanations_data.reshape((1,-1))))[0][0]
            
                  
            total_relevancetop5 = Exp_top5data.sum()
            correct_relevance_top5 = np.abs(Exp_top5data[np.array(explanations_ref_data, dtype=bool)]).sum()
            Performance_metric[i,8]                = correct_relevance_top5 / total_relevancetop5           

            
            # Relevance rank accuracy
            total_relevance = len(np.where(Exp_topKdata)[0])
            correct_relevance = len(np.where(Exp_topKdata[np.array(explanations_ref_data, dtype=bool)])[0])
            Performance_metric[i,9]                = correct_relevance / total_relevance
    
            # Relevance mass accuracy
            total_relevance = explanations_norm_data.sum()
            correct_relevance = np.abs(explanations_norm_data[np.array(explanations_ref_data, dtype=bool)]).sum()
            Performance_metric[i,10]                = correct_relevance / total_relevance      

            # Relevance mass accuracy
            total_relevance = explanations_absnorm_data.sum()
            correct_relevance = np.abs(explanations_absnorm_data[np.array(explanations_ref_data, dtype=bool)]).sum()
            Performance_metric[i,11]                = correct_relevance / total_relevance    

             
                     
            Performance_metric[i,12], pvals       =  stats.pearsonr(Exp_topKdata.reshape(-1),ExpRef_top5data.reshape(-1))
            Performance_metric[i,13], pvals       =  stats.pearsonr(explanations_absnorm_data.reshape(-1),explanations_ref_norm_data.reshape(-1))  
            Performance_metric[i,14], pvals       =  stats.pearsonr(explanations_norm_data.reshape(-1),explanations_ref_norm_data.reshape(-1))   
            Performance_metric[i,15], pvals       =  stats.pearsonr(explanations_data.reshape(-1),explanations_ref_norm_data.reshape(-1))  


            del explanations_abs_data,explanations_norm_data, Exp_top5data,ExpRef_top5data,Exp_topKdata, explanations_ref_norm_data
            del total_relevancetop5, correct_relevance_top5,correct_relevance, total_relevance    
            k=k+1     
    return Performance_metric,count




def calc_measures( explanations : np.ndarray,explanations_reference: np.ndarray,  gt_flag=1, abs_condition = 0, prctile_val=95) -> float:

    """
    Calculate overlap between the model explanation and the true signal heatmap
    Parameters
    ---
    explanations          :  Relevance heatmap 
    explanations_reference: Ground truth 
    data_range            : range of explanations
    abs_condition         : Whether to zero negative values or not
    prctile_val           : top percentile  threshold
        

    ---
    Return  Performance measures
    """
    

    # Ensure the size is the same
    assert explanations_reference.shape == explanations.shape
    
    # Make any values which are nan as zero (no relevance)
    explanations[np.isnan(explanations)]=0
    explanations_reference[np.isnan(explanations_reference)]=0
       
    
    if gt_flag:
        explanations_reference = abs(explanations_reference)
    explanation_abs = deepcopy(explanations)
    if abs_condition:
        explanation_abs = abs(explanation_abs)
    else:
        explanation_abs[explanation_abs<=0] = 0



    
    
    count, k=0,0
    Performance_metric = np.zeros((np.shape(explanations)[0],16))
    for i in range(np.shape(explanations)[0]):     
        # print(i)
        if np.logical_or(np.all(explanation_abs[i]==0),np.all(explanations_reference[i]==0)):
            print('Zeroes detected')
            count=count+1
        else:
            # print(i)
            # Different explanation versions for each example
            explanations_data = deepcopy(explanations[i])        
            explanations_norm_data = (explanations_data-np.min(explanations_data))/(np.max(explanations_data)-np.min(explanations_data))
            
            explanations_abs_data= deepcopy(explanation_abs[i])
            explanations_absnorm_data = (explanations_abs_data-np.min(explanations_abs_data))/(np.max(explanations_abs_data)-np.min(explanations_abs_data))
            
            explanations_ref_data= deepcopy(explanations_reference[i]) 
            explanations_ref_norm_data = (explanations_ref_data-np.min(explanations_ref_data))/(np.max(explanations_ref_data)-np.min(explanations_ref_data))
             
            Exp_top5data=deepcopy(explanations_data)
            Exp_top5data[explanations_data <np.percentile(explanations_data,prctile_val)]=0  
                     
            ExpRef_top5data=deepcopy(explanations_ref_norm_data)
            ExpRef_top5data[explanations_ref_norm_data <np.percentile(explanations_ref_norm_data,prctile_val)]=0    
           
            GT_count = len(np.where(explanations_ref_data>0)[0])
            exp_val = np.sort(explanations_absnorm_data.flatten())[-GT_count]
            
            Exp_topKdata=deepcopy(explanations_absnorm_data)
            Exp_topKdata[explanations_absnorm_data < exp_val]=0  
                               
            # Performance_metric[i,0], pvals         = stats.spearmanr(explanations_norm_data.reshape(-1),explanations_rand_norm_data.reshape(-1),nan_policy='omit')  # Spearman 
            # Performance_metric[i,1], pvalr         = stats.pearsonr(explanations_norm_data.reshape(-1),explanations_rand_norm_data.reshape(-1))  # Pearsons              
            Performance_metric[i,0]             = ssim((ExpRef_top5data), (Exp_top5data), win_size =7 , data_range= (np.nanmax(Exp_top5data) - np.nanmin(Exp_top5data)))    # SSIM_abs
            Performance_metric[i,1]             = ssim(explanations_ref_norm_data, explanations_absnorm_data, win_size = 7, data_range=(np.nanmax(explanations_absnorm_data) - np.nanmin(explanations_absnorm_data))) # SSIM
            Performance_metric[i,2]             = ssim(explanations_ref_norm_data, explanations_norm_data, win_size = 7, data_range= (np.nanmax((explanations_norm_data)) - np.nanmin((explanations_norm_data)))) # SSIM_abs
            Performance_metric[i,3]             = ssim(explanations_ref_norm_data, explanations_data, win_size = 7, data_range= (np.nanmax((explanations_data)) - np.nanmin((explanations_data)))) # SSIM_abs
            
                      
            
            
            Performance_metric[i,4]             = (cosine_similarity(ExpRef_top5data.reshape((1,-1)),Exp_top5data.reshape((1,-1))))[0][0]
            Performance_metric[i,5]             = (cosine_similarity(explanations_ref_norm_data.reshape((1,-1)),explanations_absnorm_data.reshape((1,-1))))[0][0]
            Performance_metric[i,6]             = (cosine_similarity(explanations_ref_norm_data.reshape((1,-1)),explanations_norm_data.reshape((1,-1))))[0][0]
            Performance_metric[i,7]             = (cosine_similarity(explanations_ref_norm_data.reshape((1,-1)),explanations_data.reshape((1,-1))))[0][0]
            
                  
            total_relevancetop5 = Exp_top5data.sum()
            correct_relevance_top5 = np.abs(Exp_top5data[np.array(explanations_ref_data, dtype=bool)]).sum()
            Performance_metric[i,8]                = correct_relevance_top5 / total_relevancetop5           

            
            # Relevance rank accuracy
            total_relevance = len(np.where(Exp_topKdata)[0])
            correct_relevance = len(np.where(Exp_topKdata[np.array(explanations_ref_data, dtype=bool)])[0])
            Performance_metric[i,9]                = correct_relevance / total_relevance
    
            # Relevance mass accuracy
            total_relevance = explanations_norm_data.sum()
            correct_relevance = np.abs(explanations_norm_data[np.array(explanations_ref_data, dtype=bool)]).sum()
            Performance_metric[i,10]                = correct_relevance / total_relevance      

            # Relevance mass accuracy
            total_relevance = explanations_absnorm_data.sum()
            correct_relevance = np.abs(explanations_absnorm_data[np.array(explanations_ref_data, dtype=bool)]).sum()
            Performance_metric[i,11]                = correct_relevance / total_relevance    

             
                     
            Performance_metric[i,12], pvals       =  stats.pearsonr(Exp_topKdata.reshape(-1),ExpRef_top5data.reshape(-1))
            Performance_metric[i,13], pvals       =  stats.pearsonr(explanations_absnorm_data.reshape(-1),explanations_ref_norm_data.reshape(-1))  
            Performance_metric[i,14], pvals       =  stats.pearsonr(explanations_norm_data.reshape(-1),explanations_ref_norm_data.reshape(-1))   
            Performance_metric[i,15], pvals       =  stats.pearsonr(explanations_data.reshape(-1),explanations_ref_norm_data.reshape(-1))   
       
            
            del explanations_abs_data,explanations_norm_data, Exp_top5data,ExpRef_top5data,Exp_topKdata, explanations_ref_norm_data
            del total_relevancetop5, correct_relevance_top5,correct_relevance, total_relevance    
            k=k+1     
    return Performance_metric,count