
"""
    Copyright 2017-2022 Department of Electrical and Computer Engineering
    University of Houston, TX/USA
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    Please contact Akshay Sujatha Ravindran for more info about licensing asujatharavindran@uh.edu
    via github issues section.
    **********************************************************************************
    Author:     Akshay Sujatha Ravindran
    Date:       04/20/21 
"""

 

import numpy as np
import torch
from captum.attr import DeepLift,LRP, IntegratedGradients, GuidedBackprop, GuidedGradCam, Deconvolution,InputXGradient,Saliency
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, LayerCAM
def set_seed(seed=None, seed_torch=True):
  """
  Function that controls randomness. NumPy and random modules must be imported.

  Args:
    seed : Integer
      A non-negative integer that defines the random state. Default is `None`.
    seed_torch : Boolean
      If `True` sets the random seed for pytorch tensors, so pytorch module
      must be imported. Default is `True`.

  Returns:
    Nothing.
  """
  if seed is None:
      seed = np.random.choice(2 ** 32)
      np.random.seed(seed)
  if seed_torch:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  print(f'Random seed {seed} has been set.')
  return seed



def ismember(A, B):
    """ Flags elements in A which are present in B
    # Arguments
        A: input matrix A.
        B: input matrix B

    # Returns
        Flag array 
    """
    return [ np.sum(a == B) for a in A ]

def isnotmember(A, B):
    """ Flags the element in A which are not present in B
    # Arguments
        A: input matrix A.
        B: input matrix B

    # Returns
        Flag array 
    """    
    return [ np.sum(a != B) for a in A ]


def get_random_target(target_exclude,num_classes):
    return np.random.choice([i for i in range(num_classes) if i != target_exclude])



def CAM_function(model, method):
    """ Flags the element in A which are not present in B
    # Arguments
        A: input matrix A.
        B: input matrix B

    # Returns
        Flag array 
    """   
    
    
    Explanations = {'Saliency':Saliency(model), 'Deconvolution':Deconvolution(model), 'GuidedBackprop':GuidedBackprop(model),
            'DeepLift':DeepLift(model),'LRP':LRP(model),
            'GuidedGradCam':GuidedGradCam(model, model.conv5), 'InputXGradient':InputXGradient(model) }  
    
    return Explanations[method]
