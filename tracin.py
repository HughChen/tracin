from tensorflow.math import reduce_sum

import tensorflow as tf
import numpy as np

def influence(expl_x, 
              expl_y, 
              refe_x, 
              refe_y, 
              mpath, 
              model, 
              eta,
              num_epochs,
              loss_fn = None):
    """ Basic implementation of TracIn that aims to
    compute influence of explicand (training example) on 
    reference (testing example)
    
    Future work (Speed):
     - Compute gradients with respect to many samples in tf
     
    Future work (Speed at cost of noise)
     - Random projections to speed up computation
     - Allow computation with subsample of checkpoints
     - Fast random projections for dense layers
    
    Args
     - expl_x : explicand features
     - expl_y : explicind labels
     - refe_x : reference features
     - refe_y : reference labels
     - mpath : model path including {} to format epoch iter
     - model : model object for loading weight parameters
     - num_epochs : number of epochs
     - loss_fn : loss function to explain
    
    Returns
     - tracin_sum : measure of influence
        
    """
    # Default to sparse categorical crossentropy
    if not loss_fn:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Compute influences
    influence = np.zeros((expl_x.shape[0],refe_x.shape[0]))
    for epoch_iter in range(1,num_epochs+1):
        
        # Load checkpoint
        model.load_weights(mpath.format(epoch_iter))

        # Compute loss gradients for explicands
        expl_grads = []
        for i in range(expl_x.shape[0]):
            with tf.GradientTape() as tape:
                curr_x = expl_x[i:i+1]
                curr_y = expl_y[i:i+1]
                curr_logits = model(curr_x)
                curr_loss   = loss_fn(curr_y, curr_logits)
            expl_grads.append(tape.gradient(curr_loss, model.trainable_variables))

        # Compute loss gradients for references
        refe_grads = []
        for j in range(refe_x.shape[0]):
            with tf.GradientTape() as tape:
                curr_x = refe_x[j:j+1]
                curr_y = refe_y[j:j+1]
                curr_logits = model(curr_x)
                curr_loss   = loss_fn(curr_y, curr_logits)
            refe_grads.append(tape.gradient(curr_loss, model.trainable_variables))

        # Accumulate sum of gradient dot products
        for i in range(len(expl_grads)):
            expl_grad = expl_grads[i]
            for j in range(len(refe_grads)):
                refe_grad = refe_grads[j]
                for k in range(len(expl_grad)):
                    influence[i,j] += eta*reduce_sum(expl_grad[k]*refe_grad[k])
            
    return(influence)