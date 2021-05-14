import tensorflow as tf


def influence(expl_x, 
              expl_y, 
              refe_x, 
              refe_y, 
              mpath, 
              model, 
              eta):
    """ Compute influence of explicand on reference
    
    Args
     - expl_x : explicand features
     - expl_y : explicind labels
     - refe_x : reference features
     - refe_y : reference labels
     - mpath : model path including {} to format epoch iter
     - model : model object for loading weight parameters
    
    Returns
     - tracin_sum : measure of influence
        
    """
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    tracin_sum = 0
    for epoch_iter in range(1,21):
        # Load checkpoint
        model.load_weights(mpath.format(epoch_iter))

        # Compute loss gradients
        with tf.GradientTape() as tape:
            expl_logits = model(expl_x)
            expl_loss   = loss_fn(expl_y, expl_logits)
        expl_grads = tape.gradient(expl_loss, model.trainable_variables)

        with tf.GradientTape() as tape:
            refe_logits = model(refe_x)
            refe_loss   = loss_fn(refe_y, refe_logits)
        refe_grads = tape.gradient(refe_loss, model.trainable_variables)

        # Accumulate sum of gradient dot products
        for i in range(len(expl_grads)):
            tracin_sum += eta*tf.math.reduce_sum(expl_grads[i]*refe_grads[i])
            
    return(tracin_sum)