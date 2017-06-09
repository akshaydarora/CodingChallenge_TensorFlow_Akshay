
# Multiorder Regression using TensorFlow with Tensorboard
# Author : Akshay Arora
# Importing all important libraries



import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Making the given data

n_observations=100
xs=np.linspace(-3,3,n_observations)
ys=np.sin(xs) + np.random.uniform(-0.5,0.5,n_observations)


# Tensorboard :: Name_scope helps in defining tensor op. 
# Here, it generates an  input node in the graph
# 
# code :: TensorFlow provides a placeholder operation that must be fed with data on execution with size 32 bit of datatype float 


with tf.name_scope('inputs'):
    X=tf.placeholder(tf.float32,name="XdataInput")
    Y=tf.placeholder(tf.float32,name="YdataInput")


# code :: Initializing the prediction with a tensorflow constant bias

with tf.name_scope('Biases'):
    prediction=tf.Variable(tf.constant(0.1,shape=[1],name='Biases'))


# Tensorboard :: Here we use tf.summary.histogram() command which produces the histogram of all the weights in the graph , thus making it possible to visualize the weight's distribution . Also, the weightname generates different weight nodes in the graph 
# 
# code :: Choosing Different orders of polynomials to train our weights and making the prediction

poly_orders=[1,2,3,4]
for orders in range(1,len(poly_orders)+1):
    weightname='weight%d'% (orders)
    with tf.name_scope(weightname):
        W=tf.Variable(tf.truncated_normal([1],stddev=0.1),name='weight')
        prediction=tf.add(tf.multiply(X**orders,W),prediction)
        tf.summary.histogram('weightname',W)


# Tensorboard :: Here loss is added to the graph and 'tf.summary.scalar' command will help to visualize the loss over all the epochs
# 
# code :: Defining the loss function as min((1/n)*[predicted-y]2) and regularizing it using L2-regularization


with tf.name_scope('loss'):
    loss =(1/n_observations)*tf.reduce_mean(tf.square(prediction-Y))
    # Regularizing the loss function using L2 -Ridge regularization by 
    # penalizing large weights
    beta=0.0001   # factor for l2-norm
    regularizer = tf.nn.l2_loss(W)
    loss = tf.reduce_mean(loss + beta * regularizer)
    tf.summary.scalar('loss',loss)


# Tensorboard :: Here train node is added to the graph which includes our Adam(Adaptive Momentum) Gradient Descent algorithm
# 
# code :: every parameter updates with Adam Gradient Descent algorithm. This optmization algorithm calculates adaptive learning rates for each parameter

with tf.name_scope('train'):
    learning_rate=0.001
    train_step=tf.train.AdamOptimizer(learning_rate).minimize(loss)


# code :: Plotting the figure and adding details using matplotlib


fig = plt.figure()
ax = plt.subplot(1,1,1)
ax.set_title('Given Data')
ax.set_xlabel('x values linearly spaced')
ax.set_ylabel('noisy sinusoid')
ax.grid(color='k', linestyle='-', linewidth=0.1)
ax.set_ylim([-3,3])
plt.scatter(xs, ys,marker='o',color='b')
plt.ion()
plt.show()


# Tensorboard :: The 'tf.summary.merge_all()' command merges all the summaries we defined above in the graph and runs in the tensorflow session and then we can write it to a destination folder using 'tf.summary.FileWriter' command 
# 
# code ::Initializing the tensorboard session and initializing all the global variables . The next step is run this for all epochs and estimate the training loss and see the fitted data and simultaneously record the summary of the result.  
# 
# Tensor Flow Session

nb_epochs=1000
sess= tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('logs/', sess.graph) 
init=tf.global_variables_initializer()
sess.run(init)
    
for i in range(nb_epochs):
    for (x,y) in zip(xs,ys):
        sess.run(train_step,feed_dict={X:x,Y:y})

        if i %10==0:
            result = sess.run(merged,feed_dict={X: x, Y: y})
            writer.add_summary(result,i)
            fitted_data=prediction.eval(feed_dict={X: xs},session=sess)
            ax.plot(xs,fitted_data,'r',alpha =i/nb_epochs,linewidth=0.5)
            plt.draw()
    cost_train=sess.run(loss,feed_dict={X:xs,Y:ys})
    print(cost_train)
red_patch = mpatches.Patch(color='red', label='fitting the data')
ax.legend(handles=[red_patch])
ax.set_ylim([-3,3])
ax.set_title('Fitting Data with different orders')
ax.grid(color='k', linestyle='-', linewidth=0.1)
fig



# # Tensorboard Instructions
# 
# Go to command line cd to working directory and type:
# 
# working dir/> python CodingChallenge_TensorFlow_Akshay.py
# 
# After compiling , logs folder will be generated in working directory
# 
# working dir/> tensorboard --logdir=logs
# 
