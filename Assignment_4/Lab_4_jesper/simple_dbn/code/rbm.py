from util import *
import sys
import matplotlib.pyplot as plt

class RestrictedBoltzmannMachine():
    '''
    For more details : A Practical Guide to Training Restricted Boltzmann Machines https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    '''
    def __init__(self, ndim_visible, ndim_hidden, is_bottom=False, image_size=[28,28], is_top=False, n_labels=10, batch_size=10):

        """
        Args:
          ndim_visible: Number of units in visible layer.
          ndim_hidden: Number of units in hidden layer.
          is_bottom: True only if this rbm is at the bottom of the stack in a deep belief net. Used to interpret visible layer as image data with dimensions "image_size".
          image_size: Image dimension for visible layer.
          is_top: True only if this rbm is at the top of stack in deep beleif net. Used to interpret visible layer as concatenated with "n_label" unit of label data at the end.
          n_label: Number of label categories.
          batch_size: Size of mini-batch.
        """

        self.ndim_visible = ndim_visible

        self.ndim_hidden = ndim_hidden

        self.is_bottom = is_bottom

        if is_bottom : self.image_size = image_size

        self.is_top = is_top

        if is_top : self.n_labels = 10

        self.batch_size = batch_size

        self.delta_bias_v = 0

        self.delta_weight_vh = 0

        self.delta_bias_h = 0

        self.bias_v = np.zeros((self.ndim_visible))

        self.weight_vh = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible,self.ndim_hidden))

        self.bias_h = np.zeros((self.ndim_hidden))

        self.delta_weight_v_to_h = 0

        self.delta_weight_h_to_v = 0

        self.weight_v_to_h = None

        self.weight_h_to_v = None

        self.learning_rate = 0.01

        self.momentum = 0.7

        self.weight_decay = 0.01

        self.print_period = 5000

        self.rf = { # receptive-fields. Only applicable when visible layer is input data
            "period" : 5000, # iteration period to visualize
            "grid" : [5,5], # size of the grid
            "ids" : np.random.randint(0,self.ndim_hidden,25) # pick some random hidden units
            }

        return


    def cd1(self,visible_trainset,plot=False, n_iterations=10000):

        """Contrastive Divergence with k=1 full alternating Gibbs sampling

        Args:
          visible_trainset: training data for this rbm, shape is (size of training set, size of visible layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        print ("learning CD1")
        n_samples = visible_trainset.shape[0]
        p = np.mean(visible_trainset, axis=0) # unactivated pixels will be = 0
        self.bias_v = np.log(p / (1-p)) # -> -Inf here. asserts that unactivated pixels will never be generated

        if self.is_bottom:
            viz_rf(weights=self.weight_vh[:,self.rf["ids"]].reshape((self.image_size[0],self.image_size[1],-1)), it=99, grid=self.rf["grid"])

        errors = []
        mbs=0
        for iter in range(1, n_iterations+1):
            # [DONE TASK 4.1] run k=1 alternating Gibbs sampling : v_0 -> h_0 ->  v_1 -> h_1.
            # you may need to use the inference functions 'get_h_given_v' and 'get_v_given_h'.
            # note that inference methods returns both probabilities and activations (samples from probablities) and you may have to decide when to use what.
            mini_batch = visible_trainset[mbs:mbs+self.batch_size,:]
            mbs += self.batch_size # update batch index for next iteration
            if mbs >= n_samples:
                mbs = mbs - n_samples

            v_0 = mini_batch
            h_0, h_activation = self.get_h_given_v(v_0) # first hidden sample is activated

            v_k = self.get_v_given_h(h_activation)[0] # Reconstruction samples are real
            h_k = self.get_h_given_v(v_k)[0]
            # [TODO TASK 4.1] update the parameters using function 'update_params'
            # if iter < int(n_iterations*0.1):
            #     self.momentum = 0.5
            # else:
            #     self.momentum = 0.9

            
            self.update_params(v_0,h_0,v_k,h_k)
            errors.append(np.linalg.norm(v_0 - v_k))

            if not iter%1000:
                print("Iteration {} / {}".format(iter, n_iterations))
                """
                plots[n].plot(errors[n], label="Error in epoch " + str(n+1))
                plots[n].set_title("Error in epoch " + str(n+1))
                plots[n].set_xlabel("Mini batch iteration")
                plots[n].set_ylabel("Reconstruction error")
                fig.show()
                """
                if plot:
                    plt.plot(errors, label="Error at iteration " + str(iter))
                    plt.title("Error during iteration {}".format(iter))
                    plt.xlabel("Mini batch iteration")
                    plt.ylabel("Reconstruction error")
                    plt.show()

                if self.is_bottom:
                    viz_rf(weights=self.weight_vh[:,self.rf["ids"]].reshape((self.image_size[0],self.image_size[1],-1)), it=iter, grid=self.rf["grid"])

                    print ("iteration=%7d recon_loss=%4.4f"%(iter, np.linalg.norm(mini_batch-v_k)))
            sys.stdout.flush()
            # visualize once in a while when visible layer is input images
        return


    def update_params(self,v_0,h_0,v_k,h_k):

        """Update the weight and bias parameters.

        You could also add weight decay and momentum for weight updates.

        Args:
           v_0: activities or probabilities of visible layer (data to the rbm)
           h_0: activities or probabilities of hidden layer
           v_k: activities or probabilities of visible layer
           h_k: activities or probabilities of hidden layer
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.1] get the gradients from the arguments (replace the 0s below) and update the weight and bias parameters

        delta_bias_v = np.mean(self.learning_rate*(v_0-v_k),axis=0)
        delta_bias_h = np.mean(self.learning_rate*(h_0-h_k),axis=0)
        delta_weight_vh = self.learning_rate * (v_0.T @ h_0 - v_k.T @ h_k) / v_0.shape[0]
        # print(self.delta_bias_h.shape, self.delta_bias_v.shape, self.delta_weight_vh.shape)

        mu = self.momentum
        delta_bias_h = mu * self.delta_bias_h + (1-mu) * delta_bias_h
        delta_bias_v = mu * self.delta_bias_v + (1-mu) * delta_bias_v
        delta_weight_vh = mu * self.delta_weight_vh + (1-mu) * delta_weight_vh

        self.bias_v += delta_bias_v
        self.bias_h += delta_bias_h
        self.weight_vh += delta_weight_vh

        # Weight decay
        # self.weight_vh *= 1 - self.learning_rate * self.weight_decay

        self.delta_bias_h = delta_bias_h
        self.delta_bias_v = delta_bias_v
        self.delta_weight_vh = delta_weight_vh

        return

    def get_h_given_v(self,visible_minibatch):

        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses undirected weight "weight_vh" and bias "bias_h"

        Args:
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:
           tuple ( p(h|v) , h)
           both are shaped (size of mini-batch, size of hidden layer)
        """

        assert self.weight_vh is not None

        n_samples = visible_minibatch.shape[0]

        # [DONE TASK 4.1] compute probabilities and activations (samples from probabilities) of hidden layer (replace the zeros below)
        h_prob = sigmoid(self.bias_h +  visible_minibatch @ self.weight_vh )
        h_activation = sample_binary(h_prob)
        return h_prob, h_activation


    def get_v_given_h(self,hidden_minibatch):

        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses undirected weight "weight_vh" and bias "bias_v"

        Args:
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:
           tuple ( p(v|h) , v)
           both are shaped (size of mini-batch, size of visible layer)
        """

        assert self.weight_vh is not None

        n_samples = hidden_minibatch.shape[0]

        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """
            # [DONE TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass below). \
            # Note that this section can also be postponed until TASK 4.2, since in this task, stand-alone RBMs do not contain labels in visible layer.
            total_input = self.bias_v + hidden_minibatch @ self.weight_vh.T
            v_images_prob = sigmoid(total_input[:, :-self.n_labels])
            v_label_prob = softmax(total_input[:, -self.n_labels:])
            v_prob = np.concatenate((v_images_prob,v_label_prob),axis=1)

            v_images_activation = sample_binary(v_images_prob) #image activation is allowed to be sample_binary
            v_label_activation = sample_categorical(v_label_prob)
            v_activation = np.concatenate((v_images_activation, v_label_activation), axis=1)
            return v_prob,v_activation

        #else:
        # [DONE TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass and zeros below)
        v_prob = sigmoid(self.bias_v + hidden_minibatch @ self.weight_vh.T)
        v_activation = sample_binary(v_prob)
        return v_prob, v_activation



    """ rbm as a belief layer : the functions below do not have to be changed until running a deep belief net """



    def untwine_weights(self):

        self.weight_v_to_h = np.copy( self.weight_vh )
        self.weight_h_to_v = np.copy( np.transpose(self.weight_vh) )
        self.weight_vh = None

    def get_h_given_v_dir(self,visible_minibatch):

        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses directed weight "weight_v_to_h" and bias "bias_h"

        Args:
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:
           tuple ( p(h|v) , h)
           both are shaped (size of mini-batch, size of hidden layer)
        """

        assert self.weight_v_to_h is not None

        n_samples = visible_minibatch.shape[0]

        # [TODO TASK 4.2] perform same computation as the function 'get_h_given_v' but with directed connections (replace the zeros below)
        h_prob = sigmoid(self.bias_h +  visible_minibatch @ self.weight_v_to_h)
        h_activation = sample_binary(h_prob)

        return h_prob, h_activation


    def get_v_given_h_dir(self,hidden_minibatch):


        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses directed weight "weight_h_to_v" and bias "bias_v"

        Args:
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:
           tuple ( p(v|h) , v)
           both are shaped (size of mini-batch, size of visible layer)
        """

        assert self.weight_h_to_v is not None

        n_samples = hidden_minibatch.shape[0]

        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """
            total_input = self.bias_v + hidden_minibatch @ self.weight_h_to_v
            v_label_prob = softmax(total_input[:, -self.n_labels:])
            v_images_prob = sigmoid(total_input[:, :-self.n_labels])
            v_prob = np.concatenate((v_images_prob, v_label_prob), axis=1)

        else:
            v_prob = sigmoid(self.bias_v + hidden_minibatch @ self.weight_h_to_v)

        v_activation = sample_binary(v_prob)
        return v_prob, v_activation

    def update_generate_params(self,inps,trgs,preds):

        """Update generative weight "weight_h_to_v" and bias "bias_v"

        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.
        self.delta_weight_h_to_v = self.learning_rate * trgs.T @ (inps - preds) / inps.shape[0]
        self.delta_bias_v = np.mean(self.learning_rate*(inps-preds),axis=0)

        self.weight_h_to_v += self.delta_weight_h_to_v
        self.bias_v += self.delta_bias_v

        return

    def update_recognize_params(self,inps,trgs,preds):

        """Update recognition weight "weight_v_to_h" and bias "bias_h"

        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.
        self.delta_weight_v_to_h = self.learning_rate * trgs.T @ (inps - preds) / inps.shape[0]
        self.delta_bias_h = np.mean(self.learning_rate*(inps - preds),axis=0)
        self.weight_v_to_h += self.delta_weight_v_to_h
        self.bias_h += self.delta_bias_h

        return
